"""Explicit T1/T2 engineering trainer; constructing it never starts training."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..model import KaloscopeBackbone
from ..model.registry import LOGICAL_DOMAINS
from .checkpoint import CHECKPOINT_VERSION, read_payload, save_payload
from .losses import LossConfig, differentiable_zero, multitask_loss


@dataclass(frozen=True)
class TrainerConfig:
    stage: str = 'frozen_heads'
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    accumulation_steps: int = 1
    amp: str = 'off'  # off, bf16, fp16 (CUDA only)
    gradient_clip: float | None = 1.0
    preservation: dict = field(default_factory=lambda: {'global_features': 0.0, 'style_embedding': 0.0, 'artist_logits': 0.0})
    identity_metric_weight: float = 0.0

    def __post_init__(self):
        if self.stage not in ('frozen_heads', 'finetune_last4', 'T1', 'T2'):
            raise ValueError('Unknown training stage')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid optimizer settings')
        if not isinstance(self.accumulation_steps, int) or self.accumulation_steps < 1 or self.amp not in ('off', 'bf16', 'fp16'):
            raise ValueError('Invalid accumulation or AMP settings')
        if self.gradient_clip is not None and (not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0):
            raise ValueError('gradient_clip must be positive or None')
        if set(self.preservation) - {'global_features', 'style_embedding', 'artist_logits'}:
            raise ValueError('Unknown preservation loss')
        if any(not math.isfinite(w) or w < 0 for w in (*self.preservation.values(), self.identity_metric_weight)):
            raise ValueError('Auxiliary weights must be finite and nonnegative')


def _chunked_artist_kd(student, teacher, chunk_size):
    def normalizer(logits):
        total = None
        for part in logits.split(chunk_size, dim=-1):
            value = checkpoint(lambda x: x.float().logsumexp(-1), part, use_reentrant=False) if part.requires_grad else part.float().logsumexp(-1)
            total = value if total is None else torch.logaddexp(total, value)
        return total
    student_z, teacher_z = normalizer(student), normalizer(teacher)
    total = differentiable_zero(student)
    for start in range(0, student.shape[-1], chunk_size):
        def divergence(prediction, target, p_z, t_z):
            p_log = prediction.float() - p_z[:, None]
            t_log = target.float() - t_z[:, None]
            return (t_log.exp() * (t_log - p_log)).sum()
        args = (student[:, start:start + chunk_size], teacher[:, start:start + chunk_size], student_z, teacher_z)
        total = total + checkpoint(divergence, *args, use_reentrant=False)
    return total / student.shape[0]


class Trainer:
    def __init__(self, model, metadata, *, config=None, loss_config=None, device='cpu',
                 teacher=None, identity_metric=None):
        self.config, self.loss_config = config or TrainerConfig(), loss_config or LossConfig()
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda') or self.config.amp == 'fp16' and self.device.type != 'cuda':
            raise ValueError('Trainer supports CPU/CUDA; FP16 AMP requires CUDA')
        self.model = model.to(self.device)
        self.model.set_other_artist_mode(self.loss_config.other_artist_mode)
        self.model.set_stage(self.config.stage)
        self.metadata = dict(metadata)
        if self.metadata.get('head_layout') != self.model.layout.to_dict() or self.metadata.get('stage_config') != {'stage': self.model.backbone.stage}:
            raise ValueError('Trainer metadata must describe the actual model layout and stage')
        enabled = any(self.config.preservation.values())
        if enabled != (teacher is not None):
            raise ValueError('Nonzero preservation requires an explicit original teacher; zero weights require teacher=None')
        if self.config.identity_metric_weight and identity_metric is None:
            raise ValueError('Identity metric weight requires a caller-supplied metric callable')
        self.identity_metric = identity_metric
        self.teacher = None
        if teacher is not None:
            if not isinstance(teacher, KaloscopeBackbone):
                raise ValueError('Teacher must be an independently loaded original KaloscopeBackbone')
            student_parameters = {id(parameter) for parameter in model.parameters()}
            if any(id(parameter) in student_parameters for parameter in teacher.parameters()):
                raise ValueError('Teacher and student must not share parameters')
            student_storage = {parameter.data_ptr() for parameter in model.parameters()}
            if any(parameter.data_ptr() in student_storage for parameter in teacher.parameters()):
                raise ValueError('Teacher and student must not share parameter storage')
            if getattr(teacher.legacy, 'class_mapping', None) != getattr(model.backbone.legacy, 'class_mapping', None):
                raise ValueError('Teacher/student original Artist mappings disagree')
            if (teacher.global_dim, teacher.style_dim, teacher.artist_classes) != (model.backbone.global_dim, model.backbone.style_dim, model.backbone.artist_classes):
                raise ValueError('Teacher/student Kaloscope dimensions disagree')
            self.teacher = teacher.to(self.device).set_stage('frozen_heads').eval()
        self.optimizer = self._new_optimizer()
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.config.amp == 'fp16')
        self.epoch, self.global_step, self.micro_step, self.pending = 0, 0, 0, 0

    def _new_optimizer(self):
        return torch.optim.AdamW((parameter for parameter in self.model.parameters() if parameter.requires_grad),
                                 lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def set_stage(self, stage):
        if self.pending:
            raise ValueError('Change stages only at an optimizer boundary')
        old = self.optimizer
        self.model.set_stage(stage)
        self.optimizer = self._new_optimizer()
        for group in self.optimizer.param_groups:
            for parameter in group['params']:
                if parameter in old.state:
                    self.optimizer.state[parameter] = old.state[parameter]
        self.metadata['stage_config'] = {'stage': self.model.backbone.stage}

    def _autocast(self):
        if self.config.amp == 'off':
            return nullcontext()
        dtype = torch.bfloat16 if self.config.amp == 'bf16' else torch.float16
        return torch.autocast(self.device.type, dtype=dtype)

    def _batch(self, batch):
        return (batch['images'].to(self.device),
                {key: value.to(self.device) for key, value in batch['targets'].items()},
                {key: value.to(self.device) for key, value in batch['masks'].items()})

    def train_batch(self, batch, *, sampler=None):
        self.model.train()
        images, targets, masks = self._batch(batch)
        selection = set(targets)
        weights = self.config.preservation
        if weights.get('artist_logits', 0):
            selection.add('artist')
        if weights.get('style_embedding', 0):
            selection.add('style_embedding')
        with self._autocast():
            output = self.model(images, selection)
            losses = multitask_loss(output.logits, targets, self.loss_config, masks)
            total = losses.total
            auxiliary = {}
            if self.teacher is not None:
                self.teacher.eval()
                with torch.no_grad():
                    reference = self.teacher(images, include_artist=bool(weights.get('artist_logits', 0)),
                                             include_style=bool(weights.get('style_embedding', 0)))
                for name in ('global_features', 'style_embedding'):
                    if weights.get(name, 0):
                        auxiliary[name] = F.mse_loss(getattr(output.features, name).float(), getattr(reference, name).float())
                if weights.get('artist_logits', 0):
                    auxiliary['artist_logits'] = _chunked_artist_kd(output.features.artist_logits, reference.artist_logits, self.loss_config.chunk_size)
                total = total + sum(weights[name] * value for name, value in auxiliary.items())
            if self.config.identity_metric_weight:
                if 'identity_relations' not in batch:
                    raise ValueError('Enabled identity metric requires caller-provided identity_relations')
                features = self.model.identity.metric_features(output.features.global_features)
                metric = self.identity_metric(features, batch['identity_relations'])
                if metric.ndim != 0:
                    raise ValueError('Identity metric must return a scalar tensor')
                auxiliary['identity_metric'] = metric
                total = total + self.config.identity_metric_weight * metric
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite training loss')
        self.scaler.scale(total / self.config.accumulation_steps).backward()
        self.pending += 1
        self.micro_step += 1
        if sampler is not None:
            sampler.advance()
        if self.pending == self.config.accumulation_steps:
            self.flush()
        return {'loss': float(total.detach()), 'by_group': {name: float(value.detach()) for name, value in losses.by_group.items()},
                'valid_counts': losses.valid_counts, 'general_statistics': losses.general_statistics,
                'auxiliary': {name: float(value.detach()) for name, value in auxiliary.items()}}

    def flush(self):
        """Commit a full or final partial accumulation; checkpointing is now safe."""
        if not self.pending:
            return
        self.scaler.unscale_(self.optimizer)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if self.pending < self.config.accumulation_steps:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(self.config.accumulation_steps / self.pending)
        # One aggregate norm pass, never a Python/GPU synchronization per tensor.
        # FP16 overflow was recorded by unscale_ and GradScaler will skip the step.
        torch.nn.utils.clip_grad_norm_(
            parameters, self.config.gradient_clip if self.config.gradient_clip is not None else float('inf'),
            error_if_nonfinite=not self.scaler.is_enabled())
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = 0
        if self.scaler.get_scale() >= old_scale:
            self.global_step += 1

    def train_epoch(self, loader, *, sampler=None):
        total, batches = 0.0, 0
        for batch in loader:
            result = self.train_batch(batch, sampler=sampler)
            total += result['loss']
            batches += 1
        self.flush()
        self.epoch += 1
        if sampler is not None:
            sampler.set_epoch(self.epoch)
        return {'batches': batches, 'mean_loss': total / batches if batches else 0.0}

    @torch.no_grad()
    def evaluate(self, loader):
        was_training = self.model.training
        self.model.eval()
        totals, counts, metrics = {}, {}, {}
        try:
            for batch in loader:
                images, targets, masks = self._batch(batch)
                with self._autocast():
                    output = self.model(images, set(targets))
                    result = multitask_loss(output.logits, targets, self.loss_config, masks)
                for name, loss in result.by_group.items():
                    count = result.valid_counts[name]
                    totals[name] = totals.get(name, 0.0) + float(loss) * count
                    counts[name] = counts.get(name, 0) + count
                    prediction, target, valid = output.logits[name], targets[name], masks[name]
                    stats = metrics.setdefault(name, {'correct': 0, 'tp': 0, 'fp': 0, 'fn': 0})
                    if name == 'other_artist' and self.loss_config.other_artist_mode == 'single_label':
                        stats['correct'] += int((prediction.argmax(-1)[valid] == target[valid]).sum())
                    else:
                        predicted = prediction[valid] >= 0
                        positive = target[valid] == 1
                        stats['correct'] += int((predicted == positive).sum())
                        stats['tp'] += int((predicted & positive).sum())
                        stats['fp'] += int((predicted & ~positive).sum())
                        stats['fn'] += int((~predicted & positive).sum())
        finally:
            self.model.train(was_training)
        return {name: {'loss': totals[name] / counts[name] if counts[name] else 0.0,
                       'valid_count': counts[name], **metrics[name]} for name in totals}

    def _settings(self):
        configuration = asdict(self.config)
        configuration['stage'] = self.model.backbone.stage
        return {'trainer': configuration, 'loss': asdict(self.loss_config)}

    def save_checkpoint(self, path, *, sampler=None):
        if self.pending:
            raise ValueError('Checkpoint requires an optimizer boundary; flush first or finish accumulation')
        numpy_rng = np.random.get_state()
        payload = {'version': CHECKPOINT_VERSION, 'metadata': self.metadata, 'settings': self._settings(),
                   'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                   'scaler': self.scaler.state_dict(), 'epoch': self.epoch, 'global_step': self.global_step,
                   'micro_step': self.micro_step, 'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                   'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(), numpy_rng[2], numpy_rng[3], numpy_rng[4]),
                   'sampler': sampler.state_dict() if sampler is not None else None,
                   'teacher': self.teacher.state_dict() if self.teacher is not None else None}
        if self.device.type == 'cuda':
            payload['cuda_rng'] = torch.cuda.get_rng_state_all()
        save_payload(path, payload)

    def load_checkpoint(self, path, *, sampler=None):
        if self.pending:
            raise ValueError('Cannot load during partial accumulation')
        payload = read_payload(path, self.metadata)
        if payload['settings'] != self._settings():
            raise ValueError('Checkpoint training/loss configuration mismatch')
        if (payload['sampler'] is None) != (sampler is None):
            raise ValueError('Checkpoint sampler presence mismatch')
        if (payload['teacher'] is None) != (self.teacher is None):
            raise ValueError('Checkpoint teacher presence mismatch')
        if sampler is not None and payload['sampler'].get('signature') != sampler._signature():
            raise ValueError('Checkpoint sampler configuration mismatch')
        # Validate tensor names/shapes before mutating either student or teacher.
        for module, state in ((self.model, payload['model']), (self.teacher, payload['teacher'])):
            if module is not None:
                current = module.state_dict()
                if current.keys() != state.keys() or any(current[key].shape != state[key].shape for key in current):
                    raise ValueError('Checkpoint tensor layout mismatch')
        self.model.load_state_dict(payload['model'], strict=True)
        if self.teacher is not None:
            self.teacher.load_state_dict(payload['teacher'], strict=True)
            self.teacher.requires_grad_(False).eval()
        self.optimizer.load_state_dict(payload['optimizer'])
        self.scaler.load_state_dict(payload['scaler'])
        self.epoch, self.global_step, self.micro_step = payload['epoch'], payload['global_step'], payload['micro_step']
        if sampler is not None:
            sampler.load_state_dict(payload['sampler'])
        torch.set_rng_state(payload['torch_rng'])
        random.setstate(payload['python_rng'])
        rng = payload['numpy_rng']
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
        if self.device.type == 'cuda' and 'cuda_rng' in payload:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
