"""Exact valid-image optimizer windows, common to CPU and torchrun DDP.

Construction never starts a run. No DDP.join, real-image rank padding, full-graph
window retention, or normalization by a configured accumulation length is used.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
import math
import os

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from ..model import KaloscopeBackbone
from ..model.registry import LOGICAL_DOMAINS
from ..data.window_sampling import window_loader
from .losses import differentiable_zero
from .positive_unknown import sparse_multitask_loss, sparse_valid_counts
from .trainer import _chunked_artist_kd

GROUPS = ('backbone_last4', 'general', 'identity', 'other_artist')


@dataclass(frozen=True)
class WindowTrainerConfig:
    stage: str = 'T1'
    learning_rates: dict = field(default_factory=lambda: dict(zip(GROUPS, (1e-5, 1e-4, 1e-4, 1e-4))))
    weight_decay: float = .01
    accumulation_steps: int = 1
    amp: str = 'off'
    gradient_clip: float | None = 1.
    warmup_updates: int = 0
    schedule_updates: int = 1000
    minimum_lr_ratio: float = 0.
    max_optimizer_steps: int | None = None
    max_epochs: int | None = None
    max_attempted_windows: int | None = None
    preservation: dict = field(default_factory=lambda: {'global_features': 0., 'style_embedding': 0., 'artist_logits': 0.})

    def __post_init__(self):
        if self.stage not in ('T1', 'T2', 'frozen_heads', 'finetune_last4') or self.amp not in ('off', 'bf16', 'fp16'):
            raise ValueError('Unknown stage or AMP mode')
        if set(self.learning_rates) != set(GROUPS) or any(not math.isfinite(v) or v <= 0 for v in self.learning_rates.values()):
            raise ValueError('Four explicit finite positive learning rates are required')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Invalid weight decay')
        if self.gradient_clip is not None and (not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0):
            raise ValueError('Invalid gradient clip')
        if any(type(v) is not int or v < 1 for v in (self.accumulation_steps, self.schedule_updates)):
            raise ValueError('Positive accumulation/schedule lengths required')
        if type(self.warmup_updates) is not int or not 0 <= self.warmup_updates < self.schedule_updates:
            raise ValueError('Warmup must be shorter than the schedule')
        if not math.isfinite(self.minimum_lr_ratio) or not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError('Invalid minimum LR ratio')
        for name in ('max_optimizer_steps', 'max_epochs', 'max_attempted_windows'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f'{name} must be positive or None')
        if set(self.preservation) - {'global_features', 'style_embedding', 'artist_logits'} or any(not math.isfinite(v) or v < 0 for v in self.preservation.values()):
            raise ValueError('Invalid preservation weights')


def initialize_distributed(device='cpu', *, timeout_seconds=120):
    """Explicit torchrun setup. Uses env:// TCP; never probes credentials or GPUs."""
    device = torch.device(device)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('Distributed timeout must be finite and positive')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world > 1 and not dist.is_initialized():
        if device.type == 'cuda':
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
            torch.cuda.set_device(device)
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', init_method='env://',
                                timeout=timedelta(seconds=timeout_seconds))
    return device


class UpdateSchedule:
    """Warmup/cosine driven only by successful updates; next LR is explicit."""
    def __init__(self, optimizer, config):
        self.optimizer, self.config, self.completed = optimizer, config, 0
        self._apply()

    def _apply(self):
        cfg, step = self.config, self.completed
        if step < cfg.warmup_updates:
            factor = (step + 1) / cfg.warmup_updates
        else:
            progress = min(1., (step - cfg.warmup_updates) / (cfg.schedule_updates - cfg.warmup_updates))
            factor = cfg.minimum_lr_ratio + (1 - cfg.minimum_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group['lr'] = group['initial_lr'] * factor

    def step(self):
        self.completed += 1
        self._apply()

    def state_dict(self):
        return {'completed': self.completed}

    def load_state_dict(self, state):
        if type(state.get('completed')) is not int or state['completed'] < 0:
            raise ValueError('Invalid successful-update scheduler state')
        self.completed = state['completed']
        self._apply()


class CoordinatedTrainingError(RuntimeError):
    pass


class WindowTrainer:
    def __init__(self, model, release, scope, loss_config, *, config=None, device='cpu', teacher=None, teacher_source=None):
        self.config = config or WindowTrainerConfig()
        self.release, self.scope, self.loss_config = release, scope, loss_config
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda') or self.config.amp == 'fp16' and self.device.type != 'cuda':
            raise ValueError('CPU/CUDA supported; FP16 requires CUDA')
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        if scope.identity['release'] != release.identity:
            raise ValueError('Loss scope belongs to a different Release')
        model.bind_vocabulary(release.vocabulary)
        expected_artist = {entry['class_id']: entry['name'] for entry in release.vocabulary.document['original_artist']}
        if expected_artist != model.backbone.legacy.class_mapping:
            raise ValueError('Original Artist mapping differs from the bound Vocabulary')
        model.set_other_artist_mode(loss_config.other_artist_mode)
        self.model = model.to(self.device).set_stage(self.config.stage)
        self.config = replace(self.config, stage=self.model.backbone.stage)
        self.teacher = self._teacher(teacher)
        from .window_checkpoint import teacher_identity
        self.teacher_identity = teacher_identity(self.teacher, teacher_source)
        self._assert_same_contract()
        self.optimizer = self._new_optimizer()
        self.scheduler = UpdateSchedule(self.optimizer, self.config)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.config.amp == 'fp16')
        self.network = self._wrap()
        self.counters = {'attempted_windows': 0, 'successful_updates': 0, 'skipped_updates': 0,
                         'real_exposures': 0, 'control_steps': 0}
        self.at_boundary = True

    def _teacher(self, teacher):
        if bool(any(self.config.preservation.values())) != (teacher is not None):
            raise ValueError('Preservation needs an explicit independent original teacher')
        if teacher is None:
            return None
        if not isinstance(teacher, KaloscopeBackbone):
            raise ValueError('Teacher must be an independent original KaloscopeBackbone')
        student_storage = {p.data_ptr() for p in self.model.parameters()}
        if any(p.data_ptr() in student_storage for p in teacher.parameters()):
            raise ValueError('Teacher and student must not share storage')
        if ((teacher.global_dim, teacher.style_dim, teacher.artist_classes)
                != (self.model.backbone.global_dim, self.model.backbone.style_dim, self.model.backbone.artist_classes)
                or teacher.legacy.class_mapping != self.model.backbone.legacy.class_mapping):
            raise ValueError('Teacher dimensions or original Artist mapping mismatch')
        return teacher.to(self.device).set_stage('T1').requires_grad_(False).eval()

    def contract(self):
        return {'trainer': asdict(self.config), 'loss': self.loss_config.to_dict(), 'scope': self.scope.identity,
                'release': self.release.identity, 'layout': self.model.layout.to_dict(),
                'normalization': 'global_valid_image_per_domain_v1', 'world_size': self.world_size,
                'teacher': self.teacher_identity}

    def _assert_same_contract(self):
        if self.distributed:
            contracts = [None] * self.world_size
            dist.all_gather_object(contracts, self.contract())
            if any(value != contracts[0] for value in contracts):
                raise ValueError('All ranks require identical training contracts')

    def _wrap(self):
        if not self.distributed:
            return self.model
        return DistributedDataParallel(self.model,
            device_ids=[self.device.index] if self.device.type == 'cuda' else None,
            broadcast_buffers=False, find_unused_parameters=False)

    def _new_optimizer(self):
        groups = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            module = 'backbone_last4' if name.startswith('backbone.') else name.split('.')[0]
            decay = 0. if parameter.ndim <= 1 or name.endswith('.bias') else self.config.weight_decay
            key = module, decay
            group = groups.setdefault(key, {'params': [], 'name': module, 'weight_decay': decay,
                                           'initial_lr': self.config.learning_rates[module]})
            group['params'].append(parameter)
        return torch.optim.AdamW(list(groups.values()))

    def set_stage(self, stage):
        if not self.at_boundary:
            raise ValueError('Stage transitions require a complete optimizer boundary')
        self.network = self.model  # Retire the old reducer before changing trainable membership.
        self.model.set_stage(stage)
        self.config = replace(self.config, stage=self.model.backbone.stage)
        self._assert_same_contract()
        old = self.optimizer
        # Rebuild membership and DDP reducer together; retain shared Adam moments.
        self.optimizer = self._new_optimizer()
        for group in self.optimizer.param_groups:
            for parameter in group['params']:
                if parameter in old.state:
                    self.optimizer.state[parameter] = old.state[parameter]
        self.scheduler = UpdateSchedule(self.optimizer, self.config)
        self.network = self._wrap()

    def _autocast(self):
        if self.config.amp == 'off':
            return nullcontext()
        return torch.autocast(self.device.type, dtype=torch.bfloat16 if self.config.amp == 'bf16' else torch.float16)

    def _all_true(self, value):
        flag = torch.as_tensor(value, dtype=torch.int32, device=self.device).clone()
        if self.distributed:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _raise_together(self, error, phase):
        if not self._all_true(error is None):
            messages = [None] * self.world_size
            message = None if error is None else f'{type(error).__name__}: {error}'
            if self.distributed:
                dist.all_gather_object(messages, message)
            else:
                messages[0] = message
            self.optimizer.zero_grad(set_to_none=True)
            raise CoordinatedTrainingError(f'{phase} failed across ranks: {messages}') from error

    def _denominators(self, sampler, steps):
        error, counts = None, None
        try:
            records = [record for step in range(sampler.cursor, sampler.cursor + steps)
                       for record in sampler.records_for(sampler.batch_at(step))]
            totals = sparse_valid_counts(records, self.loss_config, self.release)
            counts = torch.tensor([totals[name] for name in LOGICAL_DOMAINS] + [len(records)],
                                  dtype=torch.int64, device=self.device)
        except Exception as failure:
            error = failure
        self._raise_together(error, 'planned supervision')
        if self.distributed:
            dist.all_reduce(counts)
        values = counts.tolist()
        return dict(zip(LOGICAL_DOMAINS, values[:-1])), values[-1]

    def _clear_inactive(self, counts, images):
        active = {name: counts[name] > 0 and self.loss_config.weights.get(name, 1.) > 0 for name in LOGICAL_DOMAINS}
        modules = {'general': active['general'], 'identity': active['character'] or active['copyright'],
                   'other_artist': active['other_artist'],
                   'backbone': any(active.values()) or bool(images and any(self.config.preservation.values()))}
        for name, enabled in modules.items():
            if not enabled:
                for parameter in getattr(self.model, name).parameters():
                    parameter.grad = None

    def train_window(self, sampler, loader, *, steps=None):
        """Consume one exact optimizer window from a prepared E4 sampler/loader."""
        error, signature = None, None
        try:
            if not self.at_boundary:
                raise ValueError('Previous optimizer window did not finish')
            if (sampler.rank, sampler.world_size) != (self.rank, self.world_size):
                raise ValueError('Sampler/DDP rank identity mismatch')
            steps = min(self.config.accumulation_steps, len(sampler)) if steps is None else steps
            if type(steps) is not int or not 0 < steps <= len(sampler):
                raise ValueError('A nonempty bounded optimizer window is required')
            signature = dict(sampler._signature())
            signature.pop('rank')
        except Exception as failure:
            error = failure
        self._raise_together(error, 'window request')
        if self.distributed:
            local = (signature, sampler.epoch, sampler.window, sampler.cursor, steps, sampler.window_info['control_steps'])
            identities = [None] * self.world_size
            dist.all_gather_object(identities, local)
            if any(value != identities[0] for value in identities):
                raise CoordinatedTrainingError('Ranks disagree on the scheduling/preprocessing signature or optimizer window')
        counts, images_count = self._denominators(sampler, steps)
        self.at_boundary = False
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        weights = self.config.preservation
        selection = set(LOGICAL_DOMAINS)
        if weights.get('artist_logits', 0):
            selection.add('artist')
        if weights.get('style_embedding', 0):
            selection.add('style_embedding')
        bad_window = False
        numerators = torch.zeros(len(LOGICAL_DOMAINS), device=self.device, dtype=torch.float64)
        start = sampler.cursor
        for offset in range(steps):
            error, batch = None, None
            try:
                plan, batch = next(loader)
                if plan != sampler.batch_at(start + offset):
                    raise ValueError('Loader and acknowledged logical plan disagree')
                if not plan.control_only:
                    if batch['occurrence_ids'] != [value for _, value in plan.occurrences]:
                        raise ValueError('Loader occurrence IDs changed')
                    if batch['records'] != sampler.records_for(plan) or batch['ordinals'] != [value for value, _ in plan.occurrences]:
                        raise ValueError('Loader sparse records changed from planned supervision')
                    images = batch['images'].to(self.device)
                    if images.shape[0] != len(plan.occurrences) or tuple(images.shape[-2:]) != plan.target_size[::-1]:
                        raise ValueError('Real image dimensions disagree with the frozen actual-shape plan')
                else:
                    if batch is not None:
                        raise ValueError('Control-only step must not contain a real batch')
                    images = torch.zeros(1, 3, 16, 16, device=self.device)
            except Exception as failure:
                error = failure
            self._raise_together(error, 'data load')
            synchronize = offset == steps - 1
            context = self.network.no_sync() if self.distributed and not synchronize else nullcontext()
            # DDP's no_sync must cover forward as well as backward.
            with context:
                error, total, finite, output = None, None, None, None
                try:
                    with self._autocast():
                        output = self.network(images, selection)
                        zero = sum(differentiable_zero(output.logits[name]) for name in LOGICAL_DOMAINS)
                        total = zero
                        finite = torch.ones((), dtype=torch.bool, device=self.device)
                        if not plan.control_only:
                            result = sparse_multitask_loss(output.logits, batch['records'], release=self.release,
                                scope=self.scope, config=self.loss_config, epoch=sampler.epoch,
                                occurrence_ids=batch['occurrence_ids'], defer_finite_check=True)
                            finite = torch.stack([item.finite for item in result.domains.values()]).all()
                            for position, name in enumerate(LOGICAL_DOMAINS):
                                numerator = result.domains[name].numerator
                                numerators[position] += numerator.detach().double()
                                if counts[name]:
                                    total = total + self.loss_config.weights.get(name, 1.) * numerator * self.world_size / counts[name]
                            if self.teacher is not None:
                                with torch.no_grad():
                                    reference = self.teacher(images, include_artist=bool(weights.get('artist_logits', 0)),
                                                             include_style=bool(weights.get('style_embedding', 0)))
                                auxiliary = torch.zeros((), device=self.device)
                                for name in ('global_features', 'style_embedding'):
                                    if weights.get(name, 0):
                                        auxiliary = auxiliary + weights[name] * F.mse_loss(getattr(output.features, name).float(), getattr(reference, name).float())
                                if weights.get('artist_logits', 0):
                                    auxiliary = auxiliary + weights['artist_logits'] * _chunked_artist_kd(output.features.artist_logits, reference.artist_logits, 4096)
                                total = total + auxiliary * images.shape[0] * self.world_size / images_count
                        finite = finite & torch.isfinite(total)
                except Exception as failure:
                    error = failure
                self._raise_together(error, 'forward/loss')
                bad_window |= not self._all_true(finite)
                self.scaler.scale(zero if bad_window else total).backward()
        self.scaler.unscale_(self.optimizer)
        parameters = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip or float('inf'), error_if_nonfinite=False)
        finite_gradients = self._all_true(torch.isfinite(norm))
        self._clear_inactive(counts, images_count)
        has_gradients = any(p.grad is not None for p in self.model.parameters())
        success = not bad_window and finite_gradients and has_gradients
        if success:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        elif self.scaler.is_enabled():
            scale = self.scaler.get_scale()
            self.scaler.update(new_scale=scale * self.scaler.get_backoff_factor() if bad_window or not finite_gradients else scale)
        self.optimizer.zero_grad(set_to_none=True)
        sampler.advance(steps)  # Only a complete attempted window acknowledges input.
        self.counters['attempted_windows'] += 1
        self.counters['successful_updates'] += int(success)
        self.counters['skipped_updates'] += int(not success)
        self.counters['real_exposures'] += images_count
        self.counters['control_steps'] += steps
        self.at_boundary = True
        if self.distributed:
            dist.all_reduce(numerators)
        return {'success': success, 'valid_counts': counts, 'real_exposures': images_count,
                'losses': {name: (float(numerators[i].item()) / counts[name] if torch.isfinite(numerators[i]) else None) if counts[name] else 0.
                           for i, name in enumerate(LOGICAL_DOMAINS)},
                'skip_reason': None if success else 'nonfinite' if bad_window or not finite_gradients else 'empty_gradient',
                'gradient_norm': float(norm.item()) if torch.isfinite(norm) else None, 'counters': dict(self.counters)}

    def save_checkpoint(self, path, *, sampler):
        from .window_checkpoint import save_checkpoint
        return save_checkpoint(self, path, sampler)

    def load_checkpoint(self, path, *, sampler):
        from .window_checkpoint import load_checkpoint
        return load_checkpoint(self, path, sampler)

    def run(self, sampler, *, num_workers=0, worker_transport='shared_memory', worker_timeout=60., on_boundary=None):
        """Explicit finite run. A skipped update cannot create an endless loop."""
        if self.config.max_epochs is None and self.config.max_attempted_windows is None:
            raise ValueError('A finite epoch/attempt limit is required even when max_optimizer_steps is set')
        if not len(sampler.assets):
            raise ValueError('Training requires at least one eligible unique train sample')
        def finished():
            return ((self.config.max_optimizer_steps is not None and self.counters['successful_updates'] >= self.config.max_optimizer_steps)
                    or (self.config.max_attempted_windows is not None and self.counters['attempted_windows'] >= self.config.max_attempted_windows)
                    or (self.config.max_epochs is not None and sampler.epoch >= self.config.max_epochs))
        while not finished():
            if sampler.epoch_done:
                sampler.set_epoch(sampler.epoch + 1)
                continue
            error = None
            try:
                sampler.prepare_window()
            except Exception as failure:
                error = failure
            self._raise_together(error, 'window dimensions/cache')
            loader = iter(window_loader(sampler.dataset, sampler, num_workers=num_workers, worker_transport=worker_transport,
                                        worker_timeout=worker_timeout,
                                        multiprocessing_context='spawn' if self.device.type == 'cuda' and num_workers else None))
            while len(sampler) and not finished():
                result = self.train_window(sampler, loader)
                error = None
                try:
                    if on_boundary is not None:
                        on_boundary(self, sampler, result)
                except Exception as failure:
                    error = failure
                self._raise_together(error, 'boundary callback')
            if not len(sampler):
                sampler.next_window()
        return dict(self.counters)
