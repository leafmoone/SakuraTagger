"""Masked, chunked logical-domain objectives with bounded FP32 working tensors."""
from dataclasses import dataclass, field
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..model.registry import LOGICAL_DOMAINS


def differentiable_zero(tensor):
    # Summing no elements avoids masked NaNs and BF16 overflow in x.sum()*0.
    return tensor[..., :0].sum().float()


@dataclass(frozen=True)
class LossConfig:
    gamma_pos: float = 0.0
    gamma_neg: float = 4.0
    clip: float = 0.05
    chunk_size: int = 4096
    recompute: bool = True
    other_artist_mode: str = 'multilabel'
    weights: dict = field(default_factory=lambda: {name: 1.0 for name in LOGICAL_DOMAINS})
    general_groups: dict = field(default_factory=dict)

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in (self.gamma_pos, self.gamma_neg)) or not 0 <= self.clip < 1:
            raise ValueError('Invalid ASL focusing or clip configuration')
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        if self.other_artist_mode not in ('multilabel', 'single_label'):
            raise ValueError('Other Artist supervision must be multilabel or single_label')
        if set(self.weights) - set(LOGICAL_DOMAINS) or any(not math.isfinite(w) or w < 0 for w in self.weights.values()):
            raise ValueError('Invalid logical loss weights')


@dataclass(frozen=True)
class LossResult:
    total: torch.Tensor
    by_group: dict[str, torch.Tensor]
    valid_counts: dict[str, int]
    general_statistics: dict[str, float] = field(default_factory=dict)


def _asl_sum(logits, targets, valid, gamma_pos, gamma_neg, clip):
    logits, targets = logits[valid].float(), targets[valid].float()
    if targets.numel() == 0:
        return differentiable_zero(logits)
    probability = logits.sigmoid()
    negative_probability = (1 - probability + clip).clamp(max=1)
    positive_term = F.logsigmoid(logits) * (1 - probability).pow(gamma_pos)
    negative_term = negative_probability.clamp_min(torch.finfo(torch.float32).tiny).log() * (1 - negative_probability).pow(gamma_neg)
    return -(targets * positive_term + (1 - targets) * negative_term).sum()


def masked_asymmetric_loss(logits, targets, mask, *, config=None):
    config = config or LossConfig()
    if logits.ndim != 2 or targets.shape != logits.shape or mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError('ASL requires logits/targets and boolean masks with shape [B,C]')
    targets, mask = targets.to(logits.device), mask.to(logits.device)
    # Validate once per domain, outside checkpoint recomputation/chunk loops.
    if (mask & (targets != 0) & (targets != 1)).any():
        raise ValueError('Observed multi-label targets must be finite binary values')
    count = int(mask.sum().item())
    total = differentiable_zero(logits)
    if count == 0:
        return total, 0
    for start in range(0, logits.shape[-1], config.chunk_size):
        end = start + config.chunk_size
        arguments = (logits[:, start:end], targets[:, start:end], mask[:, start:end])
        def loss_part(prediction, target, valid):
            return _asl_sum(prediction, target, valid, config.gamma_pos, config.gamma_neg, config.clip)
        if config.recompute and torch.is_grad_enabled() and logits.requires_grad:
            value = checkpoint(loss_part, *arguments, use_reentrant=False)
        else:
            value = loss_part(*arguments)
        total = total + value
    return total / count, count


def masked_single_label_loss(logits, targets, mask, *, chunk_size=4096, recompute=True):
    if logits.ndim != 2 or targets.shape != logits.shape[:1] or mask.shape != targets.shape or mask.dtype != torch.bool:
        raise ValueError('Single-label supervision requires [B] targets and boolean sample masks')
    mask, targets = mask.to(logits.device), targets.to(logits.device)
    count = int(mask.sum().item())
    if not count:
        return differentiable_zero(logits), 0
    labels = targets[mask]
    if labels.is_floating_point() or labels.dtype == torch.bool or ((labels < 0) | (labels >= logits.shape[-1])).any():
        raise ValueError('Observed single-label targets must be in-range integer class IDs')
    prediction = logits[mask]
    normalizer = None
    for chunk in prediction.split(chunk_size, dim=-1):
        def partition(values):
            return values.float().logsumexp(-1)
        value = checkpoint(partition, chunk, use_reentrant=False) if recompute and chunk.requires_grad and torch.is_grad_enabled() else partition(chunk)
        normalizer = value if normalizer is None else torch.logaddexp(normalizer, value)
    positive = prediction.gather(1, labels.long().unsqueeze(1)).squeeze(1).float()
    return (normalizer - positive).sum() / count, count


def trusted_single_artist(targets, masks):
    """Convert only fully observed, exactly-one-positive rows to trusted CE."""
    if targets.ndim != 2 or masks.shape != targets.shape or masks.dtype != torch.bool:
        raise ValueError('Expected [B,C] targets and boolean masks')
    safe = torch.where(masks, targets, torch.zeros_like(targets))
    valid = masks.all(-1) & torch.isfinite(safe).all(-1) & ((safe == 0) | (safe == 1)).all(-1) & (safe.sum(-1) == 1)
    return safe.argmax(-1), valid


def multitask_loss(logits, targets, config=None, masks=None):
    config = config or LossConfig()
    if not logits:
        raise ValueError('At least one task logit tensor is required')
    masks = {} if masks is None else masks
    unknown = set(targets) - set(LOGICAL_DOMAINS)
    if unknown:
        raise ValueError(f'Unknown target domains: {sorted(unknown)}')
    by_group, counts, statistics = {}, {}, {}
    total = differentiable_zero(next(iter(logits.values())))
    for name in LOGICAL_DOMAINS:
        if name not in targets:
            continue
        if name not in logits:
            raise ValueError(f'{name}: provided target has no selected output')
        if name not in masks:
            raise ValueError(f'{name}: explicit supervision mask is required')
        prediction, target, mask = logits[name], targets[name], masks[name]
        if name == 'other_artist' and config.other_artist_mode == 'single_label':
            loss, count = masked_single_label_loss(prediction, target, mask,
                chunk_size=config.chunk_size, recompute=config.recompute)
        else:
            loss, count = masked_asymmetric_loss(prediction, target, mask, config=config)
        by_group[name], counts[name] = loss, count
        total = total + config.weights.get(name, 1.0) * loss
        if name == 'general':
            with torch.no_grad():
                for group, ids in config.general_groups.items():
                    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < prediction.shape[-1] for i in ids):
                        raise ValueError(f'Invalid General semantic group: {group}')
                    value, _ = masked_asymmetric_loss(prediction[:, ids], target[:, ids], mask[:, ids], config=config)
                    statistics[group] = float(value)
    return LossResult(total, by_group, counts, statistics)
