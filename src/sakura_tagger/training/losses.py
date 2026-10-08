"""Masked per-task losses normalized by observed elements or samples."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossResult:
    total: torch.Tensor
    by_group: dict[str, torch.Tensor]  # Unweighted means over valid supervision
    valid_counts: dict[str, int]


def multitask_loss(logits, targets, specs, masks=None):
    masks = masks or {}
    if not logits:
        raise ValueError('At least one task logit tensor is needed for loss computation')
    by_group, counts = {}, {}
    total = next(iter(logits.values())).sum() * 0
    for spec in specs:
        name = spec.group_name
        if not spec.enabled or name not in logits or name not in targets:
            continue
        prediction = logits[name]
        target = targets[name].to(device=prediction.device)
        expected = prediction.shape[:-1] if spec.group_type == 'multiclass' else prediction.shape
        if target.shape != expected:
            raise ValueError(f'{name}: target shape {tuple(target.shape)} must equal {tuple(expected)}')
        mask = masks.get(name)
        valid = torch.ones_like(target, dtype=torch.bool) if mask is None else mask.to(device=prediction.device, dtype=torch.bool)
        if valid.shape != target.shape:
            raise ValueError(f'{name}: mask shape must equal target shape')
        count = int(valid.sum().item())
        counts[name] = count
        # Select before evaluating the loss: unknown NaN or invalid CE IDs never
        # enter the loss kernel. Empty tasks preserve a differentiable zero.
        if count == 0:
            loss = prediction.sum() * 0
        elif spec.group_type == 'multiclass':
            loss = F.cross_entropy(prediction[valid], target[valid].long(), reduction='sum') / count
        else:
            loss = F.binary_cross_entropy_with_logits(prediction[valid], target[valid].to(prediction.dtype), reduction='sum') / count
        by_group[name] = loss
        total = total + spec.loss_weight * loss
    return LossResult(total, by_group, counts)
