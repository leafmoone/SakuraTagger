from dataclasses import dataclass

import torch
from torch import nn

from .feature_bundle import FeatureBundle
from .global_heads import GlobalHeads
from .local_heads import LocalHeads
from .registry import validate_specs


@dataclass(frozen=True)
class MultiTaskOutput:
    features: FeatureBundle
    logits: dict[str, torch.Tensor]


class MultiTaskModel(nn.Module):
    def __init__(self, backbone, specs, *, hidden_dim=256, num_heads=4):
        super().__init__()
        self.backbone = backbone
        self.specs = validate_specs(specs)
        active = [spec for spec in self.specs if spec.enabled]
        self.global_heads = GlobalHeads(backbone.global_dim, [s for s in active if s.feature_mode == 'global'])
        self.local_heads = LocalHeads(backbone.global_dim, backbone.patch_dim,
                                     [s for s in active if s.feature_mode == 'local'], hidden_dim, num_heads)

    def _selection(self, head_selection):
        available = {s.group_name for s in self.specs if s.enabled} | {'artist', 'style_embedding'}
        selected = available if head_selection is None else set(head_selection)
        unknown = selected - available
        if unknown:
            raise ValueError(f'Unknown or disabled heads: {sorted(unknown)}')
        return selected

    def forward(self, images, head_selection=None):
        selected = self._selection(head_selection)
        features = self.backbone.forward_features(images)
        logits = self.global_heads(features.global_features, selected)
        logits.update(self.local_heads(features.global_features, features.patch_tokens, selected))
        return MultiTaskOutput(features, logits)

    @torch.no_grad()
    def predict(self, images, head_selection=None):
        selected = self._selection(head_selection)
        output = self(images, head_selection=selected)
        result = {'general_tags_by_group': {}}
        if 'artist' in selected:
            result['artist'] = output.features.artist_logits.softmax(dim=-1)
        if 'style_embedding' in selected:
            result['style_embedding'] = output.features.style_embedding
        for spec in self.specs:
            if spec.group_name not in output.logits:
                continue
            logits = output.logits[spec.group_name]
            probabilities = logits.softmax(dim=-1) if spec.group_type == 'multiclass' else logits.sigmoid()
            if spec.group_type == 'multilabel':
                result['general_tags_by_group'][spec.group_name] = {
                    'probabilities': probabilities, 'selected': probabilities >= spec.threshold,
                    'threshold': spec.threshold,
                }
            else:
                result[spec.group_name] = probabilities
        return result
