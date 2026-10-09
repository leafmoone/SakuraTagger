"""Single-forward Kaloscope features with explicit frozen/last-four stages."""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..adapter import validate_images
from .feature_bundle import FeatureBundle

STAGES = ('frozen_heads', 'finetune_last4')


def artist_classifier_input(global_features):
    """The original Kaloscope FP32 L2/sqrt-dimension normalization protocol."""
    return F.normalize(global_features.float(), dim=-1) * math.sqrt(global_features.shape[-1])


class KaloscopeBackbone(nn.Module):
    def __init__(self, legacy: nn.Module, stage='frozen_heads'):
        super().__init__()
        if legacy.pooling != 'cls_mean' or legacy.feature_source != 'projector':
            raise ValueError('SakuraTagger requires CLS+mean pooling and the original style projector')
        if legacy.head is None or legacy.projector is None:
            raise ValueError('Both original artist head and style projector are required')
        if legacy.classifier_input_normalization != 'l2_sqrt_dim':
            raise ValueError('SakuraTagger requires the original l2_sqrt_dim artist normalization')
        self.legacy = legacy
        self.global_dim = legacy.pooled_dim
        self.patch_dim = legacy.backbone.embed_dim
        self.style_dim = legacy.feature_dim
        self.artist_classes = legacy.head.out_features
        self.set_stage(stage)
        self.train(False)

    def set_stage(self, stage):
        stage = {'T1': 'frozen_heads', 'T2': 'finetune_last4'}.get(stage, stage)
        if stage not in STAGES:
            raise ValueError(f'Unknown training stage: {stage}')
        backbone = self.legacy.backbone
        if stage == 'finetune_last4' and (not hasattr(backbone, 'blocks') or len(backbone.blocks) < 4):
            raise ValueError('finetune_last4 requires a DINOv3 with at least four blocks')
        self.stage = stage
        self.legacy.requires_grad_(False)
        for parameter in self.legacy.parameters():
            parameter.grad = None
        if stage == 'finetune_last4':
            for block in backbone.blocks[-4:]:
                block.requires_grad_(True)
            backbone.norm.requires_grad_(True)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        # Frozen embeddings, early blocks, and legacy heads remain in eval mode.
        self.legacy.eval()
        if self.stage == 'finetune_last4':
            for block in self.legacy.backbone.blocks[-4:]:
                block.train(mode)
            self.legacy.backbone.norm.train(mode)
        return self

    def forward_features(self, images, *, include_artist=True, include_style=True):
        validate_images(images, self.legacy.backbone.patch_size)
        # No outer no_grad in T2: new heads and optional preservation losses must
        # backpropagate through the last four blocks and final normalization.
        context = torch.no_grad() if self.stage == 'frozen_heads' else nullcontext()
        with context:
            tokens = self.legacy.backbone.forward_features(images)
            cls = tokens['x_norm_clstoken']
            patches = tokens['x_norm_patchtokens']
            global_features = torch.cat((cls, patches.mean(dim=1)), dim=1)
            style = self.legacy.projector(global_features) if include_style else None
            artist = None
            if include_artist:
                normalized = artist_classifier_input(global_features)
                artist = self.legacy.head(normalized.to(dtype=self.legacy.head.weight.dtype))
        return FeatureBundle(global_features, cls, patches, artist, style)

    def forward(self, images, **kwargs):
        return self.forward_features(images, **kwargs)

    def parameter_counts(self):
        return {key: sum(p.numel() for p in self.parameters() if p.requires_grad == trainable)
                for key, trainable in (('trainable', True), ('frozen', False))}
