"""Frozen single-forward feature interface for the original Kaloscope model."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..adapter import validate_images
from .feature_bundle import FeatureBundle


class KaloscopeBackbone(nn.Module):
    """Keep the entire legacy state registered while exposing trainable-head inputs.

    Pass a strictly loaded official model, or an explicitly synthetic development
    fixture. The legacy model is permanently frozen and remains in eval mode.
    """

    def __init__(self, legacy: nn.Module):
        super().__init__()
        if legacy.pooling != "cls_mean" or legacy.feature_source != "projector":
            raise ValueError("SakuraTagger requires CLS+mean pooling and the original style projector")
        if legacy.head is None or legacy.projector is None:
            raise ValueError("Both original artist head and style projector are required")
        if legacy.classifier_input_normalization != "l2_sqrt_dim":
            raise ValueError("SakuraTagger requires the original l2_sqrt_dim artist normalization")
        self.legacy = legacy.requires_grad_(False).eval()
        self.global_dim = legacy.pooled_dim
        self.patch_dim = legacy.backbone.embed_dim
        self.style_dim = legacy.feature_dim
        self.artist_classes = legacy.head.out_features
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.legacy.requires_grad_(False).eval()
        return self

    def forward_features(self, images: torch.Tensor) -> FeatureBundle:
        validate_images(images, self.legacy.backbone.patch_size)
        # no_grad produces ordinary tensors; inference_mode tensors cannot be
        # saved for backward by newly trained heads.
        with torch.no_grad():
            tokens = self.legacy.backbone.forward_features(images)
            cls = tokens["x_norm_clstoken"]
            patches = tokens["x_norm_patchtokens"]
            global_features = torch.cat((cls, patches.mean(dim=1)), dim=1)
            style = self.legacy.projector(global_features)
            artist_input = F.normalize(global_features.float(), dim=-1) * math.sqrt(self.global_dim)
            artist = self.legacy.head(artist_input.to(dtype=self.legacy.head.weight.dtype))
        return FeatureBundle(global_features, cls, patches, artist, style)

    def forward(self, images: torch.Tensor) -> FeatureBundle:
        return self.forward_features(images)
