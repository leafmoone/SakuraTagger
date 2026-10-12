"""The three physical trainable modules, independent of original Artist/Style."""
import torch
from torch import nn
from torch.nn import functional as F

from .group_attention import GroupAttention
from .kaloscope_backbone import artist_classifier_input


def _classify(classifier, features, start=0, end=None):
    """Apply a contiguous physical class block without computing other rows."""
    end = classifier.out_features if end is None else end
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= classifier.out_features:
        raise ValueError('Invalid classifier class range')
    if start == 0 and end == classifier.out_features:
        return classifier(features)
    return F.linear(features, classifier.weight[start:end], classifier.bias[start:end])


class GeneralModule(nn.Module):
    def __init__(self, global_dim, patch_dim, layout):
        super().__init__()
        self.attention = GroupAttention(patch_dim, layout.query_count, layout.attention_dim, layout.attention_heads)
        self.query_fusion = nn.Linear(layout.query_count * layout.attention_dim, layout.general_dim)
        self.global_residual = nn.Linear(global_dim, layout.general_dim)
        self.adapter = nn.Sequential(nn.LayerNorm(layout.general_dim), nn.GELU())
        self.classifier = nn.Linear(layout.general_dim, layout.general)

    def features(self, global_features, patch_tokens):
        queries = self.attention(patch_tokens)
        features = self.query_fusion(queries.flatten(1)) + self.global_residual(global_features)
        return self.adapter(features)

    def classify_features(self, features, start=0, end=None):
        return _classify(self.classifier, features, start, end)

    def forward(self, global_features, patch_tokens):
        return self.classify_features(self.features(global_features, patch_tokens))


class IdentityModule(nn.Module):
    def __init__(self, global_dim, layout, patch_dim=None):
        super().__init__()
        self.character_classes = layout.character
        self.feature_mode = layout.identity_feature_mode
        self.adapter = nn.Sequential(nn.Linear(global_dim, layout.identity_dim), nn.GELU())
        self.classifier = nn.Linear(layout.identity_dim, layout.identity)
        if self.feature_mode == 'global_local':
            if patch_dim is None:
                raise ValueError('Identity-local requires backbone patch dimension')
            self.local_attention = GroupAttention(patch_dim, layout.identity_query_count,
                                                  layout.identity_attention_dim, layout.identity_attention_heads)
            self.local_projection = nn.Linear(layout.identity_query_count * layout.identity_attention_dim,
                                              layout.identity_dim)
            self.local_norm = nn.LayerNorm(layout.identity_dim)
            self.local_scale = nn.Parameter(torch.tensor(float(layout.identity_residual_init)))

    def features(self, global_features, patch_tokens=None):
        features = self.adapter(global_features)
        if self.feature_mode == 'global_local':
            if patch_tokens is None:
                raise ValueError('Identity-local requires patch tokens from the same backbone pass')
            local = self.local_projection(self.local_attention(patch_tokens).flatten(1))
            features = features + self.local_scale * self.local_norm(local)
        return features

    def classify_features(self, features, start=0, end=None):
        """Physical Identity IDs: Character rows, then Copyright rows."""
        return _classify(self.classifier, features, start, end)

    def forward(self, global_features, patch_tokens=None):
        logits = self.classify_features(self.features(global_features, patch_tokens))
        return logits.split((self.character_classes, logits.shape[-1] - self.character_classes), dim=-1)

    def metric_features(self, global_features, patch_tokens=None):
        """Optional identity metric objective consumes this shared representation."""
        return self.features(global_features, patch_tokens)


class OtherArtistModule(nn.Module):
    def __init__(self, global_dim, classes):
        super().__init__()
        self.classifier = nn.Linear(global_dim, classes)

    def forward(self, global_features):
        normalized = artist_classifier_input(global_features)
        return self.classifier(normalized.to(dtype=self.classifier.weight.dtype))
