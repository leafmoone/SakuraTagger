"""The three physical trainable modules, independent of original Artist/Style."""
from torch import nn

from .group_attention import GroupAttention
from .kaloscope_backbone import artist_classifier_input


class GeneralModule(nn.Module):
    def __init__(self, global_dim, patch_dim, layout):
        super().__init__()
        self.attention = GroupAttention(patch_dim, layout.query_count, layout.attention_dim, layout.attention_heads)
        self.query_fusion = nn.Linear(layout.query_count * layout.attention_dim, layout.feature_dim)
        self.global_residual = nn.Linear(global_dim, layout.feature_dim)
        self.adapter = nn.Sequential(nn.LayerNorm(layout.feature_dim), nn.GELU())
        self.classifier = nn.Linear(layout.feature_dim, layout.general)

    def forward(self, global_features, patch_tokens):
        queries = self.attention(patch_tokens)
        features = self.query_fusion(queries.flatten(1)) + self.global_residual(global_features)
        return self.classifier(self.adapter(features))


class IdentityModule(nn.Module):
    def __init__(self, global_dim, layout):
        super().__init__()
        self.character_classes = layout.character
        self.adapter = nn.Sequential(nn.Linear(global_dim, layout.feature_dim), nn.GELU())
        self.classifier = nn.Linear(layout.feature_dim, layout.identity)

    def forward(self, global_features):
        features = self.adapter(global_features)
        logits = self.classifier(features)
        return logits.split((self.character_classes, logits.shape[-1] - self.character_classes), dim=-1)

    def metric_features(self, global_features):
        """Optional identity metric objective consumes this shared representation."""
        return self.adapter(global_features)


class OtherArtistModule(nn.Module):
    def __init__(self, global_dim, classes):
        super().__init__()
        self.classifier = nn.Linear(global_dim, classes)

    def forward(self, global_features):
        normalized = artist_classifier_input(global_features)
        return self.classifier(normalized.to(dtype=self.classifier.weight.dtype))
