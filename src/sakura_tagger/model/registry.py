"""Versioned three-module layout. Vocabulary order is supplied, never sorted."""
from dataclasses import dataclass
import math
from pathlib import Path

import yaml

LOGICAL_DOMAINS = ('general', 'character', 'copyright', 'other_artist')
OUTPUT_NAMES = LOGICAL_DOMAINS + ('artist', 'style_embedding')


@dataclass(frozen=True)
class HeadLayout:
    general: int
    character: int
    copyright: int
    other_artist: int
    feature_dim: int = 512
    attention_dim: int = 256
    attention_heads: int = 4
    query_count: int = 16
    version: str = 'sakura-tagger-heads-v1'
    general_feature_dim: int | None = None
    identity_feature_dim: int | None = None
    identity_feature_mode: str = 'global'
    identity_attention_dim: int = 256
    identity_attention_heads: int = 4
    identity_query_count: int = 4
    identity_residual_init: float = 0.1

    def __post_init__(self):
        for name in (*LOGICAL_DOMAINS, 'feature_dim', 'attention_dim', 'attention_heads', 'query_count'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('general_feature_dim', 'identity_feature_dim'):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f'{name} must be None or a positive integer')
        if self.attention_dim % self.attention_heads:
            raise ValueError('attention_dim must be divisible by attention_heads')
        if self.identity_feature_mode not in ('global', 'global_local'):
            raise ValueError('identity_feature_mode must be global or global_local')
        expected_version = ('sakura-tagger-heads-v1' if self.identity_feature_mode == 'global'
                            else 'sakura-tagger-heads-v2-identity-local')
        if self.version != expected_version:
            raise ValueError(f'{self.identity_feature_mode} requires structure version {expected_version}')
        if self.identity_query_count != 4 or type(self.identity_query_count) is not int:
            raise ValueError('Identity-local uses exactly four dedicated queries')
        if any(type(value) is not int or value < 1 for value in
               (self.identity_attention_dim, self.identity_attention_heads)):
            raise ValueError('Identity attention dimensions must be positive integers')
        if self.identity_attention_dim % self.identity_attention_heads:
            raise ValueError('identity_attention_dim must be divisible by identity_attention_heads')
        if not math.isfinite(self.identity_residual_init) or self.identity_residual_init == 0:
            raise ValueError('Identity residual initialization must be finite and nonzero')

    @property
    def identity(self):
        return self.character + self.copyright

    @property
    def general_dim(self):
        return self.general_feature_dim or self.feature_dim

    @property
    def identity_dim(self):
        return self.identity_feature_dim or self.feature_dim

    def to_dict(self):
        # Canonical legacy serialization is deliberately unchanged. This lets
        # old strict checkpoint metadata and state dictionaries load as before.
        result = {name: getattr(self, name) for name in (
            *LOGICAL_DOMAINS, 'feature_dim', 'attention_dim', 'attention_heads', 'query_count', 'version')}
        for name in ('general_feature_dim', 'identity_feature_dim'):
            if getattr(self, name) is not None and getattr(self, name) != self.feature_dim:
                result[name] = getattr(self, name)
        if self.identity_feature_mode == 'global_local':
            result.update({name: getattr(self, name) for name in (
                'identity_feature_mode', 'identity_attention_dim', 'identity_attention_heads',
                'identity_query_count', 'identity_residual_init')})
        return result


def read_head_config(path):
    """Read architecture settings; final class counts must match the vocabulary."""
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    return HeadLayout(**config['head_layout'])
