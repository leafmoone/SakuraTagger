"""Versioned three-module layout. Vocabulary order is supplied, never sorted."""
from dataclasses import asdict, dataclass
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

    def __post_init__(self):
        for name in (*LOGICAL_DOMAINS, 'feature_dim', 'attention_dim', 'attention_heads', 'query_count'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.attention_dim % self.attention_heads:
            raise ValueError('attention_dim must be divisible by attention_heads')
        if self.version != 'sakura-tagger-heads-v1':
            raise ValueError('Unsupported head layout version')

    @property
    def identity(self):
        return self.character + self.copyright

    def to_dict(self):
        return asdict(self)


def read_head_config(path):
    """Read architecture settings; final class counts must match the vocabulary."""
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    return HeadLayout(**config['head_layout'])
