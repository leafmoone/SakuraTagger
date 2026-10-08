"""Configuration contracts for arbitrarily sized task groups."""
from dataclasses import dataclass
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class HeadSpec:
    group_name: str
    group_type: str
    num_classes: int
    feature_mode: str
    query_count: int = 4
    loss_weight: float = 1.0
    enabled: bool = True
    threshold: float = 0.5

    def __post_init__(self):
        if not self.group_name or self.group_name in {'artist', 'style_embedding'}:
            raise ValueError('Head names must be nonempty and not reserved legacy names')
        if self.group_type not in {'multilabel', 'multiclass', 'binary'}:
            raise ValueError('group_type must be multilabel, multiclass, or binary')
        if self.feature_mode not in {'global', 'local'}:
            raise ValueError('feature_mode must be global or local')
        if self.num_classes < 1 or self.query_count < 1:
            raise ValueError('num_classes and query_count must be positive')
        if self.group_type == 'binary' and self.num_classes != 1:
            raise ValueError('Binary groups use one sigmoid logit')
        if not math.isfinite(self.loss_weight) or self.loss_weight < 0 or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid loss weight or probability threshold')


def validate_specs(specs):
    specs = tuple(specs)
    if len({spec.group_name for spec in specs}) != len(specs):
        raise ValueError('Duplicate group_name in head registry')
    return specs


def read_head_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    specs = validate_specs(HeadSpec(**entry) for entry in config['heads'])
    return specs, config.get('attention', {})
