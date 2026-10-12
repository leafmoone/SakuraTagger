"""Fixed evaluation identities, independent-train support and leakage evidence."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from ..data.release import MULTILABEL_DOMAINS
from .metrics import PROTOCOLS

DOMAINS = (*MULTILABEL_DOMAINS, 'other_artist', 'artist')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class EvaluationConfig:
    protocol: str = 'observed_label_proxy'
    min_positives: int = 1
    min_negatives: int = 1
    recall_threshold: float = .5
    max_samples: int = 2048
    class_block: int = 256
    row_batch: int = 64
    ram_bytes: int = 512 * 1024**2
    disk_bytes: int = 2 * 1024**3

    def __post_init__(self):
        if self.protocol not in PROTOCOLS:
            raise ValueError('Unknown evaluation protocol')
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 1 for name in
               ('min_positives', 'min_negatives', 'max_samples', 'class_block', 'row_batch', 'ram_bytes', 'disk_bytes')):
            raise ValueError('Positive evaluation limits and budgets required')
        if not 0 <= self.recall_threshold <= 1:
            raise ValueError('Invalid recall threshold')


def leakage_evidence(records, reference_records):
    sha = {row['image_sha256'] for row in records}
    families = {row['family_id'] for row in records if row.get('family_id') not in (None, '', 'UNKNOWN')}
    sha_overlap, family_overlap = set(), set()
    unknown_families = sum(row.get('family_id') in (None, '', 'UNKNOWN') for row in records)
    near = Counter(row.get('near_duplicate_status', 'UNKNOWN') for row in records)
    reference_count = 0
    for row in reference_records:
        reference_count += 1
        if row['image_sha256'] in sha:
            sha_overlap.add(row['image_sha256'])
        family = row.get('family_id')
        if family in families:
            family_overlap.add(family)
        unknown_families += int(family in (None, '', 'UNKNOWN'))
        near[row.get('near_duplicate_status', 'UNKNOWN')] += 1
    return {'sha_overlap': sorted(sha_overlap), 'known_family_overlap': sorted(family_overlap),
            'reference_samples': reference_count, 'samples_without_family_evidence': unknown_families,
            'sha_disjoint': not sha_overlap, 'declared_family_disjoint': not family_overlap if not unknown_families else 'NOT_VERIFIED',
            'near_duplicate_disjoint': 'NOT_VERIFIED', 'near_duplicate_status_counts': dict(near),
            'note': 'Status strings without verified duplicate-group IDs cannot establish near-duplicate disjointness.'}


class EvaluationCohort:
    def __init__(self, index, sample_ids, *, split, config=None, class_ids=None,
                 source_membership=None, source_filter=None):
        self.config = config or EvaluationConfig()
        if split not in ('val', 'test', 'calibration'):
            raise ValueError('Evaluation cohort must explicitly use val/test/calibration')
        if (not sample_ids or len(sample_ids) > self.config.max_samples
                or len(set(sample_ids)) != len(sample_ids) or any(not isinstance(value, str) for value in sample_ids)):
            raise ValueError('A nonempty fixed unique cohort within max_samples is required')
        self.sample_ids, self.split, self.release = tuple(sample_ids), split, index.release
        chosen = set(sample_ids)
        records = {row['sample_id']: row for _, row in index.iter_records(split=split) if row['sample_id'] in chosen}
        if set(records) != chosen:
            raise ValueError('Cohort IDs do not all belong to the declared published split')
        self.records = tuple(deepcopy(records[value]) for value in sample_ids)
        self.total_classes = {name: len(index.release.vocabulary.class_id_mapping[name]) for name in DOMAINS}
        if class_ids is not None and set(class_ids) != set(DOMAINS):
            raise ValueError('Explicit class scope must declare all five classification domains')
        self.sources = []
        if source_filter is not None:
            if source_membership is None or source_membership.vocab_sha256 != self.release.vocabulary.sha256:
                raise ValueError('Source scope needs a real vocabulary-bound SourceMembership')
            self.sources = list(source_membership.normalize_sources(source_filter))
        self.class_ids = {}
        for name, count in self.total_classes.items():
            ids = list(range(count)) if class_ids is None else list(class_ids[name])
            if ids != sorted(set(ids)) or any(type(value) is not int or not 0 <= value < count for value in ids):
                raise ValueError('Class scopes require sorted unique original in-range IDs')
            if self.sources:
                allowed = set(source_membership.allowed_ids(name, self.sources))
                ids = [value for value in ids if value in allowed]
            self.class_ids[name] = ids
        self.train_support = {name: index.positive_support(name) for name in MULTILABEL_DOMAINS}
        self.leakage = leakage_evidence(self.records, (row for _, row in index.iter_records(split='train')))
        if self.leakage['sha_overlap'] or self.leakage['known_family_overlap']:
            raise ValueError('Evaluation/train SHA or known-family leakage detected')
        document = {'release': self.release.identity, 'sample_ids': list(self.sample_ids), 'split': split,
                    'protocol': self.config.protocol, 'min_positives': self.config.min_positives,
                    'min_negatives': self.config.min_negatives, 'class_ids': self.class_ids,
                    'sources': self.sources, 'labels_sha256': identity(self.records)}
        self.identity = identity(document)
        self.document = document
