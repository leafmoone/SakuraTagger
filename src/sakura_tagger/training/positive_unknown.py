"""Sparse sampled weak-negative engineering objective, not unbiased PU risk.

Unknown labels remain unknown data. Sampling reduces loss terms, not the full
classifier matrix multiplication performed by the model. Legacy ASL/CE APIs are
unchanged in losses.py.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..data.contract import _ids, _sha
from ..data.release import MULTILABEL_DOMAINS, collate_artist_targets
from ..data.source_membership import SourceMembership
from ..model.registry import LOGICAL_DOMAINS
from .losses import (LossConfig, differentiable_zero, masked_asymmetric_loss,
                     masked_single_label_loss)


@dataclass(frozen=True)
class WeakNegativeConfig:
    # Explicit engineering hyperparameters, not values tuned on real data.
    lambda_u: float
    lambda_n: float
    sample_size: int
    mode: str = 'sampled_weak_negative'
    chunk_size: int = 4096
    recompute: bool = True

    def __post_init__(self):
        if self.mode not in ('sampled_weak_negative', 'full_reference', 'chunked_reference', 'masked_asl'):
            raise ValueError('Unsupported sparse multilabel loss mode')
        if any(not math.isfinite(value) or value < 0 for value in (self.lambda_u, self.lambda_n)):
            raise ValueError('Weak/explicit negative weights must be finite and nonnegative')
        if type(self.sample_size) is not int or self.sample_size < 1:
            raise ValueError('sample_size must be an explicit positive integer')
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError('chunk_size must be positive')


@dataclass(frozen=True)
class SparseLossConfig:
    domains: dict
    weights: dict = field(default_factory=lambda: {domain: 1.0 for domain in LOGICAL_DOMAINS})
    weak_negative_scope: str = 'domain'
    source_filter: tuple = ()
    other_artist_mode: str = 'single_label'
    seed: int = 0

    def __post_init__(self):
        if set(self.domains) != set(MULTILABEL_DOMAINS) or any(not isinstance(item, WeakNegativeConfig) for item in self.domains.values()):
            raise ValueError('Explicit objective settings are required for all three multilabel domains')
        if self.other_artist_mode != 'single_label':
            raise ValueError('Sparse Release training uses trusted single_label CE; legacy multilabel uses LossConfig')
        if self.weak_negative_scope not in ('domain', 'source_membership'):
            raise ValueError('Unknown weak-negative scope')
        if self.weak_negative_scope == 'source_membership' and not self.source_filter:
            raise ValueError('Source-aware scope requires explicit run source_filter')
        if self.weak_negative_scope == 'domain' and self.source_filter:
            raise ValueError('source_filter requires source_membership scope')
        if set(self.weights) - set(LOGICAL_DOMAINS) or any(not math.isfinite(w) or w < 0 for w in self.weights.values()):
            raise ValueError('Invalid task weights')
        if type(self.seed) is not int:
            raise ValueError('Sampling seed must be an integer')

    @classmethod
    def from_dict(cls, document):
        options = dict(document)
        options['domains'] = {name: WeakNegativeConfig(**value) for name, value in options['domains'].items()}
        options['source_filter'] = tuple(options.get('source_filter', ()))
        return cls(**options)

    def to_dict(self):
        result = asdict(self)
        result['source_filter'] = list(self.source_filter)
        return result


@dataclass(frozen=True)
class SamplingKey:
    sample_id: str
    epoch: int = 0
    occurrence_id: int = 0
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError('A stable sample_id is required for sampling')
        if any(type(value) is not int or value < 0 for value in (self.epoch, self.occurrence_id)) or type(self.seed) is not int:
            raise ValueError('Epoch/occurrence must be nonnegative integers and seed an integer')

    def for_domain(self, domain):
        payload = json.dumps([self.seed, self.sample_id, self.epoch, self.occurrence_id, domain], separators=(',', ':')).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')


@dataclass(frozen=True)
class WeakNegativeScope:
    ids_by_domain: dict
    identity: dict
    mode: str
    sources: tuple

    @classmethod
    def from_index(cls, index, *, source_membership=None, source_filter=None):
        mode = 'source_membership' if source_filter is not None else 'domain'
        sources = ()
        if mode == 'source_membership':
            if not isinstance(source_membership, SourceMembership):
                raise ValueError('Source-aware weak negatives require a genuinely loaded SourceMembership')
            if source_membership.vocab_sha256 != index.release.vocabulary.sha256:
                raise ValueError('Weak-negative source membership vocabulary mismatch')
            sources = source_membership.normalize_sources(source_filter)
        ids = {}
        for domain in MULTILABEL_DOMAINS:
            supported = np.flatnonzero(index.positive_support(domain) > 0)
            if sources:
                supported = np.intersect1d(supported, source_membership.allowed_ids(domain, sources), assume_unique=True)
            supported = supported.astype(np.int64, copy=False)
            supported.flags.writeable = False
            ids[domain] = supported
        identity = {'release': index.release.identity, 'mode': mode, 'sources': list(sources),
                    'eligible_sha256': _sha({domain: value.tolist() for domain, value in ids.items()})}
        return cls(ids, identity, mode, sources)


@dataclass(frozen=True)
class DomainLoss:
    numerator: torch.Tensor
    valid_count: int
    finite: torch.Tensor | None = None

    @property
    def mean(self):
        return self.numerator / self.valid_count if self.valid_count else self.numerator


@dataclass(frozen=True)
class SparseLossResult:
    domains: dict
    weights: dict
    scope_identity: dict

    @property
    def numerators(self):
        return {name: item.numerator for name, item in self.domains.items()}

    @property
    def valid_counts(self):
        return {name: item.valid_count for name, item in self.domains.items()}

    @property
    def total(self):
        result = differentiable_zero(next(iter(self.domains.values())).numerator.reshape(1, 1))
        for name, item in self.domains.items():
            result = result + self.weights.get(name, 1.0) * item.mean
        return result


def _validate_sparse_ids(positives, negatives, forbidden, class_count):
    positives = _ids(list(positives), class_count, 'positive IDs')
    negatives = _ids(list(negatives), class_count, 'explicit negative IDs')
    forbidden = _ids(list(forbidden), class_count, 'forbidden IDs')
    if set(positives) & set(negatives):
        raise ValueError('P and N must not overlap')
    return positives, negatives, forbidden


def _validate_allowed_ids(allowed_ids, class_count):
    allowed = np.asarray(allowed_ids)
    if allowed.ndim == 1 and allowed.size == 0:
        allowed = allowed.astype(np.int64)
    if allowed.ndim != 1 or allowed.dtype.kind not in 'iu' or (allowed < 0).any() or (allowed >= class_count).any():
        raise ValueError('Eligible unknown IDs must be a one-dimensional in-range integer array')
    if len(allowed) > 1 and not (allowed[1:] > allowed[:-1]).all():
        raise ValueError('Eligible unknown IDs must be unique and sorted')
    return allowed


def _eligible_unknown(positives, negatives, forbidden, allowed):
    excluded = np.asarray(sorted(set(positives) | set(negatives) | set(forbidden)), dtype=np.int64)
    return allowed[~np.isin(allowed, excluded, assume_unique=True)].astype(np.int64, copy=False)


def unknown_ids(positives, negatives, forbidden, allowed_ids, class_count):
    positives, negatives, forbidden = _validate_sparse_ids(positives, negatives, forbidden, class_count)
    return _eligible_unknown(positives, negatives, forbidden, _validate_allowed_ids(allowed_ids, class_count))


def sample_unknown(ids, size, key, domain):
    """Uniform without replacement; stable across workers and resume ordering."""
    if size >= len(ids):
        return ids
    generator = np.random.default_rng(key.for_domain(domain))
    # Sorting changes calculation order only, never inclusion probabilities.
    return np.sort(generator.choice(ids, size=size, replace=False))


def _softplus_mean(logits, ids, *, positive, chunk_size, recompute, finite_checks):
    result = differentiable_zero(logits)
    if not len(ids):
        return result
    sign = -1 if positive else 1
    for start in range(0, len(ids), chunk_size):
        selected = torch.as_tensor(np.asarray(ids[start:start + chunk_size]).copy(), dtype=torch.long, device=logits.device)
        values = logits.index_select(0, selected)
        # Excluded/unknown sentinels never enter arithmetic. Nonfinite supervised
        # logits are a genuine training failure for the trainer to coordinate.
        finite_checks.append(torch.isfinite(values).all())
        def term(part):
            return F.softplus(part.float() * sign).sum()
        value = checkpoint(term, values, use_reentrant=False) if recompute and values.requires_grad and torch.is_grad_enabled() else term(values)
        result = result + value
    return result / len(ids)


def _row_terms(logits, positives, negatives, selected_unknown, config, finite_checks=None):
    own_checks = finite_checks is None
    finite_checks = [] if own_checks else finite_checks
    chunk = len(logits) if config.mode == 'full_reference' else config.chunk_size
    value = _softplus_mean(logits, positives, positive=True, chunk_size=chunk, recompute=config.recompute, finite_checks=finite_checks)
    if config.lambda_n:
        value = value + config.lambda_n * _softplus_mean(logits, negatives, positive=False, chunk_size=chunk,
                                                        recompute=config.recompute, finite_checks=finite_checks)
    if config.lambda_u and len(positives):
        value = value + config.lambda_u * _softplus_mean(logits, selected_unknown, positive=False, chunk_size=chunk,
                                                        recompute=config.recompute, finite_checks=finite_checks)
    if own_checks and finite_checks and not torch.stack(finite_checks).all():
        raise ValueError('Nonfinite supervised/selected logits')
    return value


def positive_unknown_loss(logits, positives, *, negatives=None, forbidden=None, allowed_ids,
                          config, keys=None, domain='general', defer_finite_check=False):
    """Sum per-image P/N/U means plus valid-image count for DDP composition."""
    if logits.ndim != 2 or len(positives) != len(logits):
        raise ValueError('Sparse loss requires [B,C] logits and one sparse row per image')
    negatives = [[] for _ in positives] if negatives is None else negatives
    forbidden = [[] for _ in positives] if forbidden is None else forbidden
    if len(negatives) != len(logits) or len(forbidden) != len(logits):
        raise ValueError('Sparse supervision batch sizes disagree')
    if config.mode == 'sampled_weak_negative' and (keys is None or len(keys) != len(logits)):
        raise ValueError('Sampled loss needs stable sample/epoch/occurrence keys')
    allowed = _validate_allowed_ids(allowed_ids, logits.shape[-1])
    numerator, count, finite_checks = differentiable_zero(logits), 0, []
    for row, (p, n, blocked) in enumerate(zip(positives, negatives, forbidden)):
        p, n, blocked = _validate_sparse_ids(p, n, blocked, logits.shape[-1])
        valid = bool(p) or bool(n) and (config.mode == 'masked_asl' or config.lambda_n > 0)
        if not valid:
            continue
        count += 1
        if config.mode == 'masked_asl':
            known = torch.tensor(list(p) + list(n), dtype=torch.long, device=logits.device)
            target = torch.tensor([1.] * len(p) + [0.] * len(n), device=logits.device).unsqueeze(0)
            prediction = logits[row].index_select(0, known).unsqueeze(0)
            finite_checks.append(torch.isfinite(prediction).all())
            value, _ = masked_asymmetric_loss(prediction, target, torch.ones_like(target, dtype=torch.bool),
                                               config=LossConfig(chunk_size=config.chunk_size, recompute=config.recompute))
        else:
            selected = np.empty(0, dtype=np.int64)
            if p and config.lambda_u:
                selected = _eligible_unknown(p, n, blocked, allowed)
                if config.mode == 'sampled_weak_negative':
                    selected = sample_unknown(selected, config.sample_size, keys[row], domain)
            value = _row_terms(logits[row], p, n, selected, config, finite_checks)
        numerator = numerator + value
    finite = torch.stack(finite_checks).all() if finite_checks else torch.ones((), dtype=torch.bool, device=logits.device)
    if not defer_finite_check and not finite:
        raise ValueError('Nonfinite supervised/selected logits')
    return DomainLoss(numerator, count, finite)


def sparse_valid_counts(records, config, release):
    """Supervision-only denominators; never retain graphs to count a window."""
    counts = {}
    for domain in MULTILABEL_DOMAINS:
        setting = config.domains[domain]
        counts[domain] = sum(bool(row.get(domain + '_ids')) or
                             bool(row.get(domain + '_negative_ids')) and (setting.mode == 'masked_asl' or setting.lambda_n > 0)
                             for row in records)
    counts['other_artist'] = int(collate_artist_targets(records, release)['valid'].sum())
    return counts


def sparse_multitask_loss(logits, records, *, release, scope, config, epoch=0, occurrence_ids=None, defer_finite_check=False):
    """Formal sparse boundary. Other Artist remains an independent trusted CE."""
    if set(LOGICAL_DOMAINS) - set(logits):
        raise ValueError('Sparse multitask training requires all four logical outputs')
    for domain in LOGICAL_DOMAINS:
        if logits[domain].ndim != 2 or logits[domain].shape[1] != len(release.vocabulary.document['domains'][domain]):
            raise ValueError('Logit dimensions must match the bound five-space Vocabulary')
    if scope.identity['release'] != release.identity or scope.mode != config.weak_negative_scope:
        raise ValueError('Weak-negative scope does not match this Release/run configuration')
    if tuple(sorted(set(config.source_filter))) != scope.sources:
        raise ValueError('Weak-negative source scope must equal explicit configured sources')
    if occurrence_ids is None:
        occurrence_ids = [0] * len(records)
    if len(occurrence_ids) != len(records) or any(len(value) != len(records) for value in logits.values()):
        raise ValueError('Batch, output, and occurrence sizes disagree')
    keys = [SamplingKey(row['sample_id'], epoch, occurrence, config.seed) for row, occurrence in zip(records, occurrence_ids)]
    domains = {}
    for domain in MULTILABEL_DOMAINS:
        domains[domain] = positive_unknown_loss(logits[domain], [row.get(domain + '_ids') or [] for row in records],
            negatives=[row.get(domain + '_negative_ids') or [] for row in records],
            forbidden=[row.get(domain + '_forbidden_ids') or [] for row in records],
            allowed_ids=scope.ids_by_domain[domain], config=config.domains[domain], keys=keys, domain=domain, defer_finite_check=defer_finite_check)
    artist = collate_artist_targets(records, release)
    loss, count = masked_single_label_loss(logits['other_artist'], artist['targets'], artist['valid'])
    finite = torch.isfinite(logits['other_artist'][artist['valid'].to(logits['other_artist'].device)]).all()
    if not defer_finite_check and not finite:
        raise ValueError('Nonfinite supervised/selected artist logits')
    domains['other_artist'] = DomainLoss(loss * count, count, finite)
    return SparseLossResult(domains, config.weights, scope.identity)
