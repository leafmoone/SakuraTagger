"""Validation-only frozen probability thresholds with explicit support fallback."""
from pathlib import Path
import tempfile

import numpy as np
import torch

from ..data.release import MULTILABEL_DOMAINS
from .cohort import identity
from .features import score_spool
from .metrics import labels_for_class

VERSION = 'sakura-tagger-thresholds-v1'


def best_f1_threshold(labels, logits):
    labels, logits = np.asarray(labels, dtype=bool), np.asarray(logits)
    if not labels.any() or labels.all() or not np.isfinite(logits).all():
        raise ValueError('Threshold fitting requires finite positive and negative evidence')
    probabilities = torch.from_numpy(np.array(logits, dtype=np.float32)).sigmoid().numpy()
    order = np.argsort(-probabilities, kind='stable')
    p, y = probabilities[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(y, dtype=np.int64)[ends]
    f1 = 2 * tp / (ends + 1 + int(y.sum()))
    best = int(np.argmax(f1))  # Highest threshold wins equal-F1 ties.
    return float(p[ends[best]]), float(f1[best])


def fit_calibration(model, cache, cohort, *, spool_dir, min_class_positives=20, min_class_negatives=20):
    from .evaluator import validate_cache
    if cohort.split not in ('val', 'calibration'):
        raise ValueError('Thresholds may be fitted only on val/calibration, never test')
    if any(type(value) is not int or value < 1 for value in (min_class_positives, min_class_negatives)):
        raise ValueError('Explicit positive per-class calibration support limits required')
    validate_cache(model, cache, cohort)
    result = {'version': VERSION, 'checkpoint': cache.binding['checkpoint'], 'model_sha256': cache.binding['model_sha256'],
              'vocab_sha256': cohort.release.vocabulary.sha256, 'release': cohort.release.identity,
              'cohort_identity': cohort.identity, 'split': cohort.split, 'protocol': cohort.config.protocol,
              'sources': cohort.sources, 'class_ids': cohort.class_ids, 'preprocessing': cache.binding['preprocessing'],
              'sample_sha256': [row['image_sha256'] for row in cohort.records],
              'known_family_ids': sorted({row['family_id'] for row in cohort.records if row.get('family_id') not in (None, '', 'UNKNOWN')}),
              'min_class_positives': min_class_positives, 'min_class_negatives': min_class_negatives,
              'objective': 'maximum observed/verified F1; highest threshold tie-break', 'domains': {}}
    Path(spool_dir).mkdir(parents=True, exist_ok=True)
    for domain in MULTILABEL_DOMAINS:
        with tempfile.TemporaryDirectory(prefix='calibration-scores-', dir=spool_dir) as temporary:
            scores = score_spool(model, cache, cohort, domain, Path(temporary) / 'scores.npy')
            y_all, z_all, columns = [], [], []
            for column, class_id in enumerate(cohort.class_ids[domain]):
                y, known = labels_for_class(cohort.records, domain, class_id, cohort.config.protocol)
                y, z = y[known], np.asarray(scores[:, column])[known]
                y_all.append(y); z_all.append(z); columns.append((class_id, y, z))
            y = np.concatenate(y_all) if y_all else np.empty(0, bool)
            z = np.concatenate(z_all) if z_all else np.empty(0)
            fitted = bool(y.any() and not y.all())
            domain_threshold, domain_f1 = best_f1_threshold(y, z) if fitted else (.5, None)
            per_class, methods = {}, {}
            for class_id, y, z in columns:
                if int(y.sum()) >= min_class_positives and int((~y).sum()) >= min_class_negatives:
                    threshold, _ = best_f1_threshold(y, z)
                    method = 'class_validation_f1'
                else:
                    threshold = domain_threshold
                    method = 'domain_validation_fallback' if fitted else 'uncalibrated_default'
                per_class[str(class_id)] = threshold
                methods[str(class_id)] = {'method': method, 'positive_support': int(y.sum()), 'negative_support': int((~y).sum())}
            result['domains'][domain] = {'domain_threshold': domain_threshold, 'domain_f1': domain_f1,
                'status': 'FITTED' if fitted else 'UNAVAILABLE_USING_UNCALIBRATED_DEFAULT',
                'per_class': per_class, 'class_support_and_method': methods}
            del scores
    result['identity'] = identity(result)
    return result


def validate_calibration(calibration, cache, cohort):
    document = dict(calibration)
    digest = document.pop('identity', None)
    if (document.get('version') != VERSION or digest != identity(document)
            or document.get('split') not in ('val', 'calibration')
            or document.get('checkpoint') != cache.binding['checkpoint']
            or document.get('model_sha256') != cache.binding['model_sha256']
            or document.get('vocab_sha256') != cohort.release.vocabulary.sha256
            or document.get('release') != cohort.release.identity
            or document.get('protocol') != cohort.config.protocol or document.get('class_ids') != cohort.class_ids
            or document.get('sources') != cohort.sources or document.get('preprocessing') != cache.binding['preprocessing']):
        raise ValueError('Frozen calibration identity/checkpoint/protocol/scope mismatch')
    if cohort.split == 'test':
        sha = {row['image_sha256'] for row in cohort.records}
        families = {row['family_id'] for row in cohort.records if row.get('family_id') not in (None, '', 'UNKNOWN')}
        if sha.intersection(document['sample_sha256']) or families.intersection(document['known_family_ids']):
            raise ValueError('Calibration/test SHA or known-family leakage')
