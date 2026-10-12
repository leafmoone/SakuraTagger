"""Formal fixed-cohort evaluation with class-block score spooling."""
from dataclasses import asdict
from pathlib import Path
import tempfile

import numpy as np

from ..data.release import MULTILABEL_DOMAINS, single_artist_target
from .cohort import identity
from .features import feature_binding, classifier_blocks, score_spool
from .metrics import multilabel_metrics, artist_metrics, style_recall


def validate_cache(model, cache, cohort):
    expected = feature_binding(model, cohort, cache.binding['checkpoint'], cache.binding['preprocessing'])
    if cache.binding != expected or cache.manifest['sample_ids'] != list(cohort.sample_ids):
        raise ValueError('Feature cache is bound to a different model/checkpoint/cohort')


def top_artist_ids(model, cache, cohort, domain):
    """Streaming exact Top-5 across declared class blocks, ties by original ID."""
    n = len(cohort.records)
    values, ids = np.empty((n, 0), np.float32), np.empty((n, 0), np.int64)
    for class_ids, scores in classifier_blocks(model, cache, cohort, domain):
        candidates = np.broadcast_to(np.asarray(class_ids), scores.shape)
        values, ids = np.concatenate((values, scores), 1), np.concatenate((ids, candidates), 1)
        # Class blocks and retained ties remain in ID order; lexsort is explicit.
        order = np.lexsort((ids, -values), axis=1)[:, :5]
        values, ids = np.take_along_axis(values, order, 1), np.take_along_axis(ids, order, 1)
    return ids


def _artist_targets(cohort, domain):
    targets = []
    allowed = set(cohort.class_ids[domain])
    for row in cohort.records:
        result = single_artist_target(row, cohort.release.vocabulary,
                                      release_confirmed=cohort.release.single_artist_confirmed)
        if domain == 'other_artist':
            target = result.target if result.valid else -100
        else:
            confirmed = row.get('single_artist_confirmed')
            confirmed = cohort.release.single_artist_confirmed if confirmed is None else confirmed
            target = result.original_ids[0] if confirmed and not result.conflict and len(result.original_ids) == 1 else -100
        targets.append(target if target in allowed else -100)
    return targets


def _artist_report(model, cache, cohort, domain):
    if not cohort.class_ids[domain]:
        return {'status': 'UNAVAILABLE', 'reason': 'EMPTY_CLASS_SCOPE', 'top1': None, 'top5': None, 'macro_top1': None}
    return artist_metrics(top_artist_ids(model, cache, cohort, domain), _artist_targets(cohort, domain),
                          total_classes=cohort.total_classes[domain])


def select_model(report, *, primary='general.macro_ap', minimums=None):
    """Report-only unless explicit quality/regression floors were configured."""
    def value(path):
        current = report
        for part in path.split('.'):
            current = current.get(part) if isinstance(current, dict) else None
        return current
    score = value(primary)
    if not minimums:
        return {'status': 'REPORT_ONLY', 'primary': primary, 'value': score,
                'reason': 'Real-data quality/regression gates are not configured'}
    constraints = {path: {'value': value(path), 'minimum': float(floor),
                          'passes': value(path) is not None and value(path) >= floor}
                   for path, floor in minimums.items()}
    return {'status': 'PASSES_CONFIGURED_GATES' if score is not None and all(item['passes'] for item in constraints.values()) else 'DOES_NOT_PASS_CONFIGURED_GATES',
            'primary': primary, 'value': score, 'constraints': constraints,
            'note': 'Only these declared gates; not a claim of production readiness'}


def evaluate(model, cache, cohort, *, spool_dir, calibration=None, reference=None,
             style_gallery=None, style_relevant_ids=None, selection=None):
    validate_cache(model, cache, cohort)
    if calibration is not None:
        from .calibration import validate_calibration
        validate_calibration(calibration, cache, cohort)
    recall_policy = ({'mode': 'fixed_uncalibrated_probability', 'threshold': cohort.config.recall_threshold}
                     if calibration is None else {'mode': 'frozen_calibration', 'identity': calibration['identity'],
                        'source_split': calibration['split'], 'method': calibration['objective']})
    evaluation_settings = {'config': asdict(cohort.config), 'effective_recall_policy': recall_policy}
    report = {'evaluation_settings': evaluation_settings,
              'evaluation_identity': identity({'cohort': cohort.identity, 'settings': evaluation_settings}),
              'cohort': cohort.document, 'cohort_identity': cohort.identity,
              'checkpoint': cache.binding['checkpoint'], 'model_sha256': cache.binding['model_sha256'],
              'leakage': cohort.leakage, 'preprocessing': cache.binding['preprocessing']}
    Path(spool_dir).mkdir(parents=True, exist_ok=True)
    for domain in MULTILABEL_DOMAINS:
        with tempfile.TemporaryDirectory(prefix='eval-scores-', dir=spool_dir) as temporary:
            scores = score_spool(model, cache, cohort, domain, Path(temporary) / 'scores.npy')
            threshold = cohort.config.recall_threshold
            if calibration is not None:
                fitted = calibration['domains'][domain]
                threshold = {int(label): value for label, value in fitted['per_class'].items()}
            report[domain] = multilabel_metrics(scores, cohort.class_ids[domain], cohort.records, domain,
                cohort.train_support[domain], protocol=cohort.config.protocol, total_classes=cohort.total_classes[domain],
                min_positives=cohort.config.min_positives, min_negatives=cohort.config.min_negatives, threshold=threshold)
            del scores
    for domain in ('other_artist', 'artist'):
        report[domain] = _artist_report(model, cache, cohort, domain)
    report['artist']['reference_difference'] = {'status': 'NOT_VERIFIED', 'reason': 'NO_ORIGINAL_REFERENCE_MODEL_CACHE'}
    if reference is not None:
        reference_model, reference_cache = reference
        validate_cache(reference_model, reference_cache, cohort)
        original = _artist_report(reference_model, reference_cache, cohort, 'artist')
        report['artist']['reference_difference'] = {'reference_checkpoint': reference_cache.binding['checkpoint'],
            'status': 'AVAILABLE' if original['top1'] is not None and report['artist']['top1'] is not None else 'UNAVAILABLE',
            'deltas': {key: report['artist'][key] - original[key] if original[key] is not None and report['artist'][key] is not None else None
                       for key in ('top1', 'top5', 'macro_top1')}}
    report['style'] = style_recall(cache.arrays['style'], cohort.sample_ids, style_gallery,
                                 relevant_ids=style_relevant_ids, ram_bytes=cohort.config.ram_bytes)
    report['selection'] = select_model(report, **(selection or {}))
    return report
