"""Bounded synthetic exact AP, feature cache and frozen calibration acceptance."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score
import torch

from .data import build_local_index
from .model import MultiTaskModel, KaloscopeBackbone
from .selfcheck import synthetic_fixture
from .selfcheck_release import reference_release, reference_row
from .evaluation import (EvaluationConfig, EvaluationCohort, FeatureCache, build_feature_cache,
                         exact_average_precision, merge_score_partitions, multilabel_metrics,
                         evaluate, fit_calibration, validate_calibration, leakage_evidence, style_recall)
from .evaluation.features import classifier_blocks, score_spool
from .evaluation.evaluator import validate_cache


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid evaluation operation accepted')


def check_evaluation():
    torch.set_num_threads(1)
    tied = np.asarray([[.4, .2, .8], [.4, .8, .2], [.9, .8, .8], [.1, .2, .1], [.9, .1, .2]], np.float32)
    labels = np.asarray([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]], bool)
    errors = [abs(exact_average_precision(labels[:, i], tied[:, i]) - average_precision_score(labels[:, i], tied[:, i])) for i in range(3)]
    errors.append(abs(exact_average_precision(labels.ravel(), tied.ravel()) - average_precision_score(labels, tied, average='micro')))
    assert max(errors) < 1e-12
    # This case has a genuinely different global micro result from a class mean.
    assert abs(np.mean([exact_average_precision(labels[:, i], tied[:, i]) for i in range(3)]) - exact_average_precision(labels.ravel(), tied.ravel())) > 1e-3
    ids = [f's{i}' for i in range(len(labels))]
    merged = merge_score_partitions([(ids[::2], tied[::2]), (ids[1::2], tied[1::2])], expected_ids=ids)
    np.testing.assert_array_equal(tied, merged)
    _reject(lambda: merge_score_partitions([(ids, tied), (ids[:1], tied[:1])], expected_ids=ids))
    protocol_rows = [reference_row(i, general_ids=np.flatnonzero(labels[i]).tolist(),
                                   general_negative_ids=[0] if i == 1 else []) for i in range(5)]
    proxy = multilabel_metrics(tied, [0, 1, 2], protocol_rows, 'general', np.asarray([1, 7, 100]),
                              protocol='observed_label_proxy', total_classes=3)
    verified = multilabel_metrics(tied, [0, 1, 2], protocol_rows, 'general', np.asarray([1, 7, 100]),
                                 protocol='verified_scope', total_classes=3)
    assert proxy['evaluated_classes'] == 3 and verified['evaluated_classes'] == 1
    assert verified['per_class']['1']['reason'] == 'NO_COMPARABLE_NEGATIVE'
    assert proxy['frequency_bands']['1']['positive_labels'] == 2
    assert abs(proxy['micro_ap'] - average_precision_score(labels, tied, average='micro')) < 1e-12
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = [reference_row(i, general_ids=[0] if i < 4 else [1], character_ids=[i % 2]) for i in range(6)]
        rows += [reference_row(6 + i, split='val', general_ids=[i % 3], general_negative_ids=[(i + 1) % 3],
                    character_ids=[i % 2], character_negative_ids=[(i + 1) % 2], copyright_ids=[0] if i % 2 else [],
                    copyright_negative_ids=[0] if not i % 2 else [],
                    other_artist_ids=[1] if i < 3 else [], original_artist_ids=[2] if i >= 3 else []) for i in range(6)]
        rows += [reference_row(12 + i, split='test', general_ids=[i % 3], character_ids=[i % 2],
                              other_artist_ids=[1] if i % 2 else [], original_artist_ids=[2] if not i % 2 else []) for i in range(4)]
        release = reference_release(root, rows)
        index = build_local_index(release, root / 'index.sqlite')
        config = EvaluationConfig(class_block=2, row_batch=2, ram_bytes=2 * 1024**2, disk_bytes=4 * 1024**2)
        cohort = EvaluationCohort(index, [row['sample_id'] for row in rows if row['split'] == 'val'], split='val', config=config)
        assert cohort.train_support['general'][:3].tolist() == [4, 2, 0]
        assert cohort.leakage['sha_disjoint'] and cohort.leakage['declared_family_disjoint'] == 'NOT_VERIFIED'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            model = MultiTaskModel(KaloscopeBackbone(synthetic_fixture()), release.vocabulary.layout(feature_dim=32, attention_dim=16), other_artist_mode='single_label').eval()
            images = torch.randn(6, 3, 32, 48)
        batches = [{'images': images[i:i + 2], 'sample_ids': list(cohort.sample_ids[i:i + 2])} for i in range(0, 6, 2)]
        with patch.object(model.backbone, 'forward_features', wraps=model.backbone.forward_features) as forward:
            cache = build_feature_cache(model, batches, cohort, root / 'features', checkpoint_id='synthetic-checkpoint-a',
                                        preprocessing={'protocol': 'synthetic-full-frame-32x48', 'normalization': 'ImageNet'})
            assert forward.call_count == 3
        output = model(images)
        block_error = 0.
        for domain in ('general', 'character', 'copyright', 'other_artist', 'artist'):
            complete = np.concatenate([scores for _, scores in classifier_blocks(model, cache, cohort, domain)], axis=1)
            expected = output.logits[domain].detach().numpy()
            np.testing.assert_allclose(complete, expected, rtol=2e-5, atol=2e-6)
            block_error = max(block_error, float(np.max(np.abs(complete - expected))))
        report = evaluate(model, cache, cohort, spool_dir=root / 'spool', reference=(model, cache))
        assert report['evaluation_settings']['effective_recall_policy']['threshold'] == .5
        changed_threshold = deepcopy(cohort)
        changed_threshold.config = replace(cohort.config, recall_threshold=.75)
        different_recall = evaluate(model, cache, changed_threshold, spool_dir=root / 'spool')
        assert different_recall['cohort_identity'] == report['cohort_identity']
        assert different_recall['evaluation_identity'] != report['evaluation_identity']
        assert different_recall['general']['per_class']['0']['recall_threshold'] == .75
        assert report['selection']['status'] == 'REPORT_ONLY' and report['style']['status'] == 'NOT_VERIFIED'
        assert report['other_artist']['valid_samples'] == report['artist']['valid_samples'] == 3
        assert all(value == 0 for value in report['artist']['reference_difference']['deltas'].values())
        calibration = fit_calibration(model, cache, cohort, spool_dir=root / 'spool', min_class_positives=3, min_class_negatives=3)
        assert calibration['domains']['general']['class_support_and_method']['0']['method'] == 'domain_validation_fallback'
        calibrated = evaluate(model, cache, cohort, spool_dir=root / 'spool', calibration=calibration)
        assert calibrated['general']['macro_ap'] == report['general']['macro_ap']
        assert calibrated['evaluation_settings']['effective_recall_policy']['identity'] == calibration['identity']
        test = EvaluationCohort(index, [row['sample_id'] for row in rows if row['split'] == 'test'], split='test', config=config)
        test_batches = [{'images': images[i:i + 2], 'sample_ids': list(test.sample_ids[i:i + 2])} for i in range(0, 4, 2)]
        test_cache = build_feature_cache(model, test_batches, test, root / 'test-features', checkpoint_id='synthetic-checkpoint-a',
                                             preprocessing=cache.binding['preprocessing'])
        validate_calibration(calibration, test_cache, test)
        assert evaluate(model, test_cache, test, spool_dir=root / 'spool', calibration=calibration)['general']['protocol'] == 'observed_label_proxy'
        _reject(lambda: fit_calibration(model, test_cache, test, spool_dir=root / 'spool'))
        _reject(lambda: FeatureCache(root / 'features', binding={**cache.binding, 'checkpoint': 'another-checkpoint'}))
        changed = deepcopy(model)
        with torch.no_grad():
            changed.general.classifier.bias[0] += 1
        _reject(lambda: validate_cache(changed, cache, cohort))
        tiny_budget = deepcopy(cohort); tiny_budget.config = replace(config, ram_bytes=10)
        _reject(lambda: score_spool(model, cache, tiny_budget, 'general', root / 'refused.npy'))
        tiny_disk = deepcopy(cohort); tiny_disk.config = replace(config, disk_bytes=10)
        _reject(lambda: build_feature_cache(model, batches, tiny_disk, root / 'refused-features', checkpoint_id='a', preprocessing={'v': 1}))
        assert not (root / 'refused.npy').exists() and not (root / 'refused-features').exists()
        _reject(lambda: EvaluationCohort(index, [cohort.sample_ids[0]] * 2, split='val'))
        _reject(lambda: EvaluationCohort(index, [cohort.sample_ids[0]], split='train'))
        leak = leakage_evidence([rows[0]], [rows[0]])
        assert leak['sha_overlap']
        family = leakage_evidence([{**rows[0], 'family_id': 'known-family'}], [{**rows[-1], 'family_id': 'known-family'}])
        assert family['known_family_overlap'] == ['known-family']
        retrieval = style_recall(np.eye(3), ['q0', 'q1', 'q2'], (np.eye(3), ['g0', 'g1', 'g2']),
                                 relevant_ids=[['g0'], ['g1'], ['g2']], ks=(1, 2), ram_bytes=10000)
        assert retrieval['recall_at_k'] == {'1': 1., '2': 1.}
        index.close()
        return {'tied_class_and_true_global_micro_sklearn_max_error': max(errors),
                'partition_merge_global_score_equivalence': 'PASS', 'protocol_and_unavailable_reasons': 'PASS',
                'train_only_frequency_bands': 'PASS', 'single_backbone_feature_cache_batches': 3,
                'cached_classifier_block_max_logit_error': block_error, 'cache_checkpoint_binding_and_budget_refusal': 'PASS',
                'separate_trusted_artist_metrics_and_reference_deltas': 'PASS',
                'validation_only_calibration_low_support_fallback_frozen_test': 'PASS',
                'effective_threshold_calibration_report_identity': 'PASS',
                'sha_and_known_family_leakage_evidence': 'PASS', 'synthetic_style_gallery_recall': 'PASS',
                'real_data_quality': 'NOT_VERIFIED', 'real_quality_gates': 'NOT_CONFIGURED_REPORT_ONLY'}


if __name__ == '__main__':
    print(json.dumps(check_evaluation(), indent=2))
