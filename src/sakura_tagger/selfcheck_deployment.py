"""Strict CPU deployment round trips and non-parametric output policy acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile

from PIL import Image
import torch

from .data import (SourceMembership, Vocabulary, BucketConfig, BucketSelector, NativePreprocessor,
                   build_local_index)
from .model import MultiTaskModel, KaloscopeBackbone
from .selfcheck import synthetic_fixture, synthetic_vocabulary
from .selfcheck_release import reference_release, reference_row
from .selfcheck_checkpoint import _same
from .deployment import export_predictor, load_predictor, untrained_label_metadata, preprocessing_identity
from .evaluation.cohort import identity


def _reject(call):
    try:
        call()
    except (ValueError, RuntimeError, KeyError):
        return
    raise AssertionError('Invalid deployment binding accepted')


def _edit_manifest(path, update):
    document = json.loads((path / 'manifest.json').read_text())
    update(document)
    document.pop('identity')
    document['identity'] = identity(document)
    (path / 'manifest.json').write_text(json.dumps(document))


def check_deployment():
    torch.set_num_threads(1)
    vocabulary = synthetic_vocabulary()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(321)
        model = MultiTaskModel(KaloscopeBackbone(synthetic_fixture(style_dim=256)), vocabulary.layout(feature_dim=32, attention_dim=16), other_artist_mode='single_label').eval()
        images = torch.randn(2, 3, 32, 48)
    original = model.predict(images, dense=True)
    model.bind_vocabulary(vocabulary)
    records = [{'domain': domain, 'class_id': value, 'source': 'fixture'} for domain in vocabulary.class_id_mapping for value in (0, 1, 2)]
    membership = SourceMembership.from_records(records, vocabulary, vocab_version=vocabulary.document['version'], vocab_sha256=vocabulary.sha256)
    model.set_source_membership(membership)
    thresholds = {domain: {'default': .0, 'per_class': {'0': 1.0}} for domain in vocabulary.class_id_mapping}
    selected = model.predict(images, ['general'], output_mode='threshold', thresholds=thresholds, source_filter='fixture', max_tags=1)['general']
    assert all(len(ids) == 1 and int(ids[0]) in (1, 2) for ids in selected['class_ids'])
    for i, ids in enumerate(selected['class_ids']):
        torch.testing.assert_close(selected['scores'][i], original['general'][i, ids], rtol=0, atol=0)
    none = model.predict(images, ['general'], output_mode='threshold', thresholds={'general': 1.})['general']
    assert all(len(ids) == 0 for ids in none['class_ids'])
    for artist in ('artist', 'other_artist'):
        output = model.predict(images, [artist], output_mode='threshold', thresholds={artist: 0.}, source_filter='fixture')[artist]
        for i, ids in enumerate(output['class_ids']):
            torch.testing.assert_close(output['scores'][i], original[artist][i, ids], rtol=0, atol=0)
            assert output['scores'][i].sum() < 1
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        release = reference_release(root, [reference_row(0, general_ids=[1], character_ids=[0], copyright_ids=[1], other_artist_ids=[2]),
            reference_row(1, split='val', general_ids=[1], character_ids=[0], copyright_ids=[1]),
            reference_row(2, split='val', general_ids=[0], character_ids=[1], copyright_ids=[0])], vocabulary=vocabulary)
        index = build_local_index(release, root / 'index.sqlite')
        untrained = untrained_label_metadata(index)
        model.set_untrained_labels(untrained)
        bounded = model.predict(images, top_k=20)
        assert bounded['general']['class_ids'].tolist() == [[1], [1]]
        assert bounded['other_artist']['class_ids'].tolist() == [[2], [2]]
        assert bounded['artist']['class_ids'].shape[-1] == len(vocabulary.document['original_artist'])
        dense = model.predict(images, dense=True)
        assert dense['general'].shape == original['general'].shape
        assert torch.count_nonzero(dense['general']) == 2
        torch.testing.assert_close(dense['artist'], original['artist'], rtol=0, atol=0)
        raw = model.predict(images, dense=True, suppress_untrained=False)
        for name in original:
            torch.testing.assert_close(raw[name], original[name], rtol=0, atol=0)
        preprocessor = NativePreprocessor(BucketSelector(BucketConfig(target_pixels=1024, max_pixels=4096, max_side=64)))
        export_predictor(model, vocabulary, root / 'package', checkpoint_id='synthetic-e8', preprocessor=preprocessor,
                         thresholds=thresholds, untrained=untrained, source_membership=membership)
        loaded = load_predictor(root / 'package', expected_vocabulary=vocabulary)
        comparisons = 0
        for options in ({'dense': True}, {'top_k': 3}, {'top_k': 3, 'source_filter': 'fixture'},
                        {'dense': True, 'source_filter': 'fixture'},
                        {'output_mode': 'threshold', 'thresholds': thresholds, 'source_filter': 'fixture', 'max_tags': 2}):
            assert _same(model.predict(images, **options), loaded.predict(images, **options))
            comparisons += 1
        assert _same(model.predict(images, ['general'], output_mode='threshold', thresholds=thresholds),
                     loaded.predict(images, ['general'], output_mode='threshold'))
        pil = Image.new('RGBA', (48, 32), (128, 200, 100, 120))
        actual = loaded.predict_image(pil, ['general'], top_k=3)
        prepared = preprocessor(pil)
        expected_image = model.predict(prepared.tensor.unsqueeze(0), ['general'], top_k=3)
        assert torch.equal(actual['outputs']['general']['class_ids'], expected_image['general']['class_ids'])
        torch.testing.assert_close(actual['outputs']['general']['scores'], expected_image['general']['scores'], rtol=1e-6, atol=1e-7)
        image_error = float((actual['outputs']['general']['scores'] - expected_image['general']['scores']).abs().max())
        assert actual['bucket']['target_size'] == prepared.bucket.target_size
        alternate = deepcopy(vocabulary.document)
        alternate['domains']['general'][0]['name'], alternate['domains']['general'][1]['name'] = alternate['domains']['general'][1]['name'], alternate['domains']['general'][0]['name']
        wrong_vocab = Vocabulary(alternate)
        _reject(lambda: load_predictor(root / 'package', expected_vocabulary=wrong_vocab))
        _reject(lambda: model.bind_vocabulary(wrong_vocab))
        _reject(lambda: export_predictor(model, wrong_vocab, root / 'wrong-vocab', checkpoint_id='a', preprocessor=preprocessor))
        altered_membership = SourceMembership.from_records(records, wrong_vocab, vocab_version=wrong_vocab.document['version'], vocab_sha256=wrong_vocab.sha256)
        _reject(lambda: model.set_source_membership(altered_membership))
        _reject(lambda: model.predict(images, ['general'], output_mode='threshold', thresholds={'general': {'default': .5, 'per_class': {'999': .2}}}))
        _reject(lambda: model.predict(images, ['artist'], output_mode='threshold', thresholds={'general': .5}))
        _reject(lambda: model.predict(images, output_mode='threshold', thresholds=thresholds, max_tags=0))
        for name, update in (
            ('mode', lambda doc: doc.pop('other_artist_mode')),
            ('layout', lambda doc: doc['layout'].update(general=12)),
            ('preprocessing', lambda doc: doc['preprocessing'].update(padding=True)),
            ('file_descriptor', lambda doc: doc['files']['model.safetensors'].update(path='../model.safetensors')),
            ('untrained_artist', lambda doc: doc['untrained_labels']['ids'].update(artist=[0])),
        ):
            broken = root / ('bad-' + name)
            shutil.copytree(root / 'package', broken)
            _edit_manifest(broken, update)
            _reject(lambda: load_predictor(broken))
        # A genuinely fitted validation artifact is embedded and rebound strictly.
        from .evaluation import EvaluationCohort, EvaluationConfig, build_feature_cache, fit_calibration
        cohort = EvaluationCohort(index, ['fixture_1', 'fixture_2'], split='val', config=EvaluationConfig(row_batch=2, class_block=3))
        cache = build_feature_cache(model, [{'images': images, 'sample_ids': list(cohort.sample_ids)}], cohort,
            root / 'calibration-features', checkpoint_id='synthetic-e8', preprocessing=preprocessing_identity(preprocessor))
        calibration = fit_calibration(model, cache, cohort, spool_dir=root / 'spool')
        export_predictor(model, vocabulary, root / 'calibrated-package', checkpoint_id='synthetic-e8', preprocessor=preprocessor,
                         calibration=calibration, untrained=untrained)
        calibrated = load_predictor(root / 'calibrated-package')
        assert calibrated.manifest['threshold_status'] == 'FROZEN_VALIDATION_CALIBRATION'
        assert _same(model.predict(images, ['general'], output_mode='threshold', thresholds=calibrated.manifest['thresholds']),
                     calibrated.predict(images, ['general'], output_mode='threshold'))
        wrong_calibration = {**calibration, 'checkpoint': 'different'}
        wrong_calibration.pop('identity'); wrong_calibration['identity'] = identity(wrong_calibration)
        _reject(lambda: export_predictor(model, vocabulary, root / 'wrong-calibration', checkpoint_id='synthetic-e8',
                                        preprocessor=preprocessor, calibration=wrong_calibration))
        # Frozen calibration cannot expand its declared source/class scope.
        with torch.no_grad():
            model.general.classifier.bias[2] = 20.  # Outside the calibrated class scope.
        scope = {name: list(range(len(entries))) for name, entries in vocabulary.class_id_mapping.items()}
        scope['general'] = [0, 1]
        narrow = EvaluationCohort(index, ['fixture_1', 'fixture_2'], split='val', config=EvaluationConfig(row_batch=2),
                                 class_ids=scope, source_membership=membership, source_filter='fixture')
        narrow_cache = build_feature_cache(model, [{'images': images, 'sample_ids': list(narrow.sample_ids)}], narrow,
            root / 'narrow-features', checkpoint_id='synthetic-narrow', preprocessing=preprocessing_identity(preprocessor))
        narrow_calibration = fit_calibration(model, narrow_cache, narrow, spool_dir=root / 'spool')
        export_predictor(model, vocabulary, root / 'narrow-package', checkpoint_id='synthetic-narrow', preprocessor=preprocessor,
                         calibration=narrow_calibration, source_membership=membership)
        scoped = load_predictor(root / 'narrow-package')
        assert scoped.manifest['calibration_scope']['sources'] == ['fixture']
        assert scoped.manifest['calibration_scope']['class_ids']['general'] == [0, 1]
        frozen = scoped.predict(images, ['general'], output_mode='threshold', source_filter='fixture', suppress_untrained=False)
        assert all(set(ids.tolist()) <= {0, 1} for ids in frozen['general']['class_ids'])
        diagnostic = scoped.predict(images, ['general'], output_mode='threshold', thresholds={'general': 0.},
                                    source_filter='fixture', suppress_untrained=False)
        assert all(2 in ids.tolist() for ids in diagnostic['general']['class_ids'])
        # Candidate architecture also reconstructs strictly without old-key guessing.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2024)
            candidate = MultiTaskModel(KaloscopeBackbone(synthetic_fixture()), vocabulary.layout(
                feature_dim=32, attention_dim=16, identity_feature_mode='global_local',
                version='sakura-tagger-heads-v2-identity-local', identity_attention_dim=16)).eval().bind_vocabulary(vocabulary)
            candidate_images = torch.randn(2, 3, 32, 48)
        export_predictor(candidate, vocabulary, root / 'candidate', checkpoint_id='synthetic-candidate', preprocessor=preprocessor)
        candidate_loaded = load_predictor(root / 'candidate')
        assert _same(candidate.state_dict(), candidate_loaded.model.state_dict())
        assert candidate.layout == candidate_loaded.model.layout
        assert candidate.backbone.legacy.package_config == candidate_loaded.model.backbone.legacy.package_config
        assert all(left.training == dict(candidate_loaded.model.named_modules())[name].training for name, left in candidate.named_modules())
        candidate_before, candidate_after = candidate.predict(candidate_images), candidate_loaded.predict(candidate_images)
        candidate_error = 0.
        for domain, result in candidate_before.items():
            if isinstance(result, dict):
                assert torch.equal(result['class_ids'], candidate_after[domain]['class_ids'])
                expected, actual_scores = result['scores'], candidate_after[domain]['scores']
            else:
                expected, actual_scores = result, candidate_after[domain]
            torch.testing.assert_close(expected, actual_scores, rtol=1e-6, atol=1e-7)
            candidate_error = max(candidate_error, float((expected - actual_scores).abs().max()))
        assert candidate_loaded.model.other_artist_mode == 'multilabel'
        before_threshold = candidate.predict(candidate_images, output_mode='threshold', thresholds=.5)
        after_threshold = candidate_loaded.predict(candidate_images, output_mode='threshold', thresholds=.5)
        assert all(all(torch.equal(left, right) for left, right in zip(value['class_ids'], after_threshold[domain]['class_ids']))
                   for domain, value in before_threshold.items() if isinstance(value, dict))
        index.close()
        return {'threshold_after_source_before_max_tags_with_original_ids': 'PASS',
                'full_vocabulary_artist_softmax_before_filter_no_renormalization': 'PASS',
                'NO_TRAIN_POSITIVE_suppression_new_heads_only_dense_dimensions_preserved': 'PASS',
                'legacy_unbound_and_explicit_raw_optout': 'PASS', 'strict_package_prediction_modes_bitwise': comparisons,
                'PIL_preprocessing_and_RGBA_max_score_error': image_error,
                'frozen_validation_calibration_package_roundtrip': 'PASS',
                'narrowed_calibration_class_source_scope_preserved': 'PASS', 'same_size_wrong_Vocabulary_membership_rejected': 'PASS',
                'malformed_package_and_threshold_rejection': 'PASS', 'identity_local_and_legacy_mode_roundtrip_max_score_error': candidate_error,
                'nonbitwise_score_tolerance': {'atol': 1e-7, 'rtol': 1e-6},
                'real_weights_and_quality': 'NOT_VERIFIED'}


if __name__ == '__main__':
    print(json.dumps(check_deployment(), indent=2))
