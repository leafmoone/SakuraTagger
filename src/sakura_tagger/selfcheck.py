"""Durable, CPU-only synthetic engineering acceptance (never formal training).

Run ``python -m sakura_tagger.selfcheck`` from an editable checkout. Random tiny
upstream DINOv3 weights are explicitly synthetic, not official-model evidence.
"""
import argparse
import json
import random
from pathlib import Path
import tempfile
from collections import Counter
from unittest.mock import patch

import torch
from torch import nn

from .adapter import _validate_model_input
from .model import HeadLayout, KaloscopeBackbone, MultiTaskModel, artist_classifier_input
from .upstream import model_loading


def synthetic_fixture(seed=0, *, style_dim=8):
    upstream = model_loading()
    from kaloscope_dinov3.architecture import build_backbone
    options = {'name': 'custom_vit', 'kwargs': {
        'img_size': 32, 'patch_size': 16, 'embed_dim': 32, 'depth': 12,
        'num_heads': 4, 'ffn_ratio': 2, 'pos_embed_rope_dtype': 'fp32'},
        'pooling': 'cls_mean', 'feature_source': 'projector',
        'classifier_input_normalization': 'l2_sqrt_dim'}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = upstream.DinoInferenceModel(
            build_backbone(options), 'cls_mean', nn.Linear(64, 7),
            nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, style_dim)),
            'projector', 'l2_sqrt_dim')
        model.register_buffer('log_temperature', torch.tensor(0.0))
        model.register_buffer('bias', torch.tensor(0.0))
    model.synthetic = True
    model.package_config = {'model': options}
    model.class_mapping = {i: f'synthetic_artist_{i}' for i in range(7)}
    model.register_forward_pre_hook(_validate_model_input, with_kwargs=True)
    return model.requires_grad_(False).eval()


def check_model():
    torch.manual_seed(17)
    legacy = synthetic_fixture()
    backbone = KaloscopeBackbone(legacy)
    layout = HeadLayout(11, 13, 5, 9, feature_dim=32, attention_dim=16)
    model = MultiTaskModel(backbone, layout)
    images = torch.randn(2, 3, 32, 48)
    original = {k: v.clone() for k, v in legacy.state_dict().items()}
    with patch.object(legacy.backbone, 'forward_features', wraps=legacy.backbone.forward_features) as call:
        output = model(images)
        assert call.call_count == 1
    assert {k: tuple(v.shape) for k, v in output.logits.items()} == {
        'general': (2, 11), 'character': (2, 13), 'copyright': (2, 5),
        'other_artist': (2, 9), 'artist': (2, 7)}
    style, artist = legacy(images, return_both=True)
    torch.testing.assert_close(output.features.style_embedding, style)
    torch.testing.assert_close(output.logits['artist'], artist)
    with patch.object(model.other_artist.classifier, 'forward', wraps=model.other_artist.classifier.forward) as other:
        with patch.object(legacy.head, 'forward', wraps=legacy.head.forward) as original_head:
            model(images, ['other_artist', 'artist'])
            torch.testing.assert_close(other.call_args.args[0], original_head.call_args.args[0])
    assert artist_classifier_input(output.features.global_features).dtype == torch.float32
    model.train()
    sum(x.square().mean() for k, x in output.logits.items() if k != 'artist').backward()
    assert all(not p.requires_grad and p.grad is None for p in legacy.parameters())
    assert model.general.attention.queries.grad.abs().sum(-1).gt(0).all()
    t1 = model.parameter_counts()
    model.zero_grad(set_to_none=True)
    model.set_stage('finetune_last4').train()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.01)
    output = model(images)
    sum(x.square().mean() for x in output.logits.values()).backward()
    for index, block in enumerate(legacy.backbone.blocks):
        assert block.training == (index >= 8)
        assert all(p.requires_grad == (index >= 8) for p in block.parameters())
        if index >= 8:
            assert sum(p.grad.abs().sum().item() for p in block.parameters() if p.grad is not None) > 0
        else:
            assert all(p.grad is None for p in block.parameters())
    assert legacy.backbone.norm.weight.grad.abs().sum() > 0
    assert not legacy.head.training and not legacy.projector.training
    optimizer.step()
    changed = {k for k, v in legacy.state_dict().items() if not torch.equal(v, original[k])}
    assert changed and all(k.startswith(('backbone.blocks.8.', 'backbone.blocks.9.',
                                         'backbone.blocks.10.', 'backbone.blocks.11.', 'backbone.norm.'))
                           for k in changed)
    t2 = model.parameter_counts()
    with patch.object(model.general, 'forward', wraps=model.general.forward) as general, \
         patch.object(model.identity, 'forward', wraps=model.identity.forward) as identity, \
         patch.object(legacy.head, 'forward', wraps=legacy.head.forward) as artist_call, \
         patch.object(legacy.projector, 'forward', wraps=legacy.projector.forward) as style_call:
        selected = model.predict(images, ['other_artist'], top_k=3)
        assert not any(c.call_count for c in (general, identity, artist_call, style_call))
        assert set(selected) == {'other_artist'} and selected['other_artist']['scores'].shape == (2, 3)
    model.set_stage('frozen_heads')
    assert all(p.grad is None and not p.requires_grad for p in legacy.parameters())
    return {'single_backbone_forward': 1, 'general_query_gradient_rows': 16,
            'T1': t1, 'T2': t2, 'T2_updated_legacy_tensors': len(changed),
            'legacy_output_parity': 'PASS', 'selective_execution': 'PASS'}


def synthetic_vocabulary():
    from .data.contract import Vocabulary, VOCAB_VERSION
    return Vocabulary({'version': VOCAB_VERSION,
        'domains': {domain: [{'class_id': i, 'name': f'label_{i}'} for i in range(count)]
                    for domain, count in (('general', 11), ('character', 13), ('copyright', 5), ('other_artist', 9))},
        'original_artist': [{'class_id': i, 'name': f'synthetic_artist_{i}'} for i in range(7)]})


def synthetic_records(count=12):
    return [{'image_id': f'image_{i}', 'image_path': f'image_{i}.png', 'width': 32, 'height': 32,
             'source': 'synthetic', 'RID': i, 'image_sha256': f'{i:064x}',
             'split': 'train' if i < count - 2 else 'val', 'family_id': 'UNKNOWN',
             'positive_labels': {'general': [i % 3], 'character': [0], 'copyright': [0]},
             'supervision_scope': {'general': 'all', 'character': 'positive_only', 'copyright': [0, 1]},
             'unknown_mask': {'general': [10]}} for i in range(count)]


def check_data():
    from .data import (BucketConfig, BucketSelector, BucketAspectError, BucketBatchSampler,
                       encode_supervision, read_manifest, repeat_factors, SCHEMA_VERSION, Vocabulary)
    vocabulary = synthetic_vocabulary()
    records = synthetic_records()
    metadata = {'schema_version': SCHEMA_VERSION, 'dataset_id': 'synthetic',
                'provider': 'SakuraPool', 'vocab_sha256': vocabulary.sha256}
    targets, masks = encode_supervision(records[0], vocabulary)
    assert masks['other_artist'].sum() == 0 and masks['general'].sum() == 10
    assert masks['character'].sum() == 1 and masks['copyright'].sum() == 2
    assert vocabulary.document['domains']['character'][0]['name'] == vocabulary.document['domains']['copyright'][0]['name']
    assert vocabulary.layout().identity == 18
    rejected = 0
    for operation in (
        lambda: Vocabulary({**vocabulary.document, 'version': 'wrong'}),
        lambda: Vocabulary({**vocabulary.document, 'domains': {**vocabulary.document['domains'],
                            'general': [{'class_id': 1, 'name': 'wrong'}]}}),
        lambda: encode_supervision({**records[0], 'positive_labels': {}}, vocabulary),
        lambda: encode_supervision({**records[0], 'positive_labels': {'general': [99]}}, vocabulary),
    ):
        try:
            operation()
        except ValueError:
            rejected += 1
    assert rejected == 4
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'manifest.jsonl'
        path.write_text('\n'.join(json.dumps(record) for record in records), encoding='utf-8')
        assert read_manifest(path, metadata, vocabulary) == records
        for changed in ({**metadata, 'schema_version': 'wrong'}, {**metadata, 'vocab_sha256': 'wrong'}):
            try:
                read_manifest(path, changed, vocabulary)
            except ValueError:
                rejected += 1
    assert rejected == 6
    factors, frequencies = repeat_factors(records, vocabulary, threshold=100)
    assert factors == [3.0] * 10 and frequencies[('character', 0)] == 1
    try:
        repeat_factors(records + [records[0]], vocabulary)
    except ValueError:
        pass
    else:
        raise AssertionError('RFS accepted duplicate training images')
    selector = BucketSelector(BucketConfig(target_pixels=2048, max_pixels=4096, max_side=128))
    sizes = [(32, 32)] * 5 + [(64, 32)] * 3 + [(32, 64)] * 2
    options = dict(selector=selector, shuffle=True, seed=21, repeat_weights=factors)
    samplers = [BucketBatchSampler(sizes, 2, rank=rank, world_size=3, **options) for rank in range(3)]
    occurrences = [item for sampler in samplers for batch in sampler.occurrence_batches() for item in batch]
    assert len(occurrences) == len(set(oid for oid, _ in occurrences)) == 30
    assert Counter(index for _, index in occurrences) == Counter({i: 3 for i in range(10)})
    for sampler in samplers:
        full = list(sampler)
        assert full == list(sampler)
        assert all(len({selector.select(*sizes[i]).target_size for i in batch}) == 1 for batch in full)
        sampler.advance()
        restored = BucketBatchSampler(sizes, 2, rank=sampler.rank, world_size=3, **options)
        restored.load_state_dict(sampler.state_dict())
        assert list(restored) == full[1:]
    replacement = BucketBatchSampler([(32, 32)], 2, repeat_weights=[3], repeat_mode='replacement', epoch_size=10)
    assert sum(len(batch) for batch in replacement) == 10 > 3
    # Reference is deliberately retained only inside the durable acceptance tool.
    comparisons = 0
    rng = random.Random(19)
    for config in (BucketConfig(), BucketConfig(max_pixels=512 * 512)):
        indexed = BucketSelector(config)
        sizes = [(1, 1), (16, 9), (9, 16), (400, 300), (100000, 1)]
        sizes += [(rng.randint(16, 8192), rng.randint(16, 8192)) for _ in range(256)]
        for width, height in sizes:
            ratio = max(width, height) / min(width, height)
            ranked = [(max((w / h) / ratio, ratio / (w / h)) - 1, w, h) for w, h in indexed._candidates]
            valid = [x for x in ranked if x[0] <= config.preferred_error] or [x for x in ranked if x[0] <= config.max_error]
            if not valid:
                try:
                    indexed.select(width, height)
                except BucketAspectError:
                    comparisons += 1
                    continue
                raise AssertionError('Unrepresentable ratio was accepted')
            error, w, h = min(valid, key=lambda x: (abs(x[1] * x[2] - config.target_pixels), x[0], x[1], x[2]))
            actual = indexed.select(width, height)
            assert actual.target_size == ((h, w) if height > width else (w, h)) and actual.aspect_error == error
            assert indexed.select(height, width).target_size == actual.target_size[::-1]
            comparisons += 1
    return {'bucket_reference_comparisons': comparisons, 'rank_unique_occurrences': len(occurrences),
            'sampler_resume': 'PASS', 'RFS_cap': max(factors), 'replacement_realized_count': 10,
            'manifest_and_vocabulary_rejections': rejected, 'unknown_supervision': 'PASS'}


def check_losses():
    from .training import (LossConfig, masked_asymmetric_loss, masked_single_label_loss,
                           multitask_loss, trusted_single_artist)
    torch.manual_seed(29)
    source = torch.randn(3, 17)
    targets = torch.randint(0, 2, (3, 17)).float()
    mask = torch.rand(3, 17) > 0.2
    targets[~mask] = float('nan')
    gradients, values = [], []
    for chunk_size in (4, 100):
        logits = source.clone().requires_grad_()
        value, count = masked_asymmetric_loss(logits, targets, mask, config=LossConfig(chunk_size=chunk_size))
        value.backward()
        gradients.append(logits.grad.clone())
        values.append(value)
    torch.testing.assert_close(values[0], values[1])
    torch.testing.assert_close(gradients[0], gradients[1])
    assert not gradients[0][~mask].any()
    extreme = torch.tensor([[-1000, 1000, -80, 80]], dtype=torch.bfloat16, requires_grad=True)
    value, _ = masked_asymmetric_loss(extreme, torch.tensor([[1., 0., 0., 1.]]), torch.ones(1, 4, dtype=torch.bool))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(extreme.grad).all()
    empty = torch.full((2, 5), float('nan'), requires_grad=True)
    zero, empty_count = masked_asymmetric_loss(empty, empty.detach(), torch.zeros(2, 5, dtype=torch.bool))
    zero.backward()
    assert zero == 0 and empty_count == 0 and not empty.grad.any()
    logits = source.clone().requires_grad_()
    labels, sample_mask = torch.tensor([2, -999, 10]), torch.tensor([True, False, True])
    ce, count = masked_single_label_loss(logits, labels, sample_mask, chunk_size=4)
    reference = torch.nn.functional.cross_entropy(logits[sample_mask], labels[sample_mask])
    torch.testing.assert_close(ce, reference)
    torch.testing.assert_close(torch.autograd.grad(ce, logits, retain_graph=True)[0], torch.autograd.grad(reference, logits)[0])
    _, trusted = trusted_single_artist(torch.tensor([[0., 1.], [0., 1.], [1., 1.]]),
                                      torch.tensor([[True, True], [False, True], [True, True]]))
    assert trusted.tolist() == [True, False, False]
    try:
        multitask_loss({'general': source}, {'general': targets})
    except ValueError:
        pass
    else:
        raise AssertionError('Missing masks were silently accepted')
    return {'chunked_value_and_gradient_parity': 'PASS', 'BF16_extreme_finite': 'PASS',
            'empty_and_unknown_masks': 'PASS', 'trusted_single_artist_CE': 'PASS'}


def check_training():
    from PIL import Image
    from torch.utils.data import DataLoader
    from .data import BucketConfig, BucketSelector, BucketBatchSampler, ManifestDataset, NativePreprocessor
    from .data.contract import SCHEMA_VERSION, encode_supervision, manifest_fingerprint
    from .training import Trainer, TrainerConfig, LossConfig, checkpoint_metadata
    vocabulary, records = synthetic_vocabulary(), synthetic_records()
    bucket_config = BucketConfig(target_pixels=1536, max_pixels=2048, max_side=64)
    dataset_metadata = {'schema_version': SCHEMA_VERSION, 'dataset_id': 'synthetic', 'provider': 'SakuraPool',
                        'vocab_sha256': vocabulary.sha256, 'manifest_sha256': manifest_fingerprint(records)}
    layout = vocabulary.layout(feature_dim=32, attention_dim=16)
    def make_model(seed=0):
        return MultiTaskModel(KaloscopeBackbone(synthetic_fixture(seed)), layout)
    model = make_model()
    metadata = checkpoint_metadata(model, vocabulary, dataset_metadata, bucket_config, 'synthetic-original-v1')
    config = TrainerConfig(accumulation_steps=2, gradient_clip=1.0)
    losses = LossConfig(chunk_size=4, general_groups={'sample_group': [0, 1]})
    trainer = Trainer(model, metadata, config=config, loss_config=losses)
    encoded = [encode_supervision(record, vocabulary) for record in records[:2]]
    batch = {'images': torch.randn(2, 3, 32, 48),
             'targets': {name: torch.stack([pair[0][name] for pair in encoded]) for name in encoded[0][0]},
             'masks': {name: torch.stack([pair[1][name] for pair in encoded]) for name in encoded[0][1]}}
    sampler = BucketBatchSampler([(32, 48)] * 8, 2, shuffle=True, seed=41)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'roundtrip.pt'
        trainer.train_batch(batch, sampler=sampler)
        try:
            trainer.save_checkpoint(path, sampler=sampler)
        except ValueError:
            pass
        else:
            raise AssertionError('Partial accumulation checkpoint was accepted')
        trainer.train_batch(batch, sampler=sampler)
        trainer.save_checkpoint(path, sampler=sampler)
        expected_rng = torch.rand(3)
        expected_python, expected_numpy = random.random(), __import__('numpy').random.rand()
        restored = Trainer(make_model(9), metadata, config=config, loss_config=losses)
        resumed_sampler = BucketBatchSampler([(32, 48)] * 8, 2, shuffle=True, seed=41)
        restored.load_checkpoint(path, sampler=resumed_sampler)
        torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
        assert random.random() == expected_python and __import__('numpy').random.rand() == expected_numpy
        assert list(resumed_sampler) == list(sampler)
        assert restored.global_step == 1 and restored.micro_step == 2
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
        # Same next update proves optimizer moments and the accumulation state restored.
        for _ in range(2):
            trainer.train_batch(batch, sampler=sampler)
            restored.train_batch(batch, sampler=resumed_sampler)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
        corrupted_metadata = {**metadata, 'vocab_sha256': 'wrong'}
        rejected = Trainer(make_model(), corrupted_metadata, config=config, loss_config=losses)
        before = {key: value.clone() for key, value in rejected.model.state_dict().items()}
        try:
            rejected.load_checkpoint(path, sampler=BucketBatchSampler([(32, 48)] * 8, 2, shuffle=True, seed=41))
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched checkpoint metadata was accepted')
        assert all(torch.equal(value, rejected.model.state_dict()[key]) for key, value in before.items())
        # Real manifest/image read, native preprocessing and default tensor collation.
        image_record = {**records[0], 'image_path': 'image.png'}
        Image.new('RGB', (32, 32), 'red').save(Path(directory) / 'image.png')
        (Path(directory) / 'vocab.json').write_text(json.dumps(vocabulary.document))
        (Path(directory) / 'manifest.jsonl').write_text(json.dumps(image_record) + '\n')
        image_metadata = {key: value for key, value in dataset_metadata.items() if key != 'manifest_sha256'}
        (Path(directory) / 'metadata.json').write_text(json.dumps(image_metadata))
        dataset = ManifestDataset(Path(directory) / 'manifest.jsonl', Path(directory) / 'metadata.json',
                                  Path(directory) / 'vocab.json', preprocessor=NativePreprocessor(BucketSelector(bucket_config)))
        loaded = next(iter(DataLoader(dataset, batch_size=1)))
        assert loaded['images'].shape[0:2] == (1, 3) and loaded['masks']['other_artist'].sum() == 0
    evaluation = trainer.evaluate([batch])
    assert set(evaluation) == {'general', 'character', 'copyright', 'other_artist'}
    assert evaluation['character']['valid_count'] == 2 and evaluation['copyright']['valid_count'] == 4
    trainer.set_stage('finetune_last4')
    optimized = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group['params']}
    assert optimized == {id(parameter) for parameter in trainer.model.parameters() if parameter.requires_grad}
    # Opt-in independent original teacher; zero labels isolate student KD gradients.
    student = make_model(10).set_stage('finetune_last4')
    teacher = KaloscopeBackbone(synthetic_fixture())
    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    teacher_config = TrainerConfig(stage='finetune_last4', accumulation_steps=2,
                                  preservation={'global_features': 1.0, 'style_embedding': 1.0, 'artist_logits': 1.0})
    kd_metadata = checkpoint_metadata(student, vocabulary, dataset_metadata, bucket_config, 'synthetic-original-v1')
    kd_trainer = Trainer(student, kd_metadata, config=teacher_config, loss_config=losses, teacher=teacher)
    unsupervised = {**batch, 'masks': {name: torch.zeros_like(mask) for name, mask in batch['masks'].items()}}
    kd_result = kd_trainer.train_batch(unsupervised)
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters())
    assert all(torch.equal(value, teacher.state_dict()[key]) for key, value in teacher_before.items())
    assert student.backbone.legacy.backbone.blocks[-1].attn.qkv.weight.grad.abs().sum() > 0
    kd_trainer.flush()
    # BF16 configurable CPU AMP integration, no CUDA invocation.
    amp_model = make_model()
    amp_trainer = Trainer(amp_model, metadata, config=TrainerConfig(amp='bf16'), loss_config=losses)
    amp_result = amp_trainer.train_batch(batch)
    assert __import__('math').isfinite(amp_result['loss'])
    return {'checkpoint_exact_next_update': 'PASS', 'checkpoint_RNG_and_sampler': 'PASS',
            'checkpoint_rejects_partial_or_mismatch': 'PASS', 'manifest_image_and_collation': 'PASS',
            'evaluation_domain_counts': {name: value['valid_count'] for name, value in evaluation.items()},
            'stage_optimizer_refresh': 'PASS', 'teacher_frozen_student_KD_gradient': 'PASS',
            'KD_losses': kd_result['auxiliary'], 'CPU_BF16_AMP': 'PASS'}


def check_source_membership():
    """Fifteen CPU checks; real mapping/weights remain explicitly unverified."""
    from .data import SourceMembership, SOURCE_MEMBERSHIP_VERSION, BucketConfig
    from .data.contract import SCHEMA_VERSION, manifest_fingerprint
    from .training import LossConfig, Trainer, checkpoint_metadata, multitask_loss
    vocabulary = synthetic_vocabulary()
    binding = {'vocab_version': vocabulary.document['version'], 'vocab_sha256': vocabulary.sha256}
    records = [{'domain': domain, 'class_id': class_id, 'source': source}
               for domain in vocabulary.class_id_mapping
               for class_id, source in ((0, 'danbooru'), (2, 'danbooru'), (1, 'zerochan'),
                                        (2, 'zerochan'), (3, 'gamecg'))]
    # A known source with candidates only in General exercises empty domains.
    records.append({'domain': 'general', 'class_id': 4, 'source': 'general_only'})
    membership = SourceMembership.from_records(records, vocabulary, **binding)
    model = MultiTaskModel(KaloscopeBackbone(synthetic_fixture(style_dim=256)),
                           vocabulary.layout(feature_dim=32, attention_dim=16)).eval()
    images = torch.randn(2, 3, 32, 48)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    parameters = [(id(p), p.requires_grad) for p in model.parameters()]
    raw = model(images)
    old = {name: logits.softmax(-1) if name == 'artist' else logits.sigmoid()
           for name, logits in raw.logits.items()}
    original_dense = model.predict(images, dense=True)
    model.set_source_membership(membership)
    dense = model.predict(images, source_filter=None, dense=True)
    for name, reference in old.items():
        torch.testing.assert_close(dense[name], reference, rtol=0, atol=0)
        torch.testing.assert_close(dense[name], original_dense[name], rtol=0, atol=0)
        scores, ids = reference.topk(3, dim=-1)
        actual = model.predict(images, [name], top_k=3)[name]
        torch.testing.assert_close(actual['scores'], scores, rtol=0, atol=0)
        assert torch.equal(actual['class_ids'], ids)
    checks = {'01_legacy_no_filter_parity': 'PASS'}
    for number, source, expected in ((2, 'danbooru', [0, 2]), (3, 'zerochan', [1, 2]),
                                      (4, ['danbooru', 'zerochan', 'danbooru'], [0, 1, 2])):
        filtered = model.predict(images, source_filter=source, dense=True)
        for name in raw.logits:
            assert filtered[name]['class_ids'].tolist() == [expected] * 2
            torch.testing.assert_close(filtered[name]['scores'], old[name][:, expected])
        checks[f'{number:02}_source_{number}_IDs_scores'] = 'PASS'
    # Force all unrestricted maxima outside Danbooru; filtering after Top-K fails.
    forced_logits = {name: torch.arange(logits.shape[-1], dtype=logits.dtype).expand_as(logits)
                     for name, logits in raw.logits.items()}
    from .model.multi_task_model import MultiTaskOutput
    with patch.object(model, 'forward', return_value=MultiTaskOutput(raw.features, forced_logits)):
        filtered = model.predict(images, source_filter='danbooru', top_k=1)
        assert all(value['class_ids'].tolist() == [[2], [2]] for name, value in filtered.items()
                   if name != 'style_embedding')
    checks['05_filter_before_top_k'] = 'PASS'
    filtered = model.predict(images, source_filter='danbooru', top_k=50)
    for name in raw.logits:
        assert filtered[name]['scores'].shape == (2, 2)
        assert all(set(row) == {0, 2} for row in filtered[name]['class_ids'].tolist())
        expected = old[name].gather(1, filtered[name]['class_ids'])
        torch.testing.assert_close(filtered[name]['scores'], expected)
    assert membership.class_name('general', 0) == membership.class_name('character', 0) == 'label_0'
    assert model.predict(images, (name for name in ['general']), source_filter='danbooru')['general']['scores'].shape == (2, 2)
    checks['06_original_IDs_and_domain_name_lookup'] = 'PASS'
    torch.testing.assert_close(filtered['artist']['scores'], old['artist'].gather(1, filtered['artist']['class_ids']))
    assert (filtered['artist']['scores'].sum(-1) < 1).all()
    checks['07_original_artist_full_softmax'] = 'PASS'
    model.set_other_artist_mode('single_label')
    probabilities = model.predict(images, ['other_artist'], dense=True)['other_artist']
    torch.testing.assert_close(probabilities, raw.logits['other_artist'].softmax(-1))
    subset = model.predict(images, ['other_artist'], source_filter='danbooru', dense=True)['other_artist']
    torch.testing.assert_close(subset['scores'], probabilities[:, [0, 2]])
    assert (subset['scores'].sum(-1) < 1).all()
    checks['08_other_artist_configured_full_softmax'] = 'PASS'
    assert filtered['style_embedding'].shape == (2, 256)
    torch.testing.assert_close(filtered['style_embedding'], raw.features.style_embedding, rtol=0, atol=0)
    style_only = model.predict(images, ['style_embedding'], source_filter='general_only')
    assert set(style_only) == {'style_embedding'}
    torch.testing.assert_close(style_only['style_embedding'], raw.features.style_embedding)
    checks['09_unchanged_256D_style'] = 'PASS'
    def rejects(operation):
        try:
            operation()
        except (ValueError, TypeError):
            return
        raise AssertionError('Invalid source membership/filter was accepted')
    model.set_source_membership(None)
    with patch.object(model.backbone.legacy.backbone, 'forward_features') as call:
        rejects(lambda: model.predict(images, source_filter='danbooru'))
        assert call.call_count == 0
    model.set_source_membership(membership)
    for invalid in ('missing', ['danbooru', 'missing'], [], [1], 1):
        rejects(lambda: model.predict(images, source_filter=invalid))
    checks['10_missing_index_unknown_source_fail_closed'] = 'PASS'
    rejects(lambda: SourceMembership.from_records(records, vocabulary, **{**binding, 'vocab_sha256': '0' * 64}))
    rejects(lambda: SourceMembership.from_records(records, vocabulary, **{**binding, 'vocab_version': 'wrong'}))
    rejects(lambda: SourceMembership.from_records(records, vocabulary, **binding, schema_version='wrong'))
    checks['11_SHA_and_version_mismatch_rejected'] = 'PASS'
    for row in ({'domain': 'artist', 'class_id': 7, 'source': 'danbooru'},
                {'domain': 'general', 'class_id': -1, 'source': 'danbooru'},
                {'domain': 'general', 'class_id': True, 'source': 'danbooru'},
                {'domain': 'general', 'class_id': 0.0, 'source': 'danbooru'},
                {'domain': 'unknown', 'class_id': 0, 'source': 'danbooru'},
                {'domain': 'general', 'class_id': 0, 'source': 'Danbooru'}):
        rejects(lambda: SourceMembership.from_records([row], vocabulary, **binding))
    rejects(lambda: SourceMembership.from_records(records + [records[0]], vocabulary, **binding))
    for is_dense in (False, True):
        empty = model.predict(images, ['character', 'artist'], dense=is_dense, source_filter='general_only')
        assert all(value['class_ids'].shape == value['scores'].shape == (2, 0) for value in empty.values())
    checks['12_empty_candidates_invalid_IDs_duplicates'] = 'PASS'
    with patch.object(model.backbone.legacy.backbone, 'forward_features',
                      wraps=model.backbone.legacy.backbone.forward_features) as call:
        model.predict(images, source_filter=['danbooru', 'zerochan'])
        assert call.call_count == 1
    checks['13_single_DINOv3_forward'] = 'PASS'
    assert [(id(p), p.requires_grad) for p in model.parameters()] == parameters
    assert before.keys() == model.state_dict().keys()
    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    after = model(images)
    for name in raw.logits:
        torch.testing.assert_close(raw.logits[name], after.logits[name], rtol=0, atol=0)
    targets, masks = {'other_artist': torch.tensor([0, 2])}, {'other_artist': torch.ones(2, dtype=torch.bool)}
    ce = LossConfig(other_artist_mode='single_label')
    torch.testing.assert_close(multitask_loss(raw.logits, targets, ce, masks).total,
                               multitask_loss(after.logits, targets, ce, masks).total)
    metadata = checkpoint_metadata(model, vocabulary, {
        'schema_version': SCHEMA_VERSION, 'dataset_id': 'synthetic', 'provider': 'SakuraPool',
        'vocab_sha256': vocabulary.sha256, 'manifest_sha256': manifest_fingerprint(synthetic_records())},
        BucketConfig(), 'synthetic-original-v1')
    with tempfile.TemporaryDirectory() as directory:
        trainer = Trainer(model, metadata, loss_config=ce)
        assert model.other_artist_mode == 'single_label'
        path = Path(directory) / 'membership.pt'
        trainer.save_checkpoint(path)
        restored_model = MultiTaskModel(KaloscopeBackbone(synthetic_fixture(style_dim=256)), model.layout)
        restored = Trainer(restored_model, metadata, loss_config=ce)
        restored.load_checkpoint(path)
        assert restored_model._source_membership is None and restored_model.other_artist_mode == 'single_label'
        restored_model.eval()
        torch.testing.assert_close(restored_model.predict(images, ['other_artist'], dense=True)['other_artist'], probabilities)
        restored_model.set_source_membership(membership)
        torch.testing.assert_close(restored_model.predict(images, ['other_artist'], source_filter='danbooru', dense=True)['other_artist']['scores'], subset['scores'])
        Trainer(restored_model, metadata, loss_config=LossConfig())
        assert restored_model.other_artist_mode == 'multilabel'
    checks['14_parameters_forward_loss_checkpoint_unchanged'] = 'PASS'
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        checks['15_real_Parquet_read_and_schema'] = 'NOT_VERIFIED: install sakura-tagger[parquet]'
    else:
        schema = pa.schema([('domain', pa.string()), ('class_id', pa.int64()), ('source', pa.string()),
                            ('namespace', pa.string()), ('evidence_type', pa.string())],
                           metadata={key.encode(): value.encode() for key, value in {
                               'schema_version': SOURCE_MEMBERSHIP_VERSION, **binding}.items()})
        table = pa.Table.from_pylist(records, schema=schema)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source_membership.parquet'
            pq.write_table(table, path)
            loaded = SourceMembership.read(path, vocabulary)
            for domain in vocabulary.class_id_mapping:
                assert loaded.allowed_ids(domain, ['danbooru', 'zerochan']) == (0, 1, 2)
            model.set_source_membership(loaded).eval()
            torch.testing.assert_close(model.predict(images, ['general'], source_filter='danbooru', dense=True)['general']['scores'], old['general'][:, [0, 2]])
            for invalid in (table.drop(['source']), table.replace_schema_metadata({}),
                            table.set_column(1, 'class_id', pa.array([float(r['class_id']) for r in records])),
                            table.replace_schema_metadata({**schema.metadata, b'vocab_sha256': b'wrong'}),
                            pa.concat_tables([table, table.slice(0, 1)])):
                pq.write_table(invalid, path)
                rejects(lambda: SourceMembership.read(path, vocabulary))
        checks['15_real_Parquet_read_and_schema'] = 'PASS'
    return {**checks, 'real_source_mapping': 'NOT_VERIFIED'}


def production_head_counts():
    from .model.modules import GeneralModule, IdentityModule, OtherArtistModule
    layout = HeadLayout(37679, 50369, 17664, 21506)
    with torch.device('meta'):
        modules = {'general': GeneralModule(1536, 768, layout), 'identity': IdentityModule(1536, layout),
                   'other_artist': OtherArtistModule(1536, layout.other_artist)}
    counts = {name: sum(parameter.numel() for parameter in module.parameters()) for name, module in modules.items()}
    return {**counts, 'total': sum(counts.values()), 'basis': 'candidate architecture only; final vocabulary authoritative'}


def check_real_weights(path):
    from .adapter import load_kaloscope
    model = load_kaloscope(path, device='cpu')
    wrapper = KaloscopeBackbone(model)
    images = torch.zeros(1, 3, 32, 48)
    with torch.no_grad():
        style, artist = model(images, return_both=True)
        features = wrapper(images)
    torch.testing.assert_close(features.style_embedding, style)
    torch.testing.assert_close(features.artist_logits, artist)
    return {'status': 'PASS', 'artist_classes': artist.shape[-1], 'style_dimension': style.shape[-1],
            'note': 'CPU strict load/output parity only; accuracy and GPU performance unverified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-model-dir', default=None, help='Optional existing official Kaloscope package directory; never downloads')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps({'synthetic_cpu': check_model(), 'data': check_data(), 'losses': check_losses(),
                      'training': check_training(), 'source_membership': check_source_membership(),
                      'candidate_production_heads': production_head_counts(),
                      'real_weights': check_real_weights(args.real_model_dir) if args.real_model_dir else 'NOT_VERIFIED',
                      'CUDA': 'NOT_RUN', 'formal_training': 'NOT_RUN'}, indent=2))


if __name__ == '__main__':
    main()
