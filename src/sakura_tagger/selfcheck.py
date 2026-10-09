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


def synthetic_fixture(seed=0):
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
            nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 8)),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps({'synthetic_cpu': check_model(), 'data': check_data(), 'real_weights': 'NOT_VERIFIED',
                      'CUDA': 'NOT_RUN', 'formal_training': 'NOT_RUN'}, indent=2))


if __name__ == '__main__':
    main()
