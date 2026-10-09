"""Durable, CPU-only synthetic engineering acceptance (never formal training).

Run ``python -m sakura_tagger.selfcheck`` from an editable checkout. Random tiny
upstream DINOv3 weights are explicitly synthetic, not official-model evidence.
"""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps({'synthetic_cpu': check_model(), 'real_weights': 'NOT_VERIFIED',
                      'CUDA': 'NOT_RUN', 'formal_training': 'NOT_RUN'}, indent=2))


if __name__ == '__main__':
    main()
