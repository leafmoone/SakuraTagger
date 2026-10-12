"""CPU contract checks for the optional, explicitly versioned Identity branch."""
from unittest.mock import patch

import torch
from torch import nn

from .model import HeadLayout, KaloscopeBackbone, MultiTaskModel
from .model.group_attention import GroupAttention


def check_architecture():
    from .selfcheck import production_head_counts, synthetic_fixture

    # These are the pre-E0 definitions, independent of the new feature helpers.
    class LegacyGeneral(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = GroupAttention(32, 16, 16, 4)
            self.query_fusion = nn.Linear(16 * 16, 32)
            self.global_residual = nn.Linear(64, 32)
            self.adapter = nn.Sequential(nn.LayerNorm(32), nn.GELU())
            self.classifier = nn.Linear(32, 11)

        def forward(self, global_features, patch_tokens):
            queries = self.attention(patch_tokens)
            features = self.query_fusion(queries.flatten(1)) + self.global_residual(global_features)
            return self.classifier(self.adapter(features))

    class LegacyIdentity(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = nn.Sequential(nn.Linear(64, 32), nn.GELU())
            self.classifier = nn.Linear(32, 18)

        def forward(self, global_features):
            return self.classifier(self.adapter(global_features)).split((13, 5), dim=-1)

    old_layout = {'general': 11, 'character': 13, 'copyright': 5, 'other_artist': 9,
                  'feature_dim': 32, 'attention_dim': 16, 'attention_heads': 4,
                  'query_count': 16, 'version': 'sakura-tagger-heads-v1'}
    layout = HeadLayout(**old_layout)
    assert layout.to_dict() == old_layout
    assert HeadLayout(**layout.to_dict()).to_dict() == old_layout
    torch.manual_seed(371)
    general, identity = LegacyGeneral(), LegacyIdentity()
    model = MultiTaskModel(KaloscopeBackbone(synthetic_fixture()), layout)
    model.general.load_state_dict(general.state_dict(), strict=True)
    model.identity.load_state_dict(identity.state_dict(), strict=True)
    assert set(model.general.state_dict()) == set(general.state_dict())
    assert set(model.identity.state_dict()) == set(identity.state_dict())
    assert not any('local' in name for name, _ in model.identity.named_parameters())
    images = torch.randn(2, 3, 32, 48)
    with torch.no_grad():
        output = model(images)
        features = output.features
        expected_general = general(features.global_features, features.patch_tokens)
        expected_identity = identity(features.global_features)
        torch.testing.assert_close(output.logits['general'], expected_general, rtol=0, atol=0)
        for name, expected in zip(('character', 'copyright'), expected_identity):
            torch.testing.assert_close(output.logits[name], expected, rtol=0, atol=0)
        for module, arguments, expected in (
            (model.general, (features.global_features, features.patch_tokens), expected_general),
            (model.identity, (features.global_features,), torch.cat(expected_identity, dim=-1))):
            cached = module.features(*arguments)
            torch.testing.assert_close(module.classify_features(cached), expected, rtol=0, atol=0)
            blockwise = torch.cat([module.classify_features(cached, start, min(start + 4, expected.shape[-1]))
                                   for start in range(0, expected.shape[-1], 4)], dim=-1)
            torch.testing.assert_close(blockwise, expected, rtol=1e-5, atol=1e-6)

    local_options = {**old_layout, 'version': 'sakura-tagger-heads-v2-identity-local',
                     'identity_feature_mode': 'global_local', 'identity_attention_dim': 16,
                     'general_feature_dim': 24, 'identity_feature_dim': 40}
    local = MultiTaskModel(KaloscopeBackbone(synthetic_fixture()), HeadLayout(**local_options))
    assert local.general.classifier.in_features == 24
    assert local.identity.classifier.in_features == 40
    assert local.layout.to_dict() != old_layout
    try:
        local.load_state_dict(model.state_dict(), strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Local architecture accepted legacy weights as an exact resume')
    with patch.object(local.backbone.legacy.backbone, 'forward_features',
                      wraps=local.backbone.legacy.backbone.forward_features) as call:
        result = local(images)
        assert call.call_count == 1
    assert {name: value.shape for name, value in result.logits.items()} == {
        name: value.shape for name, value in output.logits.items()}
    sum(result.logits[name].square().mean() for name in ('character', 'copyright')).backward()
    gradients = {name: parameter.grad for name, parameter in local.identity.named_parameters()
                 if name.startswith('local_')}
    assert gradients and all(value is not None and torch.isfinite(value).all() and value.abs().sum() > 0
                             for value in gradients.values())
    assert local.identity.local_attention.queries.grad.abs().sum(-1).gt(0).all()
    with patch.object(local.general, 'forward', side_effect=AssertionError('Unrequested General branch')):
        assert set(local(images, ['copyright']).logits) == {'copyright'}
    return {'legacy_global_strict_load_and_exact_output': 'PASS', 'legacy_layout_metadata': 'PASS',
            'features_classifier_and_block_parity': 'PASS', 'disabled_local_has_no_parameters': 'PASS',
            'local_all_four_query_gradients': 'PASS', 'single_student_forward': 'PASS',
            'selective_outputs_and_independent_widths': 'PASS', 'incompatible_resume_rejected': 'PASS',
            'tiny_baseline': model.parameter_counts(), 'tiny_local_independent_widths': local.parameter_counts(),
            'production_baseline': production_head_counts(),
            'production_identity_local': production_head_counts('global_local'),
            'quality_improvement': 'NOT_VERIFIED'}
