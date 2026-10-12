"""Exact synthetic objective/gradient acceptance; no real-data quality claim."""
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from .training.positive_unknown import (SamplingKey, WeakNegativeConfig, SparseLossConfig,
    WeakNegativeScope, positive_unknown_loss, unknown_ids, sample_unknown, _row_terms,
    sparse_multitask_loss, sparse_valid_counts)


def check_positive_unknown():
    torch.set_num_threads(1)
    reference = WeakNegativeConfig(lambda_u=0.7, lambda_n=0.3, sample_size=2,
                                   mode='full_reference', chunk_size=2, recompute=False)
    sampled = replace(reference, mode='sampled_weak_negative')
    keys = [SamplingKey('one', epoch=2, occurrence_id=3, seed=45)]
    logits = torch.tensor([[0.4, -0.7, 1.2, -2., 0.1, 0.5]], requires_grad=True)
    p, n, blocked, allowed = [[0]], [[5]], [[4]], np.arange(6, dtype=np.int64)
    reference_result = positive_unknown_loss(logits, p, negatives=n, forbidden=blocked,
                                             allowed_ids=allowed, config=reference)
    reference_gradient = torch.autograd.grad(reference_result.numerator, logits)[0]
    u = unknown_ids(p[0], n[0], blocked[0], allowed, 6)
    assert u.tolist() == [1, 2, 3]
    assert reference_gradient[0, 0] < 0 and (reference_gradient[0, [1, 2, 3, 5]] > 0).all()
    assert reference_gradient[0, 4] == 0
    # Enumerate every equally likely without-replacement subset, rather than
    # using a noisy Monte Carlo tolerance to claim exact expectation equality.
    values, gradients = [], []
    for subset in combinations(u.tolist(), 2):
        current = logits.detach().clone().requires_grad_()
        value = _row_terms(current[0], p[0], n[0], subset, sampled)
        values.append(value.detach())
        gradients.append(torch.autograd.grad(value, current)[0])
    torch.testing.assert_close(torch.stack(values).mean(), reference_result.numerator, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(torch.stack(gradients).mean(0), reference_gradient, rtol=1e-6, atol=1e-7)
    for setting in (replace(reference, mode='chunked_reference', chunk_size=1, recompute=True),
                    replace(sampled, sample_size=9)):
        current = logits.detach().clone().requires_grad_()
        result = positive_unknown_loss(current, p, negatives=n, forbidden=blocked, allowed_ids=allowed,
                                       config=setting, keys=keys)
        torch.testing.assert_close(result.numerator, reference_result.numerator)
        torch.testing.assert_close(torch.autograd.grad(result.numerator, current)[0], reference_gradient)
    selected = sample_unknown(u, 2, keys[0], 'general')
    assert np.array_equal(selected, sample_unknown(u, 2, keys[0], 'general'))
    assert len(selected) == len(set(selected)) == 2 and set(selected) <= set(u)
    assert keys[0].for_domain('general') != replace(keys[0], occurrence_id=4).for_domain('general')
    assert keys[0].for_domain('general') != keys[0].for_domain('character')
    assert not set(selected) & (set(p[0]) | set(n[0]) | set(blocked[0]))

    # N-only rows keep explicit negatives, while completely unknown rows do not
    # receive any full-vocabulary negative gradient. Excluded NaNs are untouched.
    values = torch.tensor([[0.5, float('nan')], [float('nan'), float('nan')]], requires_grad=True)
    result = positive_unknown_loss(values, [[], []], negatives=[[0], []], allowed_ids=np.arange(2),
                                   config=sampled, keys=[SamplingKey('n'), SamplingKey('unknown')])
    assert result.valid_count == 1
    torch.testing.assert_close(result.numerator, 0.3 * F.softplus(values[0, 0]))
    gradient = torch.autograd.grad(result.numerator, values)[0]
    assert torch.isfinite(gradient).all() and gradient[0, 0] > 0 and gradient[0, 1] == 0 and (gradient[1] == 0).all()
    single = torch.tensor([[-10000.]], dtype=torch.bfloat16, requires_grad=True)
    one = positive_unknown_loss(single, [[0]], allowed_ids=np.array([0]), config=sampled, keys=[SamplingKey('single')])
    one.numerator.backward()
    assert one.numerator.dtype == torch.float32 and torch.isfinite(one.numerator) and single.grad.item() < 0
    extreme = torch.tensor([[10000., -10000., float('nan')]], dtype=torch.bfloat16, requires_grad=True)
    finite = positive_unknown_loss(extreme, [[0]], forbidden=[[2]], allowed_ids=np.arange(3),
                                   config=sampled, keys=[SamplingKey('extreme')])
    finite.numerator.backward()
    assert torch.isfinite(finite.numerator) and torch.isfinite(extreme.grad).all()
    empty = torch.tensor([[float('nan')]], requires_grad=True)
    result = positive_unknown_loss(empty, [[]], allowed_ids=np.array([0]), config=reference)
    result.numerator.backward()
    assert result.valid_count == 0 and result.numerator == 0 and empty.grad.item() == 0
    with patch('sakura_tagger.training.positive_unknown._eligible_unknown',
               side_effect=AssertionError('Inactive weak-U scanned the vocabulary')):
        positive_unknown_loss(torch.zeros(2, 6), [[], []], negatives=[[0], []], allowed_ids=allowed,
                              config=reference)
        for setting in (replace(reference, lambda_u=0), replace(reference, mode='masked_asl')):
            positive_unknown_loss(torch.zeros(1, 6), [[0]], allowed_ids=allowed, config=setting)

    learnable = torch.nn.Parameter(torch.zeros(1, 3))
    optimizer = torch.optim.SGD([learnable], lr=0.4)
    losses = []
    tiny_config = replace(reference, lambda_u=1., lambda_n=1.)
    for _ in range(40):
        optimizer.zero_grad()
        loss = positive_unknown_loss(learnable, [[0]], allowed_ids=np.arange(3), config=tiny_config).mean
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] / 3 and learnable[0, 0] > 1 and (learnable[0, 1:] < -1).all()

    with tempfile.TemporaryDirectory() as temporary:
        from .selfcheck_release import reference_release, reference_row
        from .data.release import build_local_index
        from .data.source_membership import SourceMembership
        rows = [reference_row(i, other_artist_ids=[0] if i == 0 else []) for i in range(3)]
        release = reference_release(temporary, rows)
        index = build_local_index(release, Path(temporary) / 'index.sqlite')
        scope = WeakNegativeScope.from_index(index)
        assert scope.ids_by_domain['general'].tolist() == [0, 1, 2]  # NO_TRAIN_POSITIVE excluded.
        configs = {domain: sampled for domain in ('general', 'character', 'copyright')}
        config = SparseLossConfig(configs)
        assert SparseLossConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
        logits = {domain: torch.randn(3, count, requires_grad=True) for domain, count in
                  (('general', 11), ('character', 13), ('copyright', 5), ('other_artist', 9))}
        loss = sparse_multitask_loss(logits, rows, release=release, scope=scope, config=config,
                                     epoch=4, occurrence_ids=[2, 3, 4])
        assert loss.valid_counts == sparse_valid_counts(rows, config, release)
        assert loss.valid_counts == {'general': 3, 'character': 3, 'copyright': 0, 'other_artist': 1}
        loss.total.backward()
        assert torch.count_nonzero(logits['copyright'].grad) == 0
        assert torch.count_nonzero(logits['general'].grad[:, 3:]) == 0
        records = [{'domain': 'general', 'class_id': label, 'source': 'known'} for label in (0, 2)]
        membership = SourceMembership.from_records(records, release.vocabulary,
            vocab_version=release.vocabulary.document['version'], vocab_sha256=release.vocabulary.sha256)
        source_scope = WeakNegativeScope.from_index(index, source_membership=membership, source_filter=['known'])
        assert source_scope.ids_by_domain['general'].tolist() == [0, 2]
        assert source_scope.identity != scope.identity
        source_config = replace(config, weak_negative_scope='source_membership', source_filter=('known',))
        sparse_multitask_loss(logits, rows, release=release, scope=source_scope, config=source_config)
        try:
            WeakNegativeScope.from_index(index, source_filter=['known'])
        except ValueError:
            pass
        else:
            raise AssertionError('Source-aware scope invented missing SourceMembership')
        index.close()
    return {'reference_subset_expectation_and_gradient': 'PASS', 'gradient_signs_and_P_U_exclusion': 'PASS',
            'chunked_and_K_ge_U_reference_parity': 'PASS', 'sample_epoch_occurrence_determinism': 'PASS',
            'empty_N_only_one_class_and_excluded_NaN': 'PASS', 'BF16_extreme_stability': 'PASS',
            'inactive_weak_scope_avoids_full_U_scan': 'PASS',
            'NO_TRAIN_POSITIVE_and_real_source_scope': 'PASS', 'sparse_multitask_numerators_counts_and_CE': 'PASS',
            'tiny_learnable': {'initial_loss': losses[0], 'final_loss': losses[-1]},
            'real_missing_label_quality': 'NOT_VERIFIED', 'classifier_matmul': 'full logits; loss sampling only'}
