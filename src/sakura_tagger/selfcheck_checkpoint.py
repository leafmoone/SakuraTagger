"""CPU exact-window checkpoint acceptance; no formal training or data downloads."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

from .data import window_loader
from .selfcheck_ddp import training_fixture, make_components, _state
from .training import CoordinatedTrainingError
from .training.window_checkpoint import _descriptor, state_digest


def _same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return set(left) == set(right) and all(_same(value, right[name]) for name, value in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _snapshot(trainer, sampler):
    return {'model': _state(trainer.model), 'optimizer': deepcopy(trainer.optimizer.state_dict()),
            'scheduler': trainer.scheduler.state_dict(), 'scaler': trainer.scaler.state_dict(),
            'counters': dict(trainer.counters), 'sampler': deepcopy(sampler.state_dict()),
            'rng': torch.get_rng_state().clone(), 'python': random.getstate(),
            'numpy': deepcopy(np.random.get_state())}


def _assert_unchanged(trainer, sampler, before):
    after = _snapshot(trainer, sampler)
    np.testing.assert_equal(after.pop('numpy'), before['numpy'])
    assert _same(after, {key: value for key, value in before.items() if key != 'numpy'})


def _edit_file(checkpoint, name, edit):
    """Synthetic corrupt metadata with a matching file hash, to reach validation."""
    path = checkpoint / name
    value = torch.load(path, map_location='cpu', weights_only=True)
    edit(value)
    torch.save(value, path)
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if name == 'common.pt':
        manifest['common'] = _descriptor(path)
    else:
        manifest['ranks'][0] = _descriptor(path)
    (checkpoint / 'manifest.json').write_text(json.dumps(manifest))


def _continuation(root, *, rank=0, world_size=1, stage='T2'):
    trainer, sampler, cache = make_components(root, rank=rank, world_size=world_size, stage=stage)
    sampler.prepare_window()
    trainer.train_window(sampler, iter(window_loader(sampler.dataset, sampler)))
    saved_cursor = sampler.cursor
    # Explicit lookahead loads a future real image without acknowledging it.
    lookahead = iter(window_loader(sampler.dataset, sampler))
    expected_plan, _ = next(lookahead)
    assert sampler.cursor == saved_cursor
    del lookahead
    random.seed(719 + rank); np.random.seed(719 + rank); torch.manual_seed(719 + rank)
    checkpoint = root / f'checkpoint-{stage}'
    trainer.save_checkpoint(checkpoint, sampler=sampler)
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    expected_result = trainer.train_window(sampler, iter(window_loader(sampler.dataset, sampler)))
    expected = _state(trainer.model)
    expected_optimizer = deepcopy(trainer.optimizer.state_dict())
    resumed, restored, second_cache = make_components(root, rank=rank, world_size=world_size, stage=stage,
                                                              cache_root=root / ('cold-cache-' + stage))
    resumed.load_checkpoint(checkpoint, sampler=restored)
    assert restored.cursor == saved_cursor and restored.batch_at(restored.cursor) == expected_plan
    actual_rng = (random.random(), np.random.random(), torch.rand(3))
    assert _same(expected_rng, actual_rng)
    with patch.object(second_cache.provider, 'fetch', wraps=second_cache.provider.fetch) as redownload:
        actual_result = resumed.train_window(restored, iter(window_loader(restored.dataset, restored)))
        if world_size == 1:
            assert redownload.call_count >= 1
    assert _same(expected_result, actual_result)
    assert _same(expected, _state(resumed.model)) and _same(expected_optimizer, resumed.optimizer.state_dict())
    assert _same(trainer.counters, resumed.counters)
    cache.close(); second_cache.close()
    return checkpoint, resumed, restored, {
        'next_sample_ids_occurrences_and_bucket': 'PASS', 'next_loss_counters_and_rng': 'PASS',
        'next_model_and_Adam_state_bitwise': 'PASS', 'lookahead_not_committed': 'PASS',
        'cold_cache_refetched_from_fixed_release': 'PASS'}


def check_checkpoint():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        training_fixture(root / 'published')
        _, _, _, t1 = _continuation(root, stage='T1')
        checkpoint, trainer, sampler, t2 = _continuation(root, stage='T2')
        before = _snapshot(trainer, sampler)
        cases = 0
        for name, mutate in (
            ('cursor', lambda p: _edit_file(p, 'rank-00000.pt', lambda v: v['sampler'].update(cursor=10**9))),
            ('model', lambda p: _edit_file(p, 'common.pt', lambda v: v['model'].pop(next(iter(v['model']))))),
            ('optimizer', lambda p: _edit_file(p, 'common.pt', lambda v: v['optimizer']['param_groups'][0].update(betas=(.1, .2)))),
            ('rng', lambda p: _edit_file(p, 'rank-00000.pt', lambda v: v['rng'].update(torch=torch.zeros(2, dtype=torch.uint8)))),
            ('rank_missing', lambda p: (p / 'rank-00000.pt').unlink()),
            ('manifest_missing', lambda p: (p / 'manifest.json').unlink()),
            ('checksum', lambda p: (p / 'common.pt').write_bytes(b'bad checkpoint')),
        ):
            broken = root / ('bad-' + name)
            shutil.copytree(checkpoint, broken)
            mutate(broken)
            try:
                trainer.load_checkpoint(broken, sampler=sampler)
            except CoordinatedTrainingError:
                _assert_unchanged(trainer, sampler, before)
            else:
                raise AssertionError(f'Accepted invalid checkpoint: {name}')
            cases += 1
        for field, value in (('world_size', 2), ('complete', False)):
            broken = root / ('bad-' + field)
            shutil.copytree(checkpoint, broken)
            manifest = json.loads((broken / 'manifest.json').read_text())
            manifest[field] = value
            (broken / 'manifest.json').write_text(json.dumps(manifest))
            try:
                trainer.load_checkpoint(broken, sampler=sampler)
            except CoordinatedTrainingError:
                _assert_unchanged(trainer, sampler, before)
            else:
                raise AssertionError('Accepted incompatible checkpoint manifest')
            cases += 1
        # Effective loss/structure/Release/preprocessing bindings reject before mutation.
        for key in ('training', 'preprocessing', 'teacher'):
            broken = root / ('bad-binding-' + key)
            shutil.copytree(checkpoint, broken)
            manifest = json.loads((broken / 'manifest.json').read_text())
            manifest['binding'][key] = {'changed': True}
            (broken / 'manifest.json').write_text(json.dumps(manifest))
            try:
                trainer.load_checkpoint(broken, sampler=sampler)
            except CoordinatedTrainingError:
                _assert_unchanged(trainer, sampler, before)
            else:
                raise AssertionError('Accepted changed checkpoint binding')
            cases += 1
        trainer.at_boundary = False
        try:
            trainer.save_checkpoint(root / 'partial-window', sampler=sampler)
        except CoordinatedTrainingError:
            assert not (root / 'partial-window').exists()
        else:
            raise AssertionError('Saved an incomplete optimizer window')
        trainer.at_boundary = True
        with patch('sakura_tagger.training.window_checkpoint._atomic_write', side_effect=OSError('synthetic disk failure')):
            try:
                trainer.save_checkpoint(root / 'failed-write', sampler=sampler)
            except CoordinatedTrainingError:
                assert not (root / 'failed-write/manifest.json').exists()
                _assert_unchanged(trainer, sampler, before)
            else:
                raise AssertionError('Published a failed checkpoint')
        # Rejecting a sampler cursor alone must preserve the existing consumer.
        invalid = sampler.state_dict(); invalid['cursor'] = 10**9
        try:
            sampler.load_state_dict(invalid)
        except ValueError:
            _assert_unchanged(trainer, sampler, before)
        else:
            raise AssertionError('Invalid cursor accepted')
        # A fixed teacher is rebound by its source and actual weights, not copied.
        teacher, teacher_sampler, teacher_cache = make_components(root, stage='T2', preservation={'global_features': .1})
        teacher_sampler.prepare_window()
        teacher.train_window(teacher_sampler, iter(window_loader(teacher_sampler.dataset, teacher_sampler)))
        teacher.save_checkpoint(root / 'with-teacher', sampler=teacher_sampler)
        common = torch.load(root / 'with-teacher/common.pt', weights_only=True)
        assert 'teacher' not in common and common['binding']['teacher']['state_sha256'] == state_digest(teacher.teacher)
        teacher.load_checkpoint(root / 'with-teacher', sampler=teacher_sampler)
        old_identity = teacher.teacher_identity
        teacher.teacher_identity = {**old_identity, 'state_sha256': '0' * 64}
        try:
            teacher.load_checkpoint(root / 'with-teacher', sampler=teacher_sampler)
        except CoordinatedTrainingError:
            pass
        else:
            raise AssertionError('Different teacher weights accepted')
        teacher_cache.close()
        return {'T1_same_world_continuation': t1, 'T2_same_world_continuation': t2,
                'invalid_checkpoint_cases_no_live_mutation': cases,
                'atomic_complete_manifest_and_partial_boundary_rejection': 'PASS',
                'sampler_invalid_cursor_no_mutation': 'PASS', 'fixed_teacher_rebind_without_copy': 'PASS',
                'CPU_DDP_RESUME': 'NOT_VERIFIED; separate actual Gloo suite required',
                'CUDA_RESUME': 'NOT_VERIFIED'}


def check_distributed_checkpoint(root):
    """Called by the real torchrun worker only after Gloo initialized successfully."""
    root = Path(root)
    for stage in ('T1', 'T2'):
        _continuation(root, rank=dist.get_rank(), world_size=dist.get_world_size(), stage=stage)
    return {'same_world_next_ids_loss_update_rng': 'PASS'}


if __name__ == '__main__':
    print(json.dumps(check_checkpoint(), indent=2))
