"""Actual torchrun CPU/Gloo acceptance against identical single-process windows."""
import argparse
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from .data import (BucketConfig, BucketSelector, NativePreprocessor, DatasetRelease, build_local_index,
                   build_sampling_assets, SamplingAssets, SparseReleaseIndex, ShardCache, LocalShardProvider,
                   DimensionCorrections, TarDataset, ShardWindowSampler, window_loader)
from .data.release import artifact_descriptor, reference_index_schema
from .data.tar_dataset import reference_sample_json
from .model import MultiTaskModel, KaloscopeBackbone
from .selfcheck import synthetic_fixture
from .selfcheck_tar import tiny_tar_release
from .training import (WindowTrainer, WindowTrainerConfig, WeakNegativeConfig, SparseLossConfig,
                       WeakNegativeScope, initialize_distributed, CoordinatedTrainingError)


def training_fixture(root, *, splits=None, build_assets=True):
    root = Path(root)
    release = tiny_tar_release(root, count=9)
    rows = pq.read_table(root / 'index.parquet').to_pylist()
    path = root / 'training.tar'
    with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as output:
        for i, row in enumerate(rows):
            with tarfile.open(root / row['file_path'], 'r:') as source:
                body = source.extractfile(row['image_member']).read()
            if splits is not None:
                row['split'] = splits[i]
            row.update(general_ids=[i % 3] if i % 2 == 0 else [], character_ids=[1] if i == 1 else [],
                       copyright_ids=[0] if i == 6 else [], general_negative_ids=[1] if i == 7 else [],
                       other_artist_ids=[] if i in (0, 2) else [1, 2] if i == 1 else [i % 3],
                       original_artist_ids=[1] if i == 0 else [], shard_id='training', file_path='training.tar')
            for prefix, content in (('image', body), ('json', json.dumps(reference_sample_json(row, release.vocabulary)).encode())):
                member = tarfile.TarInfo(row[prefix + '_member'])
                member.size = len(content)
                row[prefix + '_offset_data'], row[prefix + '_size'] = output.offset + 512, len(content)
                output.addfile(member, BytesIO(content))
    pq.write_table(pa.Table.from_pylist(rows, schema=reference_index_schema(release.vocabulary, 'tiny_tar')), root / 'index.parquet')
    document = release.document
    document['shards'] = [{**artifact_descriptor(path, root), 'shard_id': 'training', 'status': 'published'}]
    document['indices'] = [{**artifact_descriptor(root / 'index.parquet', root), 'status': 'published'}]
    (root / 'release.json').write_text(json.dumps(document))
    release = DatasetRelease.read(root / 'release.json')
    if build_assets:
        index = build_local_index(release, root.parent / 'index.sqlite')
        build_sampling_assets(index, root.parent / 'assets')
        DimensionCorrections(root.parent / 'dimensions.sqlite', release).close()
        index.close()
    return release


def make_components(root, *, rank=0, world_size=1, stage='T1', accumulation_steps=2, amp='off', preservation=None, cache_root=None):
    root = Path(root)
    release = DatasetRelease.read(root / 'published/release.json')
    index = SparseReleaseIndex(root / 'index.sqlite', release=release)
    assets = SamplingAssets(root / 'assets', index)
    cache = ShardCache(cache_root or root / 'cache', release, LocalShardProvider(release), capacity_bytes=65536, job_id='training')
    dataset = TarDataset(index, cache, corrections=DimensionCorrections(root / 'dimensions.sqlite', release),
                         preprocessor=NativePreprocessor(BucketSelector(BucketConfig(target_pixels=1024, max_pixels=4096, max_side=64))))
    sampler = ShardWindowSampler(assets, dataset, 2, window_samples=32, seed=47, rank=rank, world_size=world_size)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(231)
        model = MultiTaskModel(KaloscopeBackbone(synthetic_fixture()), release.vocabulary.layout(feature_dim=32, attention_dim=16))
    loss_config = SparseLossConfig({domain: WeakNegativeConfig(1., 1., 2, chunk_size=4)
                                  for domain in ('general', 'character', 'copyright')}, seed=51)
    config = WindowTrainerConfig(stage=stage, accumulation_steps=accumulation_steps, max_epochs=1,
                                 schedule_updates=8, warmup_updates=2, gradient_clip=None, amp=amp,
                                 preservation=preservation or {'global_features': 0., 'style_embedding': 0., 'artist_logits': 0.})
    teacher = KaloscopeBackbone(synthetic_fixture(12)) if preservation else None
    trainer = WindowTrainer(model, release, WeakNegativeScope.from_index(index), loss_config,
                            config=config, teacher=teacher,
                            teacher_source='synthetic_fixture:v1:seed=12' if teacher is not None else None)
    return trainer, sampler, cache


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _worker(root):
    torch.set_num_threads(1)
    initialize_distributed('cpu', timeout_seconds=45)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        for stage in ('T1', 'T2'):
            trainer, sampler, cache = make_components(root, rank=rank, world_size=world, stage=stage)
            history, initial = [], _state(trainer.model)
            counters = trainer.run(sampler, on_boundary=lambda t, s, r: history.append(r))
            assert counters['real_exposures'] == 9 and counters['successful_updates'] == 2
            assert all(torch.equal(initial[name], value) for name, value in trainer.model.state_dict().items()
                       if name.startswith(('backbone.legacy.head.', 'backbone.legacy.projector.')))
            torch.save({'model': trainer.model.state_dict(), 'history': history, 'counters': counters},
                       Path(root) / f'{stage}_rank{rank}.pt')
            cache.close()
        # Rank-local scheduling differences must not silently duplicate/drop data.
        trainer, sampler, cache = make_components(root, rank=rank, world_size=world)
        sampler.seed += rank
        sampler.set_epoch(0)
        sampler.prepare_window()
        try:
            trainer.train_window(sampler, iter(window_loader(sampler.dataset, sampler)))
        except CoordinatedTrainingError:
            assert sampler.cursor == 0
        else:
            raise AssertionError('Rank-local sampler mismatch accepted')
        cache.close()
        # A rank-local ordinary data failure must stop every rank before backward.
        trainer, sampler, cache = make_components(root, rank=rank, world_size=world)
        sampler.prepare_window()
        before = sampler.state_dict()
        loader = iter(window_loader(sampler.dataset, sampler))
        def bad_data():
            if rank == 1:
                raise OSError('synthetic read failure')
            yield from loader
        try:
            trainer.train_window(sampler, iter(bad_data()))
        except CoordinatedTrainingError:
            assert sampler.state_dict() == before and not trainer.at_boundary
        else:
            raise AssertionError('Rank-local data failure was not coordinated')
        cache.close()
        # A selected NaN on just one rank skips the whole attempted window, then
        # the remaining finite short window still updates and run limits end it.
        trainer, sampler, cache = make_components(root, rank=rank, world_size=world)
        sampler.prepare_window()
        baseline = _state(trainer.model)
        loader = iter(window_loader(sampler.dataset, sampler))
        def bad_value():
            for step, (plan, batch) in enumerate(loader):
                if rank == 1 and step == 0 and batch is not None:
                    batch['images'].fill_(float('nan'))
                yield plan, batch
        bad_loader = iter(bad_value())
        result = trainer.train_window(sampler, bad_loader)
        assert not result['success'] and result['skip_reason'] == 'nonfinite'
        assert all(torch.equal(value, trainer.model.state_dict()[name]) for name, value in baseline.items())
        assert trainer.scheduler.completed == 0 and sampler.cursor == 2
        result = trainer.train_window(sampler, bad_loader)
        assert result['success'] and trainer.scheduler.completed == 1
        assert trainer.counters == {'attempted_windows': 2, 'successful_updates': 1, 'skipped_updates': 1,
                                    'real_exposures': 9, 'control_steps': 3}
        cache.close()
        # Safe stage transition rebuilds reducer and optimizer membership.
        trainer, sampler, cache = make_components(root, rank=rank, world_size=world)
        with patch.object(trainer, '_wrap', wraps=trainer._wrap) as rebuild:
            trainer.set_stage('T2')
            assert rebuild.call_count == 1
        assert trainer.config.stage == 'finetune_last4'
        assert 'backbone_last4' in {group['name'] for group in trainer.optimizer.param_groups}
        assert trainer.run(sampler)['successful_updates'] == 2
        cache.close()
        from .selfcheck_checkpoint import check_distributed_checkpoint
        checkpoint_result = check_distributed_checkpoint(root)
        from .selfcheck_integration import check_distributed_pipeline
        pipeline_result = check_distributed_pipeline(Path(root) / 'pipeline')
        (Path(root) / f'worker{rank}.json').write_text(json.dumps({'rank': rank, 'data_failure': 'PASS',
            'nonfinite_skip_then_update': 'PASS', 'stage_DDP_rebuild': 'PASS', 'sampler_signature_mismatch': 'PASS', 'checkpoint_resume': checkpoint_result, 'integrated_pipeline': pipeline_result}))
    finally:
        dist.destroy_process_group()


def _legacy_exact_check():
    from .selfcheck import synthetic_vocabulary, synthetic_records
    from .data import BucketBatchSampler
    from .data.contract import SCHEMA_VERSION, encode_supervision, manifest_fingerprint
    from .training import Trainer, TrainerConfig, LossConfig, checkpoint_metadata
    vocabulary, records = synthetic_vocabulary(), synthetic_records()
    metadata = {'schema_version': SCHEMA_VERSION, 'dataset_id': 'synthetic', 'provider': 'SakuraPool',
                'vocab_sha256': vocabulary.sha256, 'manifest_sha256': manifest_fingerprint(records)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        model = MultiTaskModel(KaloscopeBackbone(synthetic_fixture()), vocabulary.layout(feature_dim=32, attention_dim=16))
        images = torch.randn(3, 3, 32, 48)
    encoded = [encode_supervision(records[i % len(records)], vocabulary) for i in range(3)]
    targets = {name: torch.stack([item[0][name] for item in encoded]) for name in encoded[0][0]}
    masks = {name: torch.stack([item[1][name] for item in encoded]) for name in encoded[0][1]}
    # Unequal masks, including one locally empty task.
    for name in masks:
        masks[name][0] = False
    batch = {'images': images, 'targets': targets, 'masks': masks}
    splits = [{'images': images[:1], 'targets': {n: v[:1] for n, v in targets.items()}, 'masks': {n: v[:1] for n, v in masks.items()}},
              {'images': images[1:], 'targets': {n: v[1:] for n, v in targets.items()}, 'masks': {n: v[1:] for n, v in masks.items()}}]
    binding = checkpoint_metadata(model, vocabulary, metadata, BucketConfig(), 'synthetic')
    config = TrainerConfig(accumulation_steps=3, gradient_clip=None)
    a, b = Trainer(model, binding, config=config), Trainer(deepcopy(model), binding, config=config)
    sampler = BucketBatchSampler([(32, 48)] * 4, 2)
    a.train_window(splits, sampler=sampler)  # Short actual tail of 2, configured max 3.
    b.train_window([batch])
    max_error = max(float((value - b.model.state_dict()[name]).abs().max()) for name, value in a.model.state_dict().items())
    assert max_error < 2e-6
    assert sampler.cursor == 2 and a.config.normalization == 'observed_entry_window'
    empty = {**batch, 'masks': {n: torch.zeros_like(v) for n, v in masks.items()}}
    before = _state(a.model)
    a.train_window([empty])
    assert a.global_step == 1 and a.skipped_updates == 1
    assert all(torch.equal(value, a.model.state_dict()[name]) for name, value in before.items())
    legacy_sampler = BucketBatchSampler([(32, 48)] * 8, 2)
    a.train_batch(batch, sampler=legacy_sampler)
    assert legacy_sampler.cursor == 0
    a.flush()
    assert legacy_sampler.cursor == 1
    a.set_stage('T2')
    assert a.config.stage == a.model.backbone.stage == 'finetune_last4'
    return max_error


def check_window_trainer():
    """Single-process acceptance kept independent from Gloo runtime capability."""
    from .training import sparse_multitask_loss, sparse_valid_counts
    from .model.registry import LOGICAL_DOMAINS
    torch.set_num_threads(1)
    legacy_error = _legacy_exact_check()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        training_fixture(root / 'published')
        errors = {}
        for stage in ('T1', 'T2'):
            trainer, sampler, cache = make_components(root, stage=stage, accumulation_steps=4)
            reference, reference_sampler, reference_cache = make_components(root, stage=stage, accumulation_steps=4)
            sampler.prepare_window()
            reference_sampler.prepare_window()
            loader = iter(window_loader(sampler.dataset, sampler))
            reference_loader = iter(window_loader(reference_sampler.dataset, reference_sampler))
            while len(sampler):
                steps = min(4, len(sampler))
                batches = [next(reference_loader)[1] for _ in range(steps)]
                records = [row for batch in batches for row in batch['records']]
                counts = sparse_valid_counts(records, reference.loss_config, reference.release)
                total = 0.
                reference.model.train()
                for batch in batches:
                    output = reference.model(batch['images'], LOGICAL_DOMAINS)
                    loss = sparse_multitask_loss(output.logits, batch['records'], release=reference.release,
                        scope=reference.scope, config=reference.loss_config, epoch=0, occurrence_ids=batch['occurrence_ids'])
                    total = total + sum(value.numerator / counts[name] if counts[name] else value.numerator
                                        for name, value in loss.domains.items())
                total.backward()
                reference._clear_inactive(counts, len(records))
                reference.optimizer.step()
                reference.optimizer.zero_grad(set_to_none=True)
                reference.scheduler.step()
                result = trainer.train_window(sampler, loader)
                assert result['valid_counts'] == counts and result['success']
            errors[stage] = max(float((value - reference.model.state_dict()[name]).abs().max())
                                for name, value in trainer.model.state_dict().items())
            assert errors[stage] < 2e-6
            assert trainer.counters['real_exposures'] == 9 and trainer.counters['successful_updates'] == 2
            optimized = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
            assert optimized == {id(p) for p in trainer.model.parameters() if p.requires_grad}
            for group in trainer.optimizer.param_groups:
                if group['weight_decay']:
                    assert all(p.ndim > 1 for p in group['params'])
            cache.close(); reference_cache.close()
        trainer, sampler, cache = make_components(root)
        trainer.loss_config = replace(trainer.loss_config, weights={name: 0. for name in LOGICAL_DOMAINS})
        trainer.config = replace(trainer.config, max_epochs=None, max_attempted_windows=2)
        before = _state(trainer.model)
        result = trainer.run(sampler)
        assert result['successful_updates'] == 0 and result['skipped_updates'] == 2
        assert trainer.scheduler.completed == 0 and all(torch.equal(value, trainer.model.state_dict()[name]) for name, value in before.items())
        cache.close()
        trainer, sampler, cache = make_components(root, accumulation_steps=4)
        sampler.prepare_window()
        cursor = sampler.cursor
        def fail():
            raise OSError('synthetic local read failure')
            yield
        try:
            trainer.train_window(sampler, iter(fail()))
        except CoordinatedTrainingError:
            assert sampler.cursor == cursor and not trainer.at_boundary
        else:
            raise AssertionError('Data failure accepted')
        cache.close()
        trainer, sampler, cache = make_components(root, accumulation_steps=4)
        sampler.prepare_window()
        initial = _state(trainer.model)
        def nonfinite():
            for step, (plan, batch) in enumerate(window_loader(sampler.dataset, sampler)):
                if step == 0:
                    batch['images'].fill_(float('nan'))
                yield plan, batch
        loader = iter(nonfinite())
        assert trainer.train_window(sampler, loader)['skip_reason'] == 'nonfinite'
        assert all(torch.equal(value, trainer.model.state_dict()[name]) for name, value in initial.items())
        assert trainer.scheduler.completed == 0 and sampler.cursor == 4
        assert trainer.train_window(sampler, loader)['success'] and trainer.scheduler.completed == 1
        cache.close()
        teacher, sampler, cache = make_components(root, stage='T2', amp='bf16', accumulation_steps=4,
            preservation={'global_features': .2, 'style_embedding': .1, 'artist_logits': .1})
        frozen = _state(teacher.teacher)
        assert teacher.run(sampler)['successful_updates'] == 2
        assert all(not p.requires_grad and p.grad is None for p in teacher.teacher.parameters())
        assert all(torch.equal(value, teacher.teacher.state_dict()[name]) for name, value in frozen.items())
        cache.close()
        # An empty metadata-only train split must fail before any image/cache read.
        from .selfcheck_release import reference_release, reference_row
        empty_root = root / 'empty'
        empty_root.mkdir()
        empty_release = reference_release(empty_root, [reference_row(0, split='val')])
        empty_index = build_local_index(empty_release, empty_root / 'index.sqlite')
        empty_assets = build_sampling_assets(empty_index, empty_root / 'assets')
        empty_cache = ShardCache(empty_root / 'cache', empty_release, LocalShardProvider(empty_release), capacity_bytes=65536)
        empty_dataset = TarDataset(empty_index, empty_cache, corrections=DimensionCorrections(empty_root / 'dimensions', empty_release))
        empty_sampler = ShardWindowSampler(empty_assets, empty_dataset, 2)
        empty_trainer = WindowTrainer(deepcopy(trainer.model), empty_release, WeakNegativeScope.from_index(empty_index), trainer.loss_config,
                                     config=replace(trainer.config, max_epochs=None, max_attempted_windows=2))
        try:
            empty_trainer.run(empty_sampler)
        except ValueError as error:
            assert 'eligible unique train sample' in str(error)
        else:
            raise AssertionError('Empty assets did not terminate')
        empty_cache.close()
        phase, sampler, cache = make_components(root, accumulation_steps=4)
        sampler.prepare_window()
        phase.train_window(sampler, iter(window_loader(sampler.dataset, sampler)))
        assert phase.scheduler.completed == phase.counters['successful_updates'] == 1
        phase.set_stage('T2')
        assert phase.scheduler.completed == 0 and phase.counters['successful_updates'] == 1
        assert phase.config.stage == phase.model.backbone.stage == 'finetune_last4'
        phase.run(sampler)
        assert phase.scheduler.completed == 1 and phase.counters['successful_updates'] == 2
        cache.close()
        return {'exact_sparse_window_vs_full_graph_reference_error': errors,
                'legacy_exact_observed_entry_unequal_mask_short_tail_error': legacy_error,
                'T1_T2_optimizer_decay_groups': 'PASS', 'empty_gradient_no_decay_finite_end': 'PASS',
                'data_failure_no_cursor_commit': 'PASS', 'nonfinite_skip_then_finite_update': 'PASS',
                'successful_update_scheduler_and_explicit_phase_reset': 'PASS', 'empty_assets_finite_rejection': 'PASS', 'CPU_BF16_independent_frozen_teacher': 'PASS',
                'DDP': 'separate actual Gloo suite required; never inferred from this check'}


def check_ddp():
    torch.set_num_threads(1)
    legacy_error = _legacy_exact_check()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        training_fixture(root / 'published')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1', '--nproc-per-node=2',
                   '--master-addr=127.0.0.1', f'--master-port={port}', '-m',
                   'sakura_tagger.selfcheck_ddp', '--worker', '--root', str(root)]
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'GLOO_SOCKET_IFNAME': 'lo'}
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f'torchrun CPU/Gloo check failed:\n{result.stdout}\n{result.stderr}')
        errors, counts = {}, {}
        for stage in ('T1', 'T2'):
            rank0 = torch.load(root / f'{stage}_rank0.pt', weights_only=True)
            rank1 = torch.load(root / f'{stage}_rank1.pt', weights_only=True)
            for name in rank0['model']:
                torch.testing.assert_close(rank0['model'][name], rank1['model'][name], rtol=0, atol=0)
            reference, sampler, cache = make_components(root, stage=stage, accumulation_steps=4)
            history = []
            reference.run(sampler, on_boundary=lambda t, s, r: history.append(r))
            assert [r['valid_counts'] for r in history] == [r['valid_counts'] for r in rank0['history']]
            for name, expected in reference.model.state_dict().items():
                torch.testing.assert_close(rank0['model'][name], expected, rtol=2e-5, atol=2e-6)
            errors[stage] = max(float((value - reference.model.state_dict()[name]).abs().max())
                                for name, value in rank0['model'].items())
            counts[stage] = [r['valid_counts'] for r in history]
            cache.close()
        assert all(json.loads((root / f'worker{rank}.json').read_text())['nonfinite_skip_then_update'] == 'PASS' for rank in range(2))
        # Completely unknown sparse windows: all exclusive modules stay bitwise
        # fixed despite AdamW decay, and finite attempt limits still terminate.
        trainer, sampler, cache = make_components(root)
        trainer.loss_config = replace(trainer.loss_config, weights={name: 0. for name in trainer.loss_config.weights})
        trainer.config = replace(trainer.config, max_epochs=None, max_attempted_windows=2)
        before = _state(trainer.model)
        counters = trainer.run(sampler)
        assert counters['successful_updates'] == 0 and counters['skipped_updates'] == 2
        assert all(torch.equal(value, trainer.model.state_dict()[name]) for name, value in before.items())
        cache.close()
        # CPU BF16 and independent frozen teacher use the same actual image path.
        teacher, sampler, cache = make_components(root, stage='T2', amp='bf16', accumulation_steps=4,
            preservation={'global_features': .2, 'style_embedding': .1, 'artist_logits': .1})
        frozen = _state(teacher.teacher)
        assert teacher.run(sampler)['successful_updates'] == 2
        assert all(not p.requires_grad and p.grad is None for p in teacher.teacher.parameters())
        assert all(torch.equal(value, teacher.teacher.state_dict()[name]) for name, value in frozen.items())
        cache.close()
        return {'CPU_DDP_PASS': True, 'backend': 'Gloo via torchrun/TCP loopback, 2 ranks x 1 compute thread',
                'single_process_same_samples_max_parameter_error': errors, 'tolerance': {'atol': 2e-6, 'rtol': 2e-5},
                'whole_window_valid_image_counts': counts, 'real_exposures': 9, 'updates': 2,
                'unequal_batches_local_empty_control_tail_short_window': 'PASS',
                'rank_data_failure_and_nonfinite_coordination': 'PASS', 'T1_T2_optimizer_DDP_rebuild': 'PASS',
                'empty_gradient_no_AdamW_decay_finite_end': 'PASS',
                'legacy_exact_unequal_mask_short_tail_error': legacy_error,
                'legacy_committed_cursor_and_normalization_identity': 'PASS',
                'CPU_BF16_independent_teacher': 'PASS', 'same_world_checkpoint_resume': 'PASS', 'distributed_Release_to_predict_pipeline': 'PASS', 'CUDA_NCCL_FP16': 'NOT_VERIFIED',
                'default_worker_tensor_FD_IPC': 'NOT_VERIFIED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--single-process', action='store_true')
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    if args.worker:
        _worker(args.root)
    else:
        print(json.dumps(check_window_trainer() if args.single_process else check_ddp(), indent=2))


if __name__ == '__main__':
    main()
