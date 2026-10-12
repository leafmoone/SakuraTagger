"""Bounded RFS windows, actual-image geometry and exact logical cursor checks."""
from collections import Counter, defaultdict
from copy import deepcopy
from io import BytesIO
import json
import math
from pathlib import Path
import random
import tarfile
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from .data import (BucketBatchSampler, BucketConfig, BucketSelector, NativePreprocessor,
                   build_local_index, build_sampling_assets, ShardWindowSampler, window_loader,
                   TarDataset, ShardCache, LocalShardProvider, DimensionCorrections)
from .data.release import DatasetRelease, artifact_descriptor, reference_index_schema
from .selfcheck_release import reference_release, reference_row
from .selfcheck_tar import tiny_tar_release


def _single_shard_fixture(root):
    release = tiny_tar_release(root, count=8)
    rows = pq.read_table(root / 'index.parquet').to_pylist()
    path = root / 'combined.tar'
    with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as output:
        for row in rows:
            with tarfile.open(root / row['file_path'], 'r:') as source:
                for prefix in ('image', 'json'):
                    member = source.getmember(row[prefix + '_member'])
                    body = source.extractfile(member).read()
                    row[prefix + '_offset_data'] = output.offset + 512
                    output.addfile(member, BytesIO(body))
            row.update(shard_id='combined', file_path='combined.tar')
    pq.write_table(pa.Table.from_pylist(rows, schema=reference_index_schema(release.vocabulary, 'tiny_tar')),
                   root / 'index.parquet')
    document = release.document
    document['shards'] = [{**artifact_descriptor(path, root), 'shard_id': 'combined', 'status': 'published'}]
    document['indices'] = [{**artifact_descriptor(root / 'index.parquet', root), 'status': 'published'}]
    return DatasetRelease(document, release.vocabulary, root=root)


def _legacy_reference(sizes, weights, selector, batch_size, seed, epoch):
    rng, groups = random.Random(seed + epoch), defaultdict(list)
    occurrence = 0
    for sample, (size, weight) in enumerate(zip(sizes, weights)):
        for _ in range(math.floor(weight) + (rng.random() < weight % 1)):
            groups[selector.select(*size).target_size].append((occurrence, sample))
            occurrence += 1
    batches = []
    for size in sorted(groups):
        rng.shuffle(groups[size])
        for start in range(0, len(groups[size]), batch_size):
            batches.append(tuple(groups[size][start:start + batch_size]))
    rng.shuffle(batches)
    return batches


def check_sampling():
    selector = BucketSelector(BucketConfig(target_pixels=32 * 32, max_pixels=64 * 64, max_side=64))
    sizes = [(32, 32), (48, 32), (32, 48)] * 20
    weights = [1, 1.5, 3] * 20
    legacy = BucketBatchSampler(sizes, 4, selector=selector, shuffle=True, seed=21, repeat_weights=weights)
    assert list(legacy._global_batches()) == _legacy_reference(sizes, weights, selector, 4, 21, 0)
    assert len(json.dumps(legacy.state_dict())) < 1100
    # Version-1 full signatures still load without changing their next batch.
    old = deepcopy(legacy.state_dict())
    old['version'] = 1
    for field in ('asset_sha256', 'sample_count', 'max_repeat_factor'):
        old['signature'].pop(field)
    old['signature'].update(sizes=tuple(sizes), weights=tuple(weights))
    old['cursor'] = 1
    legacy.load_state_dict(old)
    with patch.object(legacy, '_global_batches', side_effect=AssertionError('No slicing/replanning')):
        before = len(legacy)
        legacy.advance()
        assert len(legacy) == before - 1
    input_sizes, input_weights = np.array(sizes), np.array(weights)
    snapshot = BucketBatchSampler(input_sizes, 4, selector=selector, shuffle=True, seed=21, repeat_weights=input_weights)
    binding = snapshot.state_dict()
    input_sizes[:] = 16
    input_weights[:] = 2
    assert snapshot.state_dict() == binding
    assert list(snapshot._global_batches()) == _legacy_reference(sizes, weights, selector, 4, 21, 0)
    assert not snapshot.sizes.flags.writeable and not snapshot.weights.flags.writeable
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Default t=.001 has all three factor groups, with 10,000 unique train
        # rows, plus excluded val/planned and invalid raw artist evidence.
        metadata_root = root / 'metadata'
        metadata_root.mkdir()
        rows = [reference_row(i, general_ids=[0] if i < 9997 else [1] if i == 9997 else [2],
                              character_ids=[], other_artist_ids=[7] if i == 0 else [],
                              single_artist_confirmed=False if i == 0 else None,
                              original_artist_ids=[1] if i == 1 else []) for i in range(10000)]
        rows[1]['general_ids'] = []  # Original-only / no new-domain positive remains factor 1.
        rows += [reference_row(10000, split='val', general_ids=[10]),
                 reference_row(10001, status='planned', file_path=None, general_ids=[10])]
        release = reference_release(metadata_root, rows)
        document = release.document
        document['shards'][0]['size_bytes'] = 32 * 1024 * 1024
        release = DatasetRelease(document, release.vocabulary, root=metadata_root)
        index = build_local_index(release, root / 'metadata.sqlite')
        assets = build_sampling_assets(index, root / 'default_assets')
        assert isinstance(assets.data, np.memmap) and assets.data.nbytes == 16 * 10000
        np.testing.assert_allclose(assets.data['factor'][-3:], [3, math.sqrt(5), math.sqrt(5)])
        assert assets.data['factor'][0] == assets.data['factor'][1] == 1
        shares = assets.metadata['stats']['factor_shares']
        assert shares == {'factor_1': .9997, 'factor_3': .0001, 'factor_between_1_3': .0002}
        assert not assets.support['other_artist'].any() and assets.support['general'][10] == 0
        # Metadata-only bounded-window scheduling probe supplies independently
        # confirmed synthetic sizes, never pretends metadata fixtures are images.
        class Dimensions:
            def __init__(self, index):
                self.index, self.preprocessor = index, NativePreprocessor(selector)
            def inspect_dimensions(self, ordinal):
                return (32, 32)
        probe = ShardWindowSampler(assets, Dimensions(index), 32, window_samples=128, seed=71)
        def collect(sampler):
            result, realized, bands = [], 0, None
            while not sampler.epoch_done:
                info = sampler.prepare_window()
                realized += info['real_exposures']
                if bands is None:
                    bands = deepcopy(info['train_frequency_band_exposures_after'])
                else:
                    for domain, values in bands.items():
                        for band in values:
                            values[band] += info['train_frequency_band_exposures_after'][domain][band]
                result.extend(pair for batch in sampler for pair in batch.occurrences)
                sampler.advance(len(sampler))
                sampler.next_window()
            return result, realized, bands
        order, realized, bands = collect(probe)
        counts = Counter(ordinal for ordinal, _ in order)
        assert set(counts) == set(range(10000)) and min(counts.values()) == 1 and max(counts.values()) == 3
        assert len(set(occurrence for _, occurrence in order)) == realized
        assert all(2 <= counts[i] <= 3 for i in (9998, 9999)) and counts[9997] == 3
        same = ShardWindowSampler(assets, Dimensions(index), 32, window_samples=128, seed=71)
        assert collect(same)[0] == order
        probe.set_epoch(1)
        assert collect(probe)[0] != order
        # Stochastic rounding's expectation is r, not floor(r), and cap is exact.
        rounding = []
        tiny = ShardWindowSampler(assets, Dimensions(index), 32, window_samples=10000, seed=3)
        for epoch in range(200):
            from .data.window_sampling import _rng
            rng = _rng(tiny.seed, epoch, 0, 'window')
            factors = assets.data['factor']
            realized_counts = np.floor(factors) + (rng.random(len(factors)) < factors % 1)
            rounding.append(realized_counts[-1])
            assert realized_counts.min() == 1 and realized_counts.max() == 3
        assert abs(np.mean(rounding) - math.sqrt(5)) < .1
        # Actual TARs: stale orientation metadata is corrected before a plan is
        # frozen; worker prefetch creates no logical consumption or order drift.
        image_root = root / 'images'
        image_release = _single_shard_fixture(image_root)
        image_index = build_local_index(image_release, root / 'images.sqlite')
        image_assets = build_sampling_assets(image_index, root / 'saturated', threshold=100.)
        assert image_assets.metadata['stats']['factor_shares']['factor_3'] == 1.
        assert image_assets.metadata['stats']['saturation_warning']
        cache = ShardCache(root / 'cache', image_release, LocalShardProvider(image_release), capacity_bytes=65536)
        corrections = DimensionCorrections(root / 'dimensions.sqlite', image_release)
        dataset = TarDataset(image_index, cache, corrections=corrections, preprocessor=NativePreprocessor(selector))
        original = image_index[0]
        sampler = ShardWindowSampler(image_assets, dataset, 3, seed=14, world_size=3)
        cold = sampler.state_dict()
        info = sampler.prepare_window()
        planned = list(sampler)
        global_ordinals = sampler._plan[0][:, 0]
        assert np.all(global_ordinals[1:] != global_ordinals[:-1])
        assert sampler.state_dict() == cold
        assert corrections.get(original['sample_id'])['effective_conflict']
        ranks, controls = [], 0
        for rank in range(3):
            local = ShardWindowSampler(image_assets, dataset, 3, seed=14, rank=rank, world_size=3)
            local.prepare_window()
            assert len(local) == len(sampler)
            occurrences = []
            for plan in local:
                controls += plan.control_only
                assert len({a for a, _ in plan.occurrences}) == len(plan.occurrences)
                occurrences.extend(plan.occurrences)
            ranks.append(set(occurrences))
        assert all(not ranks[a] & ranks[b] for a in range(3) for b in range(a + 1, 3))
        assert controls > 0
        combined = set.union(*ranks)
        assert len(combined) == 24 and Counter(a for a, _ in combined) == {i: 3 for i in range(8)}
        rng_before = torch.get_rng_state().clone()
        loader = window_loader(dataset, sampler, num_workers=2, prefetch_factor=2, worker_transport='numpy')
        first_plan, first = next(loader)
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert sampler.state_dict() == cold  # Worker lookahead is not a commit.
        sampler.advance()
        state = sampler.state_dict()
        assert len(json.dumps(state)) < 1200
        restored = ShardWindowSampler(image_assets, dataset, 3, seed=14, world_size=3)
        restored.load_state_dict(state)
        assert list(restored) == planned[1:]
        resumed = list(window_loader(dataset, restored))
        assert torch.equal(torch.get_rng_state(), rng_before)
        rest = list(loader)
        assert [plan for plan, _ in rest] == [plan for plan, _ in resumed]
        for plan, batch in [(first_plan, first), *rest]:
            if batch is not None:
                assert batch['images'].shape[-2:] == plan.target_size[::-1]
                assert batch['occurrence_ids'] == [b for _, b in plan.occurrences]
        for (_, a), (_, b) in zip(rest, resumed):
            if a is not None:
                torch.testing.assert_close(a['images'], b['images'], atol=0, rtol=0)
        assert image_index[0] == original  # Source dimensions/provenance unchanged.
        replay_cold = ShardWindowSampler(image_assets, dataset, 3, seed=14, world_size=3)
        replay_cold.load_state_dict(cold)
        assert list(replay_cold) == planned
        with patch.object(sampler, 'prepare_window', side_effect=AssertionError('No replan')):
            old_len = len(sampler)
            sampler.advance(old_len)
            assert len(sampler) == 0
        wrong = ShardWindowSampler(image_assets, dataset, 3, seed=14, world_size=2)
        try:
            wrong.load_state_dict(state)
        except ValueError:
            pass
        else:
            raise AssertionError('World-size change silently resumed')
        cache.close()
        corrections.close()
        return {'legacy_exact_order_v1_resume_and_input_snapshot': 'PASS', 'compact_mmap_bytes_per_unique_sample': 16,
                'default_factor_shares': shares, 'default_realized_epoch_exposures': realized,
                'train_frequency_bands_before': assets.metadata['stats']['train_frequency_band_exposures_before'],
                'realized_frequency_bands_after': bands,
                'stochastic_intermediate_mean': float(np.mean(rounding)),
                'strict_per_epoch_1_to_3': 'PASS', 'rank_disjoint_real_exposures': len(combined),
                'explicit_control_only_tail_steps': controls,
                'actual_EXIF_rebucket_and_prefetch_restart': 'PASS',
                'two_worker_transport': 'numpy over multiprocessing pipes; tensor-FD IPC NOT_VERIFIED',
                'O1_cursor_and_compact_checkpoint': 'PASS', 'loader_model_RNG_isolation': 'PASS',
                'saturated_fixture_warning': image_assets.metadata['stats']['saturation_warning'],
                'shuffle_scope': 'shard-local bounded windows, not uniform global shuffle'}
