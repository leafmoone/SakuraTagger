"""Immutable compact RFS assets and bounded, replayable shard-local windows.

The shuffle is window-local, not a uniform global permutation. Only acknowledged
control steps are committed; iterator lookahead and image/cache reads are not.
"""
from collections import Counter, defaultdict
from copy import copy
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from .release import MULTILABEL_DOMAINS, artifact_descriptor, single_artist_target, verify_artifact
from .tar_dataset import collate_prepared

ASSET_VERSION = 'sakura-tagger-rfs-assets-v1'
SAMPLER_VERSION = 'sakura-tagger-shard-windows-v1'
DOMAINS = (*MULTILABEL_DOMAINS, 'other_artist')
EXPOSURE_DTYPE = np.dtype([('ordinal', '<u8'), ('factor', '<f8')])
BANDS = ('1', '2-9', '10-99', '100-999', '1000+')


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _rng(seed, epoch, window, purpose):
    value = _sha([seed, epoch, window, purpose])
    return np.random.Generator(np.random.PCG64(int(value[:16], 16)))


def _positives(row, release):
    result = {domain: row[domain + '_ids'] for domain in MULTILABEL_DOMAINS}
    artist = single_artist_target(row, release.vocabulary, release_confirmed=release.single_artist_confirmed)
    result['other_artist'] = [artist.target] if artist.valid else []
    return result


def _band(count):
    return 0 if count == 1 else 1 if count < 10 else 2 if count < 100 else 3 if count < 1000 else 4


def _empty_bands():
    return {domain: {band: 0 for band in BANDS} for domain in DOMAINS}


def build_sampling_assets(index, output, *, threshold=0.001, max_repeat_factor=3.0):
    """One index pass, no image reads; 16 bytes per unique train sample on disk.

    Frequencies come from the validated index's independent train support. No
    Original Artist, val/test, conflict claims, or repeated occurrence is counted.
    """
    if not math.isfinite(threshold) or threshold <= 0 or max_repeat_factor != 3:
        raise ValueError('Positive RFS threshold and the strict cap of 3 are required')
    config = {'threshold': threshold, 'max_repeat_factor': float(max_repeat_factor)}
    output = Path(output)
    if output.exists():
        assets = SamplingAssets(output, index)
        if assets.metadata['config'] != config:
            raise ValueError('Sampling assets have different RFS settings')
        return assets
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.partial.', dir=output.parent))
    count = index.metadata['stats']['splits'].get('train', 0)
    support = {domain: index.positive_support(domain) for domain in DOMAINS}
    shares, before, expected, shards = Counter(), _empty_bands(), _empty_bands(), []
    try:
        data = np.lib.format.open_memmap(temporary / 'samples.npy', mode='w+', dtype=EXPOSURE_DTYPE, shape=(count,))
        cursor = 0
        for shard in sorted(index.release.shards):
            start = cursor
            for ordinal, row in index.iter_records(split='train', shard_id=shard):
                positive = _positives(row, index.release)
                counts = [int(support[domain][label]) for domain, labels in positive.items() for label in labels]
                if any(value < 1 for value in counts):
                    raise ValueError('Index positive-support metadata disagrees with training facts')
                factor = max((min(3., max(1., math.sqrt(threshold * count / value))) for value in counts), default=1.)
                data[cursor] = (ordinal, factor)
                cursor += 1
                shares['factor_1' if factor == 1 else 'factor_3' if factor == 3 else 'factor_between_1_3'] += 1
                for domain, labels in positive.items():
                    for label in labels:
                        band = BANDS[_band(int(support[domain][label]))]
                        before[domain][band] += 1
                        expected[domain][band] += factor
            if cursor > start:
                shards.append({'shard_id': shard, 'start': start, 'stop': cursor})
        if cursor != count:
            raise ValueError('Unique train sample count changed while building assets')
        data.flush()
        del data
        metadata = {'version': ASSET_VERSION, 'release_identity': index.release.identity, 'config': config,
                    'sample_count': count, 'shards': shards,
                    'array': artifact_descriptor(temporary / 'samples.npy', temporary),
                    'stats': {'factor_shares': {key: shares[key] / count if count else 0. for key in
                                               ('factor_1', 'factor_between_1_3', 'factor_3')},
                              'train_frequency_band_exposures_before': before,
                              'expected_band_exposures_after': expected,
                              'frequency_basis': 'unique published train; Other Artist trusted CE only',
                              'saturation_warning': 'At least half of images hit cap=3; rebalancing is limited.'
                                                    if count and shares['factor_3'] / count >= .5 else None}}
        metadata['identity'] = _sha(metadata)
        (temporary / 'manifest.json').write_text(json.dumps(metadata, sort_keys=True), encoding='utf-8')
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return SamplingAssets(output, index)


class SamplingAssets:
    def __init__(self, path, index):
        self.path, self.index = Path(path), index
        self.metadata = json.loads((self.path / 'manifest.json').read_text(encoding='utf-8'))
        binding = dict(self.metadata)
        identity = binding.pop('identity', None)
        if (binding.get('version') != ASSET_VERSION or binding.get('release_identity') != index.release.identity
                or identity != _sha(binding)):
            raise ValueError('Incomplete or incompatible sampling assets')
        verify_artifact(self.path / 'samples.npy', self.metadata['array'])  # Once at open, never per epoch/step.
        self.data = np.load(self.path / 'samples.npy', mmap_mode='r', allow_pickle=False)
        if self.data.dtype != EXPOSURE_DTYPE or self.data.shape != (self.metadata['sample_count'],):
            raise ValueError('Invalid compact sampling array')
        self.identity = identity
        self.support = {domain: index.positive_support(domain) for domain in DOMAINS}

    def __len__(self):
        return len(self.data)


@dataclass(frozen=True)
class PlannedBatch:
    epoch: int
    window: int
    control_step: int
    global_batch: int | None
    occurrences: tuple  # (index ordinal, occurrence_id), never a replicated image.
    target_size: tuple | None

    @property
    def control_only(self):
        return self.global_batch is None


class ShardWindowSampler:
    """A shared deterministic window plan, with rank-disjoint real batches.

    Call prepare_window() before using len/advance; these two are O(1) thereafter.
    A window remains frozen until its acknowledged cursor reaches the end and the
    caller explicitly calls next_window(). Iterator prefetch cannot transition it.
    All rank control lengths match. Missing tail batches have zero occurrences.
    """
    def __init__(self, assets, dataset, batch_size, *, window_samples=4096, seed=0, rank=0, world_size=1):
        if any(type(v) is not int or v < 1 for v in (batch_size, window_samples, world_size)):
            raise ValueError('Positive batch/window/world sizes are required')
        if type(rank) is not int or not 0 <= rank < world_size or type(seed) is not int:
            raise ValueError('Invalid rank or seed')
        if dataset.index.release.identity != assets.index.release.identity:
            raise ValueError('Sampling and image Release mismatch')
        self.assets, self.dataset = assets, dataset
        self.batch_size, self.window_samples, self.seed = batch_size, window_samples, seed
        self.rank, self.world_size = rank, world_size
        self.epoch = self.window = self.cursor = 0
        self._plan = None
        self._schedule = self._schedule_epoch()

    def _schedule_epoch(self):
        rng = _rng(self.seed, self.epoch, 0, 'schedule')
        windows = []
        for shard in rng.permutation(len(self.assets.metadata['shards'])):
            spec = self.assets.metadata['shards'][int(shard)]
            starts = np.arange(spec['start'], spec['stop'], self.window_samples, dtype=np.int64)
            rng.shuffle(starts)
            windows.extend((int(shard), int(start), min(int(start) + self.window_samples, spec['stop'])) for start in starts)
        return np.asarray(windows, dtype=np.int64).reshape(-1, 3)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch, self.window, self.cursor, self._plan = epoch, 0, 0, None
        self._schedule = self._schedule_epoch()

    @property
    def epoch_done(self):
        return self.window == len(self._schedule)

    def prepare_window(self):
        if self._plan is not None:
            return self.window_info
        if self.epoch_done:
            return None
        shard, start, stop = map(int, self._schedule[self.window])
        shard_id = self.assets.metadata['shards'][shard]['shard_id']
        # Planning uses actual decoded dimensions, never mutable P2 assumptions.
        # On first encounter it decodes once without resize; later visits reuse
        # the identity-bound correction. No tensors are held for the whole window.
        entries = self.assets.data[start:stop]
        rng = _rng(self.seed, self.epoch, self.window, 'window')
        factors = entries['factor']
        repeats = np.floor(factors).astype(np.int64) + (rng.random(len(entries)) < factors % 1)
        groups, records, after = defaultdict(list), {}, _empty_bands()
        for local, (entry, repeat) in enumerate(zip(entries, repeats)):
            ordinal = int(entry['ordinal'])
            size = self.dataset.inspect_dimensions(ordinal)
            target = self.dataset.preprocessor.selector.select(*size).target_size
            groups[target].append(local)
            row = self.assets.index[ordinal]
            records[ordinal] = row
            for domain, labels in _positives(row, self.assets.index.release).items():
                for label in labels:
                    after[domain][BANDS[_band(int(self.assets.support[domain][label]))]] += int(repeat)
        batches, shapes = [], []
        # Within each bucket a highest-remaining-first heap avoids batch repeats
        # entirely and consecutive repeats whenever another image is available.
        for shape in sorted(groups):
            heap = [(-int(repeats[i]), float(rng.random()), i) for i in groups[shape]]
            heapq.heapify(heap)
            previous = None
            while heap:
                picked = [heapq.heappop(heap) for _ in range(min(self.batch_size, len(heap)))]
                if len(picked) == 1 and picked[0][2] == previous and heap:
                    alternate = heapq.heappop(heap)
                    heapq.heappush(heap, picked[0])
                    picked = [alternate]
                rng.shuffle(picked)
                if len(picked) > 1 and picked[0][2] == previous:
                    picked[0], picked[1] = picked[1], picked[0]
                batch = []
                for remaining, _, i in picked:
                    ordinal = int(entries[i]['ordinal'])
                    repetition = int(repeats[i]) + remaining
                    batch.append((ordinal, ordinal * 3 + repetition))
                    if remaining < -1:
                        heapq.heappush(heap, (remaining + 1, float(rng.random()), i))
                previous = picked[-1][2]
                batches.append(batch)
                shapes.append(shape)
        # Bucket order is already shuffled by randomizing bucket runs, preserving
        # each run's no-consecutive policy. Every repeated image stays in one run.
        run_order = rng.permutation(len(groups))
        by_shape = {shape: [] for shape in groups}
        for batch, shape in zip(batches, shapes):
            by_shape[shape].append(batch)
        batches, shapes = [], []
        for position in run_order:
            shape = sorted(groups)[int(position)]
            batches.extend(by_shape[shape])
            shapes.extend([shape] * len(by_shape[shape]))
        offsets = np.empty(len(batches) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([len(batch) for batch in batches], out=offsets[1:])
        flat = np.asarray([item for batch in batches for item in batch], dtype=np.int64).reshape(-1, 2)
        self._plan = (flat, offsets, tuple(shapes), records)
        self.window_info = {'epoch': self.epoch, 'window': self.window, 'shard_id': shard_id,
                            'unique_samples': len(entries), 'real_exposures': len(flat),
                            'global_real_batches': len(batches),
                            'control_steps': (len(batches) + self.world_size - 1) // self.world_size,
                            'train_frequency_band_exposures_after': after}
        return self.window_info

    def _require_plan(self):
        if self._plan is None:
            raise RuntimeError('Call prepare_window() before accessing the current plan')

    def __len__(self):
        self._require_plan()
        return self.window_info['control_steps'] - self.cursor

    def batch_at(self, control_step):
        self._require_plan()
        if not 0 <= control_step < self.window_info['control_steps']:
            raise IndexError(control_step)
        batch = control_step * self.world_size + self.rank
        flat, offsets, shapes, _ = self._plan
        if batch >= len(shapes):
            return PlannedBatch(self.epoch, self.window, control_step, None, (), None)
        start, stop = offsets[batch:batch + 2]
        return PlannedBatch(self.epoch, self.window, control_step, batch,
                            tuple((int(a), int(b)) for a, b in flat[start:stop]), shapes[batch])

    def __iter__(self):
        self._require_plan()
        # Range is a snapshot, intentionally independent from advance/prefetch.
        for position in range(self.cursor, self.window_info['control_steps']):
            yield self.batch_at(position)

    def records_for(self, batch):
        self._require_plan()
        if (batch.epoch, batch.window) != (self.epoch, self.window):
            raise ValueError('Batch belongs to another window')
        return [self._plan[3][ordinal] for ordinal, _ in batch.occurrences]

    def advance(self, count=1):
        if type(count) is not int or count < 0 or count > len(self):
            raise ValueError('Invalid acknowledged control-step count')
        self.cursor += count

    def next_window(self):
        self._require_plan()
        if len(self):
            raise ValueError('Cannot leave an unacknowledged window')
        self.window, self.cursor, self._plan = self.window + 1, 0, None

    def _signature(self):
        return {'asset_identity': self.assets.identity, 'batch_size': self.batch_size,
                'window_samples': self.window_samples, 'world_size': self.world_size, 'rank': self.rank,
                'seed': self.seed, 'rng': 'PCG64/SHA256-v1',
                'preprocessing': {'bucket': vars(self.dataset.preprocessor.selector.config),
                                  'alpha_background': list(self.dataset.preprocessor.alpha_background)}}

    def state_dict(self):
        return {'version': SAMPLER_VERSION, 'signature': self._signature(),
                'epoch': self.epoch, 'window': self.window, 'cursor': self.cursor}

    def prepare_restore(self, state):
        """Rebuild on a temporary sampler; rejection never advances this consumer.

        Reconstructing a valid window may refill an evictable shard cache or its
        dimension corrections, but neither is committed optimizer progress.
        """
        if state.get('version') != SAMPLER_VERSION or state.get('signature') != self._signature():
            raise ValueError('Sampler identity/configuration mismatch; use an explicit new run')
        if any(type(state.get(name)) is not int or state[name] < 0 for name in ('epoch', 'window', 'cursor')):
            raise ValueError('Invalid saved sampler position')
        prospective = copy(self)
        prospective.set_epoch(state['epoch'])
        if state['window'] > len(prospective._schedule):
            raise ValueError('Invalid saved window')
        prospective.window = state['window']
        if prospective.epoch_done:
            if state['cursor'] != 0:
                raise ValueError('Invalid completed-epoch cursor')
        else:
            prospective.prepare_window()  # Bounded deterministic actual-shape replay.
            prospective.advance(state['cursor'])
        return prospective

    def load_state_dict(self, state):
        prospective = self.prepare_restore(state)
        self.__dict__.update(prospective.__dict__)


def _collate_numpy(samples):
    batch = collate_prepared(samples)
    batch['images'] = batch['images'].numpy()
    return batch


def window_loader(dataset, sampler, *, num_workers=0, prefetch_factor=2,
                  worker_transport='shared_memory', worker_timeout=60., multiprocessing_context=None):
    """Yield (PlannedBatch, real collated batch or None), without acknowledgement.

    One loader lives for one prepared window. The consumer must acknowledge only
    completed optimizer windows; DataLoader lookahead never changes sampler state.
    """
    if worker_transport not in ('shared_memory', 'numpy') or not math.isfinite(worker_timeout) or worker_timeout <= 0:
        raise ValueError('Invalid worker transport or finite timeout')
    sampler._require_plan()
    start = sampler.cursor
    batches = (list(batch.occurrences) for batch in sampler if not batch.control_only)
    options = {'prefetch_factor': prefetch_factor, 'timeout': worker_timeout,
               'multiprocessing_context': multiprocessing_context} if num_workers else {}
    # PyTorch consumes a base seed even with zero workers. Isolate it from the
    # model RNG so recreating a prefetched iterator cannot change the next update.
    generator = torch.Generator().manual_seed(int(_sha([sampler.seed, sampler.epoch, sampler.window, 'loader'])[:16], 16))
    loader = iter(DataLoader(dataset, batch_sampler=batches, num_workers=num_workers,
                             generator=generator, collate_fn=_collate_numpy if worker_transport == 'numpy' else collate_prepared, **options))
    for step in range(start, sampler.window_info['control_steps']):
        plan = sampler.batch_at(step)
        batch = None if plan.control_only else next(loader)
        if batch is not None and worker_transport == 'numpy':
            batch['images'] = torch.from_numpy(batch['images'])
        yield plan, batch
