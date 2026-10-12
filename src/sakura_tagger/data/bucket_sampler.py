"""Legacy manifest sampler with compact plans and acknowledged consumption.

No implicit rank padding/drop. Formal sharded runs should use ShardWindowSampler;
DDP must use explicit control-only tail steps, never rely on DDP.join here.
"""
from array import array
from bisect import bisect_right
from collections.abc import Sequence
import hashlib
import math
import random

import numpy as np
from torch.utils.data import Sampler

from .buckets import BucketSelector


class _BatchView(Sequence):
    def __init__(self, sampler, rank=0, world_size=1):
        self.sampler, self.rank, self.world_size = sampler, rank, world_size

    def __len__(self):
        count = len(self.sampler._order)
        return max(0, (count - self.rank + self.world_size - 1) // self.world_size)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        batch = self.sampler._order[self.rank + index * self.world_size]
        start, stop = self.sampler._offsets[batch:batch + 2]
        return tuple((int(oid), int(sample)) for oid, sample in self.sampler._flat[start:stop])


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, sizes, batch_size, *, selector=None, shuffle=False, seed=0,
                 drop_last=False, rank=0, world_size=1, repeat_weights=None,
                 repeat_mode='stochastic_round', epoch_size=None, max_repeat_factor=3.0):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        if type(world_size) is not int or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Require 0 <= rank < world_size')
        if repeat_mode not in ('stochastic_round', 'replacement'):
            raise ValueError('Unknown repeat mode')
        if not math.isfinite(max_repeat_factor) or max_repeat_factor < 1:
            raise ValueError('max_repeat_factor must be finite and >= 1')
        self.selector = selector or BucketSelector()
        self.batch_size, self.shuffle, self.seed, self.drop_last = batch_size, shuffle, seed, drop_last
        self.rank, self.world_size = rank, world_size
        self.repeat_mode, self.max_repeat_factor = repeat_mode, max_repeat_factor
        self.epoch, self.cursor = 0, 0
        self.sizes = np.array(list(sizes) if not isinstance(sizes, np.ndarray) else sizes, dtype=np.int64, copy=True).reshape(-1, 2)
        self.weights = np.ones(len(self.sizes)) if repeat_weights is None else np.array(list(repeat_weights) if not isinstance(repeat_weights, np.ndarray) else repeat_weights, dtype=np.float64, copy=True)
        if self.weights.shape != (len(self.sizes),) or not np.isfinite(self.weights).all() or ((self.weights < 1) | (self.weights > max_repeat_factor)).any():
            raise ValueError('One finite repeat weight in [1,max_repeat_factor] is required per image')
        self.epoch_size = len(self.sizes) if epoch_size is None else epoch_size
        if type(self.epoch_size) is not int or self.epoch_size < 0 or self.epoch_size and not len(self.sizes):
            raise ValueError('Invalid epoch_size')
        if repeat_mode != 'replacement' and epoch_size is not None:
            raise ValueError('epoch_size is only used by replacement sampling')
        self.bucket_sizes = np.asarray([self.selector.select(int(w), int(h)).target_size for w, h in self.sizes], dtype=np.int32).reshape(-1, 2)
        self.sizes.flags.writeable = self.weights.flags.writeable = self.bucket_sizes.flags.writeable = False
        digest = hashlib.sha256(self.sizes.tobytes())
        digest.update(self.weights.tobytes())
        self._asset_sha256 = digest.hexdigest()
        self._flat = self._offsets = self._order = None

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch, self.cursor = epoch, 0
        self._flat = self._offsets = self._order = None

    def _prepare(self):
        if self._flat is not None:
            return
        # Retain the legacy random.Random sequence for old checkpoint replay.
        rng = random.Random(self.seed + self.epoch)
        if self.repeat_mode == 'replacement':
            cumulative = np.cumsum(self.weights)
            total = float(cumulative[-1]) if len(cumulative) else 0
            samples = np.fromiter((bisect_right(cumulative, rng.random() * total, 0, len(cumulative) - 1)
                                   for _ in range(self.epoch_size)), dtype=np.int64, count=self.epoch_size)
        else:
            repeats = np.fromiter((math.floor(weight) + (rng.random() < weight % 1) for weight in self.weights),
                                  dtype=np.int64, count=len(self.weights))
            samples = np.repeat(np.arange(len(self.sizes), dtype=np.int64), repeats)
        groups, offsets = [], array('q', [0])
        unique_sizes, bucket_codes = np.unique(self.bucket_sizes, axis=0, return_inverse=True)
        codes = bucket_codes[samples] if len(samples) else np.empty(0, dtype=np.int64)
        order = np.argsort(codes, kind='stable')
        boundaries = np.searchsorted(codes[order], np.arange(len(unique_sizes) + 1))
        for bucket in range(len(unique_sizes)):
            occurrences = order[boundaries[bucket]:boundaries[bucket + 1]].copy()
            if self.shuffle:
                rng.shuffle(occurrences)
            if self.drop_last:
                occurrences = occurrences[:len(occurrences) // self.batch_size * self.batch_size]
            if not len(occurrences):
                continue
            groups.append(np.column_stack((occurrences, samples[occurrences])))
            count = len(occurrences)
            for start in range(0, count, self.batch_size):
                offsets.append(offsets[-1] + min(self.batch_size, count - start))
        self._flat = np.concatenate(groups) if groups else np.empty((0, 2), dtype=np.int64)
        self._offsets = np.frombuffer(offsets, dtype=np.int64)
        self._order = np.arange(len(offsets) - 1, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(self._order)

    def _global_batches(self):
        self._prepare()
        return _BatchView(self)

    def occurrence_batches(self):
        self._prepare()
        return _BatchView(self, self.rank, self.world_size)

    def __iter__(self):
        batches = self.occurrence_batches()
        for position in range(self.cursor, len(batches)):
            yield [sample for _, sample in batches[position]]

    def __len__(self):
        self._prepare()
        return max(0, (len(self._order) - self.rank + self.world_size - 1) // self.world_size) - self.cursor

    def advance(self, count=1):
        if type(count) is not int or count < 0 or count > len(self):
            raise ValueError('Invalid consumed-batch cursor')
        self.cursor += count

    def _signature(self):
        return {'asset_sha256': self._asset_sha256, 'sample_count': len(self.sizes), 'batch_size': self.batch_size,
                'shuffle': self.shuffle, 'seed': self.seed, 'drop_last': self.drop_last,
                'rank': self.rank, 'world_size': self.world_size, 'repeat_mode': self.repeat_mode,
                'epoch_size': self.epoch_size, 'max_repeat_factor': self.max_repeat_factor,
                'bucket_config': vars(self.selector.config)}

    def state_dict(self):
        return {'version': 2, 'epoch': self.epoch, 'cursor': self.cursor, 'signature': self._signature()}

    def validate_state_dict(self, state):
        """Check current/legacy identities without mutating sampler or model state."""
        signature = state.get('signature', {})
        if state.get('version') == 1:
            legacy = self._signature()
            for key in ('asset_sha256', 'sample_count', 'max_repeat_factor'):
                legacy.pop(key)
            valid = (all(signature.get(key) == value for key, value in legacy.items())
                     and np.array_equal(np.asarray(signature.get('sizes')), self.sizes)
                     and np.array_equal(np.asarray(signature.get('weights')), self.weights))
        else:
            valid = state.get('version') == 2 and signature == self._signature()
        if not valid:
            raise ValueError('Sampler resume configuration mismatch')
        if any(type(state.get(key)) is not int or state[key] < 0 for key in ('epoch', 'cursor')):
            raise ValueError('Invalid saved sampler epoch/cursor')

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.set_epoch(state['epoch'])
        self.advance(state['cursor'])
