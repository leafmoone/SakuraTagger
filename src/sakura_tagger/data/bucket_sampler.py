"""Deterministic homogeneous batches, globally constructed then rank-sharded.

No implicit rank padding or dropping: rank lengths can differ. Distributed
training must use DDP join (and compatible accumulation) or an explicit caller
policy. RFS duplicate occurrences are intentional, never duplicated by sharding.
The checkpoint cursor must be acknowledged by the consumer with advance(); this
avoids recording DataLoader-prefetched batches as completed training work.
"""
from collections import defaultdict
import math
import random

from torch.utils.data import Sampler

from .buckets import BucketSelector


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, sizes, batch_size, *, selector=None, shuffle=False, seed=0,
                 drop_last=False, rank=0, world_size=1, repeat_weights=None,
                 repeat_mode='stochastic_round', epoch_size=None, max_repeat_factor=3.0):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be positive')
        if not isinstance(world_size, int) or world_size < 1 or not 0 <= rank < world_size:
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
        self.sizes = tuple(tuple(size) for size in sizes)
        self.weights = tuple(1.0 for _ in self.sizes) if repeat_weights is None else tuple(float(w) for w in repeat_weights)
        if len(self.weights) != len(self.sizes) or any(not math.isfinite(w) or not 1 <= w <= max_repeat_factor for w in self.weights):
            raise ValueError('One finite repeat weight in [1,max_repeat_factor] is required per image')
        self.epoch_size = len(self.sizes) if epoch_size is None else epoch_size
        if not isinstance(self.epoch_size, int) or self.epoch_size < 0 or (self.epoch_size and not self.sizes):
            raise ValueError('Invalid epoch_size')
        if repeat_mode != 'replacement' and epoch_size is not None:
            raise ValueError('epoch_size is only used by replacement sampling')
        self.bucket_sizes = tuple(self.selector.select(*size).target_size for size in self.sizes)
        self._batches = None

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch, self.cursor, self._batches = epoch, 0, None

    def _global_batches(self):
        if self._batches is not None:
            return self._batches
        rng = random.Random(self.seed + self.epoch)
        if self.repeat_mode == 'replacement':
            # The weight cap is NOT a cap on realized per-image draw counts.
            occurrences = rng.choices(range(len(self.sizes)), weights=self.weights, k=self.epoch_size) if self.epoch_size else []
        else:
            occurrences = []
            for index, weight in enumerate(self.weights):
                count = math.floor(weight) + (rng.random() < weight % 1)
                occurrences.extend([index] * count)
        groups = defaultdict(list)
        for occurrence_id, index in enumerate(occurrences):
            groups[self.bucket_sizes[index]].append((occurrence_id, index))
        batches = []
        for size in sorted(groups):
            indices = groups[size]
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = tuple(indices[start:start + self.batch_size])
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        self._batches = tuple(batches)
        return self._batches

    def occurrence_batches(self):
        """Rank-local (occurrence_id,image_index) batches, including consumed ones."""
        return self._global_batches()[self.rank::self.world_size]

    def __iter__(self):
        for batch in self.occurrence_batches()[self.cursor:]:
            yield [index for _, index in batch]

    def __len__(self):
        return len(self.occurrence_batches()) - self.cursor

    def advance(self, count=1):
        if not isinstance(count, int) or count < 0 or self.cursor + count > len(self.occurrence_batches()):
            raise ValueError('Invalid consumed-batch cursor')
        self.cursor += count

    def _signature(self):
        # No per-image file reads/hashing. This is an in-memory sampler contract.
        return {'sizes': self.sizes, 'weights': self.weights, 'batch_size': self.batch_size,
                'shuffle': self.shuffle, 'seed': self.seed, 'drop_last': self.drop_last,
                'rank': self.rank, 'world_size': self.world_size, 'repeat_mode': self.repeat_mode,
                'epoch_size': self.epoch_size, 'bucket_config': vars(self.selector.config)}

    def state_dict(self):
        return {'version': 1, 'epoch': self.epoch, 'cursor': self.cursor, 'signature': self._signature()}

    def load_state_dict(self, state):
        if state.get('version') != 1 or state.get('signature') != self._signature():
            raise ValueError('Sampler resume configuration mismatch')
        self.set_epoch(state['epoch'])
        self.advance(state['cursor'])
