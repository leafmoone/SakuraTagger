"""Same-resolution batches over caller-provided post-EXIF image sizes."""
from collections import defaultdict
import random
from torch.utils.data import Sampler

from .buckets import BucketSelector


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, sizes, batch_size: int, *, selector=None, shuffle=False, seed=0, drop_last=False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.selector = selector or BucketSelector()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.groups = defaultdict(list)
        for index, size in enumerate(sizes):
            choice = self.selector.select(*size)
            self.groups[choice.target_size].append(index)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for size in sorted(self.groups):
            indices = list(self.groups[size])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self):
        size = self.batch_size
        return sum(len(items) // size if self.drop_last else (len(items) + size - 1) // size
                   for items in self.groups.values())
