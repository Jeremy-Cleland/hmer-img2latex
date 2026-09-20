"""Length-based bucket batch sampler to cut padding waste."""

from typing import Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[List[int]]):
    """Group similar-length sequences, then shuffle the resulting batches."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [
            order[i : i + batch_size]
            for i in range(0, len(order), batch_size)
        ]
        if drop_last and batches and len(batches[-1]) < batch_size:
            batches = batches[:-1]
        self._base_batches = batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(self._base_batches), generator=generator).tolist()
        for idx in perm:
            yield self._base_batches[idx]

    def __len__(self) -> int:
        return len(self._base_batches)
