"""Deterministic effective batches without giving distributed padding any loss."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class SeededDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, key: tuple[int, bool, int]) -> dict:
        index, valid, seed = key
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                # torch.manual_seed also resets CUDA RNG, which belongs to the
                # training model when data loading runs in the main process.
                torch.random.default_generator.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                batch = dict(self.dataset[index])
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        batch["label_mask"] = batch["label_mask"] * int(valid)
        batch["sample_valid"] = torch.tensor(valid, dtype=torch.bool)
        return batch


class EffectiveBatchSampler(Sampler):
    """Each real study occurs once per epoch across ranks; tail padding is masked.

    Keys carry deterministic per-study augmentation seeds. Resume skips whole
    effective batches directly instead of replaying stochastic worker queues.
    """

    def __init__(
        self,
        size: int,
        batch_size: int,
        accumulation_steps: int,
        *,
        seed: int,
        epoch: int,
        rank: int = 0,
        world_size: int = 1,
        start_update: int = 0,
    ):
        if min(size, batch_size, accumulation_steps, world_size) < 1:
            raise ValueError("Batch sizes, dataset size and world size must be positive")
        if not 0 <= rank < world_size or epoch < 0:
            raise ValueError("Invalid rank or epoch")
        self.size, self.batch_size, self.accumulation_steps = size, batch_size, accumulation_steps
        self.seed, self.epoch, self.rank, self.world_size = seed, epoch, rank, world_size
        self.effective_size = batch_size * accumulation_steps * world_size
        self.updates = math.ceil(size / self.effective_size)
        if not 0 <= start_update <= self.updates:
            raise ValueError("Resume update is outside this epoch")
        self.start_update = start_update

    def __len__(self) -> int:
        return (self.updates - self.start_update) * self.accumulation_steps

    def __iter__(self) -> Iterator[list[tuple[int, bool, int]]]:
        order = torch.randperm(
            self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)
        ).tolist()
        for update in range(self.start_update, self.updates):
            for micro in range(self.accumulation_steps):
                keys = []
                for item in range(self.batch_size):
                    position = (
                        update * self.effective_size
                        + (micro * self.world_size + self.rank) * self.batch_size
                        + item
                    )
                    valid = position < self.size
                    index = order[position] if valid else order[0]
                    seed_bytes = f"{self.seed}:{self.epoch}:{index}".encode()
                    sample_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big") % (
                        2**63 - 1
                    )
                    keys.append((index, valid, sample_seed))
                yield keys
