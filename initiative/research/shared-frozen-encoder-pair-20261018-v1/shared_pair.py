"""Bounded transient encoder sharing; original reader/AdamW loop stays unchanged."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.checkpoint import atomic_json, capture_rng, restore_rng, run_lock
from rsna.competitive.spatial_train import epoch_order


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def tensors_and_arrays(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (torch.Tensor, np.ndarray)):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from tensors_and_arrays(v, seen)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from tensors_and_arrays(v, seen)
    elif is_dataclass(value):
        for f in fields(value):
            yield from tensors_and_arrays(getattr(value, f.name), seen)


class SharedBatch:
    """Exactly one effective batch; no disk cache or cross-update stale features."""

    def __init__(self, original, device, *, max_bytes=4 * 2**30):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("Positive explicit transient tensor budget required")
        self.original, self.binding, self.device = original, original.binding, device
        self.max_bytes = max_bytes
        self.rows, self.versions, self.expected, self.reads = {}, {}, {}, {}
        self.source_fetches, self.reused_fetches, self.peak_bytes = 0, 0, 0

    def __len__(self):
        return len(self.original)

    def begin(self, indices, consumers):
        if self.rows or self.expected or consumers not in (1, 2):
            raise ValueError("Finish the previous batch before a distinct1/2consumer batch")
        if not indices or len(indices) > 8 or len(set(indices)) != len(indices):
            raise ValueError("One original effective batch with at most8distinct studies required")
        self.expected = {i: consumers for i in indices}
        self.reads = {i: 0 for i in indices}

    def __getitem__(self, index):
        if index not in self.expected or self.reads[index] >= self.expected[index]:
            raise ValueError("Unexpected or repeated study access outside the original batch")
        if index not in self.rows:
            before = capture_rng(self.device)
            row = self.original[index]
            after = capture_rng(self.device)
            if not exact(before, after):
                restore_rng(before, self.device)
                raise ValueError("Encoder provider consumes randomness: sharing is not admitted")
            values = list(tensors_and_arrays(row))
            if any(v.requires_grad for v in values if isinstance(v, torch.Tensor)):
                raise ValueError("Only detached immutable frozen-encoder inputs may be shared")
            added = sum(
                v.numel() * v.element_size() if isinstance(v, torch.Tensor) else v.nbytes
                for v in values
            )
            used = sum(x[2] for x in self.versions.values())
            if used + added > self.max_bytes:
                raise MemoryError(
                    "Current effective batch exceeds transient encoder-sharing budget"
                )
            ts = [v for v in values if isinstance(v, torch.Tensor)]
            self.rows[index] = row
            self.versions[index] = (ts, [t._version for t in ts], added)
            self.source_fetches += 1
            self.peak_bytes = max(self.peak_bytes, used + added)
        else:
            self.reused_fetches += 1
        ts, versions, _ = self.versions[index]
        if [t._version for t in ts] != versions:
            raise ValueError("A reader mutated shared frozen inputs")
        self.reads[index] += 1
        return self.rows[index]

    def finish(self):
        if not self.expected or self.reads != self.expected:
            raise ValueError("Both original arm accesses must complete before clearing the batch")
        for ts, versions, _ in self.versions.values():
            if [t._version for t in ts] != versions:
                raise ValueError("A reader mutated shared frozen inputs")
        self.rows.clear()
        self.versions.clear()
        self.expected.clear()
        self.reads.clear()


def fit_pair(engines, cache, output: Path, *, resume=False, stop_after=None, between_arms=None):
    """One unchanged engine update per arm; interrupted pairs roll forward once.

    Each arm owns its complete checkpoint/RNG/AdamW. A crash between arms leaves
    one arm at most one step ahead; resume computes only the lagging arm. The
    callback is an engineering crash-injection hook, never part of production.
    """
    if set(engines) != {"aligned", "null"} or any(e.arm != arm for arm, e in engines.items()):
        raise ValueError("Exactly the original aligned/null arms required")
    a, b = engines["aligned"], engines["null"]
    if (
        a.data is not cache
        or b.data is not cache
        or a.recipe != b.recipe
        or a.total != b.total
        or a.device != b.device
    ):
        raise ValueError("Same frozen inputs, recipe, step count and device required")
    if stop_after is not None and (type(stop_after) is not int or not 0 < stop_after <= a.total):
        raise ValueError("Explicit positive complete pair boundary required")
    with run_lock(output):
        if resume:
            for arm, e in engines.items():
                e.load(output / arm / "checkpoint.pt")
        else:
            if any((output / arm).exists() for arm in engines):
                raise FileExistsError("Existing arm output requires explicit pair resume")
            initial = capture_rng(cache.device)
            for arm, e in engines.items():
                restore_rng(initial, cache.device)
                atomic_json(output / arm / "run.json", {"binding": e.binding})
                e.save(output / arm / "checkpoint.pt")
        if abs(a.completed - b.completed) > 1:
            raise ValueError("Interrupted pair may differ by at most one original update")
        target = a.total if stop_after is None else stop_after
        if target <= min(a.completed, b.completed):
            raise ValueError("Requested pair stop must advance the original trajectory")
        while min(a.completed, b.completed) < target:
            boundary = min(a.completed, b.completed) + 1
            pending = [e for e in engines.values() if e.completed < boundary]
            first = pending[0]
            order = epoch_order(len(cache), first.recipe.seed, first.epoch)
            indices = order[
                first.cursor * first.recipe.effective_batch : (first.cursor + 1)
                * first.recipe.effective_batch
            ]
            cache.begin(indices, len(pending))
            for i, e in enumerate(pending):
                e.fit(output / e.arm, resume=True, stop_after=boundary)
                if between_arms is not None and i == 0 and len(pending) == 2:
                    between_arms(boundary)
            cache.finish()
            if a.completed != b.completed:
                raise ValueError("No pair result before both complete optimizer boundaries")
            atomic_json(
                output / "pair-progress.json",
                {
                    "completed_per_arm": boundary,
                    "total_per_arm": a.total,
                    "source_fetches": cache.source_fetches,
                    "reused_fetches": cache.reused_fetches,
                    "peak_transient_tensor_bytes": cache.peak_bytes,
                    "encoder_trained": False,
                },
            )
        return {
            "completed_per_arm": a.completed,
            "total_per_arm": a.total,
            "source_fetches": cache.source_fetches,
            "reused_fetches": cache.reused_fetches,
            "peak_transient_tensor_bytes": cache.peak_bytes,
        }
