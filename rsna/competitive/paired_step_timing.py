"""Observe actual optimizer-loop cost without editing either numerical engine.

Rank zero clocks LR setup through scaler/AdamW/update history, including sampled
inputs, forward/backward, clipping and collectives. Progress-file and epoch-save
IO are outside that interval and remain visible in total trainer-call wall time.
No timing or clinical claim follows from importing this wrapper.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rsna.competitive import licensed_importance_train, paired_logit_train

_LOCK = threading.Lock()


def timed_fit(engine, model, data, output, binding, recipe, *, device):
    """Run one fresh bounded fit, retaining its actual new state on caller storage.

    The caller owns matched initialization/RNG, arm order, exact input admission,
    node-local temporary storage and repetition. This wrapper does not reset a
    model, resume, change an optimizer or admit a production run. Each invocation
    really performs up to six new updates. Only rank zero emits step timings.
    """
    output = Path(output)
    updates = math.ceil(len(data) / recipe.effective_batch) * recipe.epochs
    if (
        engine not in (licensed_importance_train, paired_logit_train)
        or recipe.role != "influence"
        or recipe.epochs != 3
        or not 1 <= len(data) <= 16
        or updates != 6
        or output.exists()
    ):
        raise ValueError("Fresh original six-update engineering trainer call required")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Only one instrumented trainer per process")
    rank = dist.get_rank() if dist.is_initialized() else 0
    old_factor, old_json = engine.lr_factor, engine.atomic_json
    pending = None
    rows: list[dict[str, float | int]] = []

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def factor(*args, **kwargs):
        nonlocal pending
        if rank == 0:
            if pending is not None:
                raise RuntimeError("Prior optimizer step has no completed progress boundary")
            sync()
            pending = time.perf_counter()
        return old_factor(*args, **kwargs)

    def atomic_json(path, value):
        nonlocal pending
        if rank == 0 and Path(path) == output / "progress.json":
            sync()
            end = time.perf_counter()
            if pending is None or not math.isfinite(end - pending) or end <= pending:
                raise RuntimeError("Missing valid complete optimizer-step interval")
            if value["completed"] != len(rows) + 1:
                raise RuntimeError("Exactly contiguous new optimizer-step progress required")
            rows.append(
                {
                    "update": value["completed"],
                    "seconds": end - pending,
                    "studies": value["last"]["studies"],
                }
            )
            pending = None
        return old_json(path, value)

    try:
        engine.lr_factor, engine.atomic_json = factor, atomic_json
        sync()
        if dist.is_initialized():
            dist.barrier()
        start = time.perf_counter()
        result = engine.fit(model, data, output, binding, recipe, device=device)
        sync()
        if dist.is_initialized():
            dist.barrier()
        elapsed = time.perf_counter() - start
        if (
            result["completed"] != updates
            or result["total"] != updates
            or pending is not None
            or (rank == 0 and len(rows) != updates)
        ):
            raise ValueError("Actual complete bounded optimizer loop and timing trace required")
        return {
            "rank": rank,
            "actual_optimizer_updates": updates,
            "step_timings": rows,
            "trainer_call_wall_seconds_including_setup_and_checkpoint_IO": elapsed,
            "step_scope": "LR_factor_entry_through_history_after_AdamW_scaler_update_and_loss_allreduce",
            "step_scope_excludes": [
                "per_step_order_construction",
                "progress_file_write",
                "epoch_snapshot_and_resume_checkpoint_IO",
                "one_time_source_and_input_admission",
            ],
            "engine_source_sha256": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
            "numerical_engine_body_changed": False,
            "new_GPU_or_accuracy_acceptance": False,
        }
    finally:
        engine.lr_factor, engine.atomic_json = old_factor, old_json
        _LOCK.release()
