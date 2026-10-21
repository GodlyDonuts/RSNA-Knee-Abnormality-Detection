"""Measure one versus two intra-op threads in the two real preparation lanes.

No learned model, labels, GPU allocation or submission. Floating image rounding
is measured explicitly; all remaining context values must match exactly.
"""

import concurrent.futures as cf
import csv
import gc
import hashlib
import json
import os
import pickle
import threading
import time
from pathlib import Path

LIMIT = 780
CAP = 24 * 2**30
IMAGE_ATOL = 1e-6


def compare(reference, candidate):
    import numpy as np

    image_max = 0.0
    exact = True

    def visit(a, b, path=()):
        nonlocal image_max, exact
        assert type(a) is type(b), "Context type changed"
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape, "Context shape/dtype changed"
            same = a.tobytes() == b.tobytes()
            exact &= same
            if path[-1:] == ("images",):
                assert a.dtype == np.float32 and np.isfinite(a).all() and np.isfinite(b).all()
                error = float(np.max(np.abs(a - b))) if a.size else 0.0
                image_max = max(image_max, error)
                assert error <= IMAGE_ATOL, "Image difference exceeds declared rounding bound"
            else:
                assert same, "Non-image context changed"
        elif isinstance(a, dict):
            assert set(a) == set(b), "Context fields changed"
            for key in sorted(a):
                visit(a[key], b[key], (*path, key))
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, (*path, index))
        else:
            assert a == b, "Context scalar changed"

    visit(reference, candidate)
    return {
        "bitwise_equal": bool(exact),
        "max_image_absolute_difference": image_max,
        "all_other_values_exact": True,
    }


def trial(threads, tasks, reference, remaining):
    import torch
    from context_reference import prepare

    # Previous pool has fully joined before changing the process-wide setting.
    torch.set_num_threads(threads)
    started = time.monotonic()
    rows, observed = [], {}
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare, ("selected", *task)) for task in tasks]
        for future in cf.as_completed(futures, timeout=remaining()):
            row = future.result()
            value = pickle.loads(row.pop("payload"))  # Our own just-created in-memory bytes.
            index = row["index"]
            assert index not in observed
            original = reference[index] if reference is not None else value
            equality = compare(original, value)
            rows.append({**row, **equality})
            if reference is None:
                observed[index] = value
            else:
                observed[index] = None
            del value
    assert set(observed) == {0, 1, 2}
    result = {
        "threads_per_lane": threads,
        "lanes": 2,
        "wall_seconds": time.monotonic() - started,
        "studies": sorted(rows, key=lambda r: r["index"]),
        "all_other_values_exact": all(r["all_other_values_exact"] for r in rows),
        "bitwise_equal": all(r["bitwise_equal"] for r in rows),
        "max_image_absolute_difference": max(r["max_image_absolute_difference"] for r in rows),
    }
    return result, observed if reference is None else reference


def main(binding):
    import _predictive_serving_mounts as mounts
    import psutil
    import pydicom
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.device_count() == 0
    assert str(torch.__version__) == binding["torch"]
    assert str(pydicom.__version__) == binding["pydicom"]
    affinity = sorted(os.sched_getaffinity(0))
    assert len(affinity) >= 4, "Four actual CPU cores required for this engineering comparison"
    os.sched_setaffinity(0, set(affinity[:4]))
    roster = mounts.resolve_competition_roster("/kaggle/input")
    with roster.open() as f:
        ids = [row["StudyInstanceUID"] for row in csv.DictReader(f)]
    assert len(ids) == len(set(ids)) == 3
    assert (
        hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        == binding["public_roster_sha256"]
    )
    tasks = [(i, str(roster.parent), uid) for i, uid in enumerate(ids)]
    start, done = time.monotonic(), threading.Event()
    usage = {"peak_RSS_bytes": 0}

    def remaining():
        value = LIMIT - (time.monotonic() - start)
        if value <= 0:
            raise TimeoutError("Bounded preparation comparison expired")
        return value

    def monitor():
        process = psutil.Process()
        while not done.wait(0.2):
            usage["peak_RSS_bytes"] = max(usage["peak_RSS_bytes"], process.memory_info().rss)
            if usage["peak_RSS_bytes"] > CAP or time.monotonic() - start > LIMIT:
                print(json.dumps({"status": "CPU_limit_exceeded", **usage}), flush=True)
                os._exit(124)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    records, reference = [], None
    try:
        for threads in (1, 2, 2, 1, 1, 2):
            row, reference = trial(threads, tasks, reference, remaining)
            records.append(row)
            print(json.dumps({"phase": "thread_trial_complete", **row}), flush=True)
            gc.collect()
        result = {
            "status": "actual_thread_budget_CPU_preparation_complete",
            "binding": binding,
            "studies": 3,
            "trials": records,
            "elapsed_seconds": time.monotonic() - start,
            **usage,
            "CPU_affinity": sorted(os.sched_getaffinity(0)),
            "GPU_count": 0,
            "model_or_label_reads": False,
            "competition_entry": False,
            "hidden_runtime_proven": False,
            "production_change_accepted": False,
            "warm_trial_indices": [2, 3, 4, 5],
            "limitations": "Three public MRI studies, CPU-only four-core affinity, sequential cache effects; no encoder/GPU overlap. Only image rounding up to1e-6 is admitted for measurement, not for production. All other context values exact. Actual trained prediction comparison required before promotion.",
        }
        Path("/kaggle/working/predictive-thread-budget-CPU.json").write_text(
            json.dumps(result, sort_keys=True) + "\n"
        )
    finally:
        done.set()
        watcher.join(timeout=2)
