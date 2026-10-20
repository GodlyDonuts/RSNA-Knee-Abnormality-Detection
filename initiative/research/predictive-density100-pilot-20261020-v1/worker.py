"""Exact, explicit operational/one-coefficient derivation of the admitted worker."""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCIENCE = "39e315c65546aa5205518713bc27349ec63e7538"
ORIGINAL = Path("/tmp/rsna-predictive-stable-warmup-20261020-v1")
ROOT = Path("/tmp/rsna-predictive-density100-pilot-20261020-v1")
PORTABLE = Path("/tmp/rsna-predictive-portable-recovery-20261020-v1")
WORKER_SHA = "78ba933ae78a7d6c3b7199e9f92d0f799c9d77dced13a341968e36e632eb641e"
ENGINEERING_SHA = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"
MEASUREMENT_SHA = "6f10fc5a21e8149731e1600088e601b97c16be42b0a9e944baa05c67f6920e2a"
PLACEMENT_SHA = "c78b01c954db21fadcdeb2d1c0ed4cbd90a736e7ef16fe18c84fad2d10b1551f"
GIB, MIB = 2**30, 2**20
CHECKPOINT_CAP, CONTROL_CAP = 6 * GIB, 4 * MIB
OTHER27 = 205676085248 + 96 * GIB + 12 * GIB + 64 * MIB
SHARED_BYTES, SHARED_FILES = 123 * MIB, 376
FINISH_BY = datetime(2026, 10, 21, 17, 15, tzinfo=timezone.utc)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def once(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    with pending.open("x") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(pending, path)
    finally:
        pending.unlink()


def admit_time(now):
    assert now.tzinfo is not None and now + timedelta(minutes=30) <= FINISH_BY, (
        "Too late for this separate pilot; preserve the primary launch deadline"
    )


def node_capacity(free, paid_checkpoints=0, paid_controls=0):
    assert 0 <= paid_checkpoints <= CHECKPOINT_CAP and 0 <= paid_controls <= CONTROL_CAP
    required = OTHER27 + 20 * GIB + CHECKPOINT_CAP - paid_checkpoints + CONTROL_CAP - paid_controls
    assert free > required, "Keep every existing C0/other scope and the 20GiB reserve"
    return {"free_bytes": free, "required_bytes": required, "remaining_bytes": free - required}


def quota_capacity(text):
    row = next(line.split() for line in text.splitlines() if line.strip().startswith("/home/sa305415"))
    free_bytes = (int(row[2]) - int(row[1])) * 1024
    free_files = int(row[6]) - int(row[5])
    assert free_bytes > 20 * GIB + SHARED_BYTES and free_files >= 5000 + SHARED_FILES
    return {"free_bytes": free_bytes, "free_files": free_files,
            "retained_future_bytes": SHARED_BYTES, "retained_future_files": SHARED_FILES,
            "new_shared_files": 0, "new_shared_bytes": 0}


def build(original):
    assert sha(original.encode()) == WORKER_SHA
    substitutions = (
        (f'D = Path("{ORIGINAL}")', f'D = Path("{ROOT}")'),
        ('assert arm in ("B", "C", "C0") and phase in ("full", "stop1", "resume")',
         'assert arm == "C" and phase in ("full", "stop1", "resume")'),
        ('density_weight=0 if arm == "C0" else 0.1,', 'density_weight=100.0,'),
        ('< 12 * 2**30', '< 6 * 2**30'),
    )
    output = original
    for old, new in substitutions:
        assert output.count(old) == 1
        output = output.replace(old, new)
    compile(output, "original_C_worker_density100", "exec")
    return output


def scientific_warnings(history, terminal, original_c):
    import math

    old = original_c["real_optimizer_history"]
    old_tail = sum(r["exam_loss"] for r in old[-6:]) / 6
    tail = sum(r["exam_loss"] for r in history[-6:]) / 6
    old_max = max(r["exam_loss"] for r in old)
    maximum = max(r["exam_loss"] for r in history)
    old_bf = original_c["terminal"]["training_only_numerical_ranges"][
        "maximum_absolute_log_bayes_factor"
    ]
    bf = terminal["training_only_numerical_ranges"]["maximum_absolute_log_bayes_factor"]
    return {
        "last_six_exam_mean": tail, "original_C_last_six_exam_mean": old_tail,
        "maximum_exam_loss": maximum, "original_C_maximum_exam_loss": old_max,
        "maximum_absolute_BF": bf, "original_C_maximum_absolute_BF": old_bf,
        "last_six_exam_mean_over_twice_original": tail > 2 * old_tail,
        "maximum_exam_over_twice_original": maximum > 2 * old_max,
        "maximum_BF_over_ten_times_original": bf > 10 * old_bf,
        "weighted_fixed_Gaussian_constant": 100 * 0.5 * math.log(2 * math.pi),
        "density_minus_constant_by_update": [
            r["density_loss"] - 0.5 * math.log(2 * math.pi) for r in history
        ],
        "clipped_update_count": sum(r["gradient_norm"] > 1 for r in history),
        "warning_is_not_automatic_clinical_rejection_or_promotion": True,
    }
