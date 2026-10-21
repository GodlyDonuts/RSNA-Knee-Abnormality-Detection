"""Read original completed optimization logs; no models, labels or new inference."""

import hashlib
import json
import math
import os
from pathlib import Path

ORIGINALS = {
    "B": ("989036", "d54d69d6e07dfc5314cf41a2830b1fbbf18557e6b81b646c1e379340447ddf9e"),
    "C": ("989035", "f207261814605bba41391a161d637f8ed2b03abdc94d77c193dcddd39739af29"),
    "C0": ("989037", "8559a899075baafd32ffa22624e4120d8ce655010e428463e63a6835599c1ebd"),
    "C100": ("995351", "f0d73e6cc04cbf7b34f06499f5aef4ba60ea2c593662764b8f64ec5105a289cd"),
}


def observe():
    output = {}
    for role, (job, binding) in ORIGINALS.items():
        suffix = "v1" if role == "C100" else "v2"
        root = Path(f"/tmp/rsna-predictive-full-{role}-20261020-{suffix}")
        terminal = json.loads((root / "terminal.json").read_text())
        assert terminal["GPU_job"] == job and terminal["binding_sha256"] == binding
        assert terminal["status"] == "actual_fixed_final_full3887_predictive_arm_complete"
        path = root / "updates.jsonl"
        assert 0 < path.stat().st_size < 4 * 2**20
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == 2916
        for step, row in enumerate(rows, 1):
            assert row["step"] == step and row["epoch"] == (step - 1) // 486 + 1
            assert row["studies"] == (7 if step % 486 == 0 else 8)
            assert all(math.isfinite(row[k]) for k in ("exam_loss", "gradient_norm",
                "encoder_gradient_norm", "head_gradient_norm"))
        epochs = []
        for epoch in range(1, 7):
            group = rows[(epoch - 1) * 486:epoch * 486]
            count = sum(r["studies"] for r in group)
            assert count == 3887
            mean = lambda k: sum(r[k] * r["studies"] for r in group) / count
            epochs.append(dict(epoch=epoch, study_exposures=count,
                weighted_training_exam_BCE=mean("exam_loss"),
                weighted_encoder_gradient_norm=mean("encoder_gradient_norm"),
                weighted_head_gradient_norm=mean("head_gradient_norm"),
                clipped_update_fraction=sum(r["gradient_norm"] > 1 for r in group) / len(group)))
        output[role] = dict(original_GPU_job=job, source_git=terminal["source_git"],
            binding_sha256=binding, fitting_order_sha256=terminal["fitting_order_sha256"],
            original_log_bytes=len(raw), original_log_sha256=hashlib.sha256(raw).hexdigest(),
            recipe=terminal["recipe"], epochs=epochs)
    assert len({r["fitting_order_sha256"] for r in output.values()}) == 1
    return dict(status="original_completed_four_arm_training_log_comparison",
        CPU_job=os.getenv("SLURM_JOB_ID"), originals=output,
        metrics_are_pre_update_training_trajectory_not_fixed_final_evaluation=True,
        new_model_forwards=0, new_GPU_jobs=0, labels_or_images_read=False,
        heldout_or_clinical_accuracy_claim=False)


if __name__ == "__main__":
    print(json.dumps(observe(), allow_nan=False))
