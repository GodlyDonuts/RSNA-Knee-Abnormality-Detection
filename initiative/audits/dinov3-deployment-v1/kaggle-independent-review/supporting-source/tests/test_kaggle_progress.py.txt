import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "projection", Path(__file__).resolve().parents[1] / "scripts/kaggle/progress.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def record(**extra):
    return dict(
        epoch=2,
        optimizer_steps=20,
        binding_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        fold=0,
        seed=42,
        state="submitted",
        submission_id=9,
        kernel_version=1,
        **extra,
    )


def test_pending_score_is_unknown_not_zero():
    row = m.project([record(remote_status="PENDING", public_score="0.9")], [])[0]
    assert row["public_score"] is None and row["training_loss_last_update"] is None


def test_bound_exact_completed_epoch_only():
    h = {
        "binding_sha256": "a" * 64,
        "updates": [
            {"epoch": 1, "optimizer_steps": 19, "loss": 0.8},
            {"epoch": 1, "optimizer_steps": 20, "loss": 0.7},
            {"epoch": 2, "optimizer_steps": 21, "loss": 0.6},
        ],
    }
    unrelated = {
        "binding_sha256": "c" * 64,
        "updates": [{"epoch": 1, "optimizer_steps": 20, "loss": 0.1}],
    }
    row = m.project(
        [record(remote_status="COMPLETE", public_score="0.0")],
        [(h, "d" * 64), (unrelated, "e" * 64)],
    )[0]
    assert row["public_score"] == 0 and row["training_loss_last_update"] == 0.7
    assert row["training_loss_history_sha256"] == ["d" * 64]


def test_conflicting_history_fails():
    a = {"binding_sha256": "a" * 64, "updates": [{"epoch": 1, "optimizer_steps": 20, "loss": 0.7}]}
    b = {"binding_sha256": "a" * 64, "updates": [{"epoch": 1, "optimizer_steps": 20, "loss": 0.8}]}
    with pytest.raises(ValueError, match="Conflicting"):
        m.project([record()], [(a, "c" * 64), (b, "d" * 64)])
