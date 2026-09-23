"""The requested baseline is an exact completed replay, never a partial update."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rsna.competitive.infer import validate_inference_checkpoint
from rsna.schema import CACHE_VERSION, TARGETS
from rsna.semantic_supervision import digest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/kaggle/baseline_infer.py"
SPEC = importlib.util.spec_from_file_location("rsna_epoch1_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


def state() -> dict:
    release = {
        "schema_version": 1,
        "manifest_sha256": "a" * 64,
        "series_sha256": "b" * 64,
        "source_revision": "revision",
        "release_receipt_sha256": "c" * 64,
        "semantic_registry_sha256": "d" * 64,
    }
    release["binding_sha256"] = digest(release)
    return {
        "schema_version": 1,
        "optimizer_boundary": True,
        "epoch": 1,
        "next_update": 0,
        "step": 281,
        "scheduled_epoch": False,
        "binding": {
            "release": release,
            "config": {
                "epochs": 10,
                "submission_every_epochs": 2,
                "diagnostic_study_limit": None,
                "fold": 0,
                "seed": 20260922,
                "encoder": {"architecture": "dinov3_l16"},
            },
            "world_size": 2,
            "targets": list(TARGETS),
            "cache_version": CACHE_VERSION,
            "source_revision": "revision",
        },
    }


def test_epoch1_requires_explicit_adapter():
    checkpoint = state()
    with pytest.raises(ValueError, match="fixed final"):
        validate_inference_checkpoint(checkpoint, allow_scheduled_epoch=True)
    assert baseline.validate_baseline(checkpoint) == checkpoint["binding"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("epoch", 0),
        ("next_update", 1),
        ("step", 280),
        ("scheduled_epoch", True),
        ("optimizer_boundary", False),
    ],
)
def test_baseline_rejects_wrong_progress(field, value):
    checkpoint = state()
    checkpoint[field] = value
    with pytest.raises(ValueError):
        baseline.validate_baseline(checkpoint)


def test_baseline_rejects_diagnostic_or_changed_cadence():
    checkpoint = state()
    checkpoint["binding"]["config"]["diagnostic_study_limit"] = 100
    with pytest.raises(ValueError):
        baseline.validate_baseline(checkpoint)
    checkpoint = state()
    checkpoint["binding"]["config"]["submission_every_epochs"] = 1
    with pytest.raises(ValueError):
        baseline.validate_baseline(checkpoint)
