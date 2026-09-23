"""Odd completed epochs survive later rolling-checkpoint replacement."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/newton/archive_odd_epochs.py"
SPEC = importlib.util.spec_from_file_location("rsna_odd_epoch_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def fixture(tmp_path: Path, *, next_update: int = 0) -> Path:
    root = tmp_path / "run"
    (root / "epoch-checkpoints").mkdir(parents=True)
    binding = {
        "config": {
            "fold": 0,
            "seed": 20260922,
            "submission_every_epochs": 2,
            "diagnostic_study_limit": None,
        },
        "release": {"binding_sha256": "a" * 64},
    }
    (root / "run.json").write_text(json.dumps({"binding": binding, "training_studies": 2247}))
    torch.save(
        {
            "schema_version": 1,
            "optimizer_boundary": True,
            "epoch": 3,
            "next_update": next_update,
            "step": 843,
            "scheduled_epoch": False,
            "binding": binding,
        },
        root / "checkpoint.pt",
    )
    return root


def test_capture_preserves_completed_checkpoint_inode(tmp_path):
    root = fixture(tmp_path)
    record = archive.capture(root, 3, "123")
    target = root / "epoch-checkpoints/epoch-003.pt"
    assert record["status"] == "verified_unscheduled_epoch_archive"
    assert record["checkpoint_sha256"] == archive.sha(target)
    assert target.stat().st_ino == (root / "checkpoint.pt").stat().st_ino
    replacement = root / "next.pt"
    replacement.write_bytes(b"future recovery checkpoint")
    replacement.replace(root / "checkpoint.pt")
    assert archive.capture(root, 3, "123") == record
    assert archive.sha(target) == record["checkpoint_sha256"]


def test_partial_epoch_is_never_archived(tmp_path):
    root = fixture(tmp_path, next_update=1)
    with pytest.raises(ValueError, match="completed full-data epoch"):
        archive.capture(root, 3, "123")
    assert not (root / "epoch-checkpoints/epoch-003.pt").exists()
