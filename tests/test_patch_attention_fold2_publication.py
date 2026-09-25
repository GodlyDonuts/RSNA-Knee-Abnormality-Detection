"""Fold-2 preservation checks on synthetic completed snapshots; no network calls."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/hf" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = load("patch_attention_publisher_fold2_test", "publish_patch_attention_epoch.py")
dispatcher = load(
    "patch_attention_dispatcher_fold2_test", "dispatch_patch_attention_fold2_epoch.py"
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_fold2_dispatcher_and_publisher_bind_the_same_schedule() -> None:
    spec = publisher.FOLD_SPECS[2]
    assert spec == {
        "prefix": dispatcher.PREFIX,
        "optimizer_updates_per_epoch": 286,
        "training_studies": 2281,
        "epoch_one_baseline": False,
        "world_size": 1,
    }
    assert dispatcher.PUBLISHER_SHA == publisher.sha(
        ROOT / "scripts/hf/publish_patch_attention_epoch.py"
    )
    assert dispatcher.RUN.endswith("patchattention-fold2-203f419-production-v1")


def test_fold2_source_accepts_only_exact_completed_snapshot(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    snapshots = run / "epoch-checkpoints"
    snapshots.mkdir(parents=True)
    checkpoint = snapshots / "epoch-001.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    license_file = tmp_path / "LICENSE.md"
    license_file.write_bytes(b"synthetic license")
    monkeypatch.setattr(publisher, "LICENSE_SHA", publisher.sha(license_file))
    code_root = tmp_path / "code"
    (code_root / "rsna").mkdir(parents=True)
    (code_root / "rsna" / "sample.py").write_text("x = 1\n")
    monkeypatch.setattr(
        publisher.subprocess,
        "check_output",
        lambda command: (publisher.TRAINING_GIT + "\n").encode(),
    )
    binding = {
        "config": {
            "submission_every_epochs": 1,
            "pooling": "patch_attention",
            "diagnostic_study_limit": None,
            "seed": 20260922,
            "fold": 2,
            "encoder": {
                "architecture": "dinov3_l16",
                "source_id": "official",
                "source_revision": "pinned",
                "weights_sha256": "a" * 64,
            },
        },
        "world_size": 1,
        "code_sha256": {"sample.py": publisher.sha(code_root / "rsna" / "sample.py")},
        "release": {"binding_sha256": "b" * 64},
    }
    binding_sha = digest(
        json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    (run / "run.json").write_text(json.dumps({"binding": binding, "training_studies": 2281}))
    sidecar = snapshots / "epoch-001.json"
    record = {
        "epoch": 1,
        "status": "scheduled_epoch_snapshot",
        "checkpoint_file": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": publisher.sha(checkpoint),
        "optimizer_steps": 286,
        "fold": 2,
        "seed": 20260922,
        "diagnostic_only": False,
        "binding_sha256": binding_sha,
        "release_binding_sha256": "b" * 64,
    }
    sidecar.write_text(json.dumps(record))
    manifest, files = publisher.source_files(sidecar, run, license_file, code_root, 2)
    assert manifest["fold"] == 2
    assert manifest["optimizer_steps"] == 286
    assert manifest["competition_score_claimed"] is False
    assert files["checkpoint.pt"] == checkpoint

    for change in ({"optimizer_steps": 285}, {"fold": 1}, {"status": "partial"}):
        sidecar.write_text(json.dumps(record | change))
        with pytest.raises(ValueError):
            publisher.source_files(sidecar, run, license_file, code_root, 2)
