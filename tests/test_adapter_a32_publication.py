"""A32 preservation checks on synthetic completed snapshots; no network calls."""

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


publisher = load("adapter_a32_publisher_test", "publish_adapter_a32_epoch.py")
dispatcher = load("adapter_a32_dispatcher_test", "dispatch_adapter_a32_epoch.py")
all_publisher = load("adapter_a32_all_publisher_test", "publish_adapter_a32_all_epoch.py")
all_dispatcher = load("adapter_a32_all_dispatcher_test", "dispatch_adapter_a32_all_epoch.py")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_adapter_a32_dispatcher_and_publisher_bind_the_same_schedule() -> None:
    spec = publisher.FOLD_SPECS[0]
    assert spec == {
        "prefix": dispatcher.PREFIX,
        "optimizer_updates_per_epoch": 281,
        "training_studies": 2247,
        "epoch_one_baseline": False,
        "world_size": 1,
    }
    assert dispatcher.PUBLISHER_SHA == publisher.sha(
        ROOT / "scripts/hf/publish_adapter_a32_epoch.py"
    )
    assert dispatcher.RUN.endswith("adapter-a32-fold0-6f1ee6d-production-v1")


def test_all_development_dispatcher_binds_an_independent_pool_and_prefix() -> None:
    spec = all_publisher.FOLD_SPECS[0]
    assert spec["prefix"] == all_dispatcher.PREFIX != dispatcher.PREFIX
    assert spec["optimizer_updates_per_epoch"] == 426
    assert spec["training_studies"] == 3403
    assert all_dispatcher.PUBLISHER_SHA == all_publisher.sha(
        ROOT / "scripts/hf/publish_adapter_a32_all_epoch.py"
    )


@pytest.mark.parametrize("publisher", [publisher, all_publisher])
def test_adapter_a32_source_accepts_only_exact_completed_snapshot(
    tmp_path, monkeypatch, publisher
) -> None:
    specification = publisher.FOLD_SPECS[0]
    all_development = publisher is all_publisher
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
            **({"fit_scope": "all_development"} if all_development else {}),
            "submission_every_epochs": 1,
            "pooling": "attention",
            "diagnostic_study_limit": None,
            "seed": 20260922,
            "fold": 0,
            "freeze_encoder": True,
            "precision": "bf16",
            "lr_schedule": "cosine",
            "epochs": 10,
            "encoder_lr": 0.0001,
            "head_lr": 0.0003,
            "encoder": {
                "adapter": {
                    "method": "rs_lora_v1",
                    "rank": 32,
                    "trainable_parameters": 10_223_616,
                    "base_frozen": True,
                    "target_modules": [str(i) for i in range(120)],
                },
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
    (run / "run.json").write_text(
        json.dumps(
            {
                "binding": binding,
                "training_studies": specification["training_studies"],
                "outer_studies": 0 if all_development else 1156,
                "encoder_trainable_parameters": 10_223_616,
            }
        )
    )
    sidecar = snapshots / "epoch-001.json"
    record = {
        "epoch": 1,
        "status": "scheduled_epoch_snapshot",
        "checkpoint_file": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": publisher.sha(checkpoint),
        "optimizer_steps": specification["optimizer_updates_per_epoch"],
        "fold": 0,
        "seed": 20260922,
        "diagnostic_only": False,
        "binding_sha256": binding_sha,
        "release_binding_sha256": "b" * 64,
    }
    sidecar.write_text(json.dumps(record))
    manifest, files = publisher.source_files(sidecar, run, license_file, code_root, 0)
    assert manifest["fold"] == 0
    assert manifest["optimizer_steps"] == specification["optimizer_updates_per_epoch"]
    assert manifest["competition_score_claimed"] is False
    assert files["checkpoint.pt"] == checkpoint

    for change in ({"optimizer_steps": 280}, {"fold": 1}, {"status": "partial"}):
        sidecar.write_text(json.dumps(record | change))
        with pytest.raises(ValueError):
            publisher.source_files(sidecar, run, license_file, code_root, 0)

    sidecar.write_text(json.dumps(record))
    run_record = json.loads((run / "run.json").read_text())
    if all_development:
        (run / "run.json").write_text(json.dumps(run_record | {"outer_studies": 1156}))
        with pytest.raises(ValueError, match="run binding"):
            publisher.source_files(sidecar, run, license_file, code_root, 0)
    (run / "run.json").write_text(json.dumps(run_record | {"encoder_trainable_parameters": 0}))
    with pytest.raises(ValueError, match="actual trainable count"):
        publisher.source_files(sidecar, run, license_file, code_root, 0)
