"""Fold-aware private epoch preservation keeps training provenance exact."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/hf/publish_epoch.py"
SPEC = importlib.util.spec_from_file_location("publish_epoch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
publish_epoch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish_epoch)


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.mark.parametrize(
    ("fold", "training_studies", "optimizer_steps", "world_size"),
    [(1, 2278, 285, 2), (2, 2281, 286, 1)],
)
def test_later_fold_first_epoch_is_a_full_archived_epoch_without_baseline_replay(
    tmp_path, monkeypatch, fold, training_studies, optimizer_steps, world_size
):
    code_root = tmp_path / "code"
    source = code_root / "rsna" / "model.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"PINNED_TRAINED_SOURCE\n")
    monkeypatch.setattr(
        publish_epoch.subprocess,
        "check_output",
        lambda *_args, **_kwargs: (publish_epoch.TRAINING_GIT + "\n").encode(),
    )
    license_file = tmp_path / "LICENSE.md"
    license_file.write_bytes(b"fixture official-license bytes")
    monkeypatch.setattr(publish_epoch, "LICENSE_SHA", publish_epoch.sha(license_file))
    run = tmp_path / "run"
    epochs = run / "epoch-checkpoints"
    epochs.mkdir(parents=True)
    checkpoint = epochs / "epoch-001.pt"
    checkpoint.write_bytes(b"completed epoch-1 weights")
    binding = {
        "config": {
            "submission_every_epochs": 2,
            "diagnostic_study_limit": None,
            "seed": 20260922,
            "fold": fold,
            "encoder": {
                "architecture": "dinov3_l16",
                "source_id": "official",
                "source_revision": "pinned",
                "weights_sha256": "a" * 64,
            },
        },
        "world_size": world_size,
        "code_sha256": {"model.py": publish_epoch.sha(source)},
        "release": {"binding_sha256": "b" * 64},
    }
    binding_sha = digest(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode())
    (run / "run.json").write_text(
        json.dumps({"binding": binding, "training_studies": training_studies})
    )
    sidecar = {
        "epoch": 1,
        "status": "verified_unscheduled_epoch_archive",
        "checkpoint_file": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": publish_epoch.sha(checkpoint),
        "optimizer_steps": optimizer_steps,
        "fold": fold,
        "seed": 20260922,
        "diagnostic_only": False,
        "binding_sha256": binding_sha,
        "release_binding_sha256": "b" * 64,
    }
    snapshot = epochs / "epoch-001.json"
    snapshot.write_text(json.dumps(sidecar))
    manifest, files = publish_epoch.source_files(snapshot, run, license_file, code_root, fold)
    assert manifest["fold"] == fold
    assert manifest["optimizer_steps"] == optimizer_steps
    assert manifest["snapshot_status"] == "verified_unscheduled_epoch_archive"
    assert set(files) == {"checkpoint.pt", "snapshot.json", "DINOv3-LICENSE.md"}
    with pytest.raises(ValueError, match="Epoch archive producer/status differs"):
        publish_epoch.source_files(snapshot, run, license_file, code_root, 0)
