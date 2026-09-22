import hashlib
import json
from pathlib import Path

import pytest

from rsna.competitive.infer import validate_inference_checkpoint
from rsna.competitive.snapshots import scheduled_epoch, write_epoch_snapshot
from rsna.schema import CACHE_VERSION, TARGETS
from rsna.semantic_supervision import digest


def binding(diagnostic=False):
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
        "release": release,
        "config": {
            "epochs": 10,
            "submission_every_epochs": 2,
            "fold": 0,
            "seed": 20260922,
            "diagnostic_study_limit": 16 if diagnostic else None,
        },
        "source_revision": "revision",
        "cache_version": CACHE_VERSION,
        "targets": list(TARGETS),
    }


def checkpoint(epoch=2, update=0, diagnostic=False):
    return {
        "schema_version": 1,
        "optimizer_boundary": True,
        "binding": binding(diagnostic),
        "epoch": epoch,
        "next_update": update,
        "scheduled_epoch": True,
    }


def test_intermediate_epoch_requires_explicit_cadence_authorization():
    with pytest.raises(ValueError, match="fixed final"):
        validate_inference_checkpoint(checkpoint())
    assert validate_inference_checkpoint(checkpoint(), allow_scheduled_epoch=True) == binding()


@pytest.mark.parametrize("epoch,update", [(1, 0), (0, 0), (2, 1), (11, 0), (12, 0), (True, 0)])
def test_wrong_epoch_or_partial_update_is_never_submission_eligible(epoch, update):
    with pytest.raises(ValueError, match="fixed final|progress counters"):
        validate_inference_checkpoint(checkpoint(epoch, update), allow_scheduled_epoch=True)


def test_diagnostic_checkpoint_stays_ineligible_for_competition():
    with pytest.raises(ValueError, match="Diagnostic"):
        validate_inference_checkpoint(checkpoint(diagnostic=True), allow_scheduled_epoch=True)


def test_final_checkpoint_still_works_without_intermediate_option():
    assert validate_inference_checkpoint(checkpoint(10)) == binding()


def test_fabricated_scheduled_flag_cannot_override_disabled_cadence():
    state = checkpoint()
    state["binding"]["config"]["submission_every_epochs"] = 0
    with pytest.raises(ValueError, match="fixed final"):
        validate_inference_checkpoint(state, allow_scheduled_epoch=True)


def test_snapshot_survives_atomic_recovery_checkpoint_replacement(tmp_path: Path):
    recovery = tmp_path / "checkpoint.pt"
    recovery.write_bytes(b"epoch two weights")
    directory = tmp_path / "epoch-checkpoints"
    record = write_epoch_snapshot(recovery, directory, binding(), 2, 20)
    assert record["submission_eligible"] and not record["diagnostic_only"]
    assert record["checkpoint_sha256"] == hashlib.sha256(recovery.read_bytes()).hexdigest()
    assert record == write_epoch_snapshot(recovery, directory, binding(), 2, 20)
    successor = tmp_path / "next.partial"
    successor.write_bytes(b"later weights")
    successor.replace(recovery)
    assert (directory / "epoch-002.pt").read_bytes() == b"epoch two weights"
    assert json.loads((directory / "epoch-002.json").read_text()) == record
    with pytest.raises(ValueError, match="replace"):
        write_epoch_snapshot(recovery, directory, binding(), 2, 20)


def test_diagnostic_snapshot_is_explicitly_marked_unsuitable(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"tiny diagnostic")
    record = write_epoch_snapshot(path, tmp_path / "snapshots", binding(True), 2, 1)
    assert record["diagnostic_only"] and not record["submission_eligible"]


def test_unscheduled_epoch_is_not_written(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"not due")
    with pytest.raises(ValueError, match="cadence"):
        write_epoch_snapshot(path, tmp_path / "snapshots", binding(), 3, 1)
    assert not (tmp_path / "snapshots").exists()
    assert not scheduled_epoch(2, 0, {"epochs": 10, "submission_every_epochs": True})
