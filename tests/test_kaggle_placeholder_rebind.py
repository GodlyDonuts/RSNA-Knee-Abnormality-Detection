"""An unversioned placeholder cannot become a duplicate code submission."""

import hashlib
import importlib.util
from pathlib import Path

import pytest
from test_kaggle_submission import workflow

path = Path(__file__).parents[1] / (
    "initiative/audits/kaggle-checkpoint-comparison-v1/rebind_unversioned_kernel.py"
)
spec = importlib.util.spec_from_file_location("placeholder_rebind", path)
rebind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebind)

rebind4 = rebind
spec2 = importlib.util.spec_from_file_location(
    "placeholder_rebind2", path.with_name("rebind_epoch2_kernel.py")
)
rebind2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(rebind2)


@pytest.fixture(autouse=True, params=[rebind4, rebind2])
def control_source(monkeypatch, request):
    monkeypatch.setitem(globals(), "rebind", request.param)
    monkeypatch.setattr(rebind, "CONTROL_SOURCE_SHA", hashlib.sha256(b"control").hexdigest())


class Remote:
    username = "godlydonuts"

    def __init__(self, mode="placeholder"):
        self.mode = mode
        self.methods = []

    def call(self, service, method, body):
        self.methods.append(method)
        assert method in {"GetKernel", "ListKernels"}
        if method == "GetKernel":
            if body["kernelSlug"] == rebind.CONTROL_SLUG.split("/")[1]:
                return {
                    "metadata": {"currentVersionNumber": 1, "isPrivate": True},
                    "blob": {"source": "bad" if self.mode == "bad_control" else "control"},
                }
            if self.mode == "saved_latest":
                return {"metadata": {"currentVersionNumber": 1}}
            raise workflow.RemoteError(
                "Kaggle GetKernel HTTP " + ("404" if body.get("versionLabel") == "1" else "500")
            )
        if body["page"] == 2:
            return {}  # Proto JSON omits its empty repeated field.
        if body["search"] == rebind.CONTROL_SLUG:
            return {"kernels": [] if self.mode == "no_control" else [{"ref": rebind.CONTROL_SLUG}]}
        row = {"ref": None, "title": "[Private Notebook]", "lastRunTime": "2010-04-01T00:00:00Z"}
        if self.mode == "named":
            row["ref"] = rebind.OLD_SLUG
        return {"kernels": [row, row] if self.mode == "ambiguous" else [row]}

    def require_absent_kernel(self, slug):
        assert slug == rebind.NEW_SLUG
        if self.mode == "new_exists":
            raise ValueError("Distinct target already exists")
        return {"method": "GetKernel404"}


def seed(db):
    old = dict(
        checkpoint_sha256=rebind.CHECKPOINT,
        kernel_slug=rebind.OLD_SLUG,
        dataset_slug="godlydonuts/exact-private-dataset",
        dataset_version=1,
        manifest_sha256="m",
        dataset_metadata_sha256="d",
        kernel_source_sha256="s",
        shared_wheelhouse={"id": 12139697},
        epoch=4,
    )
    entry = dict(
        checkpoint_sha256=rebind.CHECKPOINT,
        state="push_uncertain",
        submitted=False,
        operation_sha256=rebind.OLD_OPERATION,
        operation=old,
        dataset={"id": 123, "is_private": True, "version": 1},
        epoch=4,
    )
    workflow.save_entry(db, entry, "fixture")
    return entry, dict(old, kernel_slug=rebind.NEW_SLUG)


def test_placeholder_history_preserves_uncertainty_and_same_dataset(tmp_path):
    remote = Remote()
    with workflow.ledger(tmp_path / "ledger.sqlite3", remote.username) as db:
        original, new = seed(db)
        result = rebind.preserve_placeholder(workflow, remote, db, new)
        assert result["state"] == "queued" and "operation" not in result
        old = result["superseded_operations"][0]
        assert old["operation"] == original["operation"] and old["dataset"] == original["dataset"]
        assert old["state"] == "unreadable_uncertain_placeholder_never_retry_or_submit"
        assert old["old_absence_claimed"] is False
        assert workflow.read_entry(db, rebind.CHECKPOINT) == result
        assert set(remote.methods) == {"GetKernel", "ListKernels"}


@pytest.mark.parametrize(
    "mode", ["saved_latest", "bad_control", "no_control", "named", "ambiguous", "new_exists"]
)
def test_saved_or_ambiguous_identity_prevents_migration(tmp_path, mode):
    remote = Remote(mode)
    with workflow.ledger(tmp_path / "ledger.sqlite3", remote.username) as db:
        entry, new = seed(db)
        with pytest.raises(ValueError):
            rebind.preserve_placeholder(workflow, remote, db, new)
        db.rollback()
        assert workflow.read_entry(db, rebind.CHECKPOINT) == entry


@pytest.mark.parametrize("change", ["submitted", "source", "manifest", "operation"])
def test_submission_or_changed_payload_cannot_be_rebound(tmp_path, change):
    remote = Remote()
    with workflow.ledger(tmp_path / "ledger.sqlite3", remote.username) as db:
        entry, new = seed(db)
        if change == "submitted":
            entry.update(state="submitted", submission_id=1)
        elif change == "operation":
            entry["operation_sha256"] = "0" * 64
        else:
            new["kernel_source_sha256" if change == "source" else "manifest_sha256"] = "changed"
        workflow.save_entry(db, entry, "fixture changed")
        with pytest.raises(ValueError):
            rebind.preserve_placeholder(workflow, remote, db, new)
        db.rollback()
        assert workflow.read_entry(db, rebind.CHECKPOINT) == entry
        assert not remote.methods
