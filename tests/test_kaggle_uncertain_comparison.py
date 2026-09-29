"""Distinct replacement preserves uncertainty and cannot retry a submitted version."""

import importlib.util
import json
from pathlib import Path

import pytest
from test_kaggle_submission import workflow

path = (
    Path(__file__).parents[1]
    / "initiative/audits/kaggle-checkpoint-comparison-v1/reconcile_ledger.py"
)
spec = importlib.util.spec_from_file_location("comparison_reconcile", path)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class Remote:
    username = "godlydonuts"

    def __init__(self, fail=False):
        self.checked = []
        self.fail = fail

    def require_absent_kernel(self, slug):
        self.checked.append(slug)
        if self.fail:
            raise ValueError("Old notebook exists; source reconciliation required")
        return {"method": "complete_authenticated_profile_search_after403", "count": 54}


def seed(db):
    entry = {
        "checkpoint_sha256": recovery.LATEST_SHA,
        "state": "push_uncertain",
        "operation_sha256": recovery.OLD_OPERATION_SHA,
        "submitted": False,
        "epoch": 10,
        "operation": {
            "kernel_slug": "godlydonuts/rsna-knee-f2e010-pixelfix-5e400e82c4b5",
            "checkpoint_sha256": recovery.LATEST_SHA,
        },
        "dataset": {"id": 12173460, "is_private": True, "version": 1},
    }
    workflow.save_entry(db, entry, "fixture")
    return entry


def test_absent_uncertain_push_is_preserved_without_remote_mutation(tmp_path):
    remote = Remote()
    with workflow.ledger(tmp_path / "ledger.sqlite3", remote.username) as db:
        original = seed(db)
        result = recovery.archive_absent_push(workflow, remote, db)
        assert result["state"] == "queued"
        assert "operation" not in result and "dataset" not in result
        old = result["superseded_operations"][0]
        assert old["operation"] == original["operation"] and old["dataset"] == original["dataset"]
        assert old["state"] == "absent_uncertain_push_preserved_never_retry"
        assert remote.checked == [original["operation"]["kernel_slug"], recovery.NEW_KERNEL]
        persisted = json.loads(db.execute("select record from entries").fetchone()[0])
        assert persisted == result


@pytest.mark.parametrize(
    "change",
    [
        {"state": "submitted", "submission_id": 123},
        {"state": "pushed", "kernel_version": 1},
        {"operation_sha256": "0" * 64},
        {"attempted_at": "2026-09-29T04:00:00+00:00"},
    ],
)
def test_changed_or_submitted_entry_cannot_be_requeued(tmp_path, change):
    remote = Remote()
    with workflow.ledger(tmp_path / "ledger.sqlite3", remote.username) as db:
        entry = seed(db)
        entry.update(change)
        workflow.save_entry(db, entry, "modified fixture")
        with pytest.raises(ValueError, match="exact unsubmitted uncertain push changed"):
            recovery.archive_absent_push(workflow, remote, db)
        db.rollback()
        assert workflow.read_entry(db, recovery.LATEST_SHA) == entry
        assert remote.checked == []


def test_unreadable_or_existing_notebook_never_requeues_uncertain_state(tmp_path):
    remote = Remote(fail=True)
    with workflow.ledger(tmp_path / "ledger.sqlite3", remote.username) as db:
        entry = seed(db)
        with pytest.raises(ValueError, match="Old notebook exists"):
            recovery.archive_absent_push(workflow, remote, db)
        db.rollback()
        assert workflow.read_entry(db, recovery.LATEST_SHA) == entry
