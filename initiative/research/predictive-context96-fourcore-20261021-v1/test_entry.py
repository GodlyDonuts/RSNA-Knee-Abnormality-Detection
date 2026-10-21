import importlib.util
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


E = load("dense_entry_test", HERE / "entry.py")
W = load("dense_entry_original_workflow", HERE.parents[2] / "scripts/kaggle/workflow.py")
NOW = datetime(2026, 10, 22, 1, tzinfo=timezone.utc)


class Remote:
    def __init__(self):
        self.calls = []
        self.failed_transport = False
        self.original = dict(
            ref=57491555,
            status="COMPLETE",
            publicScore="0.950",
            totalBytes=1000,
            errorDescription=None,
        )
        self.limit = 5

    def submissions(self):
        return [self.original]

    def account(self):
        return dict(
            entered=True,
            checked_at=NOW.isoformat(),
            num_today=5 - self.limit,
            num_allowed_now=self.limit,
        )

    def competition(self, method, payload):
        self.calls.append((method, payload))
        if self.failed_transport:
            raise TimeoutError("uncertain test transport")
        return {"ref": 12345}


@pytest.fixture
def context(monkeypatch):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE entries(sha TEXT PRIMARY KEY,record TEXT NOT NULL)")
    db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    original = dict(checkpoint_sha256=E.MODEL, state="submitted", submission_id=57491555)
    db.execute("INSERT INTO entries VALUES (?,?)", (E.MODEL, json.dumps(original)))
    db.commit()
    proof = json.loads((HERE / "actual-complete.json").read_text())
    assert W.object_hash(proof) == E.PROOF
    monkeypatch.setattr(E, "inspect_original", lambda *_: proof)
    return db, Remote(), original


def test_one_entry_keeps_original_checkpoint_record_and_exact_version(context, tmp_path):
    db, remote, original = context
    answer = E.execute(W, remote, db, "submit", "test", root=tmp_path, now=NOW)
    assert answer["state"] == "submitted" and answer["submission_id"] == 12345
    assert answer["checkpoint_sha256"] == E.MODEL and answer["entry_identity"] != E.MODEL
    assert (
        json.loads(db.execute("SELECT record FROM entries WHERE sha=?", (E.MODEL,)).fetchone()[0])
        == original
    )
    assert remote.calls == [
        (
            "CreateCodeSubmission",
            dict(
                fileName="submission.csv",
                kernelOwner="godlydonuts",
                kernelSlug=E.SLUG.split("/")[1],
                kernelVersion=1,
                submissionDescription=E.DESCRIPTION,
            ),
        )
    ]
    second = E.execute(W, remote, db, "submit", "test", root=tmp_path, now=NOW)
    assert second["status"] == "existing_dense_entry_do_not_repeat" and len(remote.calls) == 1
    assert W.allowance(db, remote.account(), now=NOW) == 4


@pytest.mark.parametrize("kind", ["pending", "failed", "unscored", "slot", "early", "late"])
def test_waits_for_actual_scored_original_window_and_spare_slot(kind, context, tmp_path):
    db, remote, _ = context
    now = NOW
    if kind == "pending":
        remote.original.update(status=None, publicScore=None)
    if kind == "failed":
        remote.original.update(errorDescription="original hidden failure", publicScore=None)
    if kind == "unscored":
        remote.original.update(publicScore=None)
    if kind == "slot":
        remote.limit = 1
    if kind == "early":
        now = E.START.replace(day=21)
    if kind == "late":
        now = E.LAST.replace(hour=13)
    answer = E.execute(W, remote, db, "submit", "test", root=tmp_path, now=now)
    assert answer["new_Kaggle_mutations"] == 0 and not remote.calls
    assert not (tmp_path / "submit-intent.json").exists()


def test_uncertain_request_never_repeated_and_intent_survives(context, tmp_path):
    db, remote, _ = context
    remote.failed_transport = True
    with pytest.raises(TimeoutError):
        E.execute(W, remote, db, "submit", "test", root=tmp_path, now=NOW)
    assert (tmp_path / "submit-intent.json").exists()
    row = json.loads(
        db.execute("SELECT record FROM entries WHERE sha=?", (E.ENTRY_KEY,)).fetchone()[0]
    )
    assert row["state"] == "submit_uncertain"
    assert (
        E.execute(W, remote, db, "submit", "test", root=tmp_path, now=NOW)["status"]
        == "existing_dense_entry_do_not_repeat"
    )
    assert len(remote.calls) == 1


def test_check_never_submits_even_when_ready(context, tmp_path):
    db, remote, _ = context
    answer = E.execute(W, remote, db, "check", "test", root=tmp_path, now=NOW)
    assert answer["status"] == "ready_for_one_exact_code_entry" and not remote.calls
    assert not (tmp_path / "submit-intent.json").exists()
