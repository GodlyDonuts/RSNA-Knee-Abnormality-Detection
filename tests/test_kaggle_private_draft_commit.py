"""Synthetic draft-aware SaveKernel reconciliation; never authenticates or mutates Kaggle."""

import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.kaggle import coordinator as c
from scripts.kaggle import daily_queue as dq

SPEC = importlib.util.spec_from_file_location(
    "kaggle_workflow_draft", Path(__file__).resolve().parents[1] / "scripts/kaggle/workflow.py"
)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)

SHA = "a" * 64
SLUG = "owner/rsna-alldev-offline-e002-aaaaaaaaaaaa"
SOURCE = "source"
PLACEHOLDER = {"kernels": [dict(c.PLACEHOLDER)]}
DATASET = {"id": 12, "version": 1, "is_private": True}


def operation():
    return {
        "checkpoint_sha256": SHA,
        "kernel_slug": SLUG,
        "dataset_slug": "owner/rsna-alldev-e002-aaaaaaaaaaaa",
        "manifest_sha256": "b" * 64,
        "kernel_source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
    }


def attestation(**change):
    value = {
        "schema_version": 1,
        "kind": "owner_authenticated_ui_private_draft_observation",
        "account": "owner",
        "kernel_slug": SLUG,
        "observed_at": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
        "editor_opened": True,
        "version_history": "starting_fresh_no_saved_versions",
        "inputs_match_operation": True,
        "code_matches_operation": True,
        "ui_save_version_performed": False,
        "save_kernel_retry_performed": False,
    }
    value.update(change)
    return value


class Remote:
    username = "owner"

    def __init__(self, response=None, accessible=None, listing=None):
        self.response = {"ref": "/code/" + SLUG, "versionNumber": 1}
        if response is not None:
            self.response = response
        self.accessible = accessible
        self.listing = PLACEHOLDER if listing is None else listing
        self.saves = []
        self.reads = []

    def call(self, service, method, body):
        if method == "SaveKernel":
            self.saves.append(body)
            if isinstance(self.response, Exception):
                raise self.response
            return self.response
        self.reads.append(method)
        if method == "GetKernel":
            if self.accessible is None:
                raise workflow.RemoteError("Kaggle GetKernel HTTP 403")
            return self.accessible
        assert method == "ListKernels" and body["search"] == SLUG.split("/")[1]
        return self.listing

    def dataset(self, slug):
        return {"id": 12, "isPrivate": True, "currentVersionNumber": 1}


@pytest.fixture
def case(tmp_path, monkeypatch):
    path = tmp_path / "operation.json"
    path.write_text(json.dumps(operation()))
    monkeypatch.setattr(workflow, "load_operation", lambda p: json.loads(p.read_text()))
    monkeypatch.setattr(
        workflow, "kernel_request", lambda p, op: {"slug": op["kernel_slug"], "text": SOURCE}
    )
    original_push_response = workflow.save_kernel_response({"error": "Notebook not saved"})
    entry = {
        "checkpoint_sha256": SHA,
        "state": "push_uncertain",
        "operation": operation(),
        "operation_sha256": workflow.digest(path),
        "dataset": DATASET,
        "kernel_absence_proof": {"method": "complete_authenticated_profile_after403"},
        "kernel_create_response": original_push_response,
    }
    with workflow.ledger(tmp_path / "ledger.sqlite3", "owner") as db:
        workflow.save_entry(db, entry, "fixture")
        yield db, path, entry


def commit(db, path, remote, sha=None, value=None):
    return c.commit_private_draft_once(
        workflow, remote, db, path, sha or workflow.digest(path), value or attestation()
    )


def test_save_kernel_response_is_bounded_and_sanitized():
    token = "KGAT_" + "x" * 40
    record = workflow.save_kernel_response(
        {
            "error": f"bad {token} https://s.example/a?sig=1 Bearer zzsecretzz " + "e" * 400,
            "url": "https://s.example/u?token=1",
            "invalidDatasetSources": ["owner/data"] * 20,
            "unexpected": {"nested": True},
        }
    )
    text = json.dumps(record)
    assert token not in text and "sig=1" not in text and "zzsecretzz" not in text
    assert record["kernel_id"] is None and record["ref"] is None and record["version"] is None
    assert record["error"]["chars"] > 240 and len(record["error"]["prefix"]) == 240
    assert record["url"] == "[redacted]"
    assert record["invalidDatasetSources"]["count"] == 20
    assert len(record["invalidDatasetSources"]["items"]) == 8
    assert record["response_keys"] == ["error", "invalidDatasetSources", "unexpected", "url"]
    assert workflow.save_kernel_response(None)["type"] == "NoneType"


def test_push_preserves_sanitized_null_identity_response_and_exception(tmp_path, monkeypatch):
    operation_path = tmp_path / "operation.json"
    operation_path.write_text("{}")
    monkeypatch.setattr(workflow, "load_operation", lambda *a: operation())
    monkeypatch.setattr(workflow, "kernel_request", lambda *a: {})
    monkeypatch.setattr(workflow, "verify_remote_dataset", lambda *a: DATASET)
    remote = Remote(response={"error": "Notebook draft only"})
    remote.require_absent_kernel = lambda slug: {"method": "GetKernel404"}
    with workflow.ledger(tmp_path / "ledger", "owner") as db:
        workflow.save_entry(db, {"checkpoint_sha256": SHA, "state": "queued"}, "test")
        with pytest.raises(workflow.RemoteError, match="reconciliation"):
            workflow.push(remote, db, operation_path)
        result = workflow.read_entry(db, SHA)
    assert result["state"] == "push_uncertain" and len(remote.saves) == 1
    assert result["kernel_create_response"]["error"] == "Notebook draft only"
    assert result["kernel_create_exception"]["type"] == "RemoteError"


@pytest.mark.parametrize(
    "remote,value,kind",
    [
        (Remote(), None, "unresolved"),
        (Remote(), attestation(), "private_draft_owner_attested"),
        (Remote(accessible={"metadata": {"currentVersionNumber": 1}}), None, "committed_version"),
        (Remote(accessible={"metadata": {}}), attestation(), "unresolved"),
        (Remote(listing={"kernels": []}), attestation(), "unresolved"),
        (Remote(), attestation(kernel_slug="owner/other"), "unresolved"),
        (Remote(), attestation(account="other"), "unresolved"),
        (Remote(), attestation(ui_save_version_performed=True), "unresolved"),
        (Remote(), attestation(save_kernel_retry_performed=True), "unresolved"),
        (Remote(), attestation(version_history="one_version"), "unresolved"),
        (Remote(), attestation(code_matches_operation=False), "unresolved"),
        (Remote(), attestation(observed_at="2026-10-14T00:00:00+00:00"), "unresolved"),
        (Remote(), {**attestation(), "extra": 1}, "unresolved"),
    ],
)
def test_classification_never_treats_placeholder_alone_as_absence_or_draft(remote, value, kind):
    assert c.observe_uncertain_push(workflow, remote, SLUG, value)["classification"] == kind
    assert not remote.saves


def test_exact_draft_commits_once_as_version_one_and_preserves_original(case):
    db, path, entry = case
    remote = Remote()
    result = commit(db, path, remote)
    assert len(remote.saves) == 1 and remote.saves[0]["slug"] == SLUG
    assert result["state"] == "pushed" and result["kernel_version"] == 1
    stored = workflow.read_entry(db, SHA)
    assert stored["kernel_create_response"] == entry["kernel_create_response"]
    assert stored["kernel_absence_proof"] == entry["kernel_absence_proof"]
    assert stored["draft_commit"]["original_kernel_create_response_sha256"] == (
        workflow.object_hash(entry["kernel_create_response"])
    )
    events = [row[0] for row in db.execute("SELECT event FROM events ORDER BY id")]
    assert events[-2].startswith("durable intent") and events[-1].startswith("exact draft")
    with pytest.raises(ValueError, match="null-identity"):
        commit(db, path, remote)
    assert len(remote.saves) == 1


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"error": "still draft"},
        {"ref": "/code/" + SLUG, "versionNumber": 2},
        {"ref": "/code/owner/title-derived", "versionNumber": 1},
        workflow.RemoteError("Kaggle SaveKernel transport/result uncertain"),
    ],
)
def test_uncertain_first_commit_is_terminal_and_only_read_only_reconciled(case, response):
    db, path, _ = case
    remote = Remote(response=response)
    with pytest.raises(workflow.RemoteError):
        commit(db, path, remote)
    stored = workflow.read_entry(db, SHA)
    assert stored["state"] == "draft_commit_uncertain" and stored.get("kernel_version") is None
    assert stored["draft_commit"]["exception"]["type"] == "RemoteError"
    with pytest.raises(ValueError, match="null-identity"):
        commit(db, path, Remote())
    assert len(remote.saves) == 1
    plan = dq.plan({"entries": [{"epoch": 2, "checkpoint_sha256": SHA}]}, {SHA: stored}, None)
    assert plan.name == "reconcile-push" and plan.read_only


def test_crash_after_intent_is_never_retried(case, monkeypatch):
    db, path, _ = case
    remote = Remote()

    class Crash(BaseException):
        pass

    def crash(*args):
        raise Crash

    monkeypatch.setattr(remote, "call", lambda s, m, b: crash() if m == "SaveKernel" else 0)
    monkeypatch.setattr(
        c,
        "observe_uncertain_push",
        lambda *a, **k: {"classification": "private_draft_owner_attested"},
    )
    with pytest.raises(Crash):
        commit(db, path, remote)
    db.rollback()  # process death drops only the open post-intent transaction
    assert workflow.read_entry(db, SHA)["state"] == "draft_commit_intent"
    with pytest.raises(ValueError, match="null-identity"):
        commit(db, path, Remote())


@pytest.mark.parametrize(
    "change",
    [
        {"state": "pushed"},
        {"state": "queued"},
        {"kernel_version": 1},
        {"submission_id": 5},
        {"description": "rsna-epoch-v1:x"},
        {"draft_commit": {}},
        {"operation_sha256": "c" * 64},
        {"kernel_absence_proof": None},
        {"kernel_create_response": {"kernel_id": None, "ref": "/code/owner/x", "version": None}},
        {"kernel_create_response": {"kernel_id": 9, "ref": None, "version": None}},
        {"kernel_create_response": None},
        {"dataset": {"id": 13, "version": 1, "is_private": True}},
        {"operation": {**operation(), "manifest_sha256": "d" * 64}},
    ],
)
def test_changed_or_known_identity_cannot_commit(case, change):
    db, path, entry = case
    workflow.save_entry(db, {**copy.deepcopy(entry), **change}, "changed")
    remote = Remote()
    with pytest.raises(ValueError):
        commit(db, path, remote)
    assert not remote.saves
    assert workflow.read_entry(db, SHA).get("state") == change.get("state", "push_uncertain")


def test_stale_or_missing_owner_draft_evidence_cannot_commit(case):
    db, path, _ = case
    for remote, value in [
        (Remote(), attestation(observed_at="2026-10-14T00:00:00+00:00")),
        (Remote(listing={"kernels": []}), attestation()),
        (Remote(accessible={"metadata": {"currentVersionNumber": 1}}), attestation()),
    ]:
        with pytest.raises(ValueError):
            commit(db, path, remote, value=value)
        assert not remote.saves
    assert workflow.read_entry(db, SHA)["state"] == "push_uncertain"


def test_changed_source_or_wrong_expected_operation_cannot_commit(case, monkeypatch):
    db, path, _ = case
    remote = Remote()
    with pytest.raises(ValueError):
        commit(db, path, remote, sha="c" * 64)
    monkeypatch.setattr(workflow, "kernel_request", lambda p, op: {"slug": SLUG, "text": "other"})
    with pytest.raises(ValueError, match="exact prepared source"):
        commit(db, path, remote)
    assert not remote.saves


def test_draft_states_are_reconcilable_read_only(case):
    db, path, entry = case
    for state in ("draft_commit_intent", "draft_commit_uncertain"):
        workflow.save_entry(db, {**entry, "state": state}, "state")
        remote = Remote()
        remote.kernel = lambda method, slug, version: (
            {
                "metadata": {
                    "isPrivate": True,
                    "enableInternet": False,
                    "machineShape": "NvidiaTeslaT4",
                    "competitionDataSources": [workflow.COMPETITION],
                    "datasetDataSources": [entry["operation"]["dataset_slug"]],
                },
                "blob": {"source": SOURCE},
            }
            if method == "GetKernel"
            else {"status": "RUNNING"}
        )
        assert workflow.reconcile_push(remote, db, SHA, 1)["state"] == "pushed"
        assert not remote.saves
