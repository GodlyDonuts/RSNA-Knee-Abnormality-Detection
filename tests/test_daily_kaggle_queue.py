"""Synthetic exact-admission and finite queue guards; never authenticates or mutates Kaggle."""

import copy
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.kaggle import daily_queue as dq
from scripts.kaggle import reviewed_infer as reviewed

AUDIT = Path("initiative/audits/daily-kaggle-comparison-queue-20261014-v1")
QUEUE = AUDIT / "queue.json"
SPEC = importlib.util.spec_from_file_location(
    "queue_workflow", Path(__file__).resolve().parents[1] / "scripts/kaggle/workflow.py"
)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def root_sidecars():
    return json.loads((AUDIT / "root-sidecar-bindings.json").read_text())


def bind(monkeypatch, binding):
    """Synthetic binding stands in for the private original; only its hashes are patched."""
    monkeypatch.setattr(reviewed, "ALL_DEVELOPMENT_BINDING_SHA", reviewed.object_sha(binding))
    monkeypatch.setattr(
        reviewed, "ALL_DEVELOPMENT_CODE_SHA", reviewed.object_sha(binding["code_sha256"])
    )


def state(monkeypatch, epoch=1):
    root = Path("initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final")
    binding = json.loads((root / "public-run.json").read_text())["binding"]
    sidecar = root_sidecars()
    binding["config"] = copy.deepcopy(sidecar["config"])
    binding["experimental_fitting_supervision"] = copy.deepcopy(
        sidecar["experimental_fitting_supervision"]
    )
    binding["training_uids_sha256"] = reviewed.ALL_DEVELOPMENT_SUPERVISION["fitting_uid_sha256"]
    binding["outer_uids_sha256"] = hashlib.sha256(b"[]").hexdigest()
    bind(monkeypatch, binding)
    sha, steps = reviewed.ALL_DEVELOPMENT_EPOCHS[epoch]
    checkpoint = dict(
        binding=binding,
        schema_version=2,
        optimizer_boundary=True,
        epoch=epoch,
        next_update=0,
        step=steps,
        scheduled_epoch=True,
    )
    return checkpoint, reviewed.queue_admission(epoch), sha


def test_committed_queue_equals_root_sidecars_and_admission_constants():
    queue = dq.load_queue(QUEUE)
    sidecar = root_sidecars()
    assert [
        (e["epoch"], e["checkpoint_sha256"], e["optimizer_steps"]) for e in queue["entries"]
    ] == [(s["epoch"], s["checkpoint_sha256"], s["optimizer_steps"]) for s in sidecar["snapshots"]]
    assert all(
        s["binding_sha256"] == reviewed.ALL_DEVELOPMENT_BINDING_SHA for s in sidecar["snapshots"]
    )
    assert sidecar["training_binding_sha256"] == reviewed.ALL_DEVELOPMENT_BINDING_SHA
    assert sidecar["training_code_sha256"] == reviewed.ALL_DEVELOPMENT_CODE_SHA
    assert sidecar["scientific_source_git"] == reviewed.ALL_DEVELOPMENT_SOURCE
    receipt = sidecar["experimental_fitting_supervision"]
    expected = reviewed.ALL_DEVELOPMENT_SUPERVISION
    assert receipt["source"] == expected["source"]
    assert receipt["outer_uid_sha256"] == expected["outer_uid_sha256"]
    assert sidecar["config"]["fitting_supervision"]["sha256"] == expected["overlay_sha256"]
    encoder = sidecar["config"]["encoder"]
    assert all(encoder[k] == v for k, v in reviewed.OFFICIAL_BASE.items())
    assert len(encoder["adapter"]["target_modules"]) == 120


@pytest.mark.parametrize("epoch", [1, 2, 3, 4, 5])
def test_exact_queue_admission_preserves_checkpoint_and_default_guard(monkeypatch, epoch):
    checkpoint, admission, sha = state(monkeypatch, epoch)
    before = copy.deepcopy(checkpoint)
    with pytest.raises(ValueError, match="not admitted"):
        reviewed.infer.validate_inference_checkpoint(checkpoint, allow_scheduled_epoch=True)
    assert reviewed.validate_reviewed(checkpoint, admission, sha) is checkpoint["binding"]
    assert checkpoint == before


def change_binding(checkpoint, change):
    b = checkpoint["binding"]
    c, r = b["config"], b["experimental_fitting_supervision"]
    edits = {
        "side": lambda: c.update(side=224),
        "rank": lambda: c["encoder"]["adapter"].update(rank=64),
        "base_weights": lambda: c["encoder"].update(weights_sha256="0" * 64),
        "base_source": lambda: c["encoder"].update(source_id="other/dinov3"),
        "pretrained": lambda: c["encoder"].update(pretrained=False),
        "supervision_kind": lambda: c["fitting_supervision"].update(kind="label_source_v1"),
        "overlay": lambda: r.update(overlay_sha256="0" * 64),
        "cells": lambda: r.update(applied_cells=40815),
        "label_source": lambda: r["source"].update(name="llm_labels_v4_blend"),
        "fitting_studies": lambda: r.update(fitting_studies=2247),
        "outer": lambda: b.update(outer_uids_sha256="0" * 64),
        "evaluation_masks": lambda: r.update(evaluation_masks_changed=True),
        "fit_scope": lambda: c.update(fit_scope="outer_fold"),
        "diagnostic": lambda: c.update(diagnostic_study_limit=16),
        "external": lambda: b.update(external_initialization={"source": "KneeCoT"}),
        "external_none_key": lambda: b.update(external_initialization=None),
        "private": lambda: b.update(private_research_only=True),
        "public_input": lambda: c.update(public_input={"kind": "triplet"}),
        "epoch_counter": lambda: checkpoint.update(epoch=checkpoint["epoch"] + 1),
        "step_counter": lambda: checkpoint.update(step=checkpoint["step"] - 1),
        "next_update": lambda: checkpoint.update(next_update=3),
        "unscheduled": lambda: checkpoint.update(scheduled_epoch=False),
    }
    edits[change]()


@pytest.mark.parametrize(
    "change",
    [
        "side",
        "rank",
        "base_weights",
        "base_source",
        "pretrained",
        "supervision_kind",
        "overlay",
        "cells",
        "label_source",
        "fitting_studies",
        "outer",
        "evaluation_masks",
        "fit_scope",
        "diagnostic",
        "external",
        "external_none_key",
        "private",
        "public_input",
        "epoch_counter",
        "step_counter",
        "next_update",
        "unscheduled",
    ],
)
def test_changed_checkpoint_rejected_even_when_binding_hash_rebound(monkeypatch, change):
    checkpoint, admission, sha = state(monkeypatch)
    change_binding(checkpoint, change)
    bind(monkeypatch, checkpoint["binding"])
    admission = reviewed.queue_admission(1)
    with pytest.raises(ValueError, match="exact final admission"):
        reviewed.validate_reviewed(checkpoint, admission, sha)


@pytest.mark.parametrize(
    "key,value",
    [
        ("checkpoint_sha256", "0" * 64),
        ("training_binding_sha256", "0" * 64),
        ("training_code_sha256", "0" * 64),
        ("scientific_source_git", "a9902f3db0cf56995c074cddef1f0c2370e3ce47"),
        ("epoch", 6),
        ("optimizer_steps", 2810),
        ("training_studies", 2247),
        ("outer_studies", 1156),
        ("fit_scope", "outer_fold"),
        ("diagnostic_run", True),
        ("protected_confirmation_gold_used", True),
        ("external_initialization", {"source": "Stanford"}),
        ("private_research_only", True),
        ("MRI_only_inference", False),
        ("original_checkpoint_and_numerical_source_unchanged", False),
        ("status", "owner_authorized_fixed_final_kaggle_comparison"),
        ("queue_id", "mutable-latest"),
        ("guessed", "metadata"),
    ],
)
def test_changed_admission_rejected(monkeypatch, key, value):
    checkpoint, admission, sha = state(monkeypatch)
    admission[key] = value
    with pytest.raises(ValueError, match="exact final admission"):
        reviewed.validate_reviewed(checkpoint, admission, sha)


def test_other_epoch_checkpoint_hash_and_admission_cannot_be_swapped(monkeypatch):
    checkpoint, admission, _ = state(monkeypatch, 2)
    with pytest.raises(ValueError, match="exact final admission"):
        reviewed.validate_reviewed(checkpoint, admission, reviewed.ALL_DEVELOPMENT_EPOCHS[3][0])
    with pytest.raises(ValueError, match="exact final admission"):
        reviewed.validate_reviewed(
            checkpoint, reviewed.queue_admission(3), reviewed.ALL_DEVELOPMENT_EPOCHS[3][0]
        )


def test_existing_admissions_cannot_cross_authorize(monkeypatch):
    checkpoint, admission, sha = state(monkeypatch)
    prior = json.loads(
        Path(
            "initiative/audits/kaggle-publicsource-e10-ebeb3c394286-v1/fitting-admission.json"
        ).read_text()
    )
    with pytest.raises(ValueError, match="exact final admission"):
        reviewed.validate_reviewed(checkpoint, prior, sha)
    root = Path("initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final")
    prior_binding = json.loads((root / "public-run.json").read_text())["binding"]
    old = dict(checkpoint, binding=prior_binding, epoch=10, step=2810)
    with pytest.raises(ValueError, match="exact final admission"):
        reviewed.validate_reviewed(old, admission, prior["checkpoint_sha256"])


def test_original_non_authorization_validator_still_runs(monkeypatch):
    checkpoint, admission, sha = state(monkeypatch)
    checkpoint["optimizer_boundary"] = False
    with pytest.raises(ValueError, match="Unsupported competitive checkpoint"):
        reviewed.validate_reviewed(checkpoint, admission, sha)
    checkpoint["optimizer_boundary"] = True
    checkpoint["binding"]["targets"] = checkpoint["binding"]["targets"][:-1]
    bind(monkeypatch, checkpoint["binding"])
    with pytest.raises(ValueError, match="targets"):
        reviewed.validate_reviewed(checkpoint, reviewed.queue_admission(1), sha)


@pytest.mark.parametrize("change", ["sha", "extra", "slug", "csv", "retry", "supervision", "order"])
def test_queue_file_must_be_exact(tmp_path, change):
    queue = json.loads(QUEUE.read_text())
    if change == "sha":
        queue["entries"][0]["checkpoint_sha256"] = "0" * 64
    elif change == "extra":
        queue["entries"].append(dict(queue["entries"][-1], epoch=6))
    elif change == "slug":
        queue["entries"][1]["kernel_slug"] = "godlydonuts/rsna-latest"
    elif change == "csv":
        queue["deployment"]["public_example_csv_upload"] = True
    elif change == "retry":
        queue["deployment"]["automatic_mutation_retry"] = True
    elif change == "supervision":
        queue["run"]["supervision"]["applied_cells"] = 1
    else:
        queue["entries"].reverse()
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(queue))
    with pytest.raises(ValueError, match="exact"):
        dq.load_queue(path)


# ---- finite driver -------------------------------------------------------


def account(remaining=5, today=0, checked=None):
    return {
        "checked_at": (checked or datetime.now(timezone.utc)).isoformat(),
        "entered": True,
        "num_today": today,
        "num_allowed_now": remaining,
    }


class Runner:
    def __init__(self, db, fail=None, acceptance="accepted", defer=False):
        self.db, self.fail, self.acceptance, self.defer = db, fail, acceptance, defer
        self.calls = []

    def __call__(self, phase, entry, directory):
        self.calls.append((entry["epoch"], phase))
        sha = entry["checkpoint_sha256"]
        if phase == self.fail:
            raise RuntimeError("simulated crash after intent")
        if phase == "reconcile-dataset":
            return {"status": "all_payload_bytes_verified"}
        if phase == "push-kernel":
            record = {"checkpoint_sha256": sha, "state": "pushed", "kernel_version": 1}
            workflow.save_entry(self.db, record, "synthetic push")
            return record
        if phase == "public-acceptance":
            if self.acceptance == "accepted":
                return {"status": "exact_private_offline_public_MRI_only_prefetch_verified"}
            return {"status": self.acceptance}
        if phase == "submit":
            record = workflow.read_entry(self.db, sha)
            if self.defer:
                record["next_quota_check_utc"] = "2026-10-15T00:05:00+00:00"
            else:
                record.update(state="submitted", attempted_at=workflow.utcnow(), submission_id=9)
            workflow.save_entry(self.db, record, "synthetic submit")
            return record
        return {"status": "ok"}


@pytest.fixture
def setup(tmp_path):
    queue = dq.load_queue(QUEUE)
    with workflow.ledger(tmp_path / "ledger.sqlite3", "godlydonuts") as db:
        root = tmp_path / "journal"
        root.mkdir()
        yield queue, root, db


def drive(queue, root, db, runner, acct=None):
    return dq.drive(queue, root, db, workflow, acct or account, runner)


def test_one_candidate_full_exact_sequence_then_next_without_duplicate_phase(setup):
    queue, root, db = setup
    runner = Runner(db)
    steps = drive(queue, root, db, runner)
    assert runner.calls == [(1, phase) for phase in dq.PHASES]
    assert steps[-1]["phase"] == "submit" and steps[-1]["action"] == "ran"
    drive(queue, root, db, runner)
    assert runner.calls[len(dq.PHASES) :] == [(2, phase) for phase in dq.PHASES]
    assert sum(1 for call in runner.calls if call == (1, "submit")) == 1
    assert not (root / "driver-claim").exists()


@pytest.mark.parametrize("kind", ["zero_live", "account_today", "local_today", "stale"])
def test_account_limit_and_fresh_quota_precede_any_mutation_intent(setup, kind):
    queue, root, db = setup
    if kind == "local_today":
        for i in range(5):
            workflow.save_entry(
                db,
                {
                    "checkpoint_sha256": f"{i}" * 64,
                    "state": "submitted",
                    "attempted_at": workflow.utcnow(),
                },
                "other run",
            )
    acct = {
        "zero_live": lambda: account(remaining=0),
        "account_today": lambda: account(today=5),
        "local_today": account,
        "stale": lambda: account(checked=datetime.now(timezone.utc) - timedelta(minutes=10)),
    }[kind]
    runner = Runner(db)
    if kind == "stale":
        with pytest.raises(ValueError, match="stale"):
            drive(queue, root, db, runner, acct)
    else:
        assert drive(queue, root, db, runner, acct)[-1]["action"] == "wait_for_allowance"
    assert runner.calls == [(1, "prepare"), (1, "prepare-transport")]
    directory = dq.candidate_dir(root, queue["entries"][0])
    assert not (directory / "create-dataset.intent.json").exists()


@pytest.mark.parametrize(
    "phase", ["prepare", "prepare-transport", "create-dataset", "push-kernel", "submit"]
)
def test_crash_restart_forces_read_only_reconciliation_never_rerun(setup, phase):
    queue, root, db = setup
    runner = Runner(db, fail=phase)
    with pytest.raises(RuntimeError):
        drive(queue, root, db, runner)
    calls = list(runner.calls)
    runner.fail = None
    for _ in range(3):
        decision = drive(queue, root, db, runner)[-1]
        assert decision["action"] == "read_only_reconciliation" and decision["phase"] == phase
    assert runner.calls == calls
    if phase in {"prepare", "prepare-transport", "submit"}:
        with pytest.raises(ValueError, match="never rerun"):
            dq.reconcile(queue, root, db, runner)
        assert runner.calls == calls


def test_dataset_and_push_reconciliation_adopts_only_authoritative_evidence(setup):
    queue, root, db = setup
    runner = Runner(db, fail="create-dataset")
    with pytest.raises(RuntimeError):
        drive(queue, root, db, runner)
    runner.fail = None
    assert dq.reconcile(queue, root, db, runner)["action"] == "reconciled"
    runner.fail = "push-kernel"
    with pytest.raises(RuntimeError):
        drive(queue, root, db, runner)
    with pytest.raises(ValueError, match="never rerun"):
        dq.reconcile(queue, root, db, runner)
    sha = queue["entries"][0]["checkpoint_sha256"]
    workflow.save_entry(
        db,
        {"checkpoint_sha256": sha, "state": "pushed", "kernel_version": 1},
        "root reconcile-push",
    )
    runner.fail = None
    assert dq.reconcile(queue, root, db, runner)["outcome"]["kernel_version"] == 1
    drive(queue, root, db, runner)
    assert [c for c in runner.calls if c[0] == 1].count((1, "push-kernel")) == 1
    assert (1, "submit") in runner.calls


@pytest.mark.parametrize(
    "state_name", ["push_intent", "push_uncertain", "submit_intent", "submit_uncertain"]
)
def test_canonical_ledger_uncertainty_blocks_every_mutation(setup, state_name):
    queue, root, db = setup
    other = state_name.startswith("submit")
    sha = "f" * 64 if other else queue["entries"][0]["checkpoint_sha256"]
    workflow.save_entry(db, {"checkpoint_sha256": sha, "state": state_name}, "uncertain")
    runner = Runner(db)
    assert drive(queue, root, db, runner)[-1]["action"] == "read_only_reconciliation"
    assert runner.calls == []


def test_existing_submitted_or_scored_entries_are_never_resubmitted(setup):
    queue, root, db = setup
    first = queue["entries"][0]["checkpoint_sha256"]
    workflow.save_entry(
        db,
        {"checkpoint_sha256": first, "state": "submitted", "remote_outcome": "scored"},
        "earlier",
    )
    runner = Runner(db)
    drive(queue, root, db, runner)
    assert all(epoch == 2 for epoch, _ in runner.calls)


def test_unjournaled_ledger_entry_requires_read_only_reconciliation(setup):
    queue, root, db = setup
    first = queue["entries"][0]["checkpoint_sha256"]
    workflow.save_entry(db, {"checkpoint_sha256": first, "state": "pushed"}, "outside queue")
    runner = Runner(db)
    assert "not created by this queue" in drive(queue, root, db, runner)[-1]["reason"]
    assert runner.calls == []


def test_public_execution_pending_then_rejected_blocks_submission(setup):
    queue, root, db = setup
    runner = Runner(db, acceptance="pending")
    assert drive(queue, root, db, runner)[-1]["action"] == "await_public_execution"
    runner.acceptance = "rejected"
    assert drive(queue, root, db, runner)[-1]["action"] == "blocked"
    runner.acceptance = "accepted"
    assert drive(queue, root, db, runner)[-1]["action"] == "blocked"
    assert (1, "submit") not in runner.calls


def test_deferred_submission_archives_intent_and_retries_only_after_fresh_allowance(setup):
    queue, root, db = setup
    runner = Runner(db, defer=True)
    with pytest.raises(ValueError, match="deferred"):
        drive(queue, root, db, runner)
    assert dq.reconcile(queue, root, db, runner)["action"] == "deferred_intent_archived"
    directory = dq.candidate_dir(root, queue["entries"][0])
    assert (directory / "submit.intent.deferred-2026-10-15.json").exists()
    runner.defer = False
    assert drive(queue, root, db, runner)[-1]["action"] == "ran"
    assert runner.calls.count((1, "submit")) == 2


def test_driver_claim_prevents_concurrent_drivers(setup):
    queue, root, db = setup
    (root / "driver-claim").mkdir()
    with pytest.raises(FileExistsError):
        drive(queue, root, db, Runner(db))


def test_driver_never_uploads_a_csv_or_calls_competition_directly():
    text = Path(dq.__file__).read_text()
    assert ".competition(" not in text and "CreateSubmission" not in text


# ---- public execution acceptance -----------------------------------------


def acceptance_inputs():
    from rsna.schema import TARGETS

    uids = ["1.2.3", "1.2.4", "1.2.5"]
    rows = [["StudyInstanceUID", *TARGETS]] + [[u, *["0.25"] * len(TARGETS)] for u in uids]
    raw = ("\n".join(",".join(r) for r in rows) + "\n").encode()
    op = {"checkpoint_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    digest = hashlib.sha256(raw).hexdigest()
    receipt = {
        "status": "offline_inference_complete",
        "checkpoint_sha256": "a" * 64,
        "bundle_manifest_sha256": "b" * 64,
        "submission_sha256": digest,
    }
    metrics = {
        "status": "passed",
        "studies": 3,
        "resident_engine": dq.ENGINE,
        "chunks": [{"fallback_studies": 0}],
        "production_memory": {"summed_lifetime_RSS_bytes": 12 * 2**30},
        "submission_sha256": digest,
    }
    return op, receipt, metrics, raw, uids, TARGETS


def test_public_acceptance_exact_and_never_marks_csv_uploaded():
    result = dq.accept_public_execution(*acceptance_inputs())
    assert result["csv_uploaded"] is False and result["studies"] == 3


@pytest.mark.parametrize(
    "change", ["checkpoint", "fallback", "engine", "rss", "uids", "nan", "range"]
)
def test_public_acceptance_rejects_any_difference(change):
    op, receipt, metrics, raw, uids, targets = acceptance_inputs()
    if change == "checkpoint":
        receipt["checkpoint_sha256"] = "c" * 64
    elif change == "fallback":
        metrics["chunks"] = [{"fallback_studies": 1}]
    elif change == "engine":
        metrics["resident_engine"] = "resident_two_gpu_fp16"
    elif change == "rss":
        metrics["production_memory"]["summed_lifetime_RSS_bytes"] = 30 * 2**30
    elif change == "uids":
        uids = list(reversed(uids))
    else:
        raw = raw.replace(b"0.25", b"nan" if change == "nan" else b"1.5", 1)
        receipt["submission_sha256"] = metrics["submission_sha256"] = hashlib.sha256(
            raw
        ).hexdigest()
    with pytest.raises(ValueError, match="exact admitted inference"):
        dq.accept_public_execution(op, receipt, metrics, raw, uids, targets)
    assert math.isfinite(0.25)


def test_prefetch_source_binds_projected_helpers_and_accepted_hashes():
    op = {
        "manifest_sha256": "b" * 64,
        "dataset_slug": "godlydonuts/rsna-alldev-publabel-e01-74a24c6468a3",
        "shared_wheelhouse": {"archive_file": "w.bin", "dataset_slug": "godlydonuts/rsna-wheels"},
    }
    source, hashes = dq.prefetch_kernel_source(Path("."), op)
    assert set(hashes) == set(dq.PREFETCH_HELPERS) and "INFERENCE_FP32_PREFETCH = True" in source
    with pytest.raises(ValueError, match="accepted engine"):
        dq.prefetch_kernel_source(Path("."), op, dict(hashes, **{"chunked_infer.py": "0" * 64}))
