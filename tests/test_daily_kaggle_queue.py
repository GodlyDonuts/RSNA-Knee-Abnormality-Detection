"""Exact daily all-development queue: admission, rejection, single phases, restarts."""

import base64
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from rsna.schema import TARGETS
from scripts.kaggle import daily_queue as dq
from scripts.kaggle import reviewed_infer as reviewed

RECEIPTS = {
    1: "initiative/audits/final-public-source-epoch1-20261013-v1/verified.json",
    2: "initiative/audits/final-public-source-epoch2-20261013-v1/verified.json",
    3: "initiative/audits/final-public-source-epoch3-20261014-v1/verified.json",
}
UIDS = ["1.2.3", "1.2.4", "1.2.5"]


def all_development_binding():
    path = "initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final"
    binding = json.loads((Path(path) / "public-run.json").read_text())["binding"]
    admitted = "initiative/audits/final-public-source-full-fit-evc40-launch-v1"
    receipt = json.loads((Path(admitted) / "complete-input-admitted.json").read_text())
    binding["experimental_fitting_supervision"] = receipt["supervision_receipt"]
    binding["training_uids_sha256"] = reviewed.ALL_DEVELOPMENT_FIT
    binding["outer_uids_sha256"] = reviewed.EMPTY_ROSTER
    binding["config"].update(
        fit_scope="all_development",
        warmup_updates=426,
        fitting_supervision={
            "kind": "label_source_final_v1",
            "sha256": reviewed.ALL_DEVELOPMENT_OVERLAY,
            "enabled": True,
        },
    )
    return binding


@pytest.fixture
def repo(tmp_path, monkeypatch):
    binding = all_development_binding()
    # The real 908454 binding is root-owned; pin this synthetic equivalent instead.
    monkeypatch.setattr(reviewed, "ALL_DEVELOPMENT_BINDING", reviewed.object_sha(binding))
    monkeypatch.setattr(dq.reviewed(), "ALL_DEVELOPMENT_BINDING", reviewed.object_sha(binding))
    entries = []
    for epoch, relative in RECEIPTS.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True)
        shutil.copyfile(relative, target)
        receipt = json.loads(target.read_text())
        sha = receipt["checkpoint_sha256"]
        dataset, kernel = dq.slugs(epoch, sha)
        entries.append(
            {
                "epoch": epoch,
                "checkpoint_sha256": sha,
                "checkpoint_bytes": receipt["files"]["checkpoint.pt"]["bytes"],
                "optimizer_steps": 426 * epoch,
                "training_code_sha256": reviewed.object_sha(binding["code_sha256"]),
                "private_publication": {
                    "repo_id": "GodlyDonuts/RSNA-Data",
                    "revision": receipt["revision"],
                    "prefix": receipt["prefix"],
                    "file_count": 5,
                },
                "verified_receipt": relative,
                "verified_receipt_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "dataset_slug": dataset,
                "kernel_slug": kernel,
            }
        )
    queue = {
        "schema_version": 1,
        "queue_id": dq.QUEUE_ID,
        "status": dq.QUEUE_STATUS,
        "original_job": 908454,
        "scientific_source_git": reviewed.ALL_DEVELOPMENT_SOURCE,
        "training_binding_sha256": reviewed.object_sha(binding),
        "account": "godlydonuts",
        "public_csv_upload": False,
        "automatic_mutation_retry": False,
        "entries": entries,
    }
    path = tmp_path / dq.QUEUE_FILE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(queue))
    return tmp_path, path, binding


def checkpoint_for(binding, epoch):
    return {
        "binding": copy.deepcopy(binding),
        "schema_version": 2,
        "optimizer_boundary": True,
        "epoch": epoch,
        "next_update": 0,
        "step": 426 * epoch,
        "scheduled_epoch": True,
    }


def test_exact_queue_admissions_pass_original_guard_without_mutation(repo):
    root, path, binding = repo
    queue = dq.load_queue(path, root)
    for entry in queue["entries"]:
        admission = json.loads(dq.admission_bytes(queue, entry))
        assert set(admission) == reviewed.ALL_DEVELOPMENT_KEYS
        checkpoint = checkpoint_for(binding, entry["epoch"])
        before = copy.deepcopy(checkpoint)
        with pytest.raises(ValueError, match="not admitted"):
            reviewed.infer.validate_inference_checkpoint(checkpoint)
        sha = entry["checkpoint_sha256"]
        assert reviewed.validate_reviewed(checkpoint, admission, sha) is checkpoint["binding"]
        assert checkpoint == before


def test_admission_only_from_exact_entry(repo):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    guessed = dict(queue["entries"][0], epoch=4)
    with pytest.raises(ValueError, match="exact queue entry"):
        dq.admission_for(queue, guessed)
    with pytest.raises(ValueError, match="exact queue entry"):
        dq.queue_entry(queue, 5)


@pytest.mark.parametrize(
    "change",
    [
        "checkpoint",
        "bytes",
        "steps",
        "prefix",
        "revision",
        "slug",
        "receipt_bytes",
        "extra",
        "order",
        "duplicate",
        "csv_upload",
        "binding",
        "source",
        "retry",
    ],
)
def test_changed_queue_rejected(repo, change):
    root, path, _ = repo
    queue = json.loads(path.read_text())
    entry = queue["entries"][0]
    if change == "checkpoint":
        entry["checkpoint_sha256"] = "a" * 64
        entry["dataset_slug"], entry["kernel_slug"] = dq.slugs(1, "a" * 64)
    elif change == "bytes":
        entry["checkpoint_bytes"] += 1
    elif change == "steps":
        entry["optimizer_steps"] = 425
    elif change == "prefix":
        entry["private_publication"]["prefix"] = entry["private_publication"]["prefix"][:-1] + "2"
    elif change == "revision":
        entry["private_publication"]["revision"] = "0" * 40
    elif change == "slug":
        entry["kernel_slug"] = "godlydonuts/rsna-latest"
    elif change == "receipt_bytes":
        receipt = root / entry["verified_receipt"]
        receipt.write_text(receipt.read_text() + " ")
    elif change == "extra":
        entry["label"] = "best"
    elif change == "order":
        queue["entries"].reverse()
    elif change == "duplicate":
        queue["entries"].append(copy.deepcopy(entry))
    elif change == "csv_upload":
        queue["public_csv_upload"] = True
    elif change == "binding":
        queue["training_binding_sha256"] = "b" * 64
    elif change == "source":
        queue["scientific_source_git"] = "0" * 40
    elif change == "retry":
        queue["automatic_mutation_retry"] = True
    path.write_text(json.dumps(queue))
    with pytest.raises(ValueError):
        dq.load_queue(path, root)


@pytest.mark.parametrize(
    "change",
    [
        "checkpoint",
        "binding",
        "code",
        "supervision",
        "supervision_receipt",
        "base",
        "source",
        "epoch",
        "step",
        "next_update",
        "private",
        "external",
        "diagnostic",
        "public_input",
        "outer",
        "MRI",
        "extra",
        "prefix",
        "rank",
    ],
)
def test_changed_checkpoint_or_admission_rejected(repo, change):
    root, path, binding = repo
    queue = dq.load_queue(path, root)
    entry = queue["entries"][1]
    admission = json.loads(dq.admission_bytes(queue, entry))
    checkpoint = checkpoint_for(binding, 2)
    sha = entry["checkpoint_sha256"]
    b = checkpoint["binding"]
    if change == "checkpoint":
        sha = "a" * 64
    elif change == "binding":
        b["config"]["side"] = 224
    elif change == "code":
        admission["training_code_sha256"] = "c" * 64
    elif change == "supervision":
        b["config"]["fitting_supervision"]["sha256"] = "d" * 64
    elif change == "supervision_receipt":
        b["experimental_fitting_supervision"]["source"]["csv_sha256"] = "e" * 64
    elif change == "base":
        b["config"]["encoder"]["weights_sha256"] = "f" * 64
    elif change == "source":
        admission["scientific_source_git"] = "0" * 40
    elif change == "epoch":
        checkpoint["epoch"] = 3
    elif change == "step":
        checkpoint["step"] = 851
    elif change == "next_update":
        checkpoint["next_update"] = 1
    elif change == "private":
        b["private_research_only"] = True
    elif change == "external":
        b["external_initialization"] = {"source": "KneeCoT"}
    elif change == "diagnostic":
        b["config"]["diagnostic_study_limit"] = 16
    elif change == "public_input":
        b["config"]["public_input"] = {"kind": "ordered_triplet"}
    elif change == "outer":
        admission["outer_studies"] = 1156
    elif change == "MRI":
        admission["MRI_only_inference"] = False
    elif change == "extra":
        admission["allow_diagnostic"] = True
    elif change == "prefix":
        admission["private_publication"]["prefix"] += "x"
    elif change == "rank":
        b["config"]["encoder"]["adapter"]["rank"] = 16
    with pytest.raises(ValueError, match="exact queue admission"):
        reviewed.validate_reviewed(checkpoint, admission, sha)


def test_prior_public_source_and_reviewed_roles_still_exact(repo):
    root, path, binding = repo
    queue = dq.load_queue(path, root)
    admission = json.loads(dq.admission_bytes(queue, queue["entries"][0]))
    admission["role"] = "public_source_final10"
    with pytest.raises(ValueError, match="exact final admission"):
        reviewed.validate_reviewed(
            checkpoint_for(binding, 1), admission, queue["entries"][0]["checkpoint_sha256"]
        )


def test_snapshot_sidecar_must_match_entry(repo):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    entry = queue["entries"][0]
    sidecar = {
        "status": "scheduled_epoch_snapshot",
        "epoch": 1,
        "checkpoint_file": "epoch-001.pt",
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "checkpoint_bytes": entry["checkpoint_bytes"],
        "optimizer_steps": 426,
        "binding_sha256": queue["training_binding_sha256"],
        "diagnostic_only": False,
        "submission_eligible": True,
    }
    dq.check_snapshot(entry, sidecar)
    for key, value in (("diagnostic_only", True), ("optimizer_steps", 427), ("epoch", 2)):
        with pytest.raises(ValueError):
            dq.check_snapshot(entry, dict(sidecar, **{key: value}))


def public_outputs(queue, entry, **change):
    rows = [["StudyInstanceUID", *TARGETS], *[[uid, *["0.25"] * len(TARGETS)] for uid in UIDS]]
    raw = ("\n".join(",".join(r) for r in rows) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    operation = {
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "epoch": entry["epoch"],
        "dataset_slug": entry["dataset_slug"],
        "kernel_slug": entry["kernel_slug"],
        "manifest_sha256": "9" * 64,
        "fitting_admission_sha256": hashlib.sha256(dq.admission_bytes(queue, entry)).hexdigest(),
        "inference_execution_receipts": {
            "engine": dq.ENGINE,
            "new_checkpoint_public_acceptance_required": True,
        },
    }
    receipt = {
        "status": "offline_inference_complete",
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "bundle_manifest_sha256": "9" * 64,
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
    if change.get("fallback"):
        metrics["chunks"][0]["fallback_studies"] = 1
    if change.get("memory"):
        metrics["production_memory"]["summed_lifetime_RSS_bytes"] = 30 * 2**30
    if change.get("engine"):
        metrics["resident_engine"] = "legacy"
    if change.get("checkpoint"):
        receipt["checkpoint_sha256"] = "a" * 64
    if change.get("bytes"):
        raw += b"\n"
    if change.get("order"):
        raw = raw.replace(b"1.2.3", b"1.2.9")
        receipt["submission_sha256"] = metrics["submission_sha256"] = hashlib.sha256(
            raw
        ).hexdigest()
    if change.get("range"):
        raw = raw.replace(b"0.25", b"1.25", 1)
        receipt["submission_sha256"] = metrics["submission_sha256"] = hashlib.sha256(
            raw
        ).hexdigest()
    return {
        "operation": operation,
        "receipt": receipt,
        "metrics": metrics,
        "csv": base64.b64encode(raw).decode(),
    }


@pytest.mark.parametrize(
    "change", ["fallback", "memory", "engine", "checkpoint", "bytes", "order", "range"]
)
def test_public_execution_acceptance_is_exact(repo, change):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    entry = queue["entries"][0]
    good = public_outputs(queue, entry)
    accepted = dq.accept_public(
        queue,
        entry,
        good["operation"],
        good["receipt"],
        good["metrics"],
        base64.b64decode(good["csv"]),
        UIDS,
    )
    assert accepted["public_csv_uploaded"] is False and accepted["studies"] == 3
    bad = public_outputs(queue, entry, **{change: True})
    with pytest.raises(ValueError):
        dq.accept_public(
            queue,
            entry,
            bad["operation"],
            bad["receipt"],
            bad["metrics"],
            base64.b64decode(bad["csv"]),
            UIDS,
        )


class Crash(RuntimeError):
    pass


class FakeExecutor:
    """Simulates the canonical ledger transitions of workflow.push/submit."""

    def __init__(self, queue, allowance=5):
        self.queue = queue
        self.records = {}
        self.allow = allowance
        self.calls = []
        self.crash = set()
        self.kernel = []
        self.outcome = "pending_or_unknown"

    def _entry(self, sha):
        return [x for x in self.queue["entries"] if x["checkpoint_sha256"] == sha][0]

    def ledger_records(self):
        return copy.deepcopy(self.records)

    def allowance(self, entry):
        self.calls.append(("allowance", entry["epoch"]))
        return self.allow

    def prepare(self, entry, admission):
        self.calls.append(("prepare", entry["epoch"]))
        if "prepare" in self.crash:
            raise Crash
        return {
            "operation": public_outputs(self.queue, entry)["operation"],
            "competition_submission_created": False,
            "original_source_and_weights_unchanged": True,
        }

    def inspect_prepared(self, entry):
        self.calls.append(("inspect-prepared", entry["epoch"]))
        if "prepare_done" not in self.crash:
            return {}
        return {
            "operation": public_outputs(self.queue, entry)["operation"],
            "competition_submission_created": False,
            "original_source_and_weights_unchanged": True,
        }

    def create_dataset(self, entry):
        self.calls.append(("create-dataset", entry["epoch"]))
        if "create" in self.crash:
            raise Crash
        return {"status": "created_once"}

    def reconcile_dataset(self, entry):
        self.calls.append(("reconcile-dataset", entry["epoch"]))
        return {"status": "all_payload_bytes_verified"}

    def push_kernel(self, entry):
        self.calls.append(("push-kernel", entry["epoch"]))
        sha = entry["checkpoint_sha256"]
        self.records[sha] = {"checkpoint_sha256": sha, "state": "push_intent"}
        if "push" in self.crash:
            self.records[sha]["state"] = "push_uncertain"
            raise Crash
        self.records[sha].update(state="pushed", kernel_version=1)
        return copy.deepcopy(self.records[sha])

    def reconcile_push(self, entry):
        self.calls.append(("reconcile-push", entry["epoch"]))
        self.records[entry["checkpoint_sha256"]].update(state="pushed", kernel_version=1)
        return {"state": "pushed"}

    def status(self, entry):
        self.calls.append(("status", entry["epoch"]))
        return {"kernel_status": {"status": self.kernel.pop(0) if self.kernel else "COMPLETE"}}

    def collect_public(self, entry):
        self.calls.append(("collect-public", entry["epoch"]))
        return public_outputs(self.queue, entry)

    def submit(self, entry, acceptance):
        self.calls.append(("submit", entry["epoch"]))
        assert acceptance["checkpoint_sha256"] == entry["checkpoint_sha256"]
        sha = entry["checkpoint_sha256"]
        self.allow -= 1
        self.records[sha]["state"] = "submit_intent"
        if "submit" in self.crash:
            self.records[sha]["state"] = "submit_uncertain"
            raise Crash
        self.records[sha].update(state="submitted", submission_id=1000 + entry["epoch"])
        return copy.deepcopy(self.records[sha])

    def reconcile_submission(self, entry):
        self.calls.append(("reconcile-submission", entry["epoch"]))
        record = self.records[entry["checkpoint_sha256"]]
        record.update(state="submitted", remote_outcome=self.outcome)
        record.setdefault("submission_id", 2000 + entry["epoch"])
        return copy.deepcopy(record)


def driver(queue, ex, tmp_path, **kw):
    return dq.Driver(queue, ex, dq.Journal(tmp_path / "journal"), UIDS, sleep=lambda _: None, **kw)


def mutations(ex):
    return [c for c in ex.calls if c[0] in dq.MUTATIONS]


def test_full_queue_one_candidate_at_a_time_each_phase_once(repo):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    ex = FakeExecutor(queue)
    ex.kernel = ["QUEUED", "RUNNING"]
    history = driver(queue, ex, root).run()
    assert history[-1]["status"] == "complete"
    expected = [(p, e) for e in (1, 2, 3) for p in dq.MUTATIONS]
    assert mutations(ex) == expected
    assert [c for c in ex.calls if c[0] == "submit"] == [
        ("submit", 1),
        ("submit", 2),
        ("submit", 3),
    ]
    # Every cloud mutation was preceded immediately by a fresh allowance read.
    for i, call in enumerate(ex.calls):
        if call[0] in dq.CLOUD_MUTATIONS:
            assert ex.calls[i - 1] == ("allowance", call[1])
    # Restart after completion is read-only and never resubmits.
    before = list(ex.calls)
    assert driver(queue, ex, root).run()[-1]["status"] == "complete"
    assert mutations(ex) == mutations(type("E", (), {"calls": before})())


def test_account_limit_stops_submission_after_prevalidation(repo):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    ex = FakeExecutor(queue, allowance=2)
    history = driver(queue, ex, root).run()
    assert history[-1] == {"status": "allowance_exhausted", "epoch": 3}
    assert [c for c in ex.calls if c[0] == "submit"] == [("submit", 1), ("submit", 2)]
    assert ("create-dataset", 3) in ex.calls
    assert ("push-kernel", 3) in ex.calls
    assert (
        driver(queue, ex, root, max_submissions=0).run()[-1]["status"] == "submission_cap_reached"
    )


def test_submission_cap_per_invocation(repo):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    ex = FakeExecutor(queue)
    assert driver(queue, ex, root, max_submissions=1).run()[-1]["status"] == (
        "submission_cap_reached"
    )
    assert [c for c in ex.calls if c[0] == "submit"] == [("submit", 1)]


@pytest.mark.parametrize("phase", ["prepare", "create", "push", "submit"])
def test_crash_restart_reconciles_read_only_and_never_repeats(repo, phase):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    ex = FakeExecutor(queue)
    ex.crash = {phase}
    with pytest.raises(Crash):
        driver(queue, ex, root).run()
    first = mutations(ex)
    ex.crash = {"prepare_done"} if phase == "prepare" else set()
    history = driver(queue, ex, root).run()
    name = {"prepare": "prepare", "create": "create-dataset"}.get(phase, phase + "-kernel")
    if phase == "submit":
        name = "submit"
    epoch1 = [c for c in mutations(ex) if c[1] == 1]
    assert epoch1.count((name, 1)) == 1, epoch1
    assert history[-1]["status"] == "complete"
    assert mutations(ex)[: len(first)] == first
    if phase == "push":
        assert ("reconcile-push", 1) in ex.calls
    if phase == "submit":
        assert ("reconcile-submission", 1) in ex.calls
        assert ex.records[queue["entries"][0]["checkpoint_sha256"]]["state"] == "submitted"


def test_uncertain_prepare_without_node_record_halts(repo):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    ex = FakeExecutor(queue)
    ex.crash = {"prepare"}
    with pytest.raises(Crash):
        driver(queue, ex, root).run()
    ex.crash = set()
    assert driver(queue, ex, root).run()[-1]["status"] == "halt"
    assert mutations(ex) == [("prepare", 1)]


def test_existing_submitted_or_uncertain_entries_force_reconciliation(repo):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    journal = dq.Journal(root / "journal")
    first, second = (
        queue["entries"][0]["checkpoint_sha256"],
        queue["entries"][1]["checkpoint_sha256"],
    )
    records = {first: {"state": "submitted", "submission_id": 1}}
    assert dq.plan(queue, records, journal).name == "reconcile-outcome"
    records[first]["remote_outcome"] = "failed"
    assert dq.plan(queue, records, journal).name == "halt"
    records[first]["remote_outcome"] = "scored"
    assert dq.plan(queue, records, journal) == dq.Action("prepare", 2, second, False)
    records[second] = {"state": "submit_uncertain"}
    assert dq.plan(queue, records, journal).name == "reconcile-submission"
    records[second] = {"state": "push_intent"}
    assert dq.plan(queue, records, journal).name == "reconcile-push"
    foreign = {"f" * 64: {"state": "submit_intent"}}
    assert dq.plan(queue, foreign, journal).name == "halt"
    assert dq.plan(queue, {first: {"state": "mystery"}}, journal).name == "halt"


def test_ledger_change_between_fresh_observations_blocks_mutation(repo):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    ex = FakeExecutor(queue)
    d = driver(queue, ex, root)
    d.step()  # prepare
    original = ex.ledger_records
    reads = iter([{}, {queue["entries"][0]["checkpoint_sha256"]: {"state": "submit_intent"}}])
    ex.ledger_records = lambda: next(reads)
    with pytest.raises(ValueError, match="changed"):
        d.step()
    ex.ledger_records = original
    assert ("create-dataset", 1) not in ex.calls


def test_terminal_notebook_failure_and_deadline_halt_before_submission(repo):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    ex = FakeExecutor(queue)
    ex.kernel = ["ERROR"]
    assert driver(queue, ex, root).run()[-1]["status"] == "halt"
    assert ("submit", 1) not in ex.calls
    ex2 = FakeExecutor(queue)
    ex2.kernel = ["RUNNING"] * 10
    out = dq.Driver(
        queue,
        ex2,
        dq.Journal(root / "journal2"),
        UIDS,
        sleep=lambda _: None,
        poll_seconds=60,
        deadline_seconds=30,
    ).run()
    assert out[-1]["status"] == "deadline" and ("submit", 1) not in ex2.calls


def test_zero_slots_prevalidates_once_then_submits_only_after_fresh_reset(repo):
    root, path, _ = repo
    queue = dq.load_queue(path, root)
    ex = FakeExecutor(queue, allowance=0)
    history = driver(queue, ex, root).run()
    assert history[-1] == {"status": "allowance_exhausted", "epoch": 1}
    assert ("create-dataset", 1) in ex.calls
    assert ("push-kernel", 1) in ex.calls
    assert not any(name == "submit" for name, _ in ex.calls)
    before = list(mutations(ex))
    assert driver(queue, ex, root).run()[-1]["status"] == "allowance_exhausted"
    assert mutations(ex) == before
    ex.allow = 1
    driver(queue, ex, root, max_submissions=1).run()
    assert [call for call in ex.calls if call[0] == "submit"] == [("submit", 1)]
    assert ex.calls.count(("create-dataset", 1)) == 1
    assert ex.calls.count(("push-kernel", 1)) == 1
