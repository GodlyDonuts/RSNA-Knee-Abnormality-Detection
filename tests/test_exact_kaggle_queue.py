"""Synthetic only: no Kaggle, Slurm, HF, checkpoint bytes or MRI are touched."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.kaggle import exact_queue as queue
from scripts.kaggle import reviewed_infer as reviewed

AUDITS = Path("initiative/audits")


def checkpoint_for(epoch, monkeypatch):
    root = AUDITS / "public-label-source-full-fit-preparation-v1/paired-fixed-final"
    binding = json.loads((root / "public-run.json").read_text())["binding"]
    config = binding["config"]
    config.update(
        fit_scope="all_development",
        epochs=10,
        warmup_updates=426,
        seed=20260922,
        fitting_supervision={
            "kind": "label_source_final_v1",
            "sha256": reviewed.ALL_DEVELOPMENT_OVERLAY,
            "enabled": True,
        },
    )
    binding["experimental_fitting_supervision"].update(
        overlay_sha256=reviewed.ALL_DEVELOPMENT_OVERLAY,
        release_sha256=reviewed.ALL_DEVELOPMENT_RELEASE,
        folds_sha256=reviewed.ALL_DEVELOPMENT_FOLDS,
        fitting_uid_sha256=reviewed.ALL_DEVELOPMENT_FIT,
        outer_uid_sha256=reviewed.EMPTY_ROSTER,
        fitting_studies=3403,
        fit_scope="all_development",
        final_fit_only=True,
        internal_validation_available=False,
        evaluation_masks_changed=False,
        experimental_only=True,
        enabled=True,
        source={"csv_sha256": reviewed.ALL_DEVELOPMENT_SOURCE_CSV},
    )
    binding["training_uids_sha256"] = reviewed.ALL_DEVELOPMENT_FIT
    binding["outer_uids_sha256"] = reviewed.EMPTY_ROSTER
    binding["release"]["manifest_sha256"] = reviewed.ALL_DEVELOPMENT_RELEASE
    monkeypatch.setattr(reviewed, "ALL_DEVELOPMENT_BINDING", reviewed.object_sha(binding))
    monkeypatch.setattr(
        reviewed, "ALL_DEVELOPMENT_CODE", reviewed.object_sha(binding["code_sha256"])
    )
    checkpoint = dict(
        binding=binding,
        schema_version=2,
        optimizer_boundary=True,
        epoch=epoch,
        next_update=0,
        step=426 * epoch,
        scheduled_epoch=True,
    )
    admission = reviewed.all_development_admission(epoch)
    return checkpoint, admission, admission["checkpoint_sha256"]


def test_exact_registry_is_bound_to_committed_production_receipts():
    for epoch, name in ((1, "epoch1-20261013"), (2, "epoch2-20261013"), (3, "epoch3-20261014")):
        text = (AUDITS / f"final-public-source-{name}-v1/verified.json").read_text()
        sha, revision = reviewed.ALL_DEVELOPMENT_EPOCHS[epoch]
        assert sha in text and revision in text
    admission = json.loads(
        (
            AUDITS
            / "final-public-source-fit-production-start-20261013-v1/production-admission.json"
        ).read_text()
    )
    assert reviewed.ALL_DEVELOPMENT_BINDING in json.dumps(admission)
    cpu = json.loads(
        (AUDITS / "public-label-final-inputs-20261013-v1/actual-CPU-accepted.json").read_text()
    )
    code = {
        k.removeprefix("code/rsna/"): v
        for k, v in cpu["source_modules"].items()
        if k.startswith("code/rsna/")
    }
    assert reviewed.object_sha(code) == reviewed.ALL_DEVELOPMENT_CODE
    assert cpu["source_git"] == reviewed.ALL_DEVELOPMENT_SOURCE
    assert reviewed.EMPTY_ROSTER == hashlib.sha256(b"[]").hexdigest()
    assert [x["epoch"] for x in queue.candidates()] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("epoch", [1, 2, 3, 4, 5])
def test_each_exact_queued_epoch_is_admitted_without_changing_checkpoint(epoch, monkeypatch):
    checkpoint, admission, sha = checkpoint_for(epoch, monkeypatch)
    before = copy.deepcopy(checkpoint)
    with pytest.raises(ValueError):
        reviewed.infer.validate_inference_checkpoint(checkpoint)
    assert reviewed.validate_reviewed(checkpoint, admission, sha) is checkpoint["binding"]
    assert checkpoint == before


CHANGES = [
    "checkpoint",
    "binding",
    "code",
    "supervision_kind",
    "supervision_sha",
    "supervision_disabled",
    "base",
    "source",
    "epoch",
    "step",
    "fitting",
    "outer",
    "private",
    "diagnostic",
    "external",
    "label",
    "MRI",
    "revision",
    "unknown_epoch",
]


@pytest.mark.parametrize("change", CHANGES)
def test_changed_identity_counters_or_flags_are_rejected(change, monkeypatch):
    checkpoint, admission, sha = checkpoint_for(2, monkeypatch)
    binding, config = checkpoint["binding"], checkpoint["binding"]["config"]
    if change == "checkpoint":
        sha = reviewed.ALL_DEVELOPMENT_EPOCHS[3][0]
    elif change == "binding":
        config["side"] = 224
    elif change == "code":
        admission["training_code_sha256"] = "b" * 64
    elif change == "supervision_kind":
        config["fitting_supervision"]["kind"] = "public_v2_probability"
    elif change == "supervision_sha":
        binding["experimental_fitting_supervision"]["overlay_sha256"] = "c" * 64
    elif change == "supervision_disabled":
        config["fitting_supervision"]["enabled"] = False
    elif change == "base":
        config["encoder"]["weights_sha256"] = "d" * 64
    elif change == "source":
        admission["scientific_source_git"] = "a9902f3db0cf56995c074cddef1f0c2370e3ce47"
    elif change == "epoch":
        checkpoint["epoch"] = 3
    elif change == "step":
        checkpoint["step"] = 851
    elif change == "fitting":
        binding["experimental_fitting_supervision"]["fitting_studies"] = 2247
    elif change == "outer":
        binding["outer_uids_sha256"] = reviewed.OUTER
    elif change == "private":
        binding["private_research_only"] = True
    elif change == "diagnostic":
        config["diagnostic_study_limit"] = 16
    elif change == "external":
        binding["external_initialization"] = {"source": "KneeCoT"}
    elif change == "label":
        admission["note"] = "owner approved"
    elif change == "MRI":
        admission["MRI_only_inference"] = False
    elif change == "revision":
        admission["private_HF_revision"] = "0" * 40
    elif change == "unknown_epoch":
        admission["epoch"] = 6
    # Rebinding the hashes inside an edited checkpoint must not help.
    if change not in {"code", "binding"}:
        monkeypatch.setattr(reviewed, "ALL_DEVELOPMENT_BINDING", reviewed.object_sha(binding))
    with pytest.raises(ValueError, match="exact queue admission"):
        reviewed.validate_reviewed(checkpoint, admission, sha)


def test_existing_final10_and_diagnostic_roles_cannot_cross_admit(monkeypatch):
    checkpoint, admission, sha = checkpoint_for(1, monkeypatch)
    final10 = dict(admission, role="public_source_final10")
    with pytest.raises(ValueError):
        reviewed.validate_reviewed(checkpoint, final10, sha)
    with pytest.raises(ValueError):
        reviewed.validate_reviewed(checkpoint, dict(admission, role="diagnostic"), sha)
    with pytest.raises(ValueError, match="exact queue admission"):
        reviewed.validate_reviewed(
            checkpoint, admission, reviewed.ALL_DEVELOPMENT_EPOCHS[1][0][:-1]
        )
    old = reviewed.all_development_admission(1)
    old["checkpoint_sha256"] = "ebeb3c39428612dc85e60fb6e5d3c0e4fde0acc6987cbe9a1b3f088aa273cfca"
    with pytest.raises(ValueError, match="exact queue admission"):
        reviewed.validate_reviewed(checkpoint, old, old["checkpoint_sha256"])


def owner_queue(tmp_path, change=None):
    items = [dict(x, deployment_state="not_prepared_not_submitted") for x in queue.candidates()]
    if change == "sha":
        items[4]["checkpoint_sha256"] = "e" * 64
    elif change == "extra":
        items.append(dict(items[0], epoch=6))
    elif change == "bytes":
        items[0]["bytes"] -= 1
    value = {
        "candidates": items,
        "same_native_FP32_deployment_engine_as_scored_source_e10_required": True,
        "no_diagnostic_model_or_Stanford_initialization": change != "flag",
    }
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize("change", ["sha", "extra", "bytes", "flag"])
def test_queue_admits_only_exact_owner_entries(tmp_path, change):
    assert queue.load_queue(owner_queue(tmp_path)) == queue.candidates()
    with pytest.raises(ValueError, match="exact owner"):
        queue.load_queue(owner_queue(tmp_path, change))


UIDS = ["1.2.3", "1.2.4", "1.2.5"]


def public_csv(value="0.5"):
    from rsna.schema import TARGETS

    rows = [",".join(["StudyInstanceUID", *TARGETS])]
    rows += [",".join([uid, *[value] * len(TARGETS)]) for uid in UIDS]
    return "\n".join(rows) + "\n"


class Fake:
    """Canonical ledger and remote phases, simulated in memory."""

    def __init__(self, remaining=5, today=0):
        self.entries = {}
        self.calls = []
        self.remaining, self.today = remaining, today
        self.fail = set()
        self.engine = "two_device_exact_FP32_prefetch_v1"
        self.outcome = "scored"

    def account(self):
        return {
            "checked_at": queue.utcnow(),
            "entered": True,
            "num_today": self.today,
            "num_allowed_now": self.remaining,
        }

    def __call__(self, request):
        phase = request["phase"]
        self.calls.append((phase, request.get("candidate", {}).get("epoch")))
        if phase == "ledger":
            return {"entries": [copy.deepcopy(x) for x in self.entries.values()]}
        if phase == "account":
            return self.account()
        item = request["candidate"]
        sha = item["checkpoint_sha256"]
        assert json.loads(Path(request["fitting_admission"]).read_text()) == (
            reviewed.all_development_admission(item["epoch"])
        )
        manifest = hashlib.sha256(sha.encode()).hexdigest()
        if phase == "push-kernel":
            self.entries[sha]["state"] = "push_intent"
        if phase == "submit":
            self.entries[sha].update(state="submit_intent", attempted_at=queue.utcnow())
        if phase in self.fail:
            raise RuntimeError("crash after durable remote intent")
        out = {"phase": phase, "checkpoint_sha256": sha}
        if phase == "queue":
            self.entries[sha] = {"checkpoint_sha256": sha, "state": "queued"}
        elif phase == "prepare":
            out["manifest_sha256"] = manifest
        elif phase == "push-kernel":
            self.entries[sha].update(state="pushed", operation={"manifest_sha256": manifest})
        elif phase == "public-acceptance":
            csv = public_csv()
            digest = hashlib.sha256(csv.encode()).hexdigest()
            out.update(
                kernel_status="COMPLETE",
                submission_csv=csv,
                expected_public_uids=UIDS,
                receipt={
                    "status": "offline_inference_complete",
                    "checkpoint_sha256": sha,
                    "bundle_manifest_sha256": manifest,
                    "studies": 3,
                    "submission_sha256": digest,
                },
                metrics={
                    "status": "passed",
                    "diagnostic_only": False,
                    "checkpoint_sha256": sha,
                    "checkpoint_epoch": item["epoch"],
                    "submission_sha256": digest,
                    "studies": 3,
                    "resident_engine": self.engine,
                    "chunks": [{"fallback_studies": 0}],
                    "production_memory": {"summed_lifetime_RSS_bytes": 12 * 2**30},
                },
            )
        elif phase == "submit":
            self.entries[sha].update(state="submitted", submission_id=1)
            self.remaining -= 1
            self.today += 1
        elif phase == "reconcile-submission":
            self.entries[sha].update(state="submitted", remote_outcome=self.outcome)
            out["remote_outcome"] = self.outcome
        return out


def driver(tmp_path):
    return queue.load_queue(owner_queue(tmp_path)), queue.Journal(tmp_path / "journal")


def mutations(fake):
    return [x for x in fake.calls if x[0] in queue.MUTATIONS]


def test_one_candidate_at_a_time_with_fresh_allowance_before_each_mutation(tmp_path):
    items, journal = driver(tmp_path)
    fake = Fake()
    seen = []
    for _ in range(len(queue.PHASES) * 2):
        result = queue.step(items, journal, fake)
        seen.append((result["epoch"], result["phase"]))
        if result["epoch"] == 2:
            break
    assert [p for e, p in seen if e == 1] == list(queue.PHASES)
    assert seen[-1] == (2, "validate")
    assert mutations(fake) == [("create-dataset", 1), ("push-kernel", 1), ("submit", 1)]
    for index, call in enumerate(fake.calls):
        if call[0] in queue.MUTATIONS:
            assert fake.calls[index - 2 : index] == [("account", None), ("ledger", None)]


def test_completed_phase_is_never_repeated_and_intents_are_exclusive(tmp_path):
    items, journal = driver(tmp_path)
    fake = Fake()
    for _ in range(len(queue.PHASES)):
        queue.step(items, journal, fake)
    assert queue.plan(items, journal, list(fake.entries.values()), "outcome")["epoch"] == 2
    with pytest.raises(FileExistsError):
        queue.write_once(journal.path(1, "submit", "intent"), {})
    assert mutations(fake).count(("submit", 1)) == 1


def test_next_epoch_waits_for_actual_outcome_by_default(tmp_path):
    items, journal = driver(tmp_path)
    fake = Fake()
    fake.outcome = "pending_or_unknown"
    for _ in range(len(queue.PHASES) + 3):
        result = queue.step(items, journal, fake)
    assert (result["epoch"], result["phase"]) == (1, "reconcile-submission")
    assert mutations(fake).count(("submit", 1)) == 1


@pytest.mark.parametrize("remaining,today", [(0, 0), (5, 5)])
def test_account_limit_blocks_mutation_before_any_remote_call(tmp_path, remaining, today):
    items, journal = driver(tmp_path)
    fake = Fake(remaining, today)
    for _ in range(4):
        queue.step(items, journal, fake)
    with pytest.raises(ValueError, match="allowance"):
        queue.step(items, journal, fake)
    assert mutations(fake) == []
    assert not journal.path(1, "create-dataset", "intent").exists()


def test_local_ledger_count_enforces_five_daily_attempts(tmp_path):
    items, journal = driver(tmp_path)
    fake = Fake()
    for n in range(5):
        fake.entries[str(n)] = {
            "checkpoint_sha256": str(n),
            "state": "submitted",
            "attempted_at": queue.utcnow(),
        }
    for _ in range(4):
        queue.step(items, journal, fake)
    with pytest.raises(ValueError, match="allowance"):
        queue.step(items, journal, fake)


def test_crashed_submission_restarts_only_into_read_only_reconciliation(tmp_path):
    items, journal = driver(tmp_path)
    fake = Fake()
    fake.fail = {"submit"}
    for _ in range(8):
        queue.step(items, journal, fake)
    with pytest.raises(RuntimeError):
        queue.step(items, journal, fake)
    fake.fail = set()
    result = queue.step(items, journal, fake)
    assert result["phase"] == "reconcile-submission"
    assert mutations(fake).count(("submit", 1)) == 1


def test_any_uncertain_ledger_entry_blocks_all_mutations(tmp_path):
    items, journal = driver(tmp_path)
    fake = Fake()
    fake.entries["x"] = {"checkpoint_sha256": "x", "state": "push_uncertain"}
    for _ in range(4):
        queue.step(items, journal, fake)
    with pytest.raises(ValueError, match="Uncertain"):
        queue.step(items, journal, fake)
    assert mutations(fake) == []


def test_crashed_dataset_and_push_never_retry(tmp_path):
    items, journal = driver(tmp_path)
    fake = Fake()
    fake.fail = {"create-dataset"}
    for _ in range(4):
        queue.step(items, journal, fake)
    with pytest.raises(RuntimeError):
        queue.step(items, journal, fake)
    fake.fail = set()
    assert queue.step(items, journal, fake)["phase"] == "reconcile-dataset"
    fake.fail = {"push-kernel"}
    with pytest.raises(RuntimeError):
        queue.step(items, journal, fake)
    fake.fail = set()
    assert queue.step(items, journal, fake)["blocked"] == "push_requires_reconcile_push"
    assert mutations(fake) == [("create-dataset", 1), ("push-kernel", 1)]


@pytest.mark.parametrize("state", ["submit_intent", "submit_uncertain", "submitted"])
def test_existing_submission_entries_are_reconciled_never_resubmitted(tmp_path, state):
    items, journal = driver(tmp_path)
    fake = Fake()
    sha = items[0]["checkpoint_sha256"]
    fake.entries[sha] = {"checkpoint_sha256": sha, "state": state}
    result = queue.step(items, journal, fake)
    assert result["phase"] == "reconcile-submission"
    assert mutations(fake) == []


def test_public_execution_must_match_exact_engine_before_submission(tmp_path):
    items, journal = driver(tmp_path)
    fake = Fake()
    fake.engine = "legacy"
    for _ in range(8):
        result = queue.step(items, journal, fake)
    assert result["status"] == "await_or_blocked_public_execution"
    assert journal.result(1, "public-acceptance") is None
    entry = fake.entries[items[0]["checkpoint_sha256"]]
    with pytest.raises(ValueError, match="public acceptance"):
        queue.gate("submit", items[0], [entry], fake.account(), None)
    assert queue.step(items, journal, fake)["phase"] == "public-acceptance"
    assert mutations(fake) == [("create-dataset", 1), ("push-kernel", 1)]
