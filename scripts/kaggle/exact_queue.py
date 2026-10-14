"""Finite exact owner queue: one phase of one all-development epoch per invocation.

The driver never talks to Kaggle, Slurm or HF itself. Every phase runs through a
root-owned executor (JSON on stdin, one JSON object on stdout) that invokes the
original workflow/coordinator phase on Newton. Mutating phases need a fresh
authoritative allowance and the canonical Newton ledger first; an interrupted
mutation is never retried, only reconciled read-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from scripts.kaggle import reviewed_infer as reviewed  # noqa: E402

CHECKPOINT_BYTES = 125556557
PHASES = (
    "validate",
    "queue",
    "prepare",
    "prepare-transport",
    "create-dataset",
    "reconcile-dataset",
    "push-kernel",
    "public-acceptance",
    "submit",
    "reconcile-submission",
)
MUTATIONS = {"create-dataset", "push-kernel", "submit"}
READ_ONLY = {"validate", "reconcile-dataset", "public-acceptance", "reconcile-submission"}
RECONCILE = {"create-dataset": "reconcile-dataset", "submit": "reconcile-submission"}
SUBMISSION_STATES = {"submit_intent", "submit_uncertain", "submitted"}
PUSH_UNCERTAIN = {"push_intent", "push_uncertain"}
FINAL_OUTCOMES = {"scored", "failed", "complete_unscored"}


def candidates() -> list[dict]:
    return [
        {
            "epoch": epoch,
            "checkpoint_sha256": sha,
            "private_HF_revision": revision,
            "private_HF_prefix": f"{reviewed.ALL_DEVELOPMENT_PREFIX}/epoch-{epoch:03d}",
            "bytes": CHECKPOINT_BYTES,
        }
        for epoch, (sha, revision) in sorted(reviewed.ALL_DEVELOPMENT_EPOCHS.items())
    ]


def load_queue(path: Path) -> list[dict]:
    """Accept only the exact five owner entries; mutable labels never drive state."""
    queue = json.loads(Path(path).read_text())
    found = queue.get("candidates") if isinstance(queue, dict) else None
    keys = ("epoch", "checkpoint_sha256", "private_HF_revision", "private_HF_prefix", "bytes")
    if (
        not isinstance(found, list)
        or [{k: x.get(k) for k in keys} if isinstance(x, dict) else x for x in found]
        != candidates()
        or queue.get("no_diagnostic_model_or_Stanford_initialization") is not True
        or queue.get("same_native_FP32_deployment_engine_as_scored_source_e10_required") is not True
    ):
        raise ValueError("Queue differs from the exact owner all-development entries")
    return candidates()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_once(path: Path, value: dict) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


class Journal:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def path(self, epoch: int, phase: str, kind: str) -> Path:
        return self.root / f"epoch-{epoch:03d}" / f"{phase}.{kind}.json"

    def result(self, epoch: int, phase: str) -> dict | None:
        return read(self.path(epoch, phase, "result"))

    def interrupted(self, epoch: int, phase: str) -> bool:
        return self.path(epoch, phase, "intent").exists() and self.result(epoch, phase) is None

    def admission(self, epoch: int) -> Path:
        path = self.root / f"epoch-{epoch:03d}" / "fitting-admission.json"
        path.parent.mkdir(exist_ok=True, mode=0o700)
        if not path.exists():
            write_once(path, reviewed.all_development_admission(epoch))
        if read(path) != reviewed.all_development_admission(epoch):
            raise ValueError("Durable admission differs from the exact queue entry")
        return path


def load_workflow():
    spec = importlib.util.spec_from_file_location("kaggle_workflow", HERE / "workflow.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remaining(records: list[dict], account: dict, now: datetime | None = None) -> int:
    """The original workflow allowance applied to the canonical ledger snapshot."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE entries (sha TEXT PRIMARY KEY, record TEXT NOT NULL)")
    db.executemany(
        "INSERT INTO entries VALUES (?, ?)",
        [(x["checkpoint_sha256"], json.dumps(x)) for x in records],
    )
    return load_workflow().allowance(db, account, now)


def ledger_entry(records: list[dict], sha: str) -> dict | None:
    found = [x for x in records if x.get("checkpoint_sha256") == sha]
    if len(found) > 1:
        raise ValueError("Canonical ledger has duplicate checkpoint identities")
    return found[0] if found else None


def candidate_done(journal: Journal, records: list[dict], item: dict, advance: str) -> bool:
    entry = ledger_entry(records, item["checkpoint_sha256"])
    result = journal.result(item["epoch"], "reconcile-submission")
    if entry is None or entry.get("state") != "submitted" or result is None:
        return False
    return advance == "submitted" or entry.get("remote_outcome") in FINAL_OUTCOMES


def plan(queue: list[dict], journal: Journal, records: list[dict], advance: str) -> dict:
    """Next single phase. Only the earliest unfinished epoch can ever act."""
    for item in queue:
        if candidate_done(journal, records, item, advance):
            continue
        epoch, sha = item["epoch"], item["checkpoint_sha256"]
        entry = ledger_entry(records, sha)
        state = None if entry is None else entry.get("state")
        if state in SUBMISSION_STATES:
            return {"epoch": epoch, "phase": "reconcile-submission", "reason": state}
        if state in PUSH_UNCERTAIN:
            return {"epoch": epoch, "phase": None, "blocked": "push_requires_reconcile_push"}
        for phase in PHASES:
            if journal.interrupted(epoch, phase) and phase not in READ_ONLY:
                if phase in RECONCILE and journal.result(epoch, RECONCILE[phase]) is None:
                    return {"epoch": epoch, "phase": RECONCILE[phase], "reason": "interrupted"}
                if phase not in RECONCILE:
                    return {"epoch": epoch, "phase": None, "blocked": f"interrupted_{phase}"}
            done = journal.result(epoch, phase) is not None
            if phase == "push-kernel" and state == "pushed":
                done = True
            if phase == "create-dataset" and journal.result(epoch, "reconcile-dataset"):
                done = True
            if not done:
                return {"epoch": epoch, "phase": phase}
        return {"epoch": epoch, "phase": "reconcile-submission", "reason": "await_outcome"}
    return {"epoch": None, "phase": None, "complete": True}


def verify_public_execution(item: dict, prepared: dict, result: dict) -> dict:
    """The accepted source-e10 public execution check, before any code submission."""
    import csv
    import hashlib
    import io
    import math

    from rsna.schema import TARGETS

    receipt, metrics = result.get("receipt", {}), result.get("metrics", {})
    raw = result.get("submission_csv")
    uids = result.get("expected_public_uids")
    manifest = prepared.get("manifest_sha256")
    if not isinstance(raw, str) or not isinstance(uids, list) or len(uids) != 3:
        raise ValueError("Public execution output is incomplete")
    csv_sha = hashlib.sha256(raw.encode()).hexdigest()
    rows = list(csv.reader(io.StringIO(raw)))
    chunks = metrics.get("chunks")
    if (
        result.get("kernel_status") != "COMPLETE"
        or not isinstance(manifest, str)
        or len(manifest) != 64
        or receipt.get("status") != "offline_inference_complete"
        or receipt.get("checkpoint_sha256") != item["checkpoint_sha256"]
        or receipt.get("bundle_manifest_sha256") != manifest
        or receipt.get("studies") != 3
        or receipt.get("submission_sha256") != csv_sha
        or metrics.get("status") != "passed"
        or metrics.get("diagnostic_only") is not False
        or metrics.get("checkpoint_sha256") != item["checkpoint_sha256"]
        or metrics.get("checkpoint_epoch") != item["epoch"]
        or metrics.get("submission_sha256") != csv_sha
        or metrics.get("studies") != 3
        or metrics.get("resident_engine") != "two_device_exact_FP32_prefetch_v1"
        or not isinstance(chunks, list)
        or sum(x.get("fallback_studies", 0) for x in chunks) != 0
        or metrics.get("production_memory", {}).get("summed_lifetime_RSS_bytes", 2**62) > 29 * 2**30
        or not rows
        or rows[0] != ["StudyInstanceUID", *TARGETS]
        or [x[0] for x in rows[1:]] != uids
        or any(len(x) != 13 for x in rows)
    ):
        raise ValueError("Public execution differs from the accepted exact engine")
    try:
        values = [float(x) for row in rows[1:] for x in row[1:]]
    except ValueError:
        raise ValueError("Public execution produced non-numeric probabilities") from None
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
        raise ValueError("Public execution produced invalid probabilities")
    return {
        "status": "exact_private_offline_public_MRI_only_prefetch_verified",
        "phase": "public-acceptance",
        "checkpoint_sha256": item["checkpoint_sha256"],
        "manifest_sha256": manifest,
        "receipt": receipt,
        "metrics": metrics,
        "csv_sha256": csv_sha,
        "columns": 13,
        "studies": 3,
        "public_example_only": True,
        "csv_uploaded": False,
    }


def gate(
    phase: str, item: dict, records: list[dict], account: dict | None, accepted: dict | None
) -> None:
    entry = ledger_entry(records, item["checkpoint_sha256"])
    state = None if entry is None else entry.get("state")
    blocking = (SUBMISSION_STATES - {"submitted"}) | PUSH_UNCERTAIN
    if any(x.get("state") in blocking for x in records):
        raise ValueError("Uncertain canonical ledger entry requires read-only reconciliation")
    if phase in {"create-dataset", "push-kernel"} and state != "queued":
        raise ValueError("Only an exact queued, never-pushed ledger entry may be deployed")
    if phase == "submit" and (
        state != "pushed"
        or accepted is None
        or accepted.get("checkpoint_sha256") != item["checkpoint_sha256"]
        or accepted.get("receipt", {}).get("bundle_manifest_sha256")
        != (entry or {}).get("operation", {}).get("manifest_sha256")
    ):
        raise ValueError("Submission requires the exact pushed entry and public acceptance")
    if account is None or remaining(records, account) < 1:
        raise ValueError("No fresh authoritative competition allowance")


class Executor:
    def __init__(self, command: list[str]):
        self.command = command

    def __call__(self, request: dict) -> dict:
        done = subprocess.run(
            self.command, input=json.dumps(request), capture_output=True, text=True, check=False
        )
        result = json.loads(done.stdout) if done.stdout.strip() else {}
        if done.returncode != 0 or not isinstance(result, dict):
            raise RuntimeError(f"{request['phase']} executor failed; no automatic retry")
        return result


def request(phase: str, item: dict, journal: Journal) -> dict:
    return {
        "phase": phase,
        "candidate": item,
        "fitting_admission": str(journal.admission(item["epoch"])),
    }


def step(queue: list[dict], journal: Journal, run, advance: str = "outcome") -> dict:
    records = run({"phase": "ledger"})["entries"]
    decision = plan(queue, journal, records, advance)
    phase = decision.get("phase")
    if phase is None:
        return decision
    item = next(x for x in queue if x["epoch"] == decision["epoch"])
    epoch = item["epoch"]
    if phase in MUTATIONS:
        account = run({"phase": "account"})
        accepted = journal.result(epoch, "public-acceptance")
        gate(phase, item, run({"phase": "ledger"})["entries"], account, accepted)
    req = request(phase, item, journal)
    intent = journal.path(epoch, phase, "intent")
    if phase in READ_ONLY:
        intent = intent.with_name(
            f"{phase}.attempt-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}.json"
        )
    write_once(intent, {**req, "at": utcnow()})
    result = run(req)
    if result.get("checkpoint_sha256") != item["checkpoint_sha256"] or result.get("phase") != phase:
        raise ValueError("Executor result identity differs; keep blocked")
    if phase == "public-acceptance":
        prepared = journal.result(epoch, "prepare") or {}
        try:
            result = verify_public_execution(item, prepared, result)
        except ValueError as exc:
            return {**decision, "status": "await_or_blocked_public_execution", "error": str(exc)}
    if phase == "reconcile-submission":
        path = journal.path(epoch, phase, "result")
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        os.replace(tmp, path)
    else:
        write_once(journal.path(epoch, phase, "result"), {**result, "at": utcnow()})
    return {**decision, "status": "phase_complete", "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "step"])
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--executor", nargs="+", required=True)
    parser.add_argument("--advance-after", choices=["outcome", "submitted"], default="outcome")
    args = parser.parse_args()
    queue, journal, run = load_queue(args.queue), Journal(args.journal), Executor(args.executor)
    if args.action == "plan":
        result = plan(queue, journal, run({"phase": "ledger"})["entries"], args.advance_after)
    else:
        result = step(queue, journal, run, args.advance_after)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
