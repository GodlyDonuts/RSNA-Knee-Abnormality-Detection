"""Finite exact Kaggle comparison queue for completed all-development epochs.

Only entries of the committed queue are admitted. Each entry binds one immutable
checkpoint of original job 908454 to its private full-download receipt; the
admission is derived from the entry, never from mutable labels or guessed
metadata. The driver advances one candidate at a time through the original
coordinator phases. Every cloud mutation is preceded by a fresh allowance and
canonical ledger read and by an exclusive durable intent. Any existing intent,
uncertain, pushed or submitted state forces read-only reconciliation; nothing is
ever resubmitted or automatically retried. The public three-row CSV is never
uploaded: submissions are competition code submissions of the exact notebook.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
QUEUE_ID = "daily-kaggle-comparison-queue-20261014-v1"
QUEUE_FILE = Path("initiative/audits") / QUEUE_ID / "queue.json"
QUEUE_STATUS = "owner_authorized_exact_daily_kaggle_queue"
RECEIPT_STATUS = "private_final_public_source_epoch_full_download_size_SHA_verified"
RECEIPT_MEMBERS = {
    "checkpoint.pt",
    "snapshot.json",
    "source-and-input-provenance.json",
    "DINOv3-LICENSE.md",
    "manifest.json",
}
ACCOUNT = "godlydonuts"
ORIGINAL_JOB = 908454
ENGINE = "two_device_exact_FP32_prefetch_v1"
MEMORY_CAP = 29 * 2**30
SHA = re.compile(r"[0-9a-f]{64}")
QUEUE_KEYS = {
    "schema_version",
    "queue_id",
    "status",
    "original_job",
    "scientific_source_git",
    "training_binding_sha256",
    "account",
    "public_csv_upload",
    "automatic_mutation_retry",
    "entries",
}
ENTRY_KEYS = {
    "epoch",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "optimizer_steps",
    "training_code_sha256",
    "private_publication",
    "verified_receipt",
    "verified_receipt_sha256",
    "dataset_slug",
    "kernel_slug",
}
CLOUD_MUTATIONS = ("create-dataset", "push-kernel", "submit")
MUTATIONS = ("prepare", *CLOUD_MUTATIONS)
LIVE_KERNEL = {"RUNNING", "QUEUED"}


@lru_cache(maxsize=1)
def reviewed():
    spec = importlib.util.spec_from_file_location(
        "rsna_reviewed_infer", Path(__file__).with_name("reviewed_infer.py")
    )
    if spec is None or spec.loader is None:
        raise ValueError("Reviewed inference adapter is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def slugs(epoch: int, sha: str) -> tuple[str, str]:
    """Dedicated never-used identities; the kernel title reproduces its slug."""
    return (
        f"{ACCOUNT}/rsna-alldev-e{epoch:03d}-{sha[:12]}",
        f"{ACCOUNT}/rsna-alldev-offline-e{epoch:03d}-{sha[:12]}",
    )


def small_json(path: Path, limit: int = 256 * 1024) -> Any:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Queue evidence must be a small regular file")
    return json.loads(path.read_text())


def validate_entry(entry: dict, repo: Path | None = ROOT) -> None:
    r = reviewed()
    epoch = entry.get("epoch")
    sha = entry.get("checkpoint_sha256")
    publication = entry.get("private_publication")
    if (
        not isinstance(entry, dict)
        or set(entry) != ENTRY_KEYS
        or type(epoch) is not int
        or epoch not in range(1, 11)
        or not isinstance(sha, str)
        or not SHA.fullmatch(sha)
        or type(entry["checkpoint_bytes"]) is not int
        or entry["checkpoint_bytes"] <= 0
        or entry["optimizer_steps"] != r.ALL_DEVELOPMENT_UPDATES * epoch
        or not SHA.fullmatch(str(entry["training_code_sha256"]))
        or not SHA.fullmatch(str(entry["verified_receipt_sha256"]))
        or not isinstance(publication, dict)
        or set(publication) != {"repo_id", "revision", "prefix", "file_count"}
        or publication["repo_id"] != "GodlyDonuts/RSNA-Data"
        or publication["prefix"] != f"{r.ALL_DEVELOPMENT_PREFIX}/epoch-{epoch:03d}"
        or publication["file_count"] != len(RECEIPT_MEMBERS)
        or not re.fullmatch(r"[0-9a-f]{40}", str(publication["revision"]))
        or (entry["dataset_slug"], entry["kernel_slug"]) != slugs(epoch, sha)
    ):
        raise ValueError("Queue entry differs from the exact admitted entry format")
    relative = Path(entry["verified_receipt"])
    if repo is None:
        # Owning-node phases receive a root-validated queue pinned by its hash.
        return
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2]
        != (
            "initiative",
            "audits",
        )
    ):
        raise ValueError("Queue receipt must be a committed initiative audit")
    path = repo / relative
    if file_sha(path) != entry["verified_receipt_sha256"]:
        raise ValueError("Committed private verification receipt changed")
    receipt = small_json(path)
    files = receipt.get("files", {})
    if (
        receipt.get("status") != RECEIPT_STATUS
        or receipt.get("epoch") != epoch
        or receipt.get("checkpoint_sha256") != sha
        or receipt.get("revision") != publication["revision"]
        or receipt.get("prefix") != publication["prefix"]
        or receipt.get("file_count") != len(RECEIPT_MEMBERS)
        or set(files) != RECEIPT_MEMBERS
        or files["checkpoint.pt"] != {"bytes": entry["checkpoint_bytes"], "sha256": sha}
    ):
        raise ValueError("Queue entry is not the exact private full-download verified epoch")


def load_queue(path: Path = ROOT / QUEUE_FILE, repo: Path = ROOT) -> dict:
    return check_queue(small_json(path), repo)


def check_queue(queue: Any, repo: Path | None = ROOT) -> dict:
    r = reviewed()
    if (
        not isinstance(queue, dict)
        or set(queue) != QUEUE_KEYS
        or queue["schema_version"] != 1
        or queue["queue_id"] != QUEUE_ID
        or queue["status"] != QUEUE_STATUS
        or queue["original_job"] != ORIGINAL_JOB
        or queue["scientific_source_git"] != r.ALL_DEVELOPMENT_SOURCE
        or queue["training_binding_sha256"] != r.ALL_DEVELOPMENT_BINDING
        or queue["account"] != ACCOUNT
        or queue["public_csv_upload"] is not False
        or queue["automatic_mutation_retry"] is not False
        or not isinstance(queue["entries"], list)
        or not 1 <= len(queue["entries"]) <= 10
    ):
        raise ValueError("Queue differs from the exact owner-admitted queue format")
    for entry in queue["entries"]:
        validate_entry(entry, repo)
    epochs = [x["epoch"] for x in queue["entries"]]
    shas = [x["checkpoint_sha256"] for x in queue["entries"]]
    if epochs != sorted(set(epochs)) or len(set(shas)) != len(shas):
        raise ValueError("Queue entries must be unique and in strictly increasing epoch order")
    return queue


def queue_entry(queue: dict, epoch: int) -> dict:
    found = [x for x in queue["entries"] if x["epoch"] == epoch]
    if len(found) != 1:
        raise ValueError("Epoch is not an exact queue entry")
    return found[0]


def admission_for(queue: dict, entry: dict) -> dict:
    """The only admission an entry can produce; no field is supplied by the caller."""
    r = reviewed()
    if entry not in queue["entries"]:
        raise ValueError("Admission requires an exact queue entry")
    return {
        "schema_version": 1,
        "status": r.ALL_DEVELOPMENT_STATUS,
        "role": r.ALL_DEVELOPMENT_ROLE,
        "queue_id": queue["queue_id"],
        "scientific_source_git": r.ALL_DEVELOPMENT_SOURCE,
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "checkpoint_bytes": entry["checkpoint_bytes"],
        "training_binding_sha256": r.ALL_DEVELOPMENT_BINDING,
        "training_code_sha256": entry["training_code_sha256"],
        "epoch": entry["epoch"],
        "optimizer_steps": entry["optimizer_steps"],
        "seed": 20260922,
        "training_studies": 3403,
        "outer_studies": 0,
        "fit_scope": "all_development",
        "fitting_supervision": dict(r.ALL_DEVELOPMENT_SUPERVISION),
        "official_base": dict(r.OFFICIAL_BASE),
        "private_publication": dict(entry["private_publication"]),
        "MRI_only_inference": True,
        "original_checkpoint_and_numerical_source_unchanged": True,
    }


def admission_bytes(queue: dict, entry: dict) -> bytes:
    return (json.dumps(admission_for(queue, entry), sort_keys=True, indent=2) + "\n").encode()


def check_snapshot(entry: dict, sidecar: dict) -> None:
    """Cheap sidecar binding; workflow.validate_snapshot then rechecks full tensors."""
    r = reviewed()
    if (
        sidecar.get("status") != "scheduled_epoch_snapshot"
        or sidecar.get("epoch") != entry["epoch"]
        or sidecar.get("checkpoint_file") != f"epoch-{entry['epoch']:03d}.pt"
        or sidecar.get("checkpoint_sha256") != entry["checkpoint_sha256"]
        or sidecar.get("checkpoint_bytes") != entry["checkpoint_bytes"]
        or sidecar.get("optimizer_steps") != entry["optimizer_steps"]
        or sidecar.get("binding_sha256") != r.ALL_DEVELOPMENT_BINDING
        or sidecar.get("diagnostic_only") is not False
        or sidecar.get("submission_eligible") is not True
    ):
        raise ValueError("Actual snapshot differs from its exact queue entry")


def check_prepared(queue: dict, entry: dict, prepared: dict) -> dict:
    op = prepared.get("operation", {})
    if (
        prepared.get("competition_submission_created") is not False
        or prepared.get("original_source_and_weights_unchanged") is not True
        or op.get("checkpoint_sha256") != entry["checkpoint_sha256"]
        or op.get("epoch") != entry["epoch"]
        or op.get("dataset_slug") != entry["dataset_slug"]
        or op.get("kernel_slug") != entry["kernel_slug"]
        or op.get("fitting_admission_sha256")
        != hashlib.sha256(admission_bytes(queue, entry)).hexdigest()
        or op.get("inference_execution_receipts", {}).get("engine") != ENGINE
        or op.get("inference_execution_receipts", {}).get(
            "new_checkpoint_public_acceptance_required"
        )
        is not True
    ):
        raise ValueError("Prepared operation does not bind the exact queue admission")
    return op


def accept_public(
    queue: dict,
    entry: dict,
    operation: dict,
    receipt: dict,
    metrics: dict,
    raw: bytes,
    expected_uids: list[str],
    targets: list[str] | None = None,
) -> dict:
    """Mandatory exact public execution acceptance before the one code submission."""
    if targets is None:
        from rsna.schema import TARGETS

        targets = list(TARGETS)
    sha = entry["checkpoint_sha256"]
    if (
        operation.get("checkpoint_sha256") != sha
        or operation.get("kernel_slug") != entry["kernel_slug"]
        or operation.get("fitting_admission_sha256")
        != hashlib.sha256(admission_bytes(queue, entry)).hexdigest()
        or receipt.get("status") != "offline_inference_complete"
        or receipt.get("checkpoint_sha256") != sha
        or receipt.get("bundle_manifest_sha256") != operation.get("manifest_sha256")
        or metrics.get("status") != "passed"
        or metrics.get("studies") != len(expected_uids)
        or metrics.get("resident_engine") != ENGINE
        or not metrics.get("chunks")
        or sum(x.get("fallback_studies", 0) for x in metrics["chunks"]) != 0
        or metrics.get("production_memory", {}).get("summed_lifetime_RSS_bytes", math.inf)
        > MEMORY_CAP
    ):
        raise ValueError("Public execution is not the exact admitted MRI-only notebook")
    digest = hashlib.sha256(raw).hexdigest()
    if not digest == receipt.get("submission_sha256") == metrics.get("submission_sha256"):
        raise ValueError("Public CSV bytes differ from the notebook receipts")
    rows = list(csv.reader(io.StringIO(raw.decode())))
    if (
        not rows
        or rows[0] != ["StudyInstanceUID", *targets]
        or [x[0] for x in rows[1:]] != expected_uids
        or any(len(x) != len(targets) + 1 for x in rows)
    ):
        raise ValueError("Public CSV columns or study order differ")
    for row in rows[1:]:
        for value in row[1:]:
            number = float(value)
            if not (math.isfinite(number) and 0 <= number <= 1):
                raise ValueError("Public CSV probabilities must be finite in [0, 1]")
    return {
        "status": "exact_private_offline_public_MRI_only_prefetch_verified",
        "checkpoint_sha256": sha,
        "epoch": entry["epoch"],
        "kernel_slug": entry["kernel_slug"],
        "bundle_manifest_sha256": operation["manifest_sha256"],
        "receipt_sha256": object_sha(receipt),
        "metrics_sha256": object_sha(metrics),
        "csv_sha256": digest,
        "columns": len(targets) + 1,
        "studies": len(expected_uids),
        "public_example_only": True,
        "public_csv_uploaded": False,
    }


@dataclass(frozen=True)
class Action:
    name: str
    epoch: int | None = None
    sha: str | None = None
    read_only: bool = True
    reason: str = ""


class Journal:
    """Driver-local exclusive intents/results, one directory per queue epoch."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, epoch: int) -> Path:
        return self.root / f"epoch-{epoch:03d}"

    def intent(self, epoch: int, phase: str) -> dict | None:
        path = self._dir(epoch) / f"{phase}.intent.json"
        return json.loads(path.read_text()) if path.exists() else None

    def result(self, epoch: int, phase: str) -> dict | None:
        path = self._dir(epoch) / f"{phase}.result.json"
        return json.loads(path.read_text()) if path.exists() else None

    def _exclusive(self, path: Path, value: dict) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_intent(self, epoch: int, phase: str, value: dict) -> None:
        self._exclusive(self._dir(epoch) / f"{phase}.intent.json", value)

    def write_result(self, epoch: int, phase: str, value: dict) -> None:
        self._exclusive(self._dir(epoch) / f"{phase}.result.json", value)

    def observe(self, epoch: int, phase: str, value: dict) -> None:
        directory = self._dir(epoch) / "observations"
        index = len(list(directory.glob("*.json"))) if directory.exists() else 0
        self._exclusive(directory / f"{index:04d}-{phase}.json", value)


def plan(queue: dict, records: dict[str, dict], journal: Journal) -> Action:
    """Pure next step from the canonical ledger and driver journal; one candidate at a time."""
    shas = {x["checkpoint_sha256"] for x in queue["entries"]}
    for sha, record in records.items():
        if sha not in shas and record.get("state") in {"submit_intent", "submit_uncertain"}:
            return Action("halt", sha=sha, reason="foreign uncertain submission; reconcile first")
    for entry in queue["entries"]:
        epoch, sha = entry["epoch"], entry["checkpoint_sha256"]
        record = records.get(sha)

        def act(name, read_only=True, reason=""):
            return Action(name, epoch, sha, read_only, reason)

        if record is None:
            if journal.intent(epoch, "push-kernel") is not None:
                return act("halt", reason="push intent without ledger entry; inspect read-only")
            if journal.result(epoch, "prepare") is None:
                if journal.intent(epoch, "prepare") is not None:
                    return act("reconcile-prepare")
                return act("prepare", read_only=False)
            if journal.intent(epoch, "create-dataset") is None:
                return act("create-dataset", read_only=False)
            if journal.result(epoch, "reconcile-dataset") is None:
                return act("reconcile-dataset")
            return act("push-kernel", read_only=False)
        state = record.get("state")
        if state in {"push_intent", "push_uncertain"}:
            return act("reconcile-push")
        if state in {"submit_intent", "submit_uncertain"}:
            return act("reconcile-submission")
        if state == "queued":
            if journal.intent(epoch, "push-kernel") is not None:
                return act("halt", reason="push intent with queued ledger entry; inspect read-only")
            if journal.result(epoch, "reconcile-dataset") is None:
                return act("halt", reason="queued without this driver's verified Dataset")
            return act("push-kernel", read_only=False)
        if state == "pushed":
            if journal.intent(epoch, "submit") is not None:
                return act("halt", reason="submit intent with pushed ledger entry; inspect")
            if journal.result(epoch, "accept-public") is None:
                return act("observe")
            return act("submit", read_only=False)
        if state == "submitted":
            outcome = record.get("remote_outcome")
            if outcome is None:
                return act("reconcile-outcome")
            if outcome == "failed":
                return act("halt", reason="submitted entry failed; review before next candidate")
            continue
        return act("halt", reason="unknown ledger state")
    return Action("complete")


class CommandExecutor:
    """Bind phases to root's own transport (SSH/Slurm stdin token); JSON in/out only."""

    def __init__(self, command: list[str], queue: dict):
        self.command = command
        self.queue = queue

    def _call(self, phase: str, entry: dict | None = None, **payload) -> Any:
        request = {
            "phase": phase,
            "queue": self.queue,
            "queue_sha256": object_sha(self.queue),
            **payload,
        }
        if entry is not None:
            request["epoch"] = entry["epoch"]
            request["checkpoint_sha256"] = entry["checkpoint_sha256"]
        x = subprocess.run(
            self.command, input=json.dumps(request), capture_output=True, text=True, timeout=3600
        )
        if x.returncode != 0:
            raise RuntimeError(f"Phase {phase} failed; reconcile read-only, never retry")
        return json.loads(x.stdout)

    def ledger_records(self) -> dict:
        return self._call("ledger")

    def allowance(self, entry: dict) -> int:
        return int(self._call("allowance", entry)["allowance"])

    def prepare(self, entry: dict, admission: bytes) -> dict:
        return self._call("prepare", entry, admission=base64.b64encode(admission).decode())

    def __getattr__(self, name: str) -> Callable[..., Any]:
        phase = name.replace("_", "-")
        if phase not in {
            "inspect-prepared",
            "create-dataset",
            "reconcile-dataset",
            "push-kernel",
            "reconcile-push",
            "status",
            "collect-public",
            "submit",
            "reconcile-submission",
        }:
            raise AttributeError(name)
        return lambda entry, **kw: self._call(phase, entry, **kw)


class Driver:
    def __init__(
        self,
        queue: dict,
        executor: Any,
        journal: Journal,
        expected_uids: list[str],
        *,
        poll_seconds: float = 60,
        deadline_seconds: float = 3 * 3600,
        max_submissions: int = 5,
        targets: list[str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.queue, self.ex, self.journal = queue, executor, journal
        self.expected_uids, self.targets = expected_uids, targets
        self.poll, self.sleep, self.clock = poll_seconds, sleep, clock
        self.deadline = clock() + deadline_seconds
        self.max_submissions = max_submissions
        self.submissions = 0

    def _entry(self, action: Action) -> dict:
        assert action.epoch is not None
        return queue_entry(self.queue, action.epoch)

    def step(self) -> dict:
        records = self.ex.ledger_records()
        action = plan(self.queue, records, self.journal)
        if action.name in {"halt", "complete"}:
            return {"status": action.name, "epoch": action.epoch, "reason": action.reason}
        entry = self._entry(action)
        epoch = entry["epoch"]
        if not action.read_only:
            if action.name == "submit" and self.submissions >= self.max_submissions:
                return {"status": "submission_cap_reached", "epoch": epoch}
            allowance = None
            if action.name in CLOUD_MUTATIONS:
                allowance = self.ex.allowance(entry)
                fresh = self.ex.ledger_records()
                if plan(self.queue, fresh, self.journal) != action:
                    raise ValueError("Canonical ledger changed between fresh observations")
                if allowance < 1:
                    return {"status": "allowance_exhausted", "epoch": epoch}
                records = fresh
            self.journal.write_intent(
                epoch,
                action.name,
                {
                    "phase": action.name,
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                    "fresh_allowance": allowance,
                    "ledger_sha256": object_sha(records),
                    "automatic_retry": False,
                },
            )
            result = self._mutate(action.name, entry)
            self.journal.write_result(epoch, action.name, result)
            if action.name == "submit":
                self.submissions += 1
            return {"status": action.name, "epoch": epoch}
        return self._read_only(action.name, entry)

    def _mutate(self, phase: str, entry: dict) -> dict:
        if phase == "prepare":
            prepared = self.ex.prepare(entry, admission_bytes(self.queue, entry))
            check_prepared(self.queue, entry, prepared)
            return prepared
        if phase == "create-dataset":
            return self.ex.create_dataset(entry)
        if phase == "push-kernel":
            record = self.ex.push_kernel(entry)
            if record.get("state") != "pushed" or record.get("kernel_version") != 1:
                raise ValueError("Kernel push needs read-only reconciliation")
            return record
        acceptance = self.journal.result(entry["epoch"], "accept-public")
        record = self.ex.submit(entry, acceptance=acceptance)
        if record.get("state") != "submitted" or not record.get("submission_id"):
            raise ValueError("Submission needs read-only reconciliation")
        return record

    def _read_only(self, phase: str, entry: dict) -> dict:
        epoch = entry["epoch"]
        if phase == "reconcile-prepare":
            prepared = self.ex.inspect_prepared(entry)
            if not prepared:
                return {"status": "halt", "epoch": epoch, "reason": "prepare uncertain"}
            check_prepared(self.queue, entry, prepared)
            self.journal.write_result(epoch, "prepare", prepared)
            return {"status": phase, "epoch": epoch}
        if phase == "reconcile-dataset":
            proof = self.ex.reconcile_dataset(entry)
            if proof.get("status") != "all_payload_bytes_verified":
                return {"status": "halt", "epoch": epoch, "reason": "Dataset not verified"}
            self.journal.write_result(epoch, phase, proof)
            return {"status": phase, "epoch": epoch}
        if phase == "observe":
            status = self.ex.status(entry)
            self.journal.observe(epoch, phase, status)
            kernel = status.get("kernel_status", {}).get("status")
            if kernel in LIVE_KERNEL:
                if self.clock() + self.poll > self.deadline:
                    return {"status": "deadline", "epoch": epoch}
                self.sleep(self.poll)
                return {"status": "waiting", "epoch": epoch}
            if kernel != "COMPLETE":
                return {"status": "halt", "epoch": epoch, "reason": "terminal notebook failure"}
            out = self.ex.collect_public(entry)
            acceptance = accept_public(
                self.queue,
                entry,
                out["operation"],
                out["receipt"],
                out["metrics"],
                base64.b64decode(out["csv"]),
                self.expected_uids,
                self.targets,
            )
            self.journal.write_result(epoch, "accept-public", acceptance)
            return {"status": "accepted", "epoch": epoch}
        method = {
            "reconcile-push": self.ex.reconcile_push,
            "reconcile-submission": self.ex.reconcile_submission,
            "reconcile-outcome": self.ex.reconcile_submission,
        }[phase]
        self.journal.observe(epoch, phase, method(entry))
        return {"status": phase, "epoch": epoch}

    def run(self, max_steps: int = 500) -> list[dict]:
        history = []
        for _ in range(max_steps):
            result = self.step()
            history.append(result)
            if result["status"] in {
                "halt",
                "complete",
                "allowance_exhausted",
                "submission_cap_reached",
                "deadline",
            }:
                break
            if (
                result["status"] == "reconcile-outcome"
                and self.ex.ledger_records()
                .get(queue_entry(self.queue, result["epoch"])["checkpoint_sha256"], {})
                .get("remote_outcome")
                is None
            ):
                # The ledger could not establish an outcome; stop, do not spin.
                break
        return history


# ---------------------------------------------------------------------------
# Owning-node phases. Root's transport adds the access token to the request on
# stdin; nothing here reads credentials from disk or the command line.

OUTPUT_HOSTS = {"storage.googleapis.com", "api.kaggle.com", "www.kaggle.com"}


def _node_context(request: dict, config: dict):
    if request.get("queue_sha256") != object_sha(request.get("queue")):
        raise ValueError("Root-validated queue hash differs")
    queue = check_queue(request["queue"], repo=None)
    entry = None
    if "epoch" in request:
        entry = queue_entry(queue, request["epoch"])
        if entry["checkpoint_sha256"] != request.get("checkpoint_sha256"):
            raise ValueError("Requested checkpoint differs from its queue entry")
    if os.uname().nodename != config["node"] or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Queue phases run only on the owning node without a GPU allocation")
    return queue, entry


def _work(config: dict, entry: dict) -> Path:
    return Path(config["work_root"]) / f"epoch-{entry['epoch']:03d}"


def _coordinator(R: Path):
    P = R / "projection"
    prepared = json.loads((R / "prepared.json").read_text())
    sys.path.insert(0, str(P / "code"))
    spec = importlib.util.spec_from_file_location("c", P / "code/scripts/kaggle/coordinator.py")
    assert spec is not None and spec.loader is not None
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    w, _ = c.load_workflow(P / "code", P / "projection.json", prepared["projection_sha"])
    return c, w, prepared


def _remote(c, w, token: str):
    sys.stdin = io.StringIO(json.dumps({"token": token}))
    remote, _ = c.remote_from_stdin(w)
    return remote


def _csv_output(remote, slug: str, version: int) -> bytes:
    from urllib.parse import urljoin, urlparse

    import requests

    owner, kernel = slug.split("/")
    remote.session.cookies.clear()
    response = remote.session.post(
        "https://api.kaggle.com/v1/kernels.KernelsApiService/DownloadKernelOutput",
        json={
            "ownerSlug": owner,
            "kernelSlug": kernel,
            "versionNumber": version,
            "filePath": "submission.csv",
        },
        timeout=(10, 60),
        allow_redirects=False,
        stream=True,
    )
    for _ in range(5):
        with response:
            if response.status_code == 200:
                raw = bytearray()
                for block in response.iter_content(65536):
                    raw.extend(block)
                    if len(raw) > 2**20:
                        raise ValueError("Public CSV exceeds its bound")
                return bytes(raw)
            if response.status_code not in {301, 302, 303, 307, 308}:
                raise ValueError("Public CSV is unavailable")
            url = urljoin(response.url, response.headers.get("Location", ""))
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not (
            host in OUTPUT_HOSTS
            or host.endswith(".googleusercontent.com")
            or host == "www.kaggleusercontent.com"
        ):
            raise ValueError("Unexpected public output redirect")
        session = remote.session if host == "api.kaggle.com" else requests
        response = session.get(url, timeout=(10, 60), allow_redirects=False, stream=True)
    raise ValueError("Output redirect budget exhausted")


def _node_prepare(request: dict, config: dict, queue: dict, entry: dict) -> dict:
    import shutil
    import tarfile

    R = _work(config, entry)
    admission = base64.b64decode(request["admission"])
    if admission != admission_bytes(queue, entry):
        raise ValueError("Admission differs from the exact queue entry")
    snapshot = Path(config["snapshot_dir"]) / f"epoch-{entry['epoch']:03d}.json"
    check_snapshot(entry, json.loads(snapshot.read_text()))
    if R.exists() or shutil.disk_usage(R.parent).free < config["min_free_bytes"]:
        raise ValueError("Existing attempt or unfunded node storage; reconcile read-only")
    v = json.loads(Path(config["prepare_envelope"]).read_text())
    raw = base64.b64decode(v["tar"])
    manifest = base64.b64decode(v["manifest"])
    if (
        hashlib.sha256(raw).hexdigest() != v["tar_sha"]
        or hashlib.sha256(manifest).hexdigest() != v["projection_sha"]
    ):
        raise ValueError("Root-reviewed projection differs")
    R.mkdir(mode=0o700, parents=True)
    P = R / "projection"
    P.mkdir()
    (P / "projection.json").write_bytes(manifest)
    code = P / "code"
    code.mkdir()
    files = json.loads(manifest)["files"]
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        if {x.name for x in archive.getmembers()} != set(files):
            raise ValueError("Projection archive membership differs")
        for member in archive.getmembers():
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError("Projection member is not a confined regular file")
            handle = archive.extractfile(member)
            assert handle is not None
            data = handle.read()
            record = files[member.name]
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Projection member bytes differ")
            target = code / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(record["mode"])
    (R / "encoder-spec.json").write_text(json.dumps(v["encoder"]))
    (R / "fitting-admission.json").write_bytes(admission)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(code),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    py = config["python"]
    # fmt: off
    command = [
        py, "-B", str(code / "scripts/kaggle/workflow.py"), "prepare",
        "--snapshot", str(snapshot), "--encoder-spec", str(R / "encoder-spec.json"),
        "--wheelhouse", config["wheelhouse"], "--shared-wheelhouse", config["shared_wheelhouse"],
        "--license-file", config["license_file"], "--license-source", config["license_source"],
        "--license-sha256", config["license_sha256"], "--output", str(R / "candidate"),
        "--dataset-slug", entry["dataset_slug"], "--kernel-slug", entry["kernel_slug"],
        "--runtime-image", config["runtime_image"], "--inference-engine", "resident_two_gpu_fp32",
        "--fitting-admission", str(R / "fitting-admission.json"),
    ]
    # fmt: on
    x = subprocess.run(command, cwd=code, env=env, capture_output=True, text=True, timeout=900)
    (R / "prepare-result.json").write_text(json.dumps({"returncode": x.returncode}))
    if x.returncode != 0:
        raise ValueError("Exact workflow preparation refused this checkpoint")
    opfile = R / "candidate/operation.json"
    op = json.loads(opfile.read_text())
    helpers = v["helpers"]
    prefix = {
        "EXPECTED_MANIFEST_SHA256": op["manifest_sha256"],
        "INFERENCE_FP32_PREFETCH": True,
        "INFERENCE_PREFETCH_SOURCES": helpers,
        "INFERENCE_PREFETCH_HASHES": {
            n: hashlib.sha256(t.encode()).hexdigest() for n, t in helpers.items()
        },
        "ATTACHED_PAYLOAD_ROOTS": {
            "bundle-manifest.json": {
                "kind": "datasets",
                "slug": op["dataset_slug"],
                "version": 1,
                "relative": "bundle-manifest.json",
            },
            op["shared_wheelhouse"]["archive_file"]: {
                "kind": "datasets",
                "slug": op["shared_wheelhouse"]["dataset_slug"],
                "version": 1,
                "relative": op["shared_wheelhouse"]["archive_file"],
            },
        },
    }
    source = "\n".join(f"{k} = {val!r}" for k, val in prefix.items()) + "\n"
    source += (code / "scripts/kaggle/notebook_entry.py").read_text()
    compile(source, "offline_prefetch.py", "exec")
    if len(source.encode()) >= 256 * 1024:
        raise ValueError("Notebook source exceeds its bound")
    (R / "candidate/kernel/run.py").write_text(source)
    op["kernel_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    op["inference_execution_receipts"] = {
        "engine": ENGINE,
        "reference_fullscale_receipt_sha256": v["fullscale_sha"],
        "reference_memory_receipt_sha256": v["memory_sha"],
        "new_checkpoint_public_acceptance_required": True,
    }
    opfile.write_text(json.dumps(op, sort_keys=True, indent=2) + "\n")
    workflow = str(code / "scripts/kaggle/workflow.py")
    x = subprocess.run(
        [py, "-B", workflow, "validate", str(opfile)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if x.returncode != 0:
        raise ValueError("Prepared notebook operation failed exact validation")
    # fmt: off
    args = [
        py, "-B", str(code / "scripts/kaggle/coordinator.py"), "prepare-transport",
        "--projection", str(code), "--projection-manifest", str(P / "projection.json"),
        "--projection-sha256", v["projection_sha"], "--snapshot", str(snapshot),
        "--operation", str(opfile),
        "--node-local-upload-root", f"/tmp/rsna-kaggle-upload-{entry['checkpoint_sha256'][:12]}",
    ]
    # fmt: on
    x = subprocess.run(args, env=env, capture_output=True, text=True, timeout=300)
    if x.returncode != 0:
        raise ValueError("Coordinator transport preparation refused")
    out = {
        "status": "exact_queue_epoch_offline_FP32_prefetch_bundle_prepared_not_published",
        "CPU_job": os.environ.get("SLURM_JOB_ID"),
        "operation": op,
        "projection_sha": v["projection_sha"],
        "snapshot": json.loads(snapshot.read_text()),
        "snapshot_path": str(snapshot),
        "original_source_and_weights_unchanged": True,
        "competition_submission_created": False,
    }
    check_prepared(queue, entry, out)
    with (R / "prepared.json").open("x") as handle:
        json.dump(out, handle, indent=2)
    return out


def _collect(remote, w, op: dict, entry_record: dict) -> dict:
    version = entry_record["kernel_version"]
    return {
        "operation": op,
        "receipt": remote.output_json(op["kernel_slug"], version, "kaggle-inference-receipt.json"),
        "metrics": remote.output_json(op["kernel_slug"], version, "submission.metrics.json"),
        "csv": base64.b64encode(_csv_output(remote, op["kernel_slug"], version)).decode(),
    }


def node_phase(request: dict, config: dict) -> Any:
    queue, entry = _node_context(request, config)
    phase = request["phase"]
    base = Path(config["kaggle_base"])
    ledger_path = base / "account-ledger.sqlite3"
    if phase == "ledger":
        import sqlite3

        db = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
        try:
            rows = db.execute("SELECT sha, record FROM entries").fetchall()
        finally:
            db.close()
        return {sha: json.loads(record) for sha, record in rows}
    if entry is None:
        raise ValueError("Unsupported queue-wide phase")
    R = _work(config, entry)
    if phase == "prepare":
        return _node_prepare(request, config, queue, entry)
    if phase == "inspect-prepared":
        path = R / "prepared.json"
        return json.loads(path.read_text()) if path.exists() else {}
    # Cloud phases use the exact projection prepared for this candidate.
    c, w, prepared = _coordinator(R)
    token = request["token"]
    if phase == "allowance":
        remote = _remote(c, w, token)
        with c.single_coordinator(base):
            with w.ledger(ledger_path, remote.username) as db:
                account = remote.account()
                return {"allowance": w.allowance(db, account), "account": account}
    sha = entry["checkpoint_sha256"]
    op = w.load_operation(R / "candidate/operation.json")
    check_prepared(queue, entry, {**prepared, "operation": op})
    if phase in {"status", "collect-public", "reconcile-push", "reconcile-submission"}:
        remote = _remote(c, w, token)
        with c.single_coordinator(base):
            with w.ledger(ledger_path, remote.username) as db:
                if phase == "status":
                    return w.status(remote, db, sha)
                if phase == "reconcile-push":
                    return w.reconcile_push(remote, db, sha, 1)
                if phase == "reconcile-submission":
                    return w.reconcile_submission(remote, db, sha)
                record = w.read_entry(db, sha)
                if w.status(remote, db, sha)["kernel_status"]["status"] != "COMPLETE":
                    raise ValueError("Public notebook has not completed")
                return _collect(remote, w, op, record)
    if phase not in {"create-dataset", "reconcile-dataset", "push-kernel", "submit"}:
        raise ValueError("Unsupported queue phase")
    if phase == "submit":
        remote = _remote(c, w, token)
        with c.single_coordinator(base):
            with w.ledger(ledger_path, remote.username) as db:
                record = w.read_entry(db, sha)
                out = _collect(remote, w, op, record)
        accepted = accept_public(
            queue,
            entry,
            op,
            out["receipt"],
            out["metrics"],
            base64.b64decode(out["csv"]),
            config["expected_uids"],
        )
        if accepted != request.get("acceptance"):
            raise ValueError("Fresh public acceptance differs from the driver's acceptance")
        with (R / "public-accepted.json").open("x") as handle:
            json.dump(accepted, handle, indent=2)
    args = argparse.Namespace(
        action=phase,
        projection=R / "projection/code",
        projection_manifest=R / "projection/projection.json",
        projection_sha256=prepared["projection_sha"],
        operation=R / "candidate/operation.json",
        snapshot=Path(prepared["snapshot_path"]),
        publication_revision=False,
        superseded_operation_sha256=None,
        dataset_publication_operation=None,
        management_python=Path(config["management_python"]),
        node_local_upload_root=Path(f"/tmp/rsna-kaggle-upload-{sha[:12]}"),
        node_local_journal_root=None,
    )
    sys.stdin = io.StringIO(json.dumps({"token": token}))
    return c.run(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-queue", "admission", "plan", "drive"):
        p = sub.add_parser(name)
        p.add_argument("--queue", type=Path, default=ROOT / QUEUE_FILE)
        if name == "admission":
            p.add_argument("--epoch", type=int, required=True)
            p.add_argument("--output", type=Path, required=True)
        if name in {"plan", "drive"}:
            p.add_argument("--journal", type=Path, required=True)
            p.add_argument("--phase-command", required=True, help="JSON argv list")
        if name == "drive":
            p.add_argument("--expected-uids", type=Path, required=True)
            p.add_argument("--max-steps", type=int, default=500)
            p.add_argument("--max-submissions", type=int, default=5)
            p.add_argument("--poll-seconds", type=float, default=60)
            p.add_argument("--deadline-seconds", type=float, default=3 * 3600)
    p = sub.add_parser("node-phase", help="owning-node phase; request JSON on stdin")
    p.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "node-phase":
            config = small_json(args.config)
            request = json.load(sys.stdin)
            result = node_phase(request, config)
        else:
            queue = load_queue(args.queue)
            if args.command == "validate-queue":
                result = {
                    "status": "exact_queue_valid",
                    "queue_sha256": object_sha(queue),
                    "epochs": [x["epoch"] for x in queue["entries"]],
                }
            elif args.command == "admission":
                entry = queue_entry(queue, args.epoch)
                with args.output.open("xb") as handle:
                    handle.write(admission_bytes(queue, entry))
                result = {"sha256": file_sha(args.output), "epoch": args.epoch}
            else:
                executor = CommandExecutor(json.loads(args.phase_command), queue)
                journal = Journal(args.journal)
                if args.command == "plan":
                    result = plan(queue, executor.ledger_records(), journal).__dict__
                else:
                    uids = small_json(args.expected_uids)
                    result = Driver(
                        queue,
                        executor,
                        journal,
                        uids,
                        poll_seconds=args.poll_seconds,
                        deadline_seconds=args.deadline_seconds,
                        max_submissions=args.max_submissions,
                    ).run(args.max_steps)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
