"""Finite one-candidate-at-a-time exact Kaggle queue for owner-admitted epochs.

Root runs bounded steps on the owning CPU allocation. Each local/mutating phase
writes an exclusive durable intent before its single call and an exclusive
result after it. An intent without result, a canonical-ledger push/submit
intent or uncertainty, or an unjournaled ledger entry stops the queue for
read-only reconciliation; nothing is retried or resubmitted. The only
competition mutation is the original workflow CreateCodeSubmission after exact
public execution acceptance; the public three-row CSV is never uploaded.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from . import reviewed_infer as reviewed
else:  # pragma: no cover - executed from an exact projection
    import reviewed_infer as reviewed

PHASES = (
    "prepare",
    "prepare-transport",
    "create-dataset",
    "reconcile-dataset",
    "push-kernel",
    "public-acceptance",
    "submit",
)
# Local exclusive writers and cloud mutations: intent before, result after.
JOURNALED = {"prepare", "prepare-transport", "create-dataset", "push-kernel", "submit"}
# Each spends nothing without a fresh live allowance of at least one submission.
ALLOWANCE_GATED = {"create-dataset", "push-kernel", "submit"}
READ_ONLY = {"reconcile-dataset", "public-acceptance"}
UNCERTAIN_LEDGER = {"push_intent", "push_uncertain", "submit_intent", "submit_uncertain"}
PREFETCH_HELPERS = (
    "prefetch_entry.py",
    "prefetch_preparation.py",
    "parallel_preparation.py",
    "process_memory.py",
    "resident_infer.py",
    "resident_pool.py",
    "resident_worker.py",
    "chunked_infer.py",
)
ENGINE = "two_device_exact_FP32_prefetch_v1"
RSS_CAP = 29 * 2**30


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def exclusive_json(path: Path, value) -> None:
    """No-overwrite durable record; contains no credentials."""
    with Path(path).open("x") as handle:
        os.chmod(path, 0o600)
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def expected_slugs(epoch: int, checkpoint_sha: str) -> tuple[str, str]:
    tag = f"e{epoch:02d}-{checkpoint_sha[:12]}"
    return (
        f"godlydonuts/rsna-alldev-publabel-{tag}",
        f"godlydonuts/rsna-alldev-publabel-offline-{tag}",
    )


def load_queue(path: Path) -> dict:
    """Admit only the exact committed queue; no mutable labels or guessed metadata."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("Queue must be a small regular file")
    queue = json.loads(path.read_text())
    run = queue.get("run", {})
    deployment = queue.get("deployment", {})
    entries = queue.get("entries")
    if (
        queue.get("schema_version") != 1
        or queue.get("queue_id") != reviewed.QUEUE_ID
        or run.get("scientific_source_git") != reviewed.ALL_DEVELOPMENT_SOURCE
        or run.get("training_binding_sha256") != reviewed.ALL_DEVELOPMENT_BINDING_SHA
        or run.get("training_code_sha256") != reviewed.ALL_DEVELOPMENT_CODE_SHA
        or run.get("supervision") != reviewed.ALL_DEVELOPMENT_SUPERVISION
        or run.get("base_encoder") != reviewed.OFFICIAL_BASE
        or (run.get("training_studies"), run.get("outer_studies")) != (3403, 0)
        or run.get("fit_scope") != "all_development"
        or run.get("diagnostic_run") is not False
        or run.get("protected_confirmation_gold_used") is not False
        or run.get("external_initialization") is not None
        or run.get("private_research_only") is not False
        or deployment.get("engine") != ENGINE
        or deployment.get("inference_engine") != "resident_two_gpu_fp32"
        or deployment.get("MRI_only_inference") is not True
        or deployment.get("public_example_csv_upload") is not False
        or deployment.get("one_candidate_at_a_time") is not True
        or deployment.get("automatic_mutation_retry") is not False
        or not isinstance(entries, list)
        or [e.get("epoch") for e in entries] != sorted(reviewed.ALL_DEVELOPMENT_EPOCHS)
    ):
        raise ValueError("Queue differs from the exact owner-admitted daily queue")
    for entry in entries:
        epoch = entry["epoch"]
        checkpoint_sha, steps = reviewed.ALL_DEVELOPMENT_EPOCHS[epoch]
        if entry != {
            "epoch": epoch,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": 125556557,
            "optimizer_steps": steps,
            "binding_sha256": reviewed.ALL_DEVELOPMENT_BINDING_SHA,
            "dataset_slug": expected_slugs(epoch, checkpoint_sha)[0],
            "kernel_slug": expected_slugs(epoch, checkpoint_sha)[1],
        }:
            raise ValueError("Queue entry differs from its exact admitted checkpoint")
    queue["sha256"] = sha(path.read_bytes())
    return queue


def admission_bytes(epoch: int) -> bytes:
    return (json.dumps(reviewed.queue_admission(epoch), sort_keys=True, indent=2) + "\n").encode()


def candidate_dir(root: Path, entry: dict) -> Path:
    return Path(root) / f"epoch-{entry['epoch']:03d}-{entry['checkpoint_sha256'][:12]}"


def phase_files(directory: Path, phase: str) -> tuple[Path, Path]:
    return directory / f"{phase}.intent.json", directory / f"{phase}.result.json"


def plan(queue: dict, root: Path, records: dict) -> dict:
    """Pure next decision from the journal and canonical ledger; never mutates."""
    for other, record in records.items():
        if record.get("state") in {"submit_intent", "submit_uncertain"}:
            return {
                "action": "read_only_reconciliation",
                "checkpoint_sha256": other,
                "reason": "canonical ledger submission " + record["state"],
            }
    for entry in queue["entries"]:
        directory = candidate_dir(root, entry)
        base = {"epoch": entry["epoch"], "checkpoint_sha256": entry["checkpoint_sha256"]}
        record = records.get(entry["checkpoint_sha256"])
        state = None if record is None else record.get("state")
        if state == "submitted":
            # Already submitted or scored: never resubmit; outcome reads stay read-only.
            continue
        if state in UNCERTAIN_LEDGER:
            return {**base, "action": "read_only_reconciliation", "reason": "ledger " + state}
        if (directory / "public-acceptance.rejected.json").exists():
            return {**base, "action": "blocked", "reason": "public execution rejected"}
        push_intent, push_result = phase_files(directory, "push-kernel")
        if state not in (None, "queued") and not (push_intent.exists() or push_result.exists()):
            return {
                **base,
                "action": "read_only_reconciliation",
                "reason": "ledger entry " + str(state) + " was not created by this queue",
            }
        for phase in PHASES:
            intent, result = phase_files(directory, phase)
            if result.exists():
                continue
            if intent.exists():
                return {
                    **base,
                    "action": "read_only_reconciliation",
                    "phase": phase,
                    "reason": "intent without result",
                }
            return {**base, "action": "run", "phase": phase}
    return {"action": "complete"}


def ledger_records(db) -> dict:
    return {
        json.loads(row[0])["checkpoint_sha256"]: json.loads(row[0])
        for row in db.execute("SELECT record FROM entries")
    }


@contextlib.contextmanager
def driver_claim(root: Path):
    """No stale-lock stealing; a killed driver leaves the claim for inspection."""
    claim = Path(root) / "driver-claim"
    claim.mkdir(mode=0o700)
    try:
        exclusive_json(claim / "owner.json", {"pid": os.getpid(), "host": os.uname().nodename})
        yield
    finally:
        (claim / "owner.json").unlink()
        claim.rmdir()


def step(queue: dict, root: Path, db, workflow, fresh_account, runner, now=None) -> dict:
    """Run at most one phase for the first unfinished queued candidate."""
    root = Path(root)
    with driver_claim(root):
        decision = plan(queue, root, ledger_records(db))
        if decision["action"] != "run":
            return decision
        phase = decision["phase"]
        entry = next(e for e in queue["entries"] if e["epoch"] == decision["epoch"])
        directory = candidate_dir(root, entry)
        directory.mkdir(mode=0o700, exist_ok=True)
        intent, result = phase_files(directory, phase)
        allowance = None
        if phase in ALLOWANCE_GATED:
            account = fresh_account()
            # Original account-wide rule: fresh live quota, local day count and
            # canonical uncertain-attempt refusal before every mutation.
            allowance = workflow.allowance(db, account, now)
            if allowance < 1:
                return {**decision, "action": "wait_for_allowance", "allowance": allowance}
        if phase == "submit":
            accepted = json.loads(phase_files(directory, "public-acceptance")[1].read_text())
            accepted = accepted.get("outcome", {})
            if accepted.get("status") != "exact_private_offline_public_MRI_only_prefetch_verified":
                raise ValueError("Exact public execution acceptance is required")
        if phase in JOURNALED:
            exclusive_json(
                intent,
                {
                    "phase": phase,
                    "epoch": entry["epoch"],
                    "checkpoint_sha256": entry["checkpoint_sha256"],
                    "queue_sha256": queue["sha256"],
                    "admission_sha256": sha(admission_bytes(entry["epoch"])),
                    "allowance_before": allowance,
                    "at": utcnow(),
                    "automatic_retry": False,
                },
            )
        # Exceptions propagate with the intent left in place: read-only next time.
        outcome = runner(phase, entry, directory)
        if phase == "push-kernel" and outcome.get("state") != "pushed":
            raise ValueError("Kernel push did not return an exact pushed entry")
        if phase == "submit" and outcome.get("state") != "submitted":
            raise ValueError("Submission deferred or uncertain; reconcile from the ledger")
        if phase == "reconcile-dataset" and outcome.get("status") != "all_payload_bytes_verified":
            raise ValueError("Private Dataset bytes are not fully verified")
        if phase == "public-acceptance":
            if outcome.get("status") == "pending":
                return {**decision, "action": "await_public_execution", "kernel": outcome}
            if outcome.get("status") != "exact_private_offline_public_MRI_only_prefetch_verified":
                exclusive_json(directory / "public-acceptance.rejected.json", outcome)
                return {**decision, "action": "blocked", "reason": "public execution rejected"}
        exclusive_json(result, {"phase": phase, "at": utcnow(), "outcome": outcome})
        return {**decision, "action": "ran", "outcome": outcome}


def reconcile(queue: dict, root: Path, db, runner) -> dict:
    """Close one intent-without-result only from authoritative read-only evidence."""
    root = Path(root)
    with driver_claim(root):
        decision = plan(queue, root, ledger_records(db))
        phase = decision.get("phase")
        if decision["action"] != "read_only_reconciliation" or phase is None:
            raise ValueError("No journaled phase awaits reconciliation; use ledger reconcile CLI")
        entry = next(e for e in queue["entries"] if e["epoch"] == decision["epoch"])
        directory = candidate_dir(root, entry)
        intent, result = phase_files(directory, phase)
        record = ledger_records(db).get(entry["checkpoint_sha256"], {})
        if phase == "create-dataset":
            evidence = runner("reconcile-dataset", entry, directory)
            if evidence.get("status") != "all_payload_bytes_verified":
                raise ValueError("Dataset creation remains unverified; never recreate")
            exclusive_json(
                phase_files(directory, "reconcile-dataset")[1],
                {"phase": "reconcile-dataset", "at": utcnow(), "outcome": evidence},
            )
        elif phase == "push-kernel" and record.get("state") == "pushed":
            evidence = {"ledger_state": "pushed", "kernel_version": record["kernel_version"]}
        elif phase == "submit" and record.get("state") == "submitted":
            evidence = {"ledger_state": "submitted", "submission_id": record.get("submission_id")}
        elif (
            phase == "submit"
            and record.get("state") == "pushed"
            and record.get("next_quota_check_utc")
            and not record.get("attempted_at")
        ):
            # Original workflow deferred before CreateCodeSubmission: archive, never delete.
            archived = (
                directory / f"submit.intent.deferred-{record['next_quota_check_utc'][:10]}.json"
            )
            os.link(intent, archived)
            intent.unlink()
            return {**decision, "action": "deferred_intent_archived", "archived": archived.name}
        else:
            raise ValueError("Authoritative evidence absent; inspect read-only, never rerun")
        exclusive_json(
            result, {"phase": phase, "at": utcnow(), "reconciled": True, "outcome": evidence}
        )
        return {**decision, "action": "reconciled", "outcome": evidence}


def drive(queue, root, db, workflow, fresh_account, runner, max_steps=len(PHASES)) -> list:
    """Finite: stops at any non-run decision, after one submission, or max_steps."""
    steps = []
    for _ in range(max_steps):
        decision = step(queue, root, db, workflow, fresh_account, runner)
        steps.append(decision)
        if decision["action"] != "ran" or decision["phase"] == "submit":
            break
    return steps


def accept_public_execution(op, receipt, metrics, raw_csv, expected_uids, targets) -> dict:
    """Mandatory exact public MRI-only acceptance before one code submission."""
    try:
        rows = list(csv.reader(io.StringIO(raw_csv.decode())))
        values = [float(x) for row in rows[1:] for x in row[1:]]
    except (UnicodeDecodeError, ValueError):
        raise ValueError("Public submission output is not a numeric CSV") from None
    if (
        receipt.get("status") != "offline_inference_complete"
        or receipt.get("checkpoint_sha256") != op["checkpoint_sha256"]
        or receipt.get("bundle_manifest_sha256") != op["manifest_sha256"]
        or metrics.get("status") != "passed"
        or metrics.get("studies") != 3
        or metrics.get("resident_engine") != ENGINE
        or sum(x.get("fallback_studies", 0) for x in metrics.get("chunks", [])) != 0
        or metrics.get("production_memory", {}).get("summed_lifetime_RSS_bytes", RSS_CAP + 1)
        > RSS_CAP
        or not sha(raw_csv) == receipt.get("submission_sha256") == metrics.get("submission_sha256")
        or rows[0] != ["StudyInstanceUID", *targets]
        or [row[0] for row in rows[1:]] != list(expected_uids)
        or any(len(row) != 13 for row in rows)
        or not all(math.isfinite(x) and 0 <= x <= 1 for x in values)
    ):
        raise ValueError("Public execution differs from the exact admitted inference")
    return {
        "status": "exact_private_offline_public_MRI_only_prefetch_verified",
        "receipt": receipt,
        "metrics": metrics,
        "csv_sha256": sha(raw_csv),
        "columns": 13,
        "studies": 3,
        "public_example_only": True,
        "csv_uploaded": False,
    }


def prefetch_kernel_source(code: Path, operation: dict, expected_hashes=None) -> tuple[str, dict]:
    """The accepted FP32 prefetch prefix over the exact projected notebook entry."""
    kaggle = Path(code) / "scripts/kaggle"
    sources = {name: (kaggle / name).read_text() for name in PREFETCH_HELPERS}
    hashes = {name: sha(text.encode()) for name, text in sources.items()}
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError("Prefetch helpers differ from the accepted engine")
    shared = operation["shared_wheelhouse"]
    prefix = {
        "EXPECTED_MANIFEST_SHA256": operation["manifest_sha256"],
        "INFERENCE_FP32_PREFETCH": True,
        "INFERENCE_PREFETCH_SOURCES": sources,
        "INFERENCE_PREFETCH_HASHES": hashes,
        "ATTACHED_PAYLOAD_ROOTS": {
            "bundle-manifest.json": {
                "kind": "datasets",
                "slug": operation["dataset_slug"],
                "version": 1,
                "relative": "bundle-manifest.json",
            },
            shared["archive_file"]: {
                "kind": "datasets",
                "slug": shared["dataset_slug"],
                "version": 1,
                "relative": shared["archive_file"],
            },
        },
    }
    text = "\n".join(k + " = " + repr(v) for k, v in prefix.items()) + "\n"
    text += (kaggle / "notebook_entry.py").read_text()
    compile(text, "offline_prefetch.py", "exec")
    if len(text.encode()) >= 256 * 1024:
        raise ValueError("Kernel source exceeds the bounded notebook size")
    return text, hashes


class CoordinatorRunner:
    """Executes each phase only through the original workflow/coordinator code.

    ``config`` holds root-owned local paths; ``token`` is read from RAM/stdin by
    root and passed in-process only. Never invoked by tests against the cloud.
    """

    def __init__(self, config: dict, token: str):
        self.c = config
        self.token = token
        projection = Path(config["projection"])
        spec = importlib.util.spec_from_file_location(
            "queue_coordinator", projection / "scripts/kaggle/coordinator.py"
        )
        self.coordinator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.coordinator)
        self.workflow, _ = self.coordinator.load_workflow(
            projection, Path(config["projection_manifest"]), config["projection_sha256"]
        )

    def snapshot(self, entry):
        path = Path(self.c["snapshots"][str(entry["epoch"])])
        meta = json.loads(path.read_text())
        if (
            meta.get("checkpoint_sha256") != entry["checkpoint_sha256"]
            or meta.get("binding_sha256") != entry["binding_sha256"]
            or meta.get("optimizer_steps") != entry["optimizer_steps"]
            or meta.get("epoch") != entry["epoch"]
        ):
            raise ValueError("Snapshot sidecar differs from the exact queue entry")
        return path

    def args(self, phase, entry, directory):
        return argparse.Namespace(
            action=phase,
            projection=Path(self.c["projection"]),
            projection_manifest=Path(self.c["projection_manifest"]),
            projection_sha256=self.c["projection_sha256"],
            operation=directory / "candidate/operation.json",
            snapshot=self.snapshot(entry),
            publication_revision=False,
            superseded_operation_sha256=None,
            dataset_publication_operation=None,
            management_python=Path(self.c["management_python"]),
            node_local_upload_root=Path(
                "/tmp/rsna-kaggle-upload-" + entry["checkpoint_sha256"][:12]
            ),
            node_local_journal_root=None,
        )

    def coordinator_run(self, args):
        stdin = sys.stdin
        sys.stdin = io.StringIO(json.dumps({"token": self.token}))
        try:
            return self.coordinator.run(args)
        finally:
            sys.stdin = stdin

    def remote(self):
        stdin = sys.stdin
        sys.stdin = io.StringIO(json.dumps({"token": self.token}))
        try:
            return self.coordinator.remote_from_stdin(self.workflow)[0]
        finally:
            sys.stdin = stdin

    def __call__(self, phase, entry, directory):
        if phase == "prepare":
            return self.prepare(entry, directory)
        if phase == "public-acceptance":
            return self.public_acceptance(entry, directory)
        if phase not in {"prepare-transport", "create-dataset", "reconcile-dataset"} | {
            "push-kernel",
            "submit",
        }:
            raise ValueError("Unsupported queue phase")
        return self.coordinator_run(self.args(phase, entry, directory))

    def prepare(self, entry, directory):
        admission = directory / "fitting-admission.json"
        with admission.open("xb") as handle:
            handle.write(admission_bytes(entry["epoch"]))
        c = self.c
        self.workflow.prepare_bundle(
            self.snapshot(entry),
            Path(c["encoder_spec"]),
            Path(c["wheelhouse"]),
            directory / "candidate",
            entry["dataset_slug"],
            entry["kernel_slug"],
            Path(c["license_file"]),
            c["license_source"],
            c["license_sha256"],
            shared_wheelhouse=Path(c["shared_wheelhouse"]),
            runtime_image=c["runtime_image"],
            fitting_admission=admission,
            inference_engine="resident_two_gpu_fp32",
        )
        operation_file = directory / "candidate/operation.json"
        operation = json.loads(operation_file.read_text())
        source, hashes = prefetch_kernel_source(
            Path(c["projection"]), operation, c.get("accepted_prefetch_helper_hashes")
        )
        (directory / "candidate/kernel/run.py").write_text(source)
        operation["kernel_source_sha256"] = sha(source.encode())
        operation["inference_execution_receipts"] = {
            "engine": ENGINE,
            "reference_fullscale_receipt_sha256": c["reference_fullscale_receipt_sha256"],
            "reference_memory_receipt_sha256": c["reference_memory_receipt_sha256"],
            "new_checkpoint_public_acceptance_required": True,
        }
        operation_file.write_text(json.dumps(operation, sort_keys=True, indent=2) + "\n")
        operation = self.workflow.load_operation(operation_file)
        return {
            "operation_sha256": sha(operation_file.read_bytes()),
            "manifest_sha256": operation["manifest_sha256"],
            "kernel_source_sha256": operation["kernel_source_sha256"],
            "prefetch_helper_hashes": hashes,
            "competition_submission_created": False,
        }

    def public_acceptance(self, entry, directory):
        from rsna.schema import TARGETS

        remote = self.remote()
        ledger = Path(self.c["ledger"])
        with self.workflow.ledger(ledger, remote.username) as db:
            status = self.workflow.status(remote, db, entry["checkpoint_sha256"])
            record = self.workflow.read_entry(db, entry["checkpoint_sha256"])
        kernel = status.get("kernel_status", {}).get("status")
        if kernel in {"RUNNING", "QUEUED"}:
            return {"status": "pending", "kernel_status": kernel}
        if kernel != "COMPLETE":
            return {"status": "rejected", "kernel_status": kernel}
        op = record["operation"]
        version = record["kernel_version"]
        receipt = remote.output_json(op["kernel_slug"], version, "kaggle-inference-receipt.json")
        metrics = remote.output_json(op["kernel_slug"], version, "submission.metrics.json")
        raw = download_output(self.workflow, remote, op["kernel_slug"], version, "submission.csv")
        try:
            return accept_public_execution(
                op, receipt, metrics, raw, self.c["expected_public_uids"], TARGETS
            )
        except ValueError as error:
            return {"status": "rejected", "reason": str(error), "receipt": receipt}


def download_output(workflow, remote, slug, version, name, limit=2**20) -> bytes:
    """Bounded versioned output read with the original redirect host policy."""
    from urllib.parse import urljoin, urlparse

    import requests

    owner, kernel = slug.split("/")
    remote.session.cookies.clear()
    response = remote.session.post(
        workflow.API + "kernels.KernelsApiService/DownloadKernelOutput",
        json={"ownerSlug": owner, "kernelSlug": kernel, "versionNumber": version, "filePath": name},
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
                    if len(raw) > limit:
                        raise ValueError("Kernel output exceeds bounded size")
                return bytes(raw)
            if response.status_code not in {301, 302, 303, 307, 308}:
                raise ValueError("Cannot retrieve versioned kernel output")
            url = urljoin(response.url, response.headers.get("Location", ""))
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not (
            host
            in {
                "storage.googleapis.com",
                "api.kaggle.com",
                "www.kaggle.com",
                "www.kaggleusercontent.com",
            }
            or host.endswith(".googleusercontent.com")
        ):
            raise ValueError("Unrecognized Kaggle output redirect host")
        session = remote.session if host == "api.kaggle.com" else requests
        response = session.get(url, timeout=(10, 60), allow_redirects=False, stream=True)
    raise ValueError("Output redirect budget exhausted")


def main() -> None:
    """Root entry: config path in argv, token as {"token": ...} JSON on stdin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--journal-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=len(PHASES))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    queue = load_queue(args.queue)
    config = json.loads(args.config.read_text())
    value = json.load(sys.stdin)
    if set(value) != {"token"}:
        raise SystemExit("Exactly one token object is required on stdin")
    runner = CoordinatorRunner(config, value["token"])
    args.journal_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    remote = runner.remote()
    with runner.workflow.ledger(Path(config["ledger"]), remote.username) as db:
        if args.plan_only:
            result = plan(queue, args.journal_root, ledger_records(db))
        else:
            result = drive(
                queue,
                args.journal_root,
                db,
                runner.workflow,
                lambda: runner.remote().account(),
                runner,
                max_steps=args.max_steps,
            )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
