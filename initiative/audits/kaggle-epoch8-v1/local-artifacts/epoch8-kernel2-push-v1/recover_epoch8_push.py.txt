"""Move the uncertain epoch-8 notebook to a distinct exact-source private slug."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

R = Path("/home/sa305415/rsna-knee")
PROJECTION = R / "kaggle/code-projections/2bc2695-b740-epoch1-v1"
SOURCE = PROJECTION / "source"
PACKAGE_SHA = "098a3b351305af4166fd7fe57f33bdf8387d3f42c3b255210985f25205e6459b"
WORKFLOW_PATH = R / "kaggle/recovery-code/epoch10-searchguard-v1/workflow.py"
WORKFLOW_SHA = "87252a0149f371e0b9fbed62ae41efde1a7f9f0ee2d0aec06c9ed66003ab8dc6"
OLD = R / "kaggle/bundles/epoch008-e156bcf1d4e0-2bc2695-v1"
NEW = R / "kaggle/bundles/epoch008-e156bcf1d4e0-kernel2-v1"
OLD_SHA = "40be2ad0df346be3ea9185ea16936113bde4b1fdb6d0a1a6edee5b97613d3adf"
NEW_SHA = "dbb632e7e3afffbbc9324eb484f4cf4185fe0142abb71d68d6042c3998cb5ce1"
SNAPSHOT = R / "runs/dinov3-l16-production-fold0-b74060b-v1/epoch-checkpoints/epoch-008.json"
CHECKPOINT_SHA = "e156bcf1d4e0d7509d9a48ac0630af80e7119ee232e23082371c575ffc207373"
DATASET_SLUG = "godlydonuts/rsna-knee-e008-e156bcf1d4e0"
OLD_SLUG = "godlydonuts/rsna-knee-epoch8-e156bcf1d4e0"
NEW_SLUG = "godlydonuts/rsna-knee-epoch8-k2-e156bcf1d4e0"
DATASET_ID = 12146499


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Pinned source module missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("CPU-only Slurm allocation required")
    coordinator = load(SOURCE / "scripts/kaggle/coordinator.py", "pinned_coordinator")
    _, projection = coordinator.load_workflow(SOURCE, PROJECTION / "projection.json", PACKAGE_SHA)
    if projection["package_git_commit"] != "2bc26959c4b7d3c82a7cba41f8b8f662970a91ac":
        raise ValueError("Immutable inference source projection differs")
    if coordinator.p.digest(WORKFLOW_PATH) != WORKFLOW_SHA:
        raise ValueError("Reviewed search guard source differs")
    workflow = load(WORKFLOW_PATH, "reviewed_search_workflow")
    old_path = OLD / "operation.json"
    new_path = NEW / "operation.json"
    if workflow.digest(old_path) != OLD_SHA or workflow.digest(new_path) != NEW_SHA:
        raise ValueError("Original or derived epoch-8 operation differs")
    op, meta, files = coordinator.validate_operation(
        workflow,
        SimpleNamespace(operation=new_path, snapshot=SNAPSHOT, projection=SOURCE),
    )
    old_op = workflow.load_operation(old_path)
    derivation = json.loads((NEW / "derivation.json").read_text())
    if (
        meta["checkpoint_sha256"] != CHECKPOINT_SHA
        or op["dataset_slug"] != DATASET_SLUG
        or op["kernel_slug"] != NEW_SLUG
        or op["dataset_version"] != 1
        or op["epoch"] != 8
        or old_op["kernel_slug"] != OLD_SLUG
        or old_op["dataset_slug"] != DATASET_SLUG
        or old_op["kernel_source_sha256"] != op["kernel_source_sha256"]
        or old_op["manifest_sha256"] != op["manifest_sha256"]
        or derivation.get("old_operation_sha256") != OLD_SHA
        or derivation.get("new_operation_sha256") != NEW_SHA
        or derivation.get("notebook_source_sha256") != op["kernel_source_sha256"]
    ):
        raise ValueError("Exact-source notebook derivation differs")
    old_files = sorted(
        p.relative_to(OLD / "dataset") for p in (OLD / "dataset").rglob("*") if p.is_file()
    )
    if (
        len(old_files) != derivation["dataset_hardlinked_files"]
        or not all((OLD / "dataset" / name).samefile(NEW / "dataset" / name) for name in old_files)
        or not (OLD / "kernel/run.py").samefile(NEW / "kernel/run.py")
    ):
        raise ValueError("Derived operation copied or changed Dataset/notebook bytes")
    base = R / "kaggle"
    journal = base / "dataset-publications" / CHECKPOINT_SHA
    verified_path = journal / "verified.json"
    verified = json.loads(verified_path.read_text())
    if (
        verified.get("status") != "all_payload_bytes_verified"
        or verified.get("files") != files
        or verified.get("operation_sha256") != OLD_SHA
        or verified.get("intent_sha256") != workflow.digest(journal / "intent.json")
        or verified.get("id") != DATASET_ID
        or verified.get("version") != 1
        or verified.get("slug") != DATASET_SLUG
        or verified.get("private") is not True
    ):
        raise ValueError("Original private Dataset byte proof differs")
    remote, _ = coordinator.remote_from_stdin(workflow)
    with coordinator.single_coordinator(base):
        if coordinator.p.metadata(remote, DATASET_SLUG) != {
            k: verified[k] for k in ("id", "version", "slug", "private")
        }:
            raise ValueError("Private Dataset identity drifted")
        with workflow.ledger(base / "account-ledger.sqlite3", remote.username) as db:
            workflow.verify_account(db, remote)
            entry = workflow.read_entry(db, CHECKPOINT_SHA)
            if (
                entry.get("state") != "push_uncertain"
                or entry.get("kernel_version") is not None
                or entry.get("operation_sha256") != OLD_SHA
                or entry.get("operation", {}).get("kernel_slug") != OLD_SLUG
                or entry.get("dataset", {}).get("id") != DATASET_ID
                or entry.get("submitted") is not False
            ):
                raise ValueError("Ledger no longer has the one uncertain original push")
            # The old slug now appears only as a private placeholder in the
            # exact authenticated search. Never repeat SaveKernel on that slug.
            try:
                remote.call(
                    "kernels.KernelsApiService",
                    "GetKernel",
                    {"userName": remote.username, "kernelSlug": OLD_SLUG.split("/")[1]},
                )
            except workflow.RemoteError as exc:
                if str(exc) != "Kaggle GetKernel HTTP 500":
                    raise
            else:
                raise ValueError("Original notebook became readable; reconcile it first")
            search_request = {
                "group": "PROFILE",
                "page": 1,
                "pageSize": 100,
                "language": "all",
                "kernelType": "all",
                "outputType": "all",
                "search": OLD_SLUG,
            }
            old_search = remote.call("kernels.KernelsApiService", "ListKernels", search_request)
            old_rows = old_search.get("kernels", [])
            if (
                not isinstance(old_rows, list)
                or len(old_rows) != 1
                or old_rows[0].get("ref") is not None
                or old_rows[0].get("title") != "[Private Notebook]"
            ):
                raise ValueError("Original reserved notebook identity changed")
            new_absence = remote.require_absent_kernel(NEW_SLUG)
            if new_absence.get("method") != "complete_authenticated_profile_search_after403":
                raise ValueError("New notebook slug lacks exact-search absence proof")
            migration = base / "kernel-push-migrations" / CHECKPOINT_SHA
            migration.mkdir(mode=0o700)
            coordinator.p.exclusive_json(
                migration / "intent.json",
                {
                    "schema_version": 1,
                    "status": "one_epoch8_new_slug_migration_intent",
                    "checkpoint_sha256": CHECKPOINT_SHA,
                    "old_operation_sha256": OLD_SHA,
                    "new_operation_sha256": NEW_SHA,
                    "old_kernel_slug": OLD_SLUG,
                    "new_kernel_slug": NEW_SLUG,
                    "original_projection_sha256": PACKAGE_SHA,
                    "reviewed_workflow_sha256": WORKFLOW_SHA,
                    "recovery_script_sha256": workflow.digest(Path(__file__)),
                    "dataset_verified_sha256": workflow.digest(verified_path),
                    "derivation_sha256": workflow.digest(NEW / "derivation.json"),
                    "initial_ledger_entry_sha256": workflow.object_hash(entry),
                    "new_absence_proof": new_absence,
                    "old_search": "one unnamed private placeholder",
                    "no_new_dataset_or_checkpoint_bytes": True,
                },
            )
            entry.update(
                state="queued",
                operation=op,
                operation_sha256=NEW_SHA,
                superseded_kernel_slug=OLD_SLUG,
                superseded_operation_sha256=OLD_SHA,
                migration_intent_sha256=workflow.digest(migration / "intent.json"),
                kernel_version=None,
            )
            workflow.save_entry(
                db, entry, "migrated uncertain epoch-8 slug to exact-source new slug"
            )
            try:
                result = workflow.push(remote, db, new_path)
                if result.get("state") != "pushed" or result.get("kernel_version") != 1:
                    raise ValueError("New notebook did not return exact version 1")
                coordinator.p.exclusive_json(
                    migration / "push-result.json",
                    {
                        "status": "pushed",
                        "kernel_slug": NEW_SLUG,
                        "version": 1,
                        "notebook_source_sha256": op["kernel_source_sha256"],
                        "ledger_entry_sha256": workflow.object_hash(result),
                    },
                )
                print(json.dumps({"status": "pushed", "kernel_slug": NEW_SLUG, "version": 1}))
            except Exception as exc:
                coordinator.p.exclusive_json(
                    migration / "push-uncertain.json",
                    {
                        "status": "new_slug_push_uncertain",
                        "error_type": type(exc).__name__,
                        "automatic_retry": False,
                    },
                )
                raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
