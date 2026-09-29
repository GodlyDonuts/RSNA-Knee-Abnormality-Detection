"""Preserve an unreadable private placeholder before one distinct notebook push.

This never repeats the uncertain SaveKernel or creates another Dataset. A real
saved old version, ambiguous search, changed payload, or submission blocks it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/home/sa305415/rsna-knee")
PROJECTION = ROOT / "kaggle/code-projections/f4f90bc-b740-inputguard-v1"
PROJECTION_SHA = "f42e065ea2e3b99a31336511911663ed90389eb0506822885e5dc9ad075f5236"
CHECKPOINT = "092c3eed133f73d7a167c6eb7b55068c2684f82a3714f337839e81874c533605"
OLD_OPERATION = "746ec14a513b1ff6f5343da1a9bddfd1aa6b94cc8240a15cf6c6ccd2709d1801"
OLD_SLUG = "godlydonuts/rsna-knee-epoch2-f2-inputguard-092c3eed133f"
NEW_SLUG = "godlydonuts/rsna-knee-epoch2-f2-inputguard-k2-092c3eed133f"
CONTROL_SLUG = "godlydonuts/rsna-knee-epoch10-f2-inputguard-5e400e82c4b5"
CONTROL_SOURCE_SHA = "cb18ba062116c739e4f04ccf4a596f548a4ebc509bbdbb24ef692cdbbf64c833"
OLD_BUNDLE = ROOT / "kaggle/bundles/fold2-epoch002-092c3eed133f-inputguard-v1"
NEW_BUNDLE = ROOT / "kaggle/bundles/fold2-epoch002-092c3eed133f-inputguard-k2-v1"
SNAPSHOT = (
    ROOT / "runs/dinov3-l16-fold2-b74060b-1gpu-production-v1/epoch-checkpoints/epoch-002.json"
)


def search(remote, query):
    rows = []
    for page in range(1, 101):
        result = remote.call(
            "kernels.KernelsApiService",
            "ListKernels",
            dict(
                group="PROFILE",
                page=page,
                pageSize=100,
                language="all",
                kernelType="all",
                outputType="all",
                search=query,
            ),
        )
        batch = result.get("kernels", [])
        if not isinstance(batch, list):
            raise ValueError("Invalid authenticated search")
        if not batch:
            return rows
        rows.extend(batch)
    raise ValueError("Authenticated search did not terminate")


def preserve_placeholder(workflow, remote, db, new):
    db.execute("BEGIN IMMEDIATE")
    workflow.verify_account(db, remote)
    entry = workflow.read_entry(db, CHECKPOINT)
    old = entry.get("operation", {})
    if (
        entry.get("state") != "push_uncertain"
        or entry.get("operation_sha256") != OLD_OPERATION
        or entry.get("kernel_version") is not None
        or entry.get("submission_id") is not None
        or entry.get("submitted") is not False
        or entry.get("attempted_at") is not None
        or entry.get("description") is not None
        or old.get("kernel_slug") != OLD_SLUG
        or new.get("kernel_slug") != NEW_SLUG
        or old.get("checkpoint_sha256") != CHECKPOINT
        or any(
            new.get(k) != old.get(k)
            for k in (
                "checkpoint_sha256",
                "dataset_slug",
                "dataset_version",
                "manifest_sha256",
                "dataset_metadata_sha256",
                "kernel_source_sha256",
                "shared_wheelhouse",
                "epoch",
            )
        )
    ):
        raise ValueError(
            "Only the exact unsubmitted placeholder with unchanged payload can migrate"
        )
    owner, slug = OLD_SLUG.split("/")
    probes = {}
    for label, extra, expected in (("latest", {}, "Kaggle GetKernel HTTP 500"),):
        try:
            remote.call(
                "kernels.KernelsApiService",
                "GetKernel",
                dict(userName=owner, kernelSlug=slug, **extra),
            )
        except workflow.RemoteError as exc:
            if str(exc) != expected:
                raise ValueError(
                    "Placeholder observation changed; inspect without mutation"
                ) from None
            probes[label] = expected
        else:
            raise ValueError("Old saved notebook appeared; reconcile its exact source instead")
    co, cs = CONTROL_SLUG.split("/")
    control = remote.call(
        "kernels.KernelsApiService", "GetKernel", dict(userName=co, kernelSlug=cs)
    )
    source = control.get("blob", {}).get("source", "")
    metadata = control.get("metadata", {})
    if (
        metadata.get("currentVersionNumber") != 1
        or metadata.get("isPrivate") is not True
        or not isinstance(source, str)
        or hashlib.sha256(source.encode()).hexdigest() != CONTROL_SOURCE_SHA
    ):
        raise ValueError("Known saved-source positive control failed")
    rows = search(remote, OLD_SLUG)
    if (
        len(rows) != 1
        or rows[0].get("ref") is not None
        or rows[0].get("title") != "[Private Notebook]"
        or rows[0].get("lastRunTime") != "2010-04-01T00:00:00Z"
    ):
        raise ValueError("Exact unreadable placeholder is not proven")
    controls = search(remote, CONTROL_SLUG)
    if len(controls) != 1 or controls[0].get("ref") != CONTROL_SLUG:
        raise ValueError("Authenticated positive control failed")
    history = entry.get("superseded_operations", [])
    if not isinstance(history, list) or any(
        x.get("operation_sha256") == OLD_OPERATION for x in history
    ):
        raise ValueError("Attempt is already archived or history changed")
    absence = remote.require_absent_kernel(NEW_SLUG)
    entry["superseded_operations"] = [
        *history,
        dict(
            state="unreadable_uncertain_placeholder_never_retry_or_submit",
            operation=old,
            operation_sha256=OLD_OPERATION,
            dataset=entry["dataset"],
            probes=probes,
            placeholder=rows[0],
            positive_control=controls[0],
            replacement_kernel_slug=NEW_SLUG,
            new_kernel_absence=absence,
            reconciled_at=workflow.utcnow(),
            old_absence_claimed=False,
            old_saved_version_unproven=True,
        ),
    ]
    for key in (
        "operation",
        "operation_sha256",
        "dataset",
        "kernel_absence_proof",
        "kernel_version",
    ):
        entry.pop(key, None)
    entry["state"] = "queued"
    workflow.save_entry(
        db, entry, "Preserved unversioned placeholder; old identity never retried/submitted"
    )
    return entry


def main():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("CPU-only Slurm allocation required")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "pinned_coordinator", PROJECTION / "code/scripts/kaggle/coordinator.py"
    )
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    workflow, _ = c.load_workflow(
        PROJECTION / "code", PROJECTION / "projection.json", PROJECTION_SHA
    )
    remote, _ = c.remote_from_stdin(workflow)
    base = ROOT / "kaggle"
    old, meta, files = c.validate_operation(
        workflow,
        SimpleNamespace(
            operation=OLD_BUNDLE / "operation.json",
            snapshot=SNAPSHOT,
            projection=PROJECTION / "code",
        ),
    )
    proof = json.loads((base / "dataset-publications" / CHECKPOINT / "verified.json").read_text())
    if (
        c.p.digest(OLD_BUNDLE / "operation.json") != OLD_OPERATION
        or proof.get("operation_sha256") != OLD_OPERATION
        or proof.get("status") != "all_payload_bytes_verified"
        or proof.get("files") != files
        or proof.get("private") is not True
        or meta["checkpoint_sha256"] != CHECKPOINT
        or c.p.metadata(remote, old["dataset_slug"])
        != {k: proof[k] for k in ("id", "version", "slug", "private")}
    ):
        raise ValueError("Original exact private Dataset verification differs")
    with (
        c.single_coordinator(base),
        workflow.ledger(base / "account-ledger.sqlite3", remote.username) as db,
    ):
        if sys.argv[1] in ("push", "push-prepared"):
            journal = base / "kernel-push-migrations/fold2-e002-inputguard-k2-v1"
            if sys.argv[1] == "push":
                journal.mkdir(mode=0o700)
                c.p.exclusive_json(
                    journal / "intent.json",
                    dict(
                        status="one_distinct_kernel_intent",
                        old_operation_sha256=OLD_OPERATION,
                        new_kernel_slug=NEW_SLUG,
                        dataset_created=False,
                        old_kernel_retried=False,
                        script_sha256=c.p.digest(Path(__file__)),
                    ),
                )
                NEW_BUNDLE.mkdir(mode=0o700)
                shutil.copytree(
                    OLD_BUNDLE / "dataset", NEW_BUNDLE / "dataset", copy_function=os.link
                )
                (NEW_BUNDLE / "kernel").mkdir(mode=0o700)
                os.link(OLD_BUNDLE / "kernel/run.py", NEW_BUNDLE / "kernel/run.py")
                km = json.loads((OLD_BUNDLE / "kernel/kernel-metadata.json").read_text())
                km.update(id=NEW_SLUG, title=NEW_SLUG.split("/")[1].replace("-", " "))
                c.p.exclusive_json(NEW_BUNDLE / "kernel/kernel-metadata.json", km)
                new = dict(
                    old,
                    kernel_slug=NEW_SLUG,
                    prepared_at=workflow.utcnow(),
                    kernel_metadata_sha256=c.p.digest(NEW_BUNDLE / "kernel/kernel-metadata.json"),
                )
                c.p.exclusive_json(NEW_BUNDLE / "operation.json", new)
            else:
                intent = json.loads((journal / "intent.json").read_text())
                if (
                    intent.get("script_sha256")
                    != "5ad4cc9e1386da022be8e3ac572bb5ae5c5305c4c0ec33f3e5d9de2e3879f2d0"
                    or intent.get("old_operation_sha256") != OLD_OPERATION
                    or intent.get("new_kernel_slug") != NEW_SLUG
                    or (journal / "derivation.json").exists()
                    or (journal / "push-result.json").exists()
                ):
                    raise ValueError("Previous phase may have reached a mutation; reconcile only")
                new = workflow.load_operation(NEW_BUNDLE / "operation.json")
                c.p.exclusive_json(
                    journal / "preflight-repair-intent.json",
                    dict(
                        status="repair_read_only_placeholder_search_before_any_new_save",
                        original_intent_sha256=c.p.digest(journal / "intent.json"),
                        script_sha256=c.p.digest(Path(__file__)),
                        automatic_retry=False,
                    ),
                )
            checked, _, new_files = c.validate_operation(
                workflow,
                SimpleNamespace(
                    operation=NEW_BUNDLE / "operation.json",
                    snapshot=SNAPSHOT,
                    projection=PROJECTION / "code",
                ),
            )
            if new_files != files or checked != new:
                raise ValueError("Derived notebook changed Dataset or source bytes")
            preserve_placeholder(workflow, remote, db, new)
            c.p.exclusive_json(
                journal / "derivation.json",
                dict(
                    old_operation_sha256=OLD_OPERATION,
                    new_operation_sha256=c.p.digest(NEW_BUNDLE / "operation.json"),
                    unchanged_dataset_proof=proof,
                    unchanged_kernel_source_sha256=new["kernel_source_sha256"],
                ),
            )
            result = workflow.push(remote, db, NEW_BUNDLE / "operation.json")
            c.p.exclusive_json(journal / "push-result.json", result)
        elif sys.argv[1] in ("submit", "reconcile-push"):
            new = workflow.load_operation(NEW_BUNDLE / "operation.json")
            entry = workflow.read_entry(db, CHECKPOINT)
            if entry.get("operation_sha256") != c.p.digest(NEW_BUNDLE / "operation.json"):
                raise ValueError("Canonical notebook differs from the derived exact source")
            result = (
                workflow.submit(remote, db, CHECKPOINT)
                if sys.argv[1] == "submit"
                else workflow.reconcile_push(remote, db, CHECKPOINT, 1)
            )
        else:
            raise ValueError("Unknown phase")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
