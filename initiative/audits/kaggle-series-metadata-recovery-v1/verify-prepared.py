"""Independently rehash the one prepared metadata-recovery Kaggle bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/home/sa305415/rsna-knee/kaggle/bundles/fold2-epoch008-eb280864bd1c-metadata-v1")
CHECKPOINT_SHA = "eb280864bd1c2f896210ce360e1ae36d59649c854a9051523c2454a34d08c7b1"
WRAPPER_SHA = "1b57a2b675a95a8866f54fb5820ff646fbce8a1202d9dc725d42e47799f1244f"
DATASET_SLUG = "godlydonuts/rsna-knee-f2e008-metaguard-eb280864bd1c"
KERNEL_SLUG = "godlydonuts/rsna-knee-epoch8-f2-metaguard-eb280864bd1c"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Independent verification must use one CPU-only Slurm job")
    operation = json.loads((ROOT / "operation.json").read_text())
    manifest_path = ROOT / "dataset/bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        operation["checkpoint_sha256"] != CHECKPOINT_SHA
        or manifest["checkpoint_sha256"] != CHECKPOINT_SHA
    ):
        raise ValueError("Checkpoint identity differs")
    if digest(manifest_path) != operation["manifest_sha256"]:
        raise ValueError("Bundle manifest changed")
    if operation["dataset_slug"] != DATASET_SLUG or operation["kernel_slug"] != KERNEL_SLUG:
        raise ValueError("A different Kaggle operation was prepared")
    if operation["epoch"] != 8 or manifest["diagnostic_only"]:
        raise ValueError("Not the completed production epoch")
    dataset = ROOT / "dataset"
    if any(p.is_symlink() for p in dataset.rglob("*")):
        raise ValueError("Dataset payload contains a symlink")
    actual = {str(p.relative_to(dataset)) for p in dataset.rglob("*") if p.is_file()}
    expected = set(manifest["files"]) | {"bundle-manifest.json", "dataset-metadata.json"}
    if actual != expected:
        raise ValueError("Dataset payload membership differs")
    verified_bytes = 0
    for name, record in manifest["files"].items():
        path = dataset / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Bundle payload is missing or linked")
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Bundle payload bytes differ")
        verified_bytes += record["bytes"]
    if manifest["files"]["checkpoint.pt"]["sha256"] != CHECKPOINT_SHA:
        raise ValueError("Checkpoint payload differs")
    if manifest["files"]["code/chunked_infer.py"]["sha256"] != WRAPPER_SHA:
        raise ValueError("Bound metadata-recovery wrapper differs")
    for name, key in (
        ("dataset/dataset-metadata.json", "dataset_metadata_sha256"),
        ("kernel/kernel-metadata.json", "kernel_metadata_sha256"),
        ("kernel/run.py", "kernel_source_sha256"),
    ):
        path = ROOT / name
        if not path.is_file() or path.is_symlink() or digest(path) != operation[key]:
            raise ValueError(f"Prepared {name} differs")
    if any(p.is_symlink() for p in (ROOT / "kernel").iterdir()) or set(
        p.name for p in (ROOT / "kernel").iterdir()
    ) != {
        "run.py",
        "kernel-metadata.json",
    }:
        raise ValueError("Kernel membership differs")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "prepared_bundle_independently_verified",
                "bundle": str(ROOT),
                "payload_files_verified": len(manifest["files"]),
                "payload_bytes_verified": verified_bytes,
                "checkpoint_sha256": CHECKPOINT_SHA,
                "manifest_sha256": operation["manifest_sha256"],
                "kernel_source_sha256": operation["kernel_source_sha256"],
                "dataset_slug": DATASET_SLUG,
                "kernel_slug": KERNEL_SLUG,
                "allocated_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
                "allocated_gpus": 0,
                "kaggle_dataset_created": False,
                "kaggle_notebook_pushed": False,
                "competition_submission_created": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
