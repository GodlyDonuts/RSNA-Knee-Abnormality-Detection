"""One verified intra-cluster Stanford restoration, using one CPU at each end.

The stream traverses RAM/stdin, never a shared pixel archive or local download.
Original acquisition/input receipts remain intact. The new owning-node receipt
requires every original volume SHA and eight exact original tensor hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def expected(root, source):
    if sha(root / "verified-files.jsonl") != source["original_journal_sha256"]:
        raise ValueError("Original whole volume journal required")
    rows = [json.loads(line) for line in (root / "verified-files.jsonl").read_text().splitlines()]
    result = {}
    for row in rows:
        name = row["split"] + "/" + row["name"]
        if not re.fullmatch(r"(train|valid)/(axial|coronal|sagittal)/[0-9]{4}\.npy", name):
            raise ValueError("Unsafe original source volume member")
        if name in result:
            raise ValueError("Repeated source volume member")
        result[name] = row
    if len(result) != 3750 or sum(row["bytes"] for row in result.values()) != 7740871424:
        raise ValueError("Complete original source volume membership required")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--mode", choices=("send", "restore"), required=True)
    args = parser.parse_args()
    control = args.control
    source = json.loads((control / "source.json").read_text())
    if (
        sha(Path(__file__)) != source["restorer_sha256"]
        or not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_JOB_GPUS")
    ):
        raise ValueError("Original one-CPU zero-GPU restoration wrapper required")
    acquisition = Path(source["acquisition_control"])
    if sha(acquisition / "accepted.json") != source["acquisition_receipt_sha256"]:
        raise ValueError("Exact original acquisition acceptance required")
    scratch = json.loads((acquisition / "scratch.json").read_text())
    meta = ["verified-files.jsonl", "downloads.json", "source-admission.json"]
    meta += [split + ".csv" for split in ("train", "valid")]
    meta += [
        f"{split}-{target}.csv"
        for split in ("train", "valid")
        for target in ("acl", "meniscus", "abnormal")
    ]
    if args.mode == "send":
        if os.uname().nodename != scratch["node"]:
            raise ValueError("Send from the original owning node only")
        root = Path(scratch["path"])
        rows = expected(root, source)
        save(
            control / "send-start.json",
            {"job_id": os.environ["SLURM_JOB_ID"], "node": os.uname().nodename},
        )
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
            for name in meta + sorted(rows):
                path = root / name
                # No tokens, arbitrary source-directory members or links enter
                # the exact whitelist stream. Volumes are checked by receiver.
                if not path.is_file() or path.is_symlink():
                    raise ValueError("Original regular source file required")
                info = tarfile.TarInfo(name)
                info.size, info.mode = path.stat().st_size, 0o600
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
        save(control / "send-complete.json", {"job_id": os.environ["SLURM_JOB_ID"], "files": 3761})
        return
    if os.uname().nodename != source["destination_node"]:
        raise ValueError("Exact new owning node required")
    if shutil.disk_usage("/tmp").free < 20 * 1024**3 + 7740871424 + 9 * 1024**3:
        raise ValueError("Funded source/runtime/checkpoints plus20GiB node-local reserve required")
    root = Path(tempfile.mkdtemp(prefix="rsna-mrnet-restored-", dir="/tmp"))
    if root.stat().st_dev == control.stat().st_dev:
        raise ValueError("No shared source pixels permitted")
    save(
        control / "restore-start.json",
        {"job_id": os.environ["SLURM_JOB_ID"], "node": os.uname().nodename, "root": str(root)},
    )
    seen, rows = set(), None
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as archive:
        for member in archive:
            name = member.name
            if (
                not member.isfile()
                or name in seen
                or name not in (meta if rows is None else [*meta, *rows])
            ):
                raise ValueError("Unsafe, repeated or unaccepted private restoration member")
            if (not seen and name != "verified-files.jsonl") or (
                name not in meta and not set(meta).issubset(seen)
            ):
                raise ValueError("Original metadata must precede volumes")
            if name in meta and member.size > 16 * 1024**2:
                raise ValueError("Oversized source metadata")
            if name not in meta and (rows is None or member.size != rows[name]["bytes"]):
                raise ValueError("Original volume byte count differs")
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            incoming = archive.extractfile(member)
            with destination.open("xb") as stream:
                for block in iter(lambda: incoming.read(1024**2), b""):
                    stream.write(block)
                    digest.update(block)
            if destination.stat().st_size != member.size:
                raise ValueError("Incomplete private restoration member")
            seen.add(name)
            if name == "verified-files.jsonl":
                rows = expected(root, source)
            elif name not in meta and digest.hexdigest() != rows[name]["sha256"]:
                raise ValueError("Copied volume differs from original whole-byte proof")
    if rows is None or seen != set(meta) | set(rows):
        raise ValueError("Incomplete source restoration")
    # Byte-identical code/runtime are borrowed from the already frozen original
    # pilot, rather than mutating its source or rerunning acquisition.
    original = Path(source["original_input_control"])
    if sha(original / "source.json") != source["original_code_manifest_sha256"]:
        raise ValueError("Original input code manifest required")
    code_source = json.loads((original / "source.json").read_text())
    if sha(original / "source.tar") != source["original_code_tar_sha256"]:
        raise ValueError("Original frozen scientific code required")
    with tempfile.TemporaryDirectory(prefix="rsna-mrnet-restore-code-", dir="/tmp") as tmp:
        code = Path(tmp)
        with tarfile.open(original / "source.tar", "r:") as archive:
            members = archive.getmembers()
            if {m.name for m in members} != set(code_source["files"]) or len(members) != len(
                code_source["files"]
            ):
                raise ValueError("Exact original scientific closure required")
            for member in members:
                name = Path(member.name)
                if not member.isfile() or name.is_absolute() or ".." in name.parts:
                    raise ValueError("Unsafe original code member")
                path = code / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
                if sha(path) != code_source["files"][member.name]["sha256"]:
                    raise ValueError("Original code bytes differ")
        os.environ.update(
            PYTHONPATH=str(code),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1",
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        helper = code / "scripts/newton/runtime_environment.py"
        runtime = json.loads(
            subprocess.check_output(
                [
                    "/usr/bin/python3.12",
                    "-B",
                    str(helper),
                    "extract",
                    "--archive",
                    code_source["environment_archive"],
                    "--sha256",
                    code_source["environment_sha256"],
                    "--expected-files",
                    "40403",
                    "--expected-bytes",
                    "6322691335",
                    "--scratch-base",
                    "/tmp",
                    "--min-free-gib",
                    "20",
                    "--shared-python",
                    "/usr/bin/python3.12",
                ],
                text=True,
                timeout=900,
            )
        )
        try:
            subprocess.run(
                [
                    runtime["python"],
                    "-B",
                    str(code / "scripts/newton/probe_mrnet_inputs.py"),
                    "--root",
                    str(root),
                    "--acquisition-receipt",
                    str(acquisition / "accepted.json"),
                    "--output",
                    str(control / "restored-inputs.json"),
                ],
                check=True,
                timeout=300,
            )
            before = json.loads(Path(source["original_input_path"]).read_text())
            after = json.loads((control / "restored-inputs.json").read_text())
            if (
                sha(Path(source["original_input_path"])) != source["original_input_sha256"]
                or before["records"] != after["records"]
            ):
                raise ValueError(
                    "Restored actual image/label tensors differ from original CPU proof"
                )
        finally:
            subprocess.run(
                [
                    "/usr/bin/python3.12",
                    "-B",
                    str(helper),
                    "cleanup",
                    "--root",
                    runtime["runtime_root"],
                ],
                check=True,
                timeout=60,
            )
    save(
        control / "restoration-verified.json",
        {
            "status": "verbatim_MRNet_private_node_restoration_verified",
            "node": os.uname().nodename,
            "job_id": os.environ["SLURM_JOB_ID"],
            "root": str(root),
            "volume_files": 3750,
            "volume_bytes": 7740871424,
            "original_acquisition_receipt_sha256": source["acquisition_receipt_sha256"],
            "original_journal_sha256": source["original_journal_sha256"],
            "original_input_sha256": source["original_input_sha256"],
            "restored_input_sha256": sha(control / "restored-inputs.json"),
            "original_eight_image_label_tensor_records_equal": True,
            "no_GPU_model_optimizer_or_new_source_download": True,
        },
    )


if __name__ == "__main__":
    os.umask(0o077)
    main()
