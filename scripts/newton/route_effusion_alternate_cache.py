"""One private native-cache placement; no labels, encoder or GPU computation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

BASE = Path("/home/sa305415/rsna-knee")
REGISTRY_SHA = "73d06c86900f03e8571fb68a8d341333c547ab984c7a4317387754949b8ab770"
ADMISSION_SHA = "81060696c0fb4312d04e8f0a68b2c0eb515db743938af723a31f58e7ed962124"
CACHE_BYTES = 140626483351


def file_sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def extract_exact(stream, root, expected):
    """Hash the new copy while writing; never overwrite or accept extra members."""
    seen = set()
    root.mkdir(mode=0o700, exist_ok=False)
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for entry in archive:
            rel = PurePosixPath(entry.name)
            if (
                rel.is_absolute()
                or ".." in rel.parts
                or str(rel) != entry.name
                or not rel.parts
                or rel.parts[0] != "cache"
                or not entry.isfile()
                or entry.name not in expected
                or entry.name in seen
            ):
                raise ValueError("Unsafe, duplicate or unexpected cache member")
            row = expected[entry.name]
            if type(row.get("bytes")) is not int or entry.size != row["bytes"]:
                raise ValueError("Exact original cache member size required")
            path = root.joinpath(*rel.parts)
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("Symlink path prohibited")
            digest = hashlib.sha256()
            with archive.extractfile(entry) as source, path.open("xb") as target:
                while block := source.read(1024**2):
                    target.write(block)
                    digest.update(block)
                target.flush()
                os.fsync(target.fileno())
            path.chmod(0o600)
            if path.stat().st_size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("Transferred original byte/hash mismatch")
            seen.add(entry.name)
    if seen != set(expected):
        raise ValueError("Incomplete original cache copy")
    # SSH must finish cleanly; consume the tar padding, rejecting appended data.
    while tail := stream.read(1024**2):
        if tail.strip(b"\0"):
            raise ValueError("Unexpected data after tar EOF")
    return {"members": len(seen), "bytes": sum(r["bytes"] for r in expected.values())}


def original():
    path = BASE / "production-launch-native384-a32-c9-v1/native384_admission.py"
    if path.is_symlink() or file_sha(path) != ADMISSION_SHA:
        raise ValueError("Exact original admission helper required")
    registry = BASE / "native384-assembly-c9-v1/registry.json"
    if registry.is_symlink() or file_sha(registry) != REGISTRY_SHA:
        raise ValueError("Exact original registry required")
    spec = importlib.util.spec_from_file_location("original_native_admission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    members = module.validate_registry(json.loads(registry.read_text()))
    if len(members) != 37812 or sum(r["bytes"] for r in members.values()) != CACHE_BYTES:
        raise ValueError("Full original native inventory required")
    return module, members


def validate_cpu_fields(fields):
    requested = dict(x.split("=", 1) for x in fields.get("ReqTRES", "").split(",") if "=" in x)
    if (
        "gres/gpu" in fields.get("AllocTRES", "")
        or "gres/gpu" in fields.get("ReqTRES", "")
        or fields.get("NumCPUs") not in {"2", "4"}
        or requested.get("cpu") != "2"
        or fields.get("NumTasks") != "2"
        or fields.get("CPUs/Task") != "1"
        or fields.get("NumNodes") != "2"
    ):
        raise ValueError("Two requested CPUs, bounded hardware threads and zero GPUs required")


def allocated(job, node):
    if not job.isdecimal() or os.uname().nodename.split(".")[0] != node:
        raise ValueError("Exact allocated endpoint required")
    row = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
    fields = dict(x.split("=", 1) for x in row.split() if "=" in x)
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", fields["NodeList"]], text=True
    ).splitlines()
    if fields.get("JobState") != "RUNNING" or set(nodes) != {"evc31", "evc32"}:
        raise ValueError("Exact two-endpoint running CPU allocation required")
    validate_cpu_fields(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("source", "receive"), required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    node = "evc31" if args.phase == "source" else "evc32"
    allocated(args.job, node)
    if args.control != Path("/tmp/rsna-effusion-native-route-evc32-20261004-v1"):
        raise ValueError("Distinct original control required")
    manifest = json.loads((args.control / "files.json").read_text())
    for name, row in manifest["files"].items():
        path = args.control / name
        if (
            path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or file_sha(path) != row["sha256"]
        ):
            raise ValueError("Immutable routing control changed")
    module, members = original()
    if args.phase == "source":
        _, _, actual, restored = module.assembly_inputs(BASE / "native384-assembly-c9-v1")
        prior = json.loads(
            (BASE / "production-launch-native384-a32-c9-v1/cpu-preflight.json").read_text()
        )
        module.verify_members(restored, actual, prior=prior["signatures"])
        if actual != members:
            raise ValueError("Source registry changed")
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
            for name, row in sorted(members.items()):
                path = restored / name
                before = module.signature(path)
                info = tarfile.TarInfo(name)
                info.size, info.mode = row["bytes"], 0o600
                with path.open("rb") as source:
                    archive.addfile(info, source)
                if module.signature(path) != before:
                    raise ValueError("Source changed while streaming")
        return
    acceptance = args.control / "copy-accepted.json"
    if acceptance.exists():
        raise ValueError("Existing placement requires read-only reconciliation")
    reserve = (20 + 7 + 3) * 2**30
    if shutil.disk_usage("/tmp").free < CACHE_BYTES + reserve:
        raise ValueError("Fresh native copy/runtime/checkpoint/reserve funding required")
    owned = Path(
        tempfile.mkdtemp(prefix="rsna-effusion-native384-evc32-route-" + args.job + "-", dir="/tmp")
    )
    if owned.stat().st_dev == BASE.stat().st_dev:
        raise ValueError("New MRI placement must remain private node-local")
    (owned / "owner.json").write_text(
        json.dumps({"job": args.job, "uid": os.getuid(), "registry_sha256": REGISTRY_SHA})
    )
    command = [
        "/usr/bin/python3.12",
        "-B",
        str(args.control / "route.py"),
        "--phase",
        "source",
        "--job",
        args.job,
        "--control",
        str(args.control),
    ]
    with (owned / "transport.stderr").open("xb") as err:
        process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "evc31", shlex.join(command)],
            stdout=subprocess.PIPE,
            stderr=err,
        )
        try:
            record = extract_exact(process.stdout, owned / "restored", members)
            if process.wait(timeout=30) != 0:
                raise ValueError("Original owning-node transport failed")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    if shutil.disk_usage("/tmp").free < reserve:
        raise ValueError("Post-copy runtime/checkpoint/reserve exhausted")
    record.update(
        status="private_original_native384_copy_fully_byte_verified",
        job_id=args.job,
        source_node="evc31",
        destination_node="evc32",
        cache_root=str(owned / "restored/cache"),
        registry_sha256=REGISTRY_SHA,
        original_cache_retained=True,
        GPU_reserved=False,
        target_values_read=False,
        clinical_accuracy_evidence=False,
    )
    with acceptance.open("x") as out:
        json.dump(record, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
