"""One-time framed original-cache relay, private cluster endpoints only."""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path, PurePosixPath

MAGIC = b"RSNANATIVE1\n"
END = b"RSNAEOF1\n"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def members_from_registry(path, digest, count):
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or sha(path) != digest:
        raise ValueError("Original registry bytes differ")
    registry = json.loads(path.read_bytes())
    members = {}
    for shard in registry["shards"]:
        for name, record in shard["members"].items():
            rel = PurePosixPath(name)
            if (
                rel.is_absolute()
                or ".." in rel.parts
                or rel.parts[0] != "cache"
                or str(rel) != name
                or name in members
                or type(record["bytes"]) is not int
                or record["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            ):
                raise ValueError("Invalid original cache member")
            members[name] = {"bytes": record["bytes"], "sha256": record["sha256"]}
    if len(members) != count:
        raise ValueError("Original complete member count differs")
    return dict(sorted(members.items()))


def source_stream(writer, root, members, proof, registry_sha):
    if set(proof["signatures"]) != set(members) or proof["registry_file_sha256"] != registry_sha:
        raise ValueError("Original whole-byte proof membership differs")
    for name, record in members.items():
        path = root / name
        if (
            path.is_symlink()
            or any(p.is_symlink() for p in path.parents)
            or signature(path) != proof["signatures"][name]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError("Previously verified source signature changed")
    total = sum(row["bytes"] for row in members.values())
    writer.write(MAGIC + bytes.fromhex(registry_sha) + struct.pack(">QQ", len(members), total))
    for name, record in members.items():
        path = root / name
        before = signature(path)
        with path.open("rb") as stream:
            remaining = record["bytes"]
            while remaining:
                block = stream.read(min(1 << 20, remaining))
                if not block:
                    raise ValueError("Source truncated during stream")
                writer.write(block)
                remaining -= len(block)
            if stream.read(1):
                raise ValueError("Source grew during stream")
        if signature(path) != before:
            raise ValueError("Source changed during stream")
    writer.write(END)
    writer.flush()


def read_exact(reader, size):
    parts = bytearray()
    while len(parts) < size:
        block = reader.read(size - len(parts))
        if not block:
            raise ValueError("Truncated original stream")
        parts.extend(block)
    return bytes(parts)


def target_stream(reader, root, members, registry_sha, *, byte_cap, remaining_funding):
    total = sum(row["bytes"] for row in members.values())
    if total > byte_cap or (root / "cache").exists():
        raise ValueError("Cache cap or existing transfer requires reconciliation")
    if shutil.disk_usage(root).free < remaining_funding:
        raise ValueError("Original future obligations are not funded")
    if (
        read_exact(reader, len(MAGIC)) != MAGIC
        or read_exact(reader, 32).hex() != registry_sha
        or struct.unpack(">QQ", read_exact(reader, 16)) != (len(members), total)
    ):
        raise ValueError("Original stream header differs")
    copied = 0
    for name, row in members.items():
        if shutil.disk_usage(root).free < remaining_funding - copied:
            raise ValueError("Future reserve breached during transfer")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if any(p.is_symlink() for p in path.parents):
            raise ValueError("Linked target directory forbidden")
        digest = hashlib.sha256()
        with path.open("xb") as output:
            os.chmod(path, 0o600)
            remaining = row["bytes"]
            while remaining:
                block = read_exact(reader, min(1 << 20, remaining))
                digest.update(block)
                output.write(block)
                remaining -= len(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != row["sha256"]:
            raise ValueError("Copied original cache hash differs")
        copied += row["bytes"]
    if read_exact(reader, len(END)) != END or reader.read(1):
        raise ValueError("Original stream trailer or EOF differs")
    return {
        "members": len(members),
        "bytes": copied,
        "registry_sha256": registry_sha,
        "all_original_member_size_SHA_verified_during_transfer": True,
    }


def reconcile_prefix(root, members):
    """Read-only incomplete-transfer reconciliation, only after original endpoints stop.

    Return verified prefix count/bytes and at most one unverified last file. Never
    delete, overwrite or retransfer any original or complete copied member.
    """
    names = list(members)
    actual = sorted(str(p.relative_to(root)) for p in (root / "cache").rglob("*") if p.is_file())
    if actual != names[: len(actual)]:
        raise ValueError("Incomplete cache is not an exact original stream prefix")
    complete, size, partial = 0, 0, None
    signatures = {}
    for index, name in enumerate(actual):
        path = root / name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError("Linked incomplete cache forbidden")
        before = signature(path)
        row = members[name]
        valid = before[2] == row["bytes"] and sha(path) == row["sha256"]
        if signature(path) != before:
            raise ValueError("Original transfer is still changing the cache")
        if not valid:
            if index != len(actual) - 1 or before[2] > row["bytes"]:
                raise ValueError("Earlier copied cache member is corrupt")
            partial = name
            break
        complete += 1
        size += row["bytes"]
        signatures[name] = before
    return {
        "verified_prefix_members": complete,
        "verified_prefix_bytes": size,
        "missing_members": len(members) - complete,
        "unverified_last_member": partial,
        "verified_signatures": signatures,
        "read_only": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("source", "target"))
    for name in ("registry", "root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--registry-sha", required=True)
    p.add_argument("--proof", type=Path)
    p.add_argument("--proof-sha")
    p.add_argument("--node", required=True)
    args = p.parse_args()
    if os.uname().nodename != args.node or os.environ.get("SLURM_STEP_GPUS"):
        raise ValueError("Exact CPU-only endpoint required")
    members = members_from_registry(args.registry, args.registry_sha, 37812)
    if args.mode == "source":
        if sha(args.proof) != args.proof_sha:
            raise ValueError("Original CPU whole-byte proof differs")
        source_stream(
            sys.stdout.buffer,
            args.root,
            members,
            json.loads(args.proof.read_bytes()),
            args.registry_sha,
        )
    else:
        intent = args.root / "full-cache-transfer-intent.json"
        with intent.open("x") as f:
            json.dump(
                {"registry_sha": args.registry_sha, "members": 37812, "automatic_retry": False}, f
            )
        result = target_stream(
            sys.stdin.buffer,
            args.root,
            members,
            args.registry_sha,
            byte_cap=160 * 2**30,
            remaining_funding=190 * 2**30 + 128 * 2**20,
        )
        result.update(
            status="original_complete_native_cache_private_relay_verified",
            node=args.node,
            CPU_job=os.environ["SLURM_JOB_ID"],
            no_Mac_or_shared_pixel_storage=True,
        )
        with (args.root / "full-cache-transfer-verified.json").open("x") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
