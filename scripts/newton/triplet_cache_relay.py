"""Once-only private CPU relay of the immutable public ordered-triplet cache.

PREPARED ONLY; Devin dispatches nothing. Root alone runs, on Newton login stdin/RAM:

    plan   (evc31, one CPU / zero GPU step overlapping 899766): read-only stat/layout plan
    source (same step) | target (evc40, one CPU / zero GPU allocation)

The private manifest is derived from the original pinned ``index.json`` (its SHA256 is
bound by CPU receipt 158c242f and the sibling merge receipt ``merged-cache-verified.json``):
the index bytes plus every expected ``members/<uid>.npz``. Member size/SHA come from that
trusted index; the source only rechecks stable stat signatures from its plan, the target
verifies every whole original byte, count, cap, EOF and exact relative layout. Per-member
names/signatures stay node-local; stdout/Git receive aggregate counts/hashes only.

No retries, overwrite, deletion credit, shared growth or GPU; an uncertain transfer is only
reconciled read-only (``reconcile``). Later PR76 destination admission verifies all paths
and provenance independently; the original trainer root/checkpoint/journals are not moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import struct
import sys
from pathlib import Path, PurePosixPath

if __package__:
    from scripts.newton.native_cache_relay import (
        END,
        MAGIC,
        read_exact,
        sha,
        signature,
        source_stream,
    )
else:
    from native_cache_relay import (  # type: ignore[no-redef]
        END,
        MAGIC,
        read_exact,
        sha,
        signature,
        source_stream,
    )

VERSION = "public_triplet_cache_private_relay_v1"
CACHE = Path("/tmp/rsna-kaggle-public-source-shards-20261013-v1/merged-cache")
MERGED_RECEIPT = CACHE.parent / "merged-cache-verified.json"
CONTROL = Path("/tmp/rsna-dense-triplet-cache-relay-20261014-v1")
DIRECTORY = "public-ordered-triplet-v1"
INDEX = f"{DIRECTORY}/index.json"
MEMBERS_DIR = f"{DIRECTORY}/members"
SOURCE_NODE, TARGET_NODE, SOURCE_JOB = "evc31", "evc40", "899766"
# Actual CPU admission 158c242f (dense-suffix-admission-root-review-20261014-v1).
CPU_RECEIPT_SHA256 = "158c242faa37c46e3b8a2e632d68b3ae690174bb3ec3b85abed2a46ffc53c834"
INDEX_SHA256 = "849855281837669efaad81596314f7770b8a936baa457e2a571418dbd87c1736"
COHORT_SHA256 = "6d00d9aa90799aca0f32b5e0918f9716f00f7a3e516db3a4625aaed46b373844"
FITTING_UIDS_SHA256 = "e9d38ee21a114973177b89f4602129c7dc047927b8499991d3be30fd89737f3a"
EVALUATION_UIDS_SHA256 = "2f365bb0ce8e052f22dcab78bb2bbb7e03c0919ab28ba94ab9d3d8d6e08f0fdb"
MERGED_RECEIPT_SHA256 = "d2c0ad74370f7613b1a9983b9437b7e8e407d02242bb3328be36bf7a57d81ee0"
STUDIES, MEMBER_BYTES, SHARDS = 3403, 8130228, 20
NPZ_BYTES = STUDIES * MEMBER_BYTES  # 27667165884, root's measured member bytes.
RESERVE_BYTES = 20 * 2**30
CONTROL_CAP_BYTES = 64 * 2**20
INDEX_KEYS = {
    "schema_version",
    "kind",
    "renderer",
    "source_sha256",
    "node_text_sha256",
    "pixel_group",
    "source_revision",
    "fitting_uids_sha256",
    "evaluation_uids_sha256",
    "cohort_sha256",
    "members",
}
PLAN, INTENT, VERIFIED = "source-plan.json", "target-intent.json", "target-verified.json"
FUNDING_STATUS = "root_admitted_node_local_triplet_cache_relay_funding"
FUNDING_KEYS = {
    "status",
    "funded_by",
    "issue",
    "relay_version",
    "node",
    "cache",
    "control",
    "index_sha256",
    "cache_bytes",
    "control_bytes",
    "reserve_bytes",
    "existing_liabilities_bytes",
    "required_free_bytes",
    "shared_future_bytes",
    "deletion_credit_bytes",
    "GPUs_funded",
    "GPU_reserved",
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def roster_sha256(uids) -> str:
    return canonical_sha256(sorted(str(uid) for uid in uids))


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe_uid(uid: object) -> bool:
    return (
        isinstance(uid, str)
        and uid not in {"", ".", ".."}
        and not any(c in uid for c in ("/", "\\", "\x00"))
        and PurePosixPath(uid).name == uid
    )


def manifest_from_index(payload: bytes) -> dict:
    """Ordered private manifest: the exact index bytes first, then every expected NPZ."""
    index_sha = INDEX_SHA256
    if not _is_sha(index_sha) or hashlib.sha256(payload).hexdigest() != index_sha:
        raise ValueError("Original pinned index bytes differ")
    index = json.loads(payload)
    if not isinstance(index, dict) or set(index) != INDEX_KEYS or index["schema_version"] != 1:
        raise ValueError("Original index fields differ")
    members = index["members"]
    if (
        not isinstance(members, dict)
        or len(members) != STUDIES
        or index["cohort_sha256"] != roster_sha256(members)
        or index["cohort_sha256"] != COHORT_SHA256
        or index["fitting_uids_sha256"] != FITTING_UIDS_SHA256
        or index["evaluation_uids_sha256"] != EVALUATION_UIDS_SHA256
    ):
        raise ValueError("Original complete 3403-study cohort differs")
    manifest: dict[str, dict] = {INDEX: {"bytes": len(payload), "sha256": index_sha}}
    for uid in sorted(members):
        row = members[uid]
        if (
            not _safe_uid(uid)
            or not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or row["path"] != f"members/{uid}.npz"
            or type(row["bytes"]) is not int
            or row["bytes"] != MEMBER_BYTES
            or not _is_sha(row["sha256"])
        ):
            raise ValueError("Invalid or traversing original member binding")
        manifest[f"{DIRECTORY}/{row['path']}"] = {"bytes": row["bytes"], "sha256": row["sha256"]}
    if sum(r["bytes"] for n, r in manifest.items() if n != INDEX) != NPZ_BYTES:
        raise ValueError("Original member bytes differ from the measured cohort")
    return manifest


def total_bytes(manifest: dict) -> int:
    return sum(row["bytes"] for row in manifest.values())


def require_endpoint(node: str, job: str | None = None) -> str:
    """Exact allocated one-CPU zero-GPU Slurm step on the named node."""
    env = os.environ
    if (
        os.uname().nodename != node
        or not env.get("SLURM_JOB_ID")
        or not env.get("SLURM_STEP_ID")
        or (job is not None and env["SLURM_JOB_ID"] != job)
        or env.get("SLURM_CPUS_PER_TASK", "1") != "1"
        or env.get("SLURM_STEP_GPUS")
        or env.get("CUDA_VISIBLE_DEVICES") not in (None, "", "NoDevFiles")
    ):
        raise ValueError("Exact allocated one-CPU zero-GPU endpoint required")
    return env["SLURM_JOB_ID"]


def _real_dir(path: Path) -> None:
    if path.is_symlink() or not path.is_dir() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Linked or absent original directory")


def _regular(path: Path) -> os.stat_result:
    s = os.lstat(path)
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise ValueError("Linked or non-regular original member")
    return s


def source_layout(cache: Path, manifest: dict) -> dict:
    """Exact NPZ directory membership; returns stable stat signatures of relayed files."""
    for directory in (cache, cache / DIRECTORY, cache / MEMBERS_DIR):
        _real_dir(directory)
    expected = {PurePosixPath(n).name for n in manifest if n != INDEX}
    actual = {entry.name for entry in os.scandir(cache / MEMBERS_DIR)}
    if actual != expected:
        raise ValueError("Missing or extra original NPZ members")
    signatures = {}
    for name, row in manifest.items():
        path = cache / name
        if _regular(path).st_size != row["bytes"]:
            raise ValueError("Original member size differs from the trusted index")
        signatures[name] = signature(path)
    return signatures


def validate_merged_receipt(payload: bytes) -> dict:
    receipt = json.loads(payload)
    if (
        hashlib.sha256(payload).hexdigest() != MERGED_RECEIPT_SHA256
        or receipt.get("index_sha256") != INDEX_SHA256
        or receipt.get("studies") != STUDIES
        or receipt.get("complete_shards") != SHARDS
        or receipt.get("labels_in_cache") is not False
        or receipt.get("exact_member_bytes") != NPZ_BYTES
    ):
        raise ValueError("Original merge receipt does not bind this index")
    return receipt


def plan(cache: Path, merged_receipt: Path, control: Path) -> dict:
    """Read-only source plan; private names/signatures stay in the node-local control file."""
    validate_merged_receipt(merged_receipt.read_bytes())
    _real_dir(cache / DIRECTORY)
    manifest = manifest_from_index(Path(cache / INDEX).read_bytes())
    signatures = source_layout(cache, manifest)
    tree = [p for p in cache.rglob("*") if not p.is_dir()]
    tree_bytes = sum(os.lstat(p).st_size for p in tree)
    private = {
        "relay_version": VERSION,
        "index_sha256": INDEX_SHA256,
        "manifest": manifest,
        "signatures": signatures,
    }
    control.mkdir(mode=0o700, exist_ok=True)
    _real_dir(control)
    with (control / PLAN).open("x") as f:
        os.chmod(control / PLAN, 0o600)
        json.dump(private, f, sort_keys=True)
    return {
        "status": "private_triplet_cache_source_plan_read_only",
        "plan_sha256": sha(control / PLAN),
        "manifest_sha256": canonical_sha256(manifest),
        "index_sha256": INDEX_SHA256,
        "cohort_sha256": COHORT_SHA256,
        "merged_receipt_sha256": MERGED_RECEIPT_SHA256,
        "CPU_receipt_sha256": CPU_RECEIPT_SHA256,
        "studies": STUDIES,
        "relayed_files": len(manifest),
        "relayed_bytes": total_bytes(manifest),
        "tree_files": len(tree),
        "tree_bytes": tree_bytes,
        "not_relayed_producer_bookkeeping_files": len(tree) - len(manifest),
        "not_relayed_producer_bookkeeping_bytes": tree_bytes - total_bytes(manifest),
        "labels_gold_or_confirmation_read": False,
        "pixels_exported": False,
    }


def source(writer, cache: Path, plan_path: Path, plan_sha: str) -> None:
    if not _is_sha(plan_sha) or sha(plan_path) != plan_sha:
        raise ValueError("Original private source plan differs")
    private = json.loads(plan_path.read_bytes())
    manifest = manifest_from_index(Path(cache / INDEX).read_bytes())
    if private.get("manifest") != manifest or private.get("index_sha256") != INDEX_SHA256:
        raise ValueError("Source plan does not bind the original index manifest")
    signatures = source_layout(cache, manifest)
    proof = {"signatures": private["signatures"], "registry_file_sha256": INDEX_SHA256}
    if {n: list(s) for n, s in signatures.items()} != private["signatures"]:
        raise ValueError("Previously planned source signature changed")
    source_stream(writer, cache, manifest, proof, INDEX_SHA256)


def load_funding(path: Path, digest: str, cache: Path, control: Path, node: str) -> dict:
    if not _is_sha(digest) or sha(path) != digest:
        raise ValueError("Separately admitted funding proof differs")
    f = json.loads(path.read_bytes())
    cache_bytes = None
    if isinstance(f, dict) and set(f) == FUNDING_KEYS:
        cache_bytes = f["cache_bytes"]
    ints = ("control_bytes", "reserve_bytes", "existing_liabilities_bytes", "required_free_bytes")
    if (
        cache_bytes is None
        or f["status"] != FUNDING_STATUS
        or f["funded_by"] != "root"
        or f["issue"] != 78
        or f["relay_version"] != VERSION
        or f["node"] != node
        or f["cache"] != str(cache)
        or f["control"] != str(control)
        or f["index_sha256"] != INDEX_SHA256
        or type(cache_bytes) is not int
        or not NPZ_BYTES < cache_bytes <= NPZ_BYTES + CONTROL_CAP_BYTES
        or any(type(f[k]) is not int or f[k] < 0 for k in ints)
        or f["control_bytes"] > CONTROL_CAP_BYTES
        or f["reserve_bytes"] < RESERVE_BYTES
        or f["required_free_bytes"]
        != f["existing_liabilities_bytes"] + cache_bytes + f["control_bytes"] + f["reserve_bytes"]
        or f["shared_future_bytes"] != 0
        or f["deletion_credit_bytes"] != 0
        or f["GPUs_funded"] != 0
        or f["GPU_reserved"] is not False
    ):
        raise ValueError("Explicit node-local cache/control/reserve funding required")
    return f


def _free(path: Path) -> int:
    return shutil.disk_usage(path).free


def require_fresh_target(cache: Path, funding: dict) -> None:
    if cache.exists() or cache.is_symlink():
        raise ValueError("Existing target requires read-only reconciliation")
    parent = cache.parent
    probe = parent if parent.exists() or parent.is_symlink() else parent.parent
    _real_dir(probe)
    if _free(probe) < funding["required_free_bytes"]:
        raise ValueError("Funded node-local reserve is insufficient")


def target_stream(reader, cache: Path, manifest: dict, funding: dict) -> dict:
    total = total_bytes(manifest)
    if total != funding["cache_bytes"]:
        raise ValueError("Cache cap differs from the funded bytes")
    require_fresh_target(cache, funding)
    parent = cache.parent
    if (
        read_exact(reader, len(MAGIC)) != MAGIC
        or read_exact(reader, 32).hex() != INDEX_SHA256
        or struct.unpack(">QQ", read_exact(reader, 16)) != (len(manifest), total)
    ):
        raise ValueError("Original stream header differs")
    parent.mkdir(mode=0o700, exist_ok=True)
    cache.mkdir(mode=0o700)
    copied = 0
    for name, row in manifest.items():
        if _free(parent) < funding["required_free_bytes"] - copied:
            raise ValueError("Funded reserve breached during transfer")
        path = cache / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _real_dir(path.parent)
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
            raise ValueError("Copied original member hash differs")
        os.chmod(path, 0o400)
        copied += row["bytes"]
    if read_exact(reader, len(END)) != END or reader.read(1):
        raise ValueError("Original stream trailer or EOF differs")
    layout = sorted(str(p.relative_to(cache)) for p in cache.rglob("*") if not p.is_dir())
    if layout != sorted(manifest) or any(
        _regular(cache / n).st_size != manifest[n]["bytes"] for n in manifest
    ):
        raise ValueError("Target layout differs from the exact original relative layout")
    for directory in (cache / MEMBERS_DIR, cache / DIRECTORY, cache, parent):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {"files": len(manifest), "bytes": copied, "free_bytes_after": _free(parent)}


def target(reader, cache: Path, control: Path, funding_path: Path, funding_sha: str, node: str):
    job = require_endpoint(node)
    _real_dir(control)
    funding = load_funding(funding_path, funding_sha, cache, control, node)
    if (control / VERIFIED).exists() or (control / INTENT).exists():
        raise ValueError("Existing intent or verified relay forbids repetition")
    require_fresh_target(cache, funding)
    with (control / INTENT).open("x") as f:
        json.dump(
            {
                "relay_version": VERSION,
                "index_sha256": INDEX_SHA256,
                "funding_sha256": funding_sha,
                "cache_bytes": funding["cache_bytes"],
                "automatic_retry": False,
                "overwrite": False,
            },
            f,
        )
    # Index arrives first and is verified against the pinned SHA before any NPZ.
    header = read_exact(reader, len(MAGIC) + 32 + 16)
    count, total = struct.unpack(">QQ", header[-16:])
    if header[: len(MAGIC)] != MAGIC or header[len(MAGIC) : -16].hex() != INDEX_SHA256:
        raise ValueError("Original stream header differs")
    if total != funding["cache_bytes"]:
        raise ValueError("Stream bytes exceed or differ from the funded cap")
    index_bytes = total - NPZ_BYTES
    if not 0 < index_bytes <= CONTROL_CAP_BYTES or count != STUDIES + 1:
        raise ValueError("Stream count/cap differs from the complete cohort")
    index = read_exact(reader, index_bytes)
    manifest = manifest_from_index(index)

    class Replay:
        def __init__(self, prefix: bytes):
            self.prefix = prefix

        def read(self, size: int) -> bytes:
            if self.prefix:
                block, self.prefix = self.prefix[:size], self.prefix[size:]
                return block
            return reader.read(size)

    result = target_stream(Replay(header + index), cache, manifest, funding)
    receipt = {
        "status": "original_complete_public_triplet_cache_private_relay_verified",
        "relay_version": VERSION,
        "node": node,
        "CPU_job": job,
        "cache": str(cache),
        "index_sha256": INDEX_SHA256,
        "cohort_sha256": COHORT_SHA256,
        "manifest_sha256": canonical_sha256(manifest),
        "merged_receipt_sha256": MERGED_RECEIPT_SHA256,
        "CPU_receipt_sha256": CPU_RECEIPT_SHA256,
        "funding_sha256": funding_sha,
        "studies": STUDIES,
        **result,
        "every_original_byte_size_SHA_count_EOF_layout_verified": True,
        "GPU_used": False,
        "no_Mac_or_shared_pixel_storage": True,
        "trainer_root_checkpoint_or_journals_moved": False,
    }
    with (control / VERIFIED).open("x") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return receipt


def reconcile(cache: Path, control: Path) -> dict:
    """Read-only after both original handles stop; aggregate-only, never deletes/retries."""
    out = {
        "read_only": True,
        "intent_present": (control / INTENT).exists(),
        "verified_present": (control / VERIFIED).exists(),
        "cache_present": cache.exists(),
    }
    if not cache.exists():
        return out | {"verified_prefix_files": 0, "unverified_last_file": False}
    _real_dir(cache)
    names = sorted(str(p.relative_to(cache)) for p in cache.rglob("*") if not p.is_dir())
    if INDEX not in names:
        return out | {"verified_prefix_files": 0, "unverified_last_file": bool(names)}
    try:
        manifest = manifest_from_index((cache / INDEX).read_bytes())
    except ValueError:
        return out | {"verified_prefix_files": 0, "unverified_last_file": True}
    order = list(manifest)
    actual = [n for n in order if n in names]
    if set(names) != set(actual) or actual != order[: len(actual)]:
        raise ValueError("Target is not an exact original stream prefix")
    complete, size, partial = 0, 0, False
    for position, name in enumerate(actual):
        path, row = cache / name, manifest[name]
        before = signature(path)
        valid = _regular(path).st_size == row["bytes"] and sha(path) == row["sha256"]
        if signature(path) != before:
            raise ValueError("Original transfer is still changing the target")
        if not valid:
            if position != len(actual) - 1:
                raise ValueError("Earlier copied member is corrupt")
            partial = True
            break
        complete += 1
        size += row["bytes"]
    return out | {
        "verified_prefix_files": complete,
        "verified_prefix_bytes": size,
        "missing_files": len(order) - complete,
        "unverified_last_file": partial,
    }


def pipeline(relay_dir: Path, plan_sha: str, funding_sha: str, *, time: str = "02:00:00"):
    """Root's single source|target command; one CPU and --gres=none per endpoint."""
    py = ["/usr/bin/python3.12", "-B", str(relay_dir / "triplet_cache_relay.py")]
    common = ["--ntasks=1", "--cpus-per-task=1", "--gres=none", "--mem=512M", f"--time={time}"]
    src = ["srun", f"--jobid={SOURCE_JOB}", "--overlap", *common, *py, "source"]
    src += ["--plan", str(CONTROL / PLAN), "--plan-sha", plan_sha]
    dst = ["srun", "--nodes=1", f"--nodelist={TARGET_NODE}", *common]
    dst += ["--job-name=rsna-triplet-cache-private-target", *py, "target"]
    dst += ["--funding", str(CONTROL / "funding.json"), "--funding-sha", funding_sha]
    return "set -o pipefail; " + shlex.join(src) + " | " + shlex.join(dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("plan", "source", "target", "reconcile"))
    p.add_argument("--plan", type=Path, default=CONTROL / PLAN)
    p.add_argument("--plan-sha")
    p.add_argument("--funding", type=Path, default=CONTROL / "funding.json")
    p.add_argument("--funding-sha")
    args = p.parse_args()
    if args.mode == "plan":
        require_endpoint(SOURCE_NODE, SOURCE_JOB)
        print(json.dumps(plan(CACHE, MERGED_RECEIPT, CONTROL), sort_keys=True))
    elif args.mode == "source":
        require_endpoint(SOURCE_NODE, SOURCE_JOB)
        source(sys.stdout.buffer, CACHE, args.plan, args.plan_sha)
    elif args.mode == "target":
        receipt = target(
            sys.stdin.buffer, CACHE, CONTROL, args.funding, args.funding_sha, TARGET_NODE
        )
        print(json.dumps(receipt, sort_keys=True))
    else:
        require_endpoint(TARGET_NODE)
        print(json.dumps(reconcile(CACHE, CONTROL), sort_keys=True))


if __name__ == "__main__":
    main()
