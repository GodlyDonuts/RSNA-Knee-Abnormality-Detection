"""Once-only private CPU relay of the remaining dense-suffix state/runtime (issue #80).

PREPARED ONLY; Devin dispatches, funds and transfers nothing. Root alone, on Newton login
stdin/RAM, after the ACTUAL original exit-75 terminal and PR76 ``001/record.json``:

    plan     evc31, one CPU / zero GPU step overlapping 899766: read-only source proof
    source   (same step) | target (evc40, one CPU / zero GPU step)
    reconcile  evc40, read-only after BOTH original handles stop (uncertain transfer only)

The private transfer list is derived only from the original record manifest: every PR76
node-local member (original root incl. immutable epochs and the exit-75 checkpoint,
admission tree, public-cache producer bookkeeping, overlay, runtime, GPU-recovery tree) at
its IDENTICAL absolute path, plus the record and coordinator bytes. Public-cache members
already verified by the once-only triplet relay (exact original index/member hashes and its
immutable evc40 receipt) are NOT retransferred; the target gives them a fresh node-local
stat/signature proof. Existing destination files need an exact full-byte proof and are never
overwritten. New bytes are written exclusively to node-local staging, size/SHA verified and
only then hard-linked into place, so an interruption never promotes a partial member.

Success writes PR76's exact ``001/transport-receipt.json``. Per-member names/signatures stay
in node-local control files; stdout/Git receive aggregates and hashes only. No GPU, retry,
overwrite, deletion credit, shared growth or original-state deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import stat
import struct
import sys
from pathlib import Path

if __package__:
    from scripts.newton import triplet_cache_relay as triplet
    from scripts.newton.native_cache_relay import (
        END,
        MAGIC,
        read_exact,
        sha,
        signature,
        source_stream,
    )
else:
    import triplet_cache_relay as triplet  # type: ignore[no-redef]
    from native_cache_relay import (  # type: ignore[no-redef]
        END,
        MAGIC,
        read_exact,
        sha,
        signature,
        source_stream,
    )

VERSION = "dense_suffix_state_runtime_private_relay_v1"
CONTROL = Path("/tmp/rsna-dense-state-relay-20261014-v1")
SOURCE_NODE, TARGET_NODE, SOURCE_JOB = "evc31", "evc40", "899766"
NODE_LOCAL_PREFIX = "/tmp/"
# PR76 transition contract (scripts/newton/transition_dense_suffix_allocation.py), copied so
# the staged relay never imports evolving code; equality is pinned by tests.
ROOT = Path("/tmp/rsna-dense-suffix-fit-20261014-v1")
SCHEMA = "dense-suffix-allocation-transition-v1"
RECORD_STATUS = "original_exit75_checkpoint_transition_recorded"
TRANSITIONS = "allocation-transitions-v1"
NODE_LOCAL = ("original_root", "admission", "public_cache", "overlay", "runtime", "recovery")
ORIGINAL_FUNDING_SHA256 = "0f5a2d17487ff59256891594b214d063728a8b16124d0daa097dc0b149dc217d"
# Actual once-only triplet relay (dense-triplet-cache-relay-20261014-v1/source-plan-verified).
TRIPLET_RECEIPT = triplet.CONTROL / triplet.VERIFIED
TRIPLET_STATUS = "original_complete_public_triplet_cache_private_relay_verified"
TRIPLET_MANIFEST_SHA256 = "09ea7bc6aaf7c52bda2bb94eff51db37aac8b066bcc795c63d137bfe837699dc"
TRIPLET_FILES, TRIPLET_BYTES = 3404, 27668082285
RESERVE_BYTES = 20 * 2**30
RECORD_CAP_BYTES = 256 * 2**20
CONTROL_CAP_BYTES = 1 * 2**30
MAX_ATTEMPTS = 3
PLAN, VERIFIED, PROOF = "source-plan.json", "target-verified.json", "destination-proof.json"
FUNDING_STATUS = "root_admitted_node_local_dense_state_relay_funding"
PREFIX_STATUS = "root_admitted_interrupted_dense_state_relay_prefix"
FUNDING_KEYS = {
    "status",
    "funded_by",
    "issue",
    "relay_version",
    "source_node",
    "node",
    "control",
    "transition_record_sha256",
    "transition_record_bytes",
    "coordinator_sha256",
    "transfer_list_sha256",
    "stream_files",
    "stream_bytes",
    "triplet_receipt_sha256",
    "envelope_bytes",
    "control_bytes",
    "reserve_bytes",
    "existing_liabilities_bytes",
    "required_free_bytes",
    "shared_future_bytes",
    "deletion_credit_bytes",
    "GPUs_funded",
    "GPU_reserved",
}
PREFIX_KEYS = {
    "status",
    "funded_by",
    "issue",
    "relay_version",
    "interrupted_attempt",
    "reconcile_sha256",
    "transition_record_sha256",
    "funding_sha256",
    "automatic_retry",
}

canonical = triplet.canonical_sha256
_is_sha = triplet._is_sha


def intent_name(attempt: int) -> str:
    return f"target-intent-{attempt:02d}.json"


def reconcile_name(attempt: int) -> str:
    return f"reconcile-{attempt:02d}.json"


def prefix_name(attempt: int) -> str:
    return f"prefix-admission-{attempt:02d}.json"


def walk(base: Path) -> list[Path]:
    """PR76 ``_walk`` semantics: files and links, never descending a linked directory."""
    if base.is_symlink() or not base.exists():
        raise ValueError("Original member root missing or linked")
    if base.is_file():
        return [base]
    items: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        here = Path(dirpath)
        items.extend(here / n for n in sorted(dirnames) if (here / n).is_symlink())
        items.extend(here / name for name in sorted(filenames))
    return sorted(items)


def _no_linked_ancestor(path: Path) -> None:
    for parent in path.parents:
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError("Linked or non-directory ancestor forbidden")


def _normal_inside(path: str, base: str) -> bool:
    return (
        path.startswith("/")
        and os.path.normpath(path) == path
        and "\x00" not in path
        and (path == base or path.startswith(base.rstrip("/") + "/"))
    )


# ------------------------------------------------------------------ record ----


def transitions(value: dict) -> tuple[Path, Path]:
    base = Path(value["member_roots"]["original_root"]) / TRANSITIONS
    return base / "001" / "record.json", base / "coordinator.py"


def validate_record(payload: bytes, record_sha: str) -> dict:
    """Exact PR76 record of the actual original exit-75 terminal; node-local paths only."""
    if not _is_sha(record_sha) or hashlib.sha256(payload).hexdigest() != record_sha:
        raise ValueError("Original transition record bytes differ")
    if len(payload) > RECORD_CAP_BYTES:
        raise ValueError("Transition record exceeds its cap")
    value = json.loads(payload)
    original = value.get("original") if isinstance(value, dict) else None
    manifest = value.get("manifest") if isinstance(value, dict) else None
    roots = value.get("member_roots") if isinstance(value, dict) else None
    if (
        not isinstance(original, dict)
        or not isinstance(manifest, list)
        or not isinstance(roots, dict)
        or value.get("schema") != SCHEMA
        or value.get("status") != RECORD_STATUS
        or value.get("index") != 1
        or original.get("node") != SOURCE_NODE
        or original.get("parent_job") != SOURCE_JOB
        or original.get("funding_sha256") != ORIGINAL_FUNDING_SHA256
        or value.get("arm") != "dense"
        or not _is_sha(value.get("checkpoint_sha256"))
        or not _is_sha(value.get("coordinator_sha256"))
        or value.get("transport_classes") != list(NODE_LOCAL)
        or roots.get("original_root") != str(ROOT)
        or value.get("original_state_deleted") is not False
        or canonical(manifest) != value.get("manifest_sha256")
        or value.get("members") != len(manifest)
    ):
        raise ValueError("Exact PR76 exit-75 transition record required")
    for cls in NODE_LOCAL:
        base = roots.get(cls)
        if not isinstance(base, str) or not _normal_inside(base, base):
            raise ValueError("Node-local member root missing")
        if not base.startswith(NODE_LOCAL_PREFIX) or base == NODE_LOCAL_PREFIX.rstrip("/"):
            raise ValueError("Node-local class outside node-local storage; no shared growth")
    local = [m for m in manifest if isinstance(m, dict) and m.get("class") in NODE_LOCAL]
    links = set()
    for m in manifest:
        if not isinstance(m, dict) or m.get("class") not in roots:
            raise ValueError("Unknown manifest class")
    for m in local:
        path = m.get("path")
        if not isinstance(path, str) or not _normal_inside(path, roots[m["class"]]):
            raise ValueError("Traversing or misplaced original member path")
        if m["class"] == "original_root" and _normal_inside(path, str(ROOT / TRANSITIONS)):
            raise ValueError("Transition controls are never manifest members")
        if m.get("kind") == "symlink":
            if (
                set(m) != {"class", "path", "kind", "target"}
                or not isinstance(m["target"], str)
                or not m["target"]
                or "\x00" in m["target"]
            ):
                raise ValueError("Invalid original link member")
            links.add(path)
        elif m.get("kind") == "file":
            if (
                set(m) != {"class", "path", "kind", "bytes", "sha256", "executable"}
                or type(m["bytes"]) is not int
                or m["bytes"] < 0
                or not _is_sha(m["sha256"])
                or type(m["executable"]) is not bool
            ):
                raise ValueError("Invalid original file member")
        else:
            raise ValueError("Unsupported original member kind")
    for m in local:
        if any(str(p) in links for p in Path(m["path"]).parents):
            raise ValueError("Member below an original link is never written through it")
    if (
        value.get("transport_members") != len(local)
        or value.get("transport_bytes") != sum(m.get("bytes", 0) for m in local)
        or len({m["path"] for m in local}) != len(local)
    ):
        raise ValueError("Record transport aggregates differ from its manifest")
    return value


def relayed_cache(cache: Path | None = None) -> dict[str, dict]:
    """Members verified by the once-only triplet relay, from the pinned original index."""
    cache = triplet.CACHE if cache is None else cache
    _no_linked_ancestor(cache / triplet.INDEX)
    rows = triplet.manifest_from_index((cache / triplet.INDEX).read_bytes())
    if canonical(rows) != TRIPLET_MANIFEST_SHA256 or len(rows) != TRIPLET_FILES:
        raise ValueError("Triplet relay manifest differs from the actual relayed manifest")
    if triplet.total_bytes(rows) != TRIPLET_BYTES:
        raise ValueError("Triplet relay bytes differ from the actual relayed bytes")
    return {str(cache / name): row for name, row in rows.items()}


def transfer_list(value: dict, payload: bytes, cached: dict[str, dict]) -> tuple[dict, dict]:
    """Ordered stream rows (record, coordinator, remaining node-local files) and skip rows."""
    local = [m for m in value["manifest"] if m["class"] in NODE_LOCAL]
    record_path, coordinator = transitions(value)
    rows = {
        str(record_path): {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        str(coordinator): {"bytes": None, "sha256": value["coordinator_sha256"]},
    }
    skip = {}
    for m in local:
        if m["kind"] != "file":
            continue
        if m["path"] in cached:
            row = cached[m["path"]]
            if (
                m["class"] != "public_cache"
                or m["bytes"] != row["bytes"]
                or m["sha256"] != row["sha256"]
                or m["executable"]
            ):
                raise ValueError("Record and triplet relay disagree on a cache member")
            skip[m["path"]] = {"bytes": m["bytes"], "sha256": m["sha256"]}
        else:
            rows[m["path"]] = {"bytes": m["bytes"], "sha256": m["sha256"]}
    if set(skip) != set(cached):
        raise ValueError("Record does not contain the complete relayed cache member set")
    return rows, skip


def list_digest(rows: dict) -> str:
    return canonical([[name, r["bytes"], r["sha256"]] for name, r in rows.items()])


def _coordinator_bytes(rows: dict, value: dict, size: int) -> dict:
    out = dict(rows)
    out[str(transitions(value)[1])] = {"bytes": size, "sha256": value["coordinator_sha256"]}
    return out


# ------------------------------------------------------------------ source ----


def source_tree(value: dict) -> None:
    """Exact original node-local member set, sizes and links at the identical paths."""
    roots = value["member_roots"]
    for cls in NODE_LOCAL:
        wanted = [m for m in value["manifest"] if m["class"] == cls]
        base = Path(roots[cls])
        _no_linked_ancestor(base)
        exclude = ROOT / TRANSITIONS
        have = {str(p) for p in walk(base) if not (p == exclude or exclude in p.parents)}
        if have != {m["path"] for m in wanted}:
            raise ValueError("Source members differ from the recorded original set")
        for m in wanted:
            path = Path(m["path"])
            if m["kind"] == "symlink":
                if not path.is_symlink() or os.readlink(path) != m["target"]:
                    raise ValueError("Source link differs from the record")
            elif not stat.S_ISREG(os.lstat(path).st_mode) or path.stat().st_size != m["bytes"]:
                raise ValueError("Source member size/type differs from the record")
    record_path, coordinator = transitions(value)
    controls = {str(p) for p in walk(ROOT / TRANSITIONS) if not p.is_dir()}
    if controls != {str(record_path), str(coordinator)}:
        raise ValueError("Unexpected transition control state at transport time")
    if sha(coordinator) != value["coordinator_sha256"]:
        raise ValueError("Recorded coordinator bytes changed")


def _source_rows(record_path: Path, record_sha: str):
    _no_linked_ancestor(record_path)
    payload = record_path.read_bytes()
    value = validate_record(payload, record_sha)
    if record_path != transitions(value)[0]:
        raise ValueError("Record not at its original transition path")
    source_tree(value)
    rows, skip = transfer_list(value, payload, relayed_cache())
    rows = _coordinator_bytes(rows, value, transitions(value)[1].stat().st_size)
    return value, rows, skip


def plan(record_path: Path, record_sha: str, control: Path) -> dict:
    value, rows, skip = _source_rows(record_path, record_sha)
    signatures = {name: signature(Path(name)) for name in rows}
    control.mkdir(mode=0o700, exist_ok=True)
    triplet._real_dir(control)
    with (control / PLAN).open("x") as f:
        os.chmod(control / PLAN, 0o600)
        json.dump({"relay_version": VERSION, "rows": rows, "signatures": signatures}, f)
    local = [m for m in value["manifest"] if m["class"] in NODE_LOCAL]
    classes = {
        cls: {
            "entries": sum(m["class"] == cls for m in local),
            "bytes": sum(m.get("bytes", 0) for m in local if m["class"] == cls),
        }
        for cls in NODE_LOCAL
    }
    return {
        "status": "private_dense_state_source_plan_read_only",
        "relay_version": VERSION,
        "plan_sha256": sha(control / PLAN),
        "transition_record_sha256": record_sha,
        "transition_record_bytes": record_path.stat().st_size,
        "coordinator_sha256": value["coordinator_sha256"],
        "manifest_sha256": value["manifest_sha256"],
        "transport_members": value["transport_members"],
        "transport_bytes": value["transport_bytes"],
        "classes": classes,
        "symlinks": sum(m["kind"] == "symlink" for m in local),
        "already_relayed_cache_files": len(skip),
        "already_relayed_cache_bytes": sum(r["bytes"] for r in skip.values()),
        "transfer_list_sha256": list_digest(rows),
        "stream_files": len(rows),
        "stream_bytes": sum(r["bytes"] for r in rows.values()),
        "GPU_used": False,
        "pixels_or_checkpoints_exported": False,
    }


def source(writer, record_path: Path, record_sha: str, plan_path: Path, plan_sha: str) -> None:
    if not _is_sha(plan_sha) or sha(plan_path) != plan_sha:
        raise ValueError("Original private source plan differs")
    private = json.loads(plan_path.read_bytes())
    _, rows, _ = _source_rows(record_path, record_sha)
    if private.get("rows") != rows or private.get("relay_version") != VERSION:
        raise ValueError("Source plan does not bind this exact transfer list")
    proof = {"signatures": private["signatures"], "registry_file_sha256": list_digest(rows)}
    source_stream(writer, Path("/"), rows, proof, list_digest(rows))


# ------------------------------------------------------------------ target ----


def load_funding(path: Path, digest: str, control: Path, node: str) -> dict:
    if not _is_sha(digest) or sha(path) != digest:
        raise ValueError("Separately admitted funding proof differs")
    f = json.loads(path.read_bytes())
    if not isinstance(f, dict) or set(f) != FUNDING_KEYS:
        raise ValueError("Complete explicit node-local relay funding required")
    ints = (
        "transition_record_bytes",
        "stream_files",
        "stream_bytes",
        "envelope_bytes",
        "control_bytes",
        "reserve_bytes",
        "existing_liabilities_bytes",
        "required_free_bytes",
    )
    if (
        any(type(f[k]) is not int or f[k] < 0 for k in ints)
        or f["status"] != FUNDING_STATUS
        or f["funded_by"] != "root"
        or f["issue"] != 80
        or f["relay_version"] != VERSION
        or f["source_node"] != SOURCE_NODE
        or f["node"] != node
        or f["control"] != str(control)
        or not all(
            _is_sha(f[k])
            for k in (
                "transition_record_sha256",
                "coordinator_sha256",
                "transfer_list_sha256",
                "triplet_receipt_sha256",
            )
        )
        or not 0 < f["transition_record_bytes"] <= RECORD_CAP_BYTES
        or f["stream_files"] < 2
        or f["envelope_bytes"] < f["stream_bytes"]
        or f["control_bytes"] > CONTROL_CAP_BYTES
        or f["reserve_bytes"] < RESERVE_BYTES
        or f["existing_liabilities_bytes"] <= 0
        or f["required_free_bytes"]
        != f["existing_liabilities_bytes"]
        + f["envelope_bytes"]
        + f["control_bytes"]
        + f["reserve_bytes"]
        or f["shared_future_bytes"] != 0
        or f["deletion_credit_bytes"] != 0
        or f["GPUs_funded"] != 0
        or f["GPU_reserved"] is not False
    ):
        raise ValueError("Explicit fresh node-local envelope/liabilities/20GiB reserve required")
    return f


def triplet_receipt(funding: dict, path: Path | None = None) -> os.stat_result:
    """The ACTUAL immutable evc40 triplet receipt; returns its stat for the ctime proof."""
    path = TRIPLET_RECEIPT if path is None else path
    _no_linked_ancestor(path)
    if path.is_symlink() or sha(path) != funding["triplet_receipt_sha256"]:
        raise ValueError("Actual immutable triplet relay receipt differs")
    r = json.loads(path.read_bytes())
    if (
        r.get("status") != TRIPLET_STATUS
        or r.get("relay_version") != triplet.VERSION
        or r.get("node") != TARGET_NODE
        or r.get("cache") != str(triplet.CACHE)
        or r.get("index_sha256") != triplet.INDEX_SHA256
        or r.get("manifest_sha256") != TRIPLET_MANIFEST_SHA256
        or r.get("members") != TRIPLET_FILES
        or r.get("bytes") != TRIPLET_BYTES
        or r.get("every_original_byte_size_SHA_count_EOF_layout_verified") is not True
        or r.get("GPU_used") is not False
    ):
        raise ValueError("Triplet receipt does not bind the complete relayed cache")
    return os.lstat(path)


def cache_proof(skip: dict, receipt: os.stat_result) -> dict:
    """Fresh node-local stat/signature proof: unchanged since the immutable receipt."""
    proof = {}
    for name, row in skip.items():
        path = Path(name)
        _no_linked_ancestor(path)
        try:
            s = os.lstat(path)
        except FileNotFoundError as error:
            raise ValueError("Relayed cache member missing since its immutable receipt") from error
        if (
            not stat.S_ISREG(s.st_mode)
            or s.st_nlink != 1
            or s.st_size != row["bytes"]
            or stat.S_IMODE(s.st_mode) != 0o400
            or s.st_ctime_ns > receipt.st_mtime_ns
            or s.st_dev != receipt.st_dev
        ):
            raise ValueError("Relayed cache member changed after its immutable receipt")
        proof[name] = signature(path)
    return proof


def _mode(cls: str | None, executable: bool) -> int:
    """Original root stays owner-writable for the resumed trainer; other members read-only."""
    if cls == "original_root":
        return 0o700 if executable else 0o600
    return 0o500 if executable else 0o400


def _exact_file(path: Path, row: dict, executable: bool | None) -> list:
    before = signature(path)
    if (
        not stat.S_ISREG(os.lstat(path).st_mode)
        or path.stat().st_size != row["bytes"]
        or sha(path) != row["sha256"]
        or (executable is not None and os.access(path, os.X_OK) != executable)
        or signature(path) != before
    ):
        raise ValueError("Existing destination member differs; never overwritten")
    return before


def _local(value: dict) -> dict[str, dict]:
    return {m["path"]: m for m in value["manifest"] if m["class"] in NODE_LOCAL}


def _nearest(path: Path) -> Path:
    while not path.exists() and not path.is_symlink():
        path = path.parent
    return path


def preflight(value: dict, rows: dict, staging_dev: int) -> dict[str, list]:
    """Before any write: existing destination members need exact proof; nothing extra."""
    local, existing = _local(value), {}
    record_path, coordinator = transitions(value)
    for cls in NODE_LOCAL:
        base = Path(value["member_roots"][cls])
        _no_linked_ancestor(base)
        if base.exists() or base.is_symlink():
            exclude = ROOT / TRANSITIONS
            have = {str(p) for p in walk(base) if not (p == exclude or exclude in p.parents)}
            if not have <= {p for p, m in local.items() if m["class"] == cls}:
                raise ValueError("Unexpected destination member; never overwritten")
    controls = ROOT / TRANSITIONS
    if controls.exists() or controls.is_symlink():
        have = {str(p) for p in walk(controls) if not p.is_dir()}
        if not have <= {str(record_path), str(coordinator)}:
            raise ValueError("Destination transition already continued; repetition refused")
    for name, row in rows.items():
        path = Path(name)
        _no_linked_ancestor(path)
        if os.lstat(_nearest(path.parent)).st_dev != staging_dev:
            raise ValueError("Destination is not on the funded node-local filesystem")
        if path.exists() or path.is_symlink():
            m = local.get(name)
            existing[name] = _exact_file(path, row, m["executable"] if m else None)
    for name, m in local.items():
        path = Path(name)
        if m["kind"] == "symlink" and (path.exists() or path.is_symlink()):
            _no_linked_ancestor(path)
            if not path.is_symlink() or os.readlink(path) != m["target"]:
                raise ValueError("Existing destination link differs; never overwritten")
    return existing


def _mkdirs(path: Path) -> None:
    missing = []
    while not path.exists() and not path.is_symlink():
        missing.append(path)
        path = path.parent
    triplet._real_dir(path)
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _check_attempt(control: Path, attempt: int, prefix: Path | None, prefix_sha: str | None):
    if type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPTS:
        raise ValueError("Bounded explicit attempt number required")
    if (control / VERIFIED).exists():
        raise ValueError("Verified relay forbids repetition")
    later = [control / intent_name(n) for n in range(attempt, MAX_ATTEMPTS + 1)]
    earlier = [control / intent_name(n) for n in range(1, attempt)]
    if any(p.exists() for p in later) or not all(p.exists() for p in earlier):
        raise ValueError("Existing intent forbids repetition; no automatic retry")
    if attempt == 1:
        if prefix is not None:
            raise ValueError("A first attempt admits no prefix")
        return None
    if prefix != control / prefix_name(attempt - 1) or not prefix_sha or sha(prefix) != prefix_sha:
        raise ValueError("Root's separate prefix admission of the interrupted attempt required")
    admitted = json.loads(prefix.read_bytes())
    reconciled = control / reconcile_name(attempt - 1)
    if (
        set(admitted) != PREFIX_KEYS
        or admitted["status"] != PREFIX_STATUS
        or admitted["funded_by"] != "root"
        or admitted["issue"] != 80
        or admitted["relay_version"] != VERSION
        or admitted["interrupted_attempt"] != attempt - 1
        or admitted["automatic_retry"] is not False
        or not reconciled.exists()
        or sha(reconciled) != admitted["reconcile_sha256"]
    ):
        raise ValueError("Prefix admission does not bind the read-only reconciliation")
    return admitted


def target(
    reader,
    control: Path,
    funding_path: Path,
    funding_sha: str,
    node: str = TARGET_NODE,
    *,
    attempt: int = 1,
    prefix: Path | None = None,
    prefix_sha: str | None = None,
) -> dict:
    job = triplet.require_endpoint(node)
    triplet._real_dir(control)
    funding = load_funding(funding_path, funding_sha, control, node)
    admitted = _check_attempt(control, attempt, prefix, prefix_sha)
    if admitted is not None and (
        admitted["transition_record_sha256"] != funding["transition_record_sha256"]
        or admitted["funding_sha256"] != funding_sha
    ):
        raise ValueError("Prefix admission binds a different record/funding")
    receipt = triplet_receipt(funding)
    cached = relayed_cache()
    if shutil.disk_usage(control).free < funding["required_free_bytes"]:
        raise ValueError("Funded node-local reserve is insufficient")
    header = read_exact(reader, len(MAGIC) + 32 + 16)
    digest = header[len(MAGIC) : -16].hex()
    count, total = struct.unpack(">QQ", header[-16:])
    if (
        header[: len(MAGIC)] != MAGIC
        or digest != funding["transfer_list_sha256"]
        or (count, total) != (funding["stream_files"], funding["stream_bytes"])
    ):
        raise ValueError("Stream header differs from the funded transfer list/cap")
    payload = read_exact(reader, funding["transition_record_bytes"])
    value = validate_record(payload, funding["transition_record_sha256"])
    if value["coordinator_sha256"] != funding["coordinator_sha256"]:
        raise ValueError("Recorded coordinator differs from the funded coordinator")
    rows, skip = transfer_list(value, payload, cached)
    coordinator = str(transitions(value)[1])
    rest = total - sum(r["bytes"] for n, r in rows.items() if n != coordinator)
    rows = _coordinator_bytes(rows, value, rest)
    if rest < 0 or list_digest(rows) != digest or len(rows) != count:
        raise ValueError("Transfer list differs from the original record")
    proofs = cache_proof(skip, receipt)
    staging = control / f"staging-{attempt:02d}"
    existing = preflight(value, rows, os.lstat(control).st_dev)
    with (control / intent_name(attempt)).open("x") as f:
        json.dump(
            {
                "relay_version": VERSION,
                "attempt": attempt,
                "transition_record_sha256": funding["transition_record_sha256"],
                "funding_sha256": funding_sha,
                "stream_files": count,
                "stream_bytes": total,
                "existing_exact_files": len(existing),
                "automatic_retry": False,
                "overwrite": False,
            },
            f,
        )
    staging.mkdir(mode=0o700)
    local = _local(value)
    written, new_bytes = {}, 0
    for position, (name, row) in enumerate(rows.items()):
        if shutil.disk_usage(control).free < funding["required_free_bytes"] - new_bytes:
            raise ValueError("Funded reserve breached during transfer")
        digest_ = hashlib.sha256()
        part = None if name in existing else staging / f"{position:08d}.part"
        output = part.open("xb") if part is not None else None
        try:
            if part is not None:
                os.chmod(part, 0o600)
            remaining = row["bytes"]
            while remaining:
                block = payload if position == 0 else read_exact(reader, min(1 << 20, remaining))
                digest_.update(block)
                if output is not None:
                    output.write(block)
                remaining -= len(block)
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
        finally:
            if output is not None:
                output.close()
        if digest_.hexdigest() != row["sha256"]:
            raise ValueError("Streamed original member hash differs")
        if part is None:
            continue
        m = local.get(name)
        os.chmod(part, _mode(m["class"] if m else None, bool(m and m["executable"])))
        path = Path(name)
        _mkdirs(path.parent)
        _no_linked_ancestor(path)
        os.link(part, path)  # exclusive: never replaces an existing member
        os.unlink(part)
        _fsync_dir(path.parent)
        written[name] = signature(path)
        new_bytes += row["bytes"]
    if read_exact(reader, len(END)) != END or reader.read(1):
        raise ValueError("Original stream trailer or EOF differs")
    links = 0
    for name, m in local.items():
        path = Path(name)
        if m["kind"] == "symlink" and not path.is_symlink():
            _mkdirs(path.parent)
            _no_linked_ancestor(path)
            os.symlink(m["target"], path)
            _fsync_dir(path.parent)
            links += 1
    staging.rmdir()  # empty by construction; never removes a member
    private = final_check(value, rows, existing, written, skip, receipt)
    with (control / PROOF).open("x") as f:
        os.chmod(control / PROOF, 0o600)
        json.dump(private | {"cache": proofs}, f, sort_keys=True)
    transport = {
        "executed_by": "root",
        "transition_record_sha256": funding["transition_record_sha256"],
        "manifest_sha256": value["manifest_sha256"],
        "source_node": SOURCE_NODE,
        "destination_node": node,
        "members": value["transport_members"],
        "bytes": value["transport_bytes"],
        "original_state_deleted": False,
        "automatic_retry": False,
    }
    receipt_path = transitions(value)[0].parent / "transport-receipt.json"
    with receipt_path.open("x") as f:
        json.dump(transport, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    result = {
        "status": "original_dense_state_runtime_private_relay_verified",
        "relay_version": VERSION,
        "attempt": attempt,
        "node": node,
        "CPU_job": job,
        "transition_record_sha256": funding["transition_record_sha256"],
        "manifest_sha256": value["manifest_sha256"],
        "funding_sha256": funding_sha,
        "triplet_receipt_sha256": funding["triplet_receipt_sha256"],
        "transport_receipt_sha256": sha(receipt_path),
        "transport_members": value["transport_members"],
        "transport_bytes": value["transport_bytes"],
        "stream_files": count,
        "stream_bytes": total,
        "new_files": len(written),
        "new_bytes": new_bytes,
        "existing_exact_files": len(existing),
        "already_relayed_cache_files": len(skip),
        "links_created": links,
        "destination_proof_sha256": sha(control / PROOF),
        "free_bytes_after": shutil.disk_usage(control).free,
        "complete_original_member_set_verified": True,
        "GPU_used": False,
        "original_state_deleted": False,
        "no_Mac_or_shared_storage": True,
    }
    with (control / VERIFIED).open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


def final_check(value, rows, existing, written, skip, receipt) -> dict:
    """Complete original node-local member set at identical paths, nothing extra."""
    roots, local = value["member_roots"], _local(value)
    exclude = ROOT / TRANSITIONS
    for cls in NODE_LOCAL:
        have = {
            str(p) for p in walk(Path(roots[cls])) if not (p == exclude or exclude in p.parents)
        }
        if have != {p for p, m in local.items() if m["class"] == cls}:
            raise ValueError("Destination members differ from the complete original set")
    for name, m in local.items():
        path = Path(name)
        if m["kind"] == "symlink":
            if not path.is_symlink() or os.readlink(path) != m["target"]:
                raise ValueError("Destination link differs from original")
        elif os.access(path, os.X_OK) != m["executable"]:
            raise ValueError("Destination mode differs from original executability")
    for name in rows:
        expected = existing.get(name) or written.get(name)
        if signature(Path(name)) != expected:
            raise ValueError("Destination member changed during the relay")
    cache_proof(skip, receipt)
    return {"existing": existing, "written": written}


def reconcile(control: Path, funding_path: Path, funding_sha: str, attempt: int) -> dict:
    """Read-only after both original handles stop; private prefix proof, never deletes."""
    funding = load_funding(funding_path, funding_sha, control, TARGET_NODE)
    out = {
        "read_only": True,
        "attempt": attempt,
        "intent_present": (control / intent_name(attempt)).exists(),
        "verified_present": (control / VERIFIED).exists(),
    }
    if not out["intent_present"] or out["verified_present"]:
        raise ValueError("Only an interrupted intent without receipt is reconciled")
    staging = control / f"staging-{attempt:02d}"
    parts = sorted(staging.glob("*.part")) if staging.exists() else []
    record_path = ROOT / TRANSITIONS / "001" / "record.json"
    verified: dict[str, list] = {}
    differing = absent = 0
    if record_path.exists() and sha(record_path) == funding["transition_record_sha256"]:
        payload = record_path.read_bytes()
        value = validate_record(payload, funding["transition_record_sha256"])
        rows, _ = transfer_list(value, payload, relayed_cache())
        for name, row in rows.items():
            path = Path(name)
            if not (path.exists() or path.is_symlink()):
                absent += 1
                continue
            before = signature(path)
            ok = row["bytes"] in (None, before[2]) and sha(path) == row["sha256"]
            if signature(path) != before:
                raise ValueError("Original transfer is still changing the destination")
            if ok:
                verified[name] = before
            else:
                differing += 1
    with (control / reconcile_name(attempt)).open("x") as f:
        os.chmod(control / reconcile_name(attempt), 0o600)
        json.dump({"verified": verified}, f, sort_keys=True)
    return out | {
        "record_present": record_path.exists(),
        "verified_files": len(verified),
        "verified_bytes": sum(s[2] for s in verified.values()),
        "absent_files": absent,
        "differing_files": differing,
        "unpromoted_staging_parts": len(parts),
        "unpromoted_staging_bytes": sum(p.stat().st_size for p in parts),
        "reconcile_sha256": sha(control / reconcile_name(attempt)),
    }


def pipeline(relay_dir: Path, record_sha: str, plan_sha: str, funding_sha: str, **kw) -> str:
    """Root's single source|target command; one CPU and --gres=none per endpoint."""
    time, attempt = kw.get("time", "03:00:00"), kw.get("attempt", 1)
    py = ["/usr/bin/python3.12", "-B", str(relay_dir / "dense_state_relay.py")]
    common = ["--ntasks=1", "--cpus-per-task=1", "--gres=none", "--mem=1G", f"--time={time}"]
    record = ROOT / TRANSITIONS / "001" / "record.json"
    src = ["srun", f"--jobid={SOURCE_JOB}", "--overlap", *common, *py, "source"]
    src += ["--record", str(record), "--record-sha", record_sha]
    src += ["--plan", str(CONTROL / PLAN), "--plan-sha", plan_sha]
    dst = ["srun", "--nodes=1", f"--nodelist={TARGET_NODE}", *common]
    dst += ["--job-name=rsna-dense-state-private-target", *py, "target"]
    dst += ["--funding", str(CONTROL / "funding.json"), "--funding-sha", funding_sha]
    dst += ["--attempt", str(attempt)]
    if attempt > 1:
        dst += ["--prefix", str(CONTROL / prefix_name(attempt - 1))]
        dst += ["--prefix-sha", kw["prefix_sha"]]
    return "set -o pipefail; " + shlex.join(src) + " | " + shlex.join(dst)


def main() -> None:
    os.umask(0o077)
    sys.dont_write_bytecode = True
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("plan", "source", "target", "reconcile"))
    p.add_argument("--record", type=Path, default=ROOT / TRANSITIONS / "001" / "record.json")
    p.add_argument("--record-sha")
    p.add_argument("--plan", type=Path, default=CONTROL / PLAN)
    p.add_argument("--plan-sha")
    p.add_argument("--funding", type=Path, default=CONTROL / "funding.json")
    p.add_argument("--funding-sha")
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--prefix", type=Path)
    p.add_argument("--prefix-sha")
    a = p.parse_args()
    if a.mode == "plan":
        triplet.require_endpoint(SOURCE_NODE, SOURCE_JOB)
        print(json.dumps(plan(a.record, a.record_sha, CONTROL), sort_keys=True))
    elif a.mode == "source":
        triplet.require_endpoint(SOURCE_NODE, SOURCE_JOB)
        source(sys.stdout.buffer, a.record, a.record_sha, a.plan, a.plan_sha)
    elif a.mode == "target":
        result = target(
            sys.stdin.buffer,
            CONTROL,
            a.funding,
            a.funding_sha,
            attempt=a.attempt,
            prefix=a.prefix,
            prefix_sha=a.prefix_sha,
        )
        print(json.dumps(result, sort_keys=True))
    else:
        triplet.require_endpoint(TARGET_NODE)
        print(json.dumps(reconcile(CONTROL, a.funding, a.funding_sha, a.attempt), sort_keys=True))


if __name__ == "__main__":
    main()
