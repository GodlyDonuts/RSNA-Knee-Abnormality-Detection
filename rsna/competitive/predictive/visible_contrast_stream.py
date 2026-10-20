"""Bounded prospective private contrast transport; no scheduler or admission.

The sender can consume visible_contrast_cache.iter_units without retaining a
second complete source-node cache. Both endpoints need separately accepted
original terminal, source, membership, storage and timeout bindings. No current
training job or owner calls this module. Receivers preserve all partial outputs
on error and never overwrite or automatically retry an existing destination.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from rsna.competitive.predictive import cache, context_cache, visible_contrast_cache

VERSION = "one_pass_normalized_context_stream_with_original_control_v1"
HEADER_CAP = 64 * 1024
CHUNK = 1024 * 1024
HASH_KEYS = {"raw_terminal_sha256", "context_terminal_sha256", "raw_index_sha256",
             "source_context_index_sha256", "roster_sha256", "candidate_sha256",
             "fast_helper_sha256", "preparer_sha256"}
MANIFEST_KEYS = HASH_KEYS | {"version", "source_git", "studies", "side", "halo", "context_cap", "classes"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate_manifest(manifest):
    if (set(manifest) != MANIFEST_KEYS or manifest["version"] != VERSION
            or not re.fullmatch(r"[0-9a-f]{40}", manifest["source_git"])
            or any(not re.fullmatch(r"[0-9a-f]{64}", manifest[k]) for k in HASH_KEYS)
            or any(type(manifest[k]) is not int or manifest[k] < 1
                   for k in ("studies", "side", "context_cap", "classes"))
            or type(manifest["halo"]) is not int or manifest["halo"] < 0):
        raise ValueError("Exact separately admitted stream manifest required")


def check_originals(original_rows, manifest):
    validate_manifest(manifest)
    uids = [r["uid"] for r in original_rows]
    if (len(uids) != manifest["studies"] or any(not isinstance(u, str) or not u for u in uids)
            or uids != sorted(set(uids))
            or hashlib.sha256(canonical(uids)).hexdigest() != manifest["roster_sha256"]):
        raise ValueError("Exact complete original control roster required")
    paths = set()
    for row in original_rows:
        path, proof, provenance = Path(row["path"]), row["proof"], row["provenance"]
        if (path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid()
                or path.resolve() in paths or proof["version"] != context_cache.VERSION
                or cache.signature(path) != proof["stat"]
                or any(provenance[k] != manifest[k] for k in ("side", "halo", "context_cap", "classes"))):
            raise ValueError("Original control file, source recipe or byte proof differs")
        paths.add(path.resolve())


def write_header(stream, value):
    body = canonical(value) + b"\n"
    if len(body) >= HEADER_CAP:
        raise ValueError("Private control header exceeds its bound")
    stream.write(body)


def read_header(stream):
    body = stream.readline(HEADER_CAP)
    if not body.endswith(b"\n") or len(body) >= HEADER_CAP:
        raise ValueError("Truncated or oversized private control header")
    return json.loads(body)


def check_row(row, original, manifest, *, per_study_cap_bytes):
    if set(row) != {"uid", "proof", "provenance"}:
        raise ValueError("Only the exact new input proof may enter a study header")
    proof, provenance = row["proof"], row["provenance"]
    if (set(proof) != {"version", "bytes", "sha256"} or proof["version"] != context_cache.VERSION
            or type(proof["bytes"]) is not int or not 0 < proof["bytes"] <= per_study_cap_bytes
            or not re.fullmatch(r"[0-9a-f]{64}", proof["sha256"])):
        raise ValueError("Bounded original normalized input proof required")
    expected = {"preparation_version": visible_contrast_cache.VERSION,
                **{k: manifest[k] for k in ("source_git", "candidate_sha256", "fast_helper_sha256",
                                            "side", "halo", "context_cap", "classes")},
                "original_raw_cache_sha256": original["provenance"]["original_raw_cache_sha256"],
                "original_context_sha256": original["proof"]["sha256"],
                "role": "normalized", "metadata_width": 10}
    if row["uid"] != original["uid"] or provenance != expected:
        raise ValueError("Study order or exact raw/control/candidate bindings differ")


def _caps(per_study_cap_bytes, total_cap_bytes):
    if any(type(x) is not int or x < 1 for x in (per_study_cap_bytes, total_cap_bytes)):
        raise ValueError("Separate positive input and total funding bounds required")


def send(stream, units, original_rows, manifest, *, per_study_cap_bytes, total_cap_bytes):
    """Synchronously consume each temporary source before advancing its iterator.

    The caller owns/closes the units iterator even on failure. Standard binary
    buffered streams are required, with timeouts enforced by the CPU wrapper.
    Only the cluster pipe receives bodies; this stream must never pass through
    the Mac or an output/logging collector.
    """
    _caps(per_study_cap_bytes, total_cap_bytes)
    check_originals(original_rows, manifest)
    write_header(stream, {"kind": "begin", "manifest": manifest})
    digest, total, count = hashlib.sha256(), 0, 0
    for ordinal, prepared in units:
        if type(ordinal) is not int or ordinal != count or count >= len(original_rows):
            raise ValueError("Contiguous, unique source study order required")
        row = {"uid": prepared["uid"], "provenance": prepared["provenance"],
               "proof": {k: prepared["proof"][k] for k in ("version", "bytes", "sha256")}}
        check_row(row, original_rows[count], manifest, per_study_cap_bytes=per_study_cap_bytes)
        path = Path(prepared["path"])
        if (path.is_symlink() or path.stat().st_uid != os.getuid()
                or cache.signature(path) != prepared["proof"]["stat"]
                or path.stat().st_size != row["proof"]["bytes"]):
            raise ValueError("Immutable prepared source changed")
        total += row["proof"]["bytes"]
        if total > total_cap_bytes:
            raise ValueError("New normalized cache exceeds separate total funding")
        write_header(stream, {"kind": "study", "ordinal": ordinal, "row": row})
        copied, file_digest = 0, hashlib.sha256()
        with path.open("rb") as src:
            while chunk := src.read(CHUNK):
                stream.write(chunk)
                file_digest.update(chunk)
                copied += len(chunk)
        if (copied != row["proof"]["bytes"] or file_digest.hexdigest() != row["proof"]["sha256"]
                or cache.signature(path) != prepared["proof"]["stat"]):
            raise ValueError("Prepared source changed while streaming; preserve receiver output")
        digest.update(canonical(row) + b"\n")
        count += 1
    if count != len(original_rows):
        raise ValueError("Incomplete source cohort cannot produce a terminal stream")
    terminal = {"kind": "end", "studies": count, "bytes": total,
                "wire_index_sha256": digest.hexdigest()}
    write_header(stream, terminal)
    stream.flush()
    return terminal


def _write_json(path, value, cap):
    body = canonical(value) + b"\n"
    if len(body) > cap:
        raise ValueError("Private control bytes exceed separate bound")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out:
        out.write(body)
        out.flush()
        os.fsync(out.fileno())
    return hashlib.sha256(body).hexdigest()


def receive(stream, destination, original_rows, manifest, *, per_study_cap_bytes,
            total_cap_bytes, index_cap_bytes):
    """Independently copy/hash each input and require complete exact stream EOF.

    A private exclusive destination and intent are created before pixel writes.
    Errors leave partial files and the intent; no accepted receipt is produced.
    A caller must reconcile those originals rather than retry this function.
    """
    _caps(per_study_cap_bytes, total_cap_bytes)
    if type(index_cap_bytes) is not int or index_cap_bytes < 1:
        raise ValueError("Separate private-index funding required")
    check_originals(original_rows, manifest)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Original destination requires read-only reconciliation")
    if read_header(stream) != {"kind": "begin", "manifest": manifest}:
        raise ValueError("Original admitted sender manifest differs")
    destination.mkdir(mode=0o700)
    _write_json(destination / "intent.private.json", {"manifest": manifest,
        "per_study_cap_bytes": per_study_cap_bytes, "total_cap_bytes": total_cap_bytes,
        "index_cap_bytes": index_cap_bytes, "automatic_retry": False}, HEADER_CAP)
    digest, total, index, index_upper = hashlib.sha256(), 0, [], 3
    for ordinal, original in enumerate(original_rows):
        header = read_header(stream)
        if (set(header) != {"kind", "ordinal", "row"} or header["kind"] != "study"
                or type(header["ordinal"]) is not int or header["ordinal"] != ordinal):
            raise ValueError("Incomplete or out-of-order private study stream")
        row = header["row"]
        check_row(row, original, manifest, per_study_cap_bytes=per_study_cap_bytes)
        remaining = row["proof"]["bytes"]
        if total + remaining > total_cap_bytes:
            raise ValueError("Refuse a write beyond independently funded destination")
        path = destination / f"normalized-{ordinal:05d}.private.npz"
        # Reserve the index record before its pixel body. Forty-digit positive
        # placeholders conservatively cover the four owning-node stat integers.
        bound_row = {**row, "path": str(path),
                     "proof": {**row["proof"], "stat": [10**40 - 1] * 4},
                     "original_control": original}
        index_upper += len(canonical(bound_row)) + 1
        if index_upper > index_cap_bytes:
            raise ValueError("Refuse pixels whose private index exceeds its funding")
        file_digest = hashlib.sha256()
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out:
            while remaining:
                chunk = stream.read(min(CHUNK, remaining))
                if not chunk:
                    raise ValueError("Truncated study body; preserve original partial output")
                out.write(chunk)
                file_digest.update(chunk)
                remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
        if path.stat().st_size != row["proof"]["bytes"] or file_digest.hexdigest() != row["proof"]["sha256"]:
            raise ValueError("Independent destination byte/SHA verification failed")
        total += row["proof"]["bytes"]
        digest.update(canonical(row) + b"\n")
        index.append({**row, "path": str(path), "proof": {**row["proof"], "stat": cache.signature(path)},
                      "original_control": original})
    terminal = {"kind": "end", "studies": len(index), "bytes": total,
                "wire_index_sha256": digest.hexdigest()}
    if read_header(stream) != terminal or stream.read(1):
        raise ValueError("Complete terminal byte/hash/count and exact EOF required")
    check_originals(original_rows, manifest)
    index_sha = _write_json(destination / "context-index.private.json", index, index_cap_bytes)
    result = {"version": VERSION, "studies": len(index), "bytes": total,
              "wire_index_sha256": digest.hexdigest(), "private_index_sha256": index_sha,
              "source_manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
              "all_destination_sizes_SHA_EOF_verified": True,
              "training_or_clinical_acceptance": False}
    _write_json(destination / "transfer-accepted.json", result, HEADER_CAP)
    return result
