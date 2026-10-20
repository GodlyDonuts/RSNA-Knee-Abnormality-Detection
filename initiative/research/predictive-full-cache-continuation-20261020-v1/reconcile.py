"""Read-only original-prefix reconciliation for a distinct terminal successor."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import stat
from pathlib import Path


def digest(body):
    return hashlib.sha256(body).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def regular(path):
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid():
        raise ValueError("Expected an owned original regular file")
    return value


def private_directory(path):
    value = Path(path).lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid() or value.st_mode & 0o077:
        raise ValueError("Expected an owned private original directory")


def bound_json(path):
    regular(path)
    return json.loads(Path(path).read_bytes())


def signature(path):
    value = regular(path)
    return [value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def terminal_receipt(sacct, squeue, *, job_id, allowed_states=("TIMEOUT",)):
    """Only exact root-job terminal accounting plus empty live queue admits work."""
    rows = [line.split("|") for line in sacct.splitlines() if line.strip()]
    matches = [row for row in rows if row[0] == str(job_id)]
    if squeue.strip() or len(matches) != 1 or len(matches[0]) < 3:
        raise ValueError("Exact original terminal state and empty live queue are required")
    _, state, exit_code, *_ = matches[0]
    if state not in allowed_states or re.fullmatch(r"\d+:\d+", exit_code) is None:
        raise ValueError("Original state does not authorize this terminal continuation")
    body = {"job_id": str(job_id), "state": state, "exit_code": exit_code, "sacct": sacct}
    return {**body, "sha256": digest(json_bytes(body)), "live_queue_empty": True}


def source_provenance(record, revision):
    rows = record["source_files"]
    return {
        "original_source_revision": revision,
        "original_fitting_case_sha256": digest(record["uid"].encode()),
        "source_file_table_sha256": digest(json_bytes(rows)),
        "source_DICOMs": len(rows),
        "source_DICOM_bytes": sum(row["bytes"] for row in rows),
    }


def source_proof_matches(proof, record):
    """Accept only complete original source-table order and exact size/SHA values."""
    expected = [
        {"source_path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in record["source_files"]
    ]
    actual = [
        {key: row[key] for key in ("source_path", "bytes", "sha256")} for row in proof["files"]
    ]
    if (
        actual != expected
        or proof["source_bytes"] != sum(row["bytes"] for row in record["source_files"])
        or proof["study_uid_sha256"] != digest(record["uid"].encode())
        or proof["series_csv_sha256"] != digest(record["series_csv"].encode())
    ):
        raise ValueError("Original whole-source proof differs from the exact source table")


def inventory(original, records, prior, *, maximum_buffers=2):
    """Map every verified case, remaining intent, source buffer and orphan cache.

    No source/cache byte is written, downloaded or removed. Complete admitted
    NPZs reuse their owning-node stat proof; unverified NPZs require later full
    admission against retained original pixels before they can be reused.
    """
    original = Path(original)
    private_directory(original)
    uid_to_index = {row["uid"]: i for i, row in enumerate(records)}
    if len(uid_to_index) != len(records):
        raise ValueError("Original source roster contains duplicates")
    complete = {}
    for old in prior:
        index = uid_to_index[old["uid"]]
        if index in complete or signature(old["path"]) != old["proof"]["stat"]:
            raise ValueError("Original prior cache binding differs")
        complete[index] = {key: old[key] for key in ("uid", "path", "proof")} | {
            "original16_reused": True
        }
    intents = {}
    partial_verified_records = {}
    expected_names = set()
    for i, record in enumerate(records):
        stem = f"case-{i:04d}"
        intent_path = original / "proofs" / (stem + ".intent.json")
        proof_path = original / "proofs" / (stem + ".source-proof.private.json.gz")
        verified_path = original / "proofs" / (stem + ".verified.private.json")
        expected_names.update((intent_path.name, proof_path.name, verified_path.name))
        if intent_path.exists():
            intent = bound_json(intent_path)
            if (
                intent["source_ordinal"] != i
                or intent["source_bytes"] != sum(x["bytes"] for x in record["source_files"])
                or intent["original_source_table_sha256"]
                != digest(json_bytes(record["source_files"]))
            ):
                raise ValueError("Original case intent differs from source order")
            intents[i] = intent
        if verified_path.exists():
            if i in complete or i not in intents:
                raise ValueError("Duplicate or unbound original verified case")
            try:
                row = bound_json(verified_path)
            except json.JSONDecodeError:
                # SIGTERM can interrupt the original non-atomic record write.
                # It remains unverified and must follow original-pixel recovery.
                partial_verified_records[i] = verified_path
                continue
            expected_path = original / "cache" / (stem + ".private.npz")
            if (
                row["uid"] != record["uid"]
                or Path(row["path"]) != expected_path
                or row["original16_reused"] is not False
                or signature(expected_path) != row["proof"]["stat"]
            ):
                raise ValueError("Original completed cache binding differs")
            regular(proof_path)
            compressed = proof_path.read_bytes()
            if digest(compressed) != row["source_proof_sha256"]:
                raise ValueError("Original source-proof archive SHA differs")
            source_proof_matches(json.loads(gzip.decompress(compressed)), record)
            complete[i] = row
    for path in (original / "proofs").iterdir():
        if path.name not in expected_names:
            raise ValueError("Unexpected member in original proof directory")
        regular(path)
    if len(set(intents) - set(complete)) > maximum_buffers:
        raise ValueError("Unfinished original intents exceed the admitted two lanes")
    buffers = {}
    partial_source_records = {}
    roots = list((original / "scratch").iterdir())
    if len(roots) > maximum_buffers:
        raise ValueError("More retained original source buffers than the admitted two lanes")
    for owned in roots:
        private_directory(owned)
        if not owned.name.startswith("fitting-study-"):
            raise ValueError("Unexpected original scratch root")
        data = owned / "data"
        private_directory(data)
        regular(data / "train.csv")
        lines = (data / "train.csv").read_text().splitlines()
        if len(lines) != 2 or lines[0] != "StudyInstanceUID" or lines[1] not in uid_to_index:
            raise ValueError("Original partial buffer cannot be mapped unambiguously")
        i = uid_to_index[lines[1]]
        if i not in intents or i in buffers:
            raise ValueError("Original buffer lacks one unique source intent")
        regular(data / "train_series.csv")
        if (data / "train_series.csv").read_text() != records[i]["series_csv"]:
            raise ValueError("Original retained series metadata differs")
        allowed = {"data", "source-proof.private.json", "source-failure.private.json"}
        if any(path.name not in allowed or path.is_symlink() for path in owned.iterdir()):
            raise ValueError("Unexpected original source-buffer member")
        proof = owned / "source-proof.private.json"
        if proof.exists():
            try:
                source_proof = bound_json(proof)
            except json.JSONDecodeError:
                # A wall-limit signal may interrupt this record after source
                # bytes finished. The missing-only helper must independently
                # re-admit every retained file; this partial record is no proof.
                partial_source_records[i] = proof
            else:
                source_proof_matches(source_proof, records[i])
        buffers[i] = data
    orphan_caches = {}
    names = {f"case-{i:04d}.private.npz": i for i in range(len(records))}
    for path in (original / "cache").iterdir():
        regular(path)
        if path.name not in names:
            raise ValueError("Unexpected original cache file")
        i = names[path.name]
        if i not in complete:
            if i not in intents or i not in buffers:
                raise ValueError("Unverified cache lacks original intent/source buffer")
            orphan_caches[i] = path
    pending = [i for i in range(len(records)) if i not in complete]
    return {
        "complete": complete,
        "pending": pending,
        "intents": intents,
        "buffers": buffers,
        "orphan_caches": orphan_caches,
        "partial_verified_records": partial_verified_records,
        "partial_source_records": partial_source_records,
    }


def physical_bytes(roots):
    """Count each existing inode once; no deletion or hardlink double-count credit."""
    seen = set()
    total = 0
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        members = [root] if root.is_file() else root.rglob("*")
        for path in members:
            if path.is_symlink():
                raise ValueError("Unexpected symlink in retained allocation")
            if not path.is_file():
                continue
            st = regular(path)
            key = st.st_dev, st.st_ino
            if key not in seen:
                seen.add(key)
                total += st.st_size
    return total
