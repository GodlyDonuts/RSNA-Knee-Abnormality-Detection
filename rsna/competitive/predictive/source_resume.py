"""Continue an original source buffer only after its owning job is terminal.

Reconciliation is read-only. Continuation creates a new private directory, links
verified complete originals without downloading them, and copies partial prefixes
before requesting only their missing suffixes. It never edits an original or
retries a request. The caller must authenticate the pinned revision, verify the
actual job-terminal receipt, and fund the new directory before calling this module.
No scheduler, network client, credential handling, or source selection lives here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import BinaryIO, Callable, Mapping, Sequence

SERIES_COLUMNS = (
    "StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive",
    "Fat_Suppression", "Anatomical_Plane",
)
TERMINAL_STATES = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"}
Signature = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class Member:
    path: str
    study_uid: str
    series_uid: str
    sop_uid: str
    size: int
    sha256: str
    retained_size: int
    retained_sha256: str | None
    signature: Signature | None

    def source_row(self) -> dict:
        return {
            "split": "train_series", "study_uid": self.study_uid,
            "series_uid": self.series_uid, "sop_uid": self.sop_uid,
            "path": self.path, "bytes": self.size, "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ResumePlan:
    original_data: Path
    uid: str
    revision: str
    series_csv: str
    members: tuple[Member, ...]
    metadata_signatures: tuple[Signature, Signature]
    source_table_sha256: str
    cap: int
    control_reserve: int


@dataclass(frozen=True)
class SourceResponse:
    """The opener reports the pinned request identity and unmodified HTTP headers.

    It must send Range: bytes=<offset>- for a positive offset, request identity
    encoding, and expose the response stream without transparent decompression.
    The helper verifies status, range, EOF, length and complete original SHA.
    """

    stream: BinaryIO
    status: int
    headers: Mapping[str, str]
    revision: str
    path: str


Opener = Callable[[dict, int], AbstractContextManager[SourceResponse]]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signature(path: Path) -> Signature:
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid():
        raise ValueError("Source member must be an owned regular file")
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _directory(path: Path) -> None:
    s = path.lstat()
    if not stat.S_ISDIR(s.st_mode) or s.st_uid != os.getuid() or s.st_mode & 0o077:
        raise ValueError("Source root must be an owned private directory")


def _member_signature(root: Path, relative: str) -> Signature | None:
    current = root
    parts = Path(relative).parts
    for part in parts[:-1]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            return None
        s = current.lstat()
        if not stat.S_ISDIR(s.st_mode) or s.st_uid != os.getuid():
            raise ValueError("Source member has an unsafe directory")
    path = root / relative
    if not path.exists() and not path.is_symlink():
        return None
    return _signature(path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(uid: str, series_csv: str) -> tuple[tuple[str, bytes], ...]:
    return (
        ("train.csv", ("StudyInstanceUID\n" + uid + "\n").encode()),
        ("train_series.csv", series_csv.encode()),
    )


def reconcile(
    original_data: Path, *, uid: str, rows: Sequence[Mapping],
    series_csv: str, revision: str, cap: int,
) -> ResumePlan:
    """Read-only classification of every exact original file as complete/partial/missing.

    cap bounds *all logical bytes in the new directory*, including linked complete
    files, metadata and a reserved proof/failure allowance. Retained originals are
    separate existing obligations; this function grants no deletion/storage credit.
    """
    def valid_uid(value: object) -> bool:
        return (
            isinstance(value, str) and len(value) <= 64
            and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value) is not None
        )

    if (
        not valid_uid(uid) or not rows or type(cap) is not int or cap <= 0
        or re.fullmatch(r"[a-f0-9]{40}", revision) is None
    ):
        raise ValueError("Invalid original source contract")
    original_data = Path(os.path.abspath(original_data))
    _directory(original_data)
    seen = set()
    normalized = []
    for row in rows:
        series, sop = row["series_uid"], row["sop_uid"]
        path = f"train_series/{uid}/{series}/{sop}.dcm"
        if (
            row["study_uid"] != uid or row["split"] != "train_series"
            or not valid_uid(series) or not valid_uid(sop)
            or row["path"] != path or path in seen
            or type(row["bytes"]) is not int or row["bytes"] <= 0
            or not isinstance(row["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is None
        ):
            raise ValueError("Original source UID/path/size/hash contract differs")
        seen.add(path)
        normalized.append({
            key: row[key] for key in (
                "split", "study_uid", "series_uid", "sop_uid", "path", "bytes", "sha256"
            )
        })
    reader = csv.DictReader(io.StringIO(series_csv))
    metadata_rows = list(reader)
    if (
        tuple(reader.fieldnames or ()) != SERIES_COLUMNS
        or any(set(row) != set(SERIES_COLUMNS) or any(v is None for v in row.values())
               or row["StudyInstanceUID"] != uid for row in metadata_rows)
        or len({row["SeriesInstanceUID"] for row in metadata_rows}) != len(metadata_rows)
        or {row["SeriesInstanceUID"] for row in metadata_rows}
        != {row["series_uid"] for row in normalized}
    ):
        raise ValueError("Original series metadata differs")
    metadata = _metadata(uid, series_csv)
    control_reserve = 65536 + 1024 * len(rows)
    required = sum(row["bytes"] for row in normalized)
    required += sum(len(body) for _, body in metadata) + control_reserve
    if required > cap:
        raise ValueError("Total new source directory exceeds funded cap")
    metadata_signatures = []
    for name, body in metadata:
        signature = _signature(original_data / name)
        if (
            signature[2] != len(body)
            or (original_data / name).read_bytes() != body
            or _signature(original_data / name) != signature
        ):
            raise ValueError("Retained original metadata differs")
        metadata_signatures.append(signature)
    allowed = seen | {name for name, _ in metadata}
    for path in original_data.rglob("*"):
        if path.is_symlink() or (
            not path.is_dir() and path.relative_to(original_data).as_posix() not in allowed
        ):
            raise ValueError("Retained source contains an unbound member")
    members = []
    for row in normalized:
        signature = _member_signature(original_data, row["path"])
        size = signature[2] if signature else 0
        if size > row["bytes"]:
            raise ValueError("Retained source exceeds original size")
        digest = _hash_file(original_data / row["path"]) if signature else None
        if _member_signature(original_data, row["path"]) != signature:
            raise ValueError("Retained source changed during reconciliation")
        if size == row["bytes"] and digest != row["sha256"]:
            raise ValueError("Retained complete source SHA differs")
        members.append(Member(
            row["path"], uid, row["series_uid"], row["sop_uid"], row["bytes"],
            row["sha256"], size, digest, signature,
        ))
    return ResumePlan(
        original_data, uid, revision, series_csv, tuple(members),
        (metadata_signatures[0], metadata_signatures[1]),
        _sha(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()),
        cap, control_reserve,
    )


def _accept_response(response: SourceResponse, plan: ResumePlan, member: Member) -> None:
    headers = {key.lower(): str(value) for key, value in response.headers.items()}
    if len(headers) != len(response.headers):
        raise ValueError("Duplicate response header")
    offset = member.retained_size
    remaining = member.size - offset
    expected_range = f"bytes {offset}-{member.size - 1}/{member.size}"
    if (
        response.revision != plan.revision or response.path != member.path
        or headers.get("content-encoding", "identity").lower() != "identity"
        or response.status != (206 if offset else 200)
        or (offset and headers.get("content-range") != expected_range)
        or (not offset and "content-range" in headers)
        or ("content-length" in headers and headers["content-length"] != str(remaining))
    ):
        raise ValueError("Original response identity/range/length differs")


def resume_missing(
    root: Path, plan: ResumePlan, *, opener: Opener,
    terminal_receipt: Mapping, workers: int = 1,
) -> tuple[Path, dict]:
    """Run one missing-only continuation after caller-verified terminal evidence.

    Complete files are hardlinked, requiring the same admitted filesystem. A
    failed link/HTTP stream/hash never falls back to a new transfer or retries.
    Every partial and failure stays private for later read-only reconciliation.
    """
    if (
        terminal_receipt.get("state") not in TERMINAL_STATES
        or re.fullmatch(r"[0-9]+", str(terminal_receipt.get("job_id", ""))) is None
        or re.fullmatch(r"[a-f0-9]{64}", str(terminal_receipt.get("sha256", ""))) is None
        or type(workers) is not int or not 1 <= workers <= 8
    ):
        raise ValueError("Actual terminal evidence and bounded parallelism are required")
    root = Path(os.path.abspath(root))
    _directory(root)
    for (name, _), signature in zip(
        _metadata(plan.uid, plan.series_csv), plan.metadata_signatures, strict=True
    ):
        if _signature(plan.original_data / name) != signature:
            raise ValueError("Original metadata changed after reconciliation")
    for member in plan.members:
        if _member_signature(plan.original_data, member.path) != member.signature:
            raise ValueError("Original source changed after reconciliation")
    owned = Path(tempfile.mkdtemp(prefix="fitting-resume-", dir=root))
    owned.chmod(0o700)
    data = owned / "data"
    data.mkdir(mode=0o700)
    for name, body in _metadata(plan.uid, plan.series_csv):
        path = data / name
        with path.open("xb") as output:
            output.write(body)
        path.chmod(0o600)

    def one(member: Member) -> dict:
        old, new = plan.original_data / member.path, data / member.path
        new.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if _member_signature(plan.original_data, member.path) != member.signature:
            raise ValueError("Original source changed before reuse")
        if member.retained_size == member.size:
            if member.signature is None:
                raise ValueError("Complete original has no reconciliation signature")
            os.link(old, new, follow_symlinks=False)
            # Linking changes ctime/nlink only; no original bytes are rewritten.
            if _signature(new)[:4] != member.signature[:4]:
                raise ValueError("Complete original changed while linking")
            action = "linked_complete"
        else:
            digest = hashlib.sha256()
            with new.open("xb") as output:
                new.chmod(0o600)
                if member.retained_size:
                    with old.open("rb") as source:
                        remaining = member.retained_size
                        while remaining:
                            chunk = source.read(min(65536, remaining))
                            if not chunk:
                                raise ValueError("Retained prefix ended early")
                            output.write(chunk)
                            digest.update(chunk)
                            remaining -= len(chunk)
                        if source.read(1):
                            raise ValueError("Retained prefix grew")
                    if digest.hexdigest() != member.retained_sha256:
                        raise ValueError("Retained prefix changed")
                if _member_signature(plan.original_data, member.path) != member.signature:
                    raise ValueError("Retained prefix changed before request")
                with opener(member.source_row(), member.retained_size) as response:
                    _accept_response(response, plan, member)
                    size = member.retained_size
                    while True:
                        chunk = response.stream.read(min(65536, member.size - size + 1))
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > member.size:
                            raise ValueError("Original response exceeded exact EOF")
                        output.write(chunk)
                        digest.update(chunk)
                if size != member.size or digest.hexdigest() != member.sha256:
                    raise ValueError("Completed original size/SHA differs")
            action = "resumed_suffix" if member.retained_size else "fetched_missing"
        return {
            "source_path": member.path, "bytes": member.size, "sha256": member.sha256,
            "action": action, "reused_bytes": member.retained_size,
            "fetched_bytes": member.size - member.retained_size,
        }

    stop = Event()

    def guarded_one(member: Member) -> dict:
        if stop.is_set():
            raise RuntimeError("Continuation stopped after a peer failure")
        try:
            return one(member)
        except BaseException:
            stop.set()
            raise

    try:
        if workers == 1:
            files = [guarded_one(member) for member in plan.members]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                files = list(pool.map(guarded_one, plan.members))
        proof = {
            "schema_version": 1, "source_revision": plan.revision,
            "source_table_sha256": plan.source_table_sha256,
            "study_uid_sha256": _sha(plan.uid.encode()),
            "series_csv_sha256": _sha(plan.series_csv.encode()),
            "terminal_job_id": str(terminal_receipt["job_id"]),
            "terminal_state": terminal_receipt["state"],
            "terminal_receipt_sha256": terminal_receipt["sha256"],
            "files": files, "source_bytes": sum(m.size for m in plan.members),
            "fetched_bytes": sum(f["fetched_bytes"] for f in files),
            "source_cap_bytes": plan.cap, "originals_preserved": True,
            "automatic_retry": False, "diagnosis_labels_read": False,
        }
        body = json.dumps(proof, sort_keys=True).encode()
        actual = sum(path.stat().st_size for path in data.rglob("*") if path.is_file())
        if len(body) > plan.control_reserve or actual + len(body) > plan.cap:
            raise ValueError("Actual new directory exceeds total cap")
        receipt = owned / "source-proof.private.json"
        receipt.write_bytes(body)
        receipt.chmod(0o600)
        return data, proof
    except BaseException as error:
        failure = owned / "source-failure.private.json"
        failure.write_text(json.dumps({
            "error_type": type(error).__name__, "automatic_retry": False,
            "originals_preserved": True, "retained_for_reconciliation": True,
        }))
        failure.chmod(0o600)
        raise
