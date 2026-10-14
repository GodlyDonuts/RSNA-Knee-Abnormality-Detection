"""Optional node-private cache of one already verified, immutable release admission.

Engineering only: it can make an identical repeated admission faster; it never
changes the roster, recipe, labels, folds or verification policy. The producer runs
the original ``load_verified_release`` under a read trace and binds every member it
consumed (original hash, stat signature, complete listing of every enumerated
directory) plus the exact frame bytes, binding, code and runtime. The consumer
refuses any difference. Fallback to the full original admission happens only when
explicitly requested and is always announced on stderr. Never commit or transfer a
cache: its frame contains reports, labels and case identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import site
import stat
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from rsna import release
from rsna.semantic_supervision import digest, parse, version

SCHEMA = "release_admission_cache_v1"
MANIFEST_NAME = "admission.json"
FRAME_NAME = "frame.parquet"
CACHE_FILES = frozenset({MANIFEST_NAME, FRAME_NAME})
MAX_MEMBERS = 4096
MAX_MEMBER_BYTES = 2 * 2**30
MAX_LISTINGS = 64
MAX_LISTING_ENTRIES = 65536
MAX_FRAME_BYTES = 64 * 2**20
MAX_MANIFEST_BYTES = 16 * 2**20
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

_TRACE: list[tuple[str, Any, Any, Any]] | None = None
_HOOKED = False


class AdmissionCacheRefused(ValueError):
    """The cache does not prove an identical admission; nothing was substituted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionCacheRefused("Release admission cache: " + message)


def _hook(event: str, args: tuple) -> None:
    trace = _TRACE
    if trace is None:
        return
    if event == "open":
        trace.append(("open", args[0], args[1], args[2]))
    elif event in {"os.listdir", "os.scandir"}:
        trace.append(("list", args[0] if args else None, None, None))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signature(result: os.stat_result) -> list[int]:
    return [
        result.st_dev,
        result.st_ino,
        result.st_mode,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    ]


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _library_roots() -> list[str]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    roots.update(site.getsitepackages())
    roots.add(site.getusersitepackages())
    return sorted(os.path.realpath(root) for root in roots if root)


def code_identity() -> dict[str, str]:
    """Exact source of the top-level ``rsna`` modules (the whole release-admission chain)."""
    package = Path(release.__file__).resolve().parent
    return {
        "rsna/" + path.name: _sha256(path.read_bytes()) for path in sorted(package.glob("*.py"))
    }


def runtime_identity() -> dict[str, str]:
    import numpy
    import pyarrow

    return {
        "python": sys.version,
        "implementation": sys.implementation.name,
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
    }


def frame_identity(frame: pd.DataFrame) -> dict[str, Any]:
    """Schema/dtype/roster identity; no row values are serialized here."""
    _require(
        {"study_uid", "split", "leakage_group"} <= set(frame.columns), "frame lacks roster columns"
    )
    roster = sorted(
        frame[["study_uid", "split", "leakage_group"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    return {
        "shape": list(frame.shape),
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index": [type(frame.index).__name__, str(frame.index.dtype), len(frame.index)],
        "index_sha256": digest([str(value) for value in frame.index]),
        "values_sha256": _sha256(
            pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        ),
        "roster_sha256": digest([list(row) for row in roster]),
    }


def _listing(path: str) -> list[list[str]]:
    with os.scandir(path) as entries:
        rows = [
            [
                entry.name,
                "link"
                if entry.is_symlink()
                else "dir"
                if entry.is_dir(follow_symlinks=False)
                else "file"
                if entry.is_file(follow_symlinks=False)
                else "other",
            ]
            for entry in entries
        ]
    _require(len(rows) <= MAX_LISTING_ENTRIES, "listing exceeds entry bound")
    return sorted(rows)


def _no_link_below(path: str, root: str) -> None:
    current = root
    for part in os.path.relpath(path, root).split(os.sep):
        if part in ("", "."):
            continue
        current = os.path.join(current, part)
        _require(not stat.S_ISLNK(os.lstat(current).st_mode), "unsafe link in source tree")


def _member_name(real: str, root: str, external: dict[str, str]) -> str:
    if real in external:
        return external[real]
    return "root/" + os.path.relpath(real, root)


def _read_member(path: str) -> tuple[bytes, list[int]]:
    before = os.lstat(path)
    _require(stat.S_ISREG(before.st_mode), "member is not a regular file")
    _require(before.st_size <= MAX_MEMBER_BYTES, "source byte bound exceeded")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        data = handle.read(MAX_MEMBER_BYTES + 1)
    after = os.lstat(path)
    _require(
        _signature(before) == _signature(opened) == _signature(after)
        and len(data) == after.st_size,
        "member changed while being read",
    )
    return data, _signature(after)


def _locations(studies: Path, artifact_root: Path) -> tuple[str, str, str]:
    root = os.path.realpath(artifact_root)
    studies_real = os.path.realpath(studies)
    receipt_real = os.path.realpath(studies.parent / release.RECEIPT_NAME)
    for given, real in (
        (studies, studies_real),
        (studies.parent / release.RECEIPT_NAME, receipt_real),
    ):
        _require(
            os.path.lexists(given) and not stat.S_ISLNK(os.lstat(given).st_mode),
            "release file is missing or a link",
        )
        _require(os.path.realpath(given.parent) == os.path.dirname(real), "release path link")
    return root, studies_real, receipt_real


def _refuse_location(cache: Path, roots: list[str]) -> str:
    parent = os.path.realpath(cache.parent)
    real = os.path.join(parent, cache.name)
    _require(bool(cache.name) and cache.name not in (".", ".."), "invalid cache name")
    for root in roots:
        _require(not _inside(real, root), "cache must stay outside source/code/release trees")
    return real


def _write_private(path: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def produce(studies: Path, artifact_root: Path, cache: Path) -> dict[str, Any]:
    """Run the ORIGINAL full admission once and retain a private exact receipt."""
    global _TRACE, _HOOKED
    root, studies_real, receipt_real = _locations(studies, artifact_root)
    package_root = str(Path(release.__file__).resolve().parents[1])
    cache_real = _refuse_location(cache, [root, package_root, os.path.dirname(studies_real)])
    _require(not os.path.lexists(cache_real), "cache path already exists")
    _require(os.path.isdir(os.path.dirname(cache_real)), "cache parent is missing")
    if not _HOOKED:
        sys.addaudithook(_hook)
        _HOOKED = True
    trace: list[tuple[str, Any, Any, Any]] = []
    started_ns = time.time_ns()
    started = time.perf_counter()
    _TRACE = trace
    try:
        frame, binding = release.load_verified_release(studies, artifact_root)
    finally:
        _TRACE = None
    admission_seconds = time.perf_counter() - started
    cwd = os.getcwd()
    libraries = _library_roots()
    external = {studies_real: "release/studies", receipt_real: "release/receipt"}
    opened: set[str] = set()
    listed: set[str] = set()
    frame_bytes = b""
    for kind, raw, mode, flags in trace:
        if raw is None and kind == "list":
            raw = "."
        if isinstance(raw, int):
            continue
        name = os.fsdecode(raw)
        if "__pycache__" in Path(name).parts:
            continue  # bytecode; executed code is bound separately by code_identity
        lexical = os.path.normpath(os.path.join(cwd, name))
        real = os.path.realpath(lexical)
        if real not in external and not _inside(real, root):
            _require(
                any(_inside(real, library) for library in libraries),
                "admission read an unbound input outside the release/source tree",
            )
            continue
        _require(lexical == real, "admission followed a link inside the release/source tree")
        if kind == "list":
            listed.add(real)
            continue
        written = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & WRITE_FLAGS
        )
        _require(not written, "admission opened a source member for writing")
        opened.add(real)
    _require({studies_real, receipt_real} <= opened, "trace missed the release files")
    _require(len(opened) <= MAX_MEMBERS and len(listed) <= MAX_LISTINGS, "member bound exceeded")
    members, total = {}, 0
    for real in sorted(opened):
        if real not in external:
            _no_link_below(real, root)
        data, signature = _read_member(real)
        total += len(data)
        _require(total <= MAX_MEMBER_BYTES, "source byte bound exceeded")
        _require(
            signature[4] < started_ns and signature[5] < started_ns,
            "member modified at or after admission start",
        )
        members[_member_name(real, root, external)] = {"sha256": _sha256(data), "stat": signature}
        if real == studies_real:
            frame_bytes = data
    listings = {}
    for real in sorted(listed):
        _no_link_below(real, root)
        listings["root/" + os.path.relpath(real, root)] = _listing(real)
    _require(len(frame_bytes) <= MAX_FRAME_BYTES, "frame byte bound exceeded")
    _require(_sha256(frame_bytes) == binding["manifest_sha256"], "frame bytes differ from binding")
    identity = frame_identity(frame)
    replay = pd.read_parquet(io.BytesIO(frame_bytes))
    _require(frame_identity(replay) == identity and replay.equals(frame), "frame replay differs")
    record = {
        "schema": SCHEMA,
        "status": "engineering_admission_receipt_not_scientific_evidence",
        "artifact_root": root,
        "studies": studies_real,
        "receipt": receipt_real,
        "code": code_identity(),
        "runtime": runtime_identity(),
        "members": members,
        "member_bytes": total,
        "listings": listings,
        "frame": {"sha256": _sha256(frame_bytes), "bytes": len(frame_bytes), **identity},
        "binding": binding,
        "original_admission_seconds": admission_seconds,
    }
    manifest = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
    _require(len(manifest) <= MAX_MANIFEST_BYTES, "manifest byte bound exceeded")
    os.mkdir(cache_real, 0o700)
    os.chmod(cache_real, 0o700)
    _write_private(os.path.join(cache_real, FRAME_NAME), frame_bytes)
    _write_private(os.path.join(cache_real, MANIFEST_NAME), manifest)
    return {
        "members": len(members),
        "member_bytes": total,
        "listings": len(listings),
        "original_admission_seconds": admission_seconds,
    }


def _private_bytes(cache: Path) -> tuple[bytes, bytes]:
    _require(os.path.lexists(cache), "cache is missing")
    info = os.lstat(cache)
    _require(stat.S_ISDIR(info.st_mode), "cache is not a real directory")
    _require(info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700, "cache mode")
    _require(set(os.listdir(cache)) == CACHE_FILES, "cache membership differs")
    payload = {}
    for name, cap in ((MANIFEST_NAME, MAX_MANIFEST_BYTES), (FRAME_NAME, MAX_FRAME_BYTES)):
        path = os.path.join(cache, name)
        entry = os.lstat(path)
        _require(stat.S_ISREG(entry.st_mode), "cache file is not regular")
        _require(
            entry.st_uid == os.geteuid() and stat.S_IMODE(entry.st_mode) == 0o600,
            "cache file mode",
        )
        _require(entry.st_size <= cap, "cache byte bound exceeded")
        data, _ = _read_member(path)
        payload[name] = data
    return payload[MANIFEST_NAME], payload[FRAME_NAME]


def consume(studies: Path, artifact_root: Path, cache: Path) -> tuple[pd.DataFrame, dict]:
    """Return the exact cached admission, or refuse; never runs or substitutes another."""
    root, studies_real, receipt_real = _locations(studies, artifact_root)
    manifest, frame_bytes = _private_bytes(cache)
    try:
        record = json.loads(manifest)
    except ValueError as exc:
        raise AdmissionCacheRefused("Release admission cache: unreadable manifest") from exc
    _require(isinstance(record, dict) and record.get("schema") == SCHEMA, "schema differs")
    _require(
        (record.get("artifact_root"), record.get("studies"), record.get("receipt"))
        == (root, studies_real, receipt_real),
        "requested release/source tree differs",
    )
    _require(record.get("code") == code_identity(), "admission code differs (stale consumer)")
    _require(record.get("runtime") == runtime_identity(), "runtime differs (stale consumer)")
    members, listings = record.get("members"), record.get("listings")
    _require(isinstance(members, dict) and isinstance(listings, dict), "malformed manifest")
    _require(len(members) <= MAX_MEMBERS and len(listings) <= MAX_LISTINGS, "member bound")
    _require({"release/studies", "release/receipt"} <= set(members), "release members missing")
    reverse = {"release/studies": studies_real, "release/receipt": receipt_real}
    total, contents = 0, {}
    for name, expected in sorted(members.items()):
        if name in reverse:
            path = reverse[name]
        else:
            _require(name.startswith("root/"), "malformed member name")
            path = os.path.normpath(os.path.join(root, name[len("root/") :]))
            _require(_inside(path, root) and path != root, "member escapes source tree")
            _require(os.path.lexists(path), "source member missing")
            _no_link_below(path, root)
        _require(os.path.realpath(path) == path, "member resolves through a link")
        data, signature = _read_member(path)
        total += len(data)
        _require(total <= MAX_MEMBER_BYTES, "source byte bound exceeded")
        _require(signature == expected.get("stat"), "source member stat signature changed")
        _require(_sha256(data) == expected.get("sha256"), "source member content changed")
        if name in reverse:
            contents[name] = data
    _require(total == record.get("member_bytes"), "source byte total differs")
    for name, expected in sorted(listings.items()):
        _require(name.startswith("root/"), "malformed listing name")
        path = os.path.normpath(os.path.join(root, name[len("root/") :]))
        _require(_inside(path, root), "listing escapes source tree")
        _require(os.path.isdir(path), "listed directory missing")
        _no_link_below(path, root)
        _require(_listing(path) == expected, "directory membership changed (missing/extra member)")
    frame_record = record.get("frame")
    _require(isinstance(frame_record, dict), "malformed frame record")
    _require(
        _sha256(frame_bytes) == frame_record.get("sha256") == members["release/studies"]["sha256"]
        and contents["release/studies"] == frame_bytes,
        "private frame differs from the bound release table",
    )
    frame = pd.read_parquet(io.BytesIO(frame_bytes))
    identity = {key: frame_record.get(key) for key in frame_identity(frame)}
    _require(frame_identity(frame) == identity, "frame schema/dtype/roster differs")
    receipt_bytes = contents["release/receipt"]
    receipt = version(parse(receipt_bytes.decode()))
    binding = release._checkpoint_binding(receipt, _sha256(receipt_bytes))
    release.validate_checkpoint_binding(binding)
    _require(binding == record.get("binding"), "release binding differs")
    _require(binding["manifest_sha256"] == frame_record["sha256"], "binding/frame mismatch")
    return frame, binding


def admit_release(
    studies: Path, artifact_root: Path, cache: Path, fallback: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Cached admission; the full original admission runs only on explicit request."""
    try:
        return consume(studies, artifact_root, cache)
    except AdmissionCacheRefused as exc:
        if not fallback:
            raise
        print(
            json.dumps(
                {
                    "release_admission": "cache_refused_running_full_original_admission",
                    "reason": str(exc),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return release.load_verified_release(studies, artifact_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("produce", "verify"))
    for name in ("studies", "artifact-root", "cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "produce":
        summary = produce(args.studies, args.artifact_root, args.cache)
    else:
        started = time.perf_counter()
        _, binding = consume(args.studies, args.artifact_root, args.cache)
        summary = {
            "binding_sha256": binding["binding_sha256"],
            "cached_admission_seconds": time.perf_counter() - started,
        }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
