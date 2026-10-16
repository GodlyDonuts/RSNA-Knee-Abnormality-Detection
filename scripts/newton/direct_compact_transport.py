"""Private TLS file transport with exact verified-prefix reuse; no automatic retry.

The caller owns endpoint allocation, TLS pinning, authentication and source manifest.
Files are created exclusively or appended ONLY after their existing prefix equals
original source bytes. Complete existing files are hashed, acknowledged and skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path, PurePosixPath

BLOCK = 2**20
MAX_JSON = 64 * 2**20


def read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        block = stream.read(count - len(result))
        if not block:
            raise ValueError("Truncated authenticated transport")
        result.extend(block)
    return bytes(result)


def send_json(stream, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if len(raw) > MAX_JSON:
        raise ValueError("Bounded manifest exceeded")
    stream.write(struct.pack(">Q", len(raw)) + raw)
    stream.flush()


def receive_json(stream):
    size = struct.unpack(">Q", read_exact(stream, 8))[0]
    if size > MAX_JSON:
        raise ValueError("Bounded manifest exceeded")
    return json.loads(read_exact(stream, size))


def confined(root: Path, name: str) -> Path:
    rel = PurePosixPath(name)
    if not name or str(rel) != name or rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Noncanonical transport member")
    path = root / name
    if (
        not path.resolve().is_relative_to(root.resolve())
        or path.is_symlink()
        or any(p.is_symlink() for p in path.parents if p != root.parent)
    ):
        raise ValueError("Linked/outside transport target")
    return path


def digest_prefix(path: Path, count: int):
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = count
        while remaining:
            block = f.read(min(BLOCK, remaining))
            if not block:
                raise ValueError("Original source prefix truncated")
            h.update(block)
            remaining -= len(block)
    return h


def serve_file(reader, writer, source: Path, record):
    size = record["bytes"]
    before = source.stat()
    if before.st_size != size:
        raise ValueError("Original source size changed")
    request = receive_json(reader)
    offset = request.get("offset")
    if type(offset) is not int or not 0 <= offset <= size:
        raise ValueError("Invalid target prefix size")
    if offset == size:
        if request.get("prefix_sha256") != record["sha256"]:
            raise ValueError("Existing complete target differs")
    elif digest_prefix(source, offset).hexdigest() != request.get("prefix_sha256"):
        raise ValueError("Existing partial target differs from source prefix")
    writer.write(b"1")
    writer.flush()
    if offset != size:
        with source.open("rb") as f:
            f.seek(offset)
            remaining = size - offset
            while remaining:
                block = f.read(min(BLOCK, remaining))
                if not block:
                    raise ValueError("Original source truncated")
                writer.write(block)
                remaining -= len(block)
        writer.flush()
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Original source changed during transport")
    if read_exact(reader, 1) != b"1":
        raise ValueError("Target did not accept original file")


def receive_file(reader, writer, target: Path, record):
    size = record["bytes"]
    if type(size) is not int or size < 0:
        raise ValueError("Nonnegative bounded file size required")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    offset = target.stat().st_size if target.exists() else 0
    if offset > size:
        raise ValueError("Existing target exceeds original size")
    h = digest_prefix(target, offset) if offset else hashlib.sha256()
    send_json(writer, {"offset": offset, "prefix_sha256": h.hexdigest()})
    if read_exact(reader, 1) != b"1":
        raise ValueError("Source refused existing target prefix")
    if size == 0 and not target.exists():
        with target.open("xb"):
            os.chmod(target, 0o600 | record.get("executable", 0))
    if offset < size:
        mode = "ab" if offset else "xb"
        if offset and target.stat().st_size != offset:
            raise ValueError("Existing prefix changed")
        with target.open(mode) as f:
            os.chmod(target, 0o600 | record.get("executable", 0))
            remaining = size - offset
            while remaining:
                block = read_exact(reader, min(BLOCK, remaining))
                f.write(block)
                h.update(block)
                remaining -= len(block)
            f.flush()
            os.fsync(f.fileno())
    if target.stat().st_size != size or h.hexdigest() != record["sha256"]:
        raise ValueError("Exact original whole-file SHA/size differs")
    writer.write(b"1")
    writer.flush()
    return {"copied_bytes": size - offset, "reused_bytes": offset}
