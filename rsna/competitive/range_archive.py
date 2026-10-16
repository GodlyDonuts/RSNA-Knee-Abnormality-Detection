"""Recover an immutable large archive by bounded, overlap-verified byte ranges."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def complete(path, size, digest, request, journal, *, chunk_bytes=256 * 2**20):
    path, journal = Path(path), Path(journal)
    if path.is_symlink() or not 0 < size <= 14 * 2**30 or not 0 < chunk_bytes <= 256 * 2**20:
        raise ValueError("A bounded original immutable archive required")
    journal.mkdir(mode=0o700)
    if not path.exists():
        path.touch(mode=0o600, exist_ok=False)
    n = path.stat().st_size
    if not 0 <= n <= size:
        raise ValueError("Existing immutable prefix exceeds its whole-byte receipt")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 2**20), b""):
            h.update(block)
    original = n
    for number in range(128):
        if n == size:
            break
        overlap = min(n, 65536)
        end = min(n + chunk_bytes, size) - 1
        with path.open("rb") as f:
            f.seek(n - overlap)
            boundary = f.read(overlap)
        with (journal / f"chunk-{number:03d}.intent.json").open("x") as f:
            json.dump(
                {
                    "original_prefix_bytes": n,
                    "first": n - overlap,
                    "last": end,
                    "total": size,
                    "automatic_retry": False,
                },
                f,
                sort_keys=True,
            )
        response = request(n - overlap, end)
        try:
            if (
                response.status_code != 206
                or response.headers.get("Content-Range") != f"bytes {n - overlap}-{end}/{size}"
                or response.headers.get("Content-Encoding", "identity") != "identity"
            ):
                raise ValueError(
                    "Exact bounded unencoded HTTP range required; original prefix retained"
                )
            first = bytearray()
            written = 0
            with path.open("ab") as f:
                try:
                    for block in response.iter_content(2**20):
                        if len(first) < overlap:
                            need = overlap - len(first)
                            first.extend(block[:need])
                            block = block[need:]
                            if len(first) == overlap and bytes(first) != boundary:
                                raise ValueError("Existing original prefix overlap differs")
                        if block:
                            if n + written + len(block) > end + 1:
                                raise ValueError("Remote bytes exceed the bounded suffix")
                            f.write(block)
                            h.update(block)
                            written += len(block)
                finally:
                    f.flush()
                    os.fsync(f.fileno())
            if len(first) != overlap or n + written != end + 1 or path.stat().st_size != end + 1:
                raise ValueError("Incomplete suffix retained for read-only reconciliation")
            n = end + 1
            with (journal / f"chunk-{number:03d}.verified.json").open("x") as f:
                json.dump({"whole_prefix_bytes": n, "new_suffix_bytes": written}, f)
        finally:
            response.close()
    if n != size or h.hexdigest() != digest:
        raise ValueError("Original whole archive size/SHA differs; preserve bytes for review")
    return {
        "original_prefix_bytes": original,
        "new_suffix_bytes": n - original,
        "whole_bytes": n,
        "sha256": h.hexdigest(),
        "original_prefix_not_downloaded_again": True,
    }
