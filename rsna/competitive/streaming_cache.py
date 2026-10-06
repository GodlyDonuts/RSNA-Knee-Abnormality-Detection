"""Bounded disposable cache rebuilt from original DICOMs, not reduced MRI views."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import OrderedDict
from pathlib import Path, PurePosixPath

from rsna.preprocess import Source, cache_series


class StreamingCache:
    """Own a fresh scratch directory; retained source archives are never evicted."""

    def __init__(
        self, source: Source, scratch: Path, *, capacity_bytes: int, min_free_gib: float = 3
    ):
        if (
            type(capacity_bytes) is not int
            or capacity_bytes < 1
            or type(min_free_gib) not in (int, float)
            or not math.isfinite(min_free_gib)
            or min_free_gib < 3
        ):
            raise ValueError("Positive scratch cap and at least3GiB reserve required")
        if scratch.exists():
            raise ValueError("Streaming scratch must be a fresh owned directory")
        scratch.mkdir(mode=0o700, parents=True)
        self.source, self.scratch = source, scratch.resolve()
        self.capacity_bytes, self.min_free_gib = capacity_bytes, min_free_gib
        self.entries: OrderedDict[str, tuple[Path, int, str]] = OrderedDict()
        self.used_bytes = 0

    def read(self, metadata: dict) -> bytes:
        relative = PurePosixPath(metadata["cache_path"])
        if (
            relative.is_absolute()
            or len(relative.parts) != 3
            or ".." in relative.parts
            or relative.parts[0] not in {"train_series", "test_series"}
            or relative.suffix != ".npz"
            or relative.parts[1] != metadata["study_uid"]
            or relative.stem != metadata["series_uid"]
        ):
            raise ValueError("Original split/study/series cache identity required")
        key = relative.as_posix()
        size = metadata["cache_bytes"]
        if type(size) is not int or size < 1:
            raise ValueError("Positive original volume size required")
        metadata_bytes = json.dumps(metadata, sort_keys=True).encode()
        retained_bytes = size + len(metadata_bytes)
        metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
        if retained_bytes > self.capacity_bytes:
            raise ValueError("One original volume exceeds the admitted scratch cap")
        if key in self.entries:
            path, retained_size, digest = self.entries.pop(key)
            self.entries[key] = (path, retained_size, digest)
            if (retained_size, digest) != (retained_bytes, metadata_sha):
                raise ValueError("A retained original volume binding changed")
        else:
            while self.entries and self.used_bytes + retained_bytes > self.capacity_bytes:
                _, (old, old_size, _) = self.entries.popitem(last=False)
                old.unlink()
                old.with_suffix(".json").unlink()
                self.used_bytes -= old_size
            # This disk check funds a full uint16 write buffer; actual decode peak
            # RAM remains a separate hardware admission requirement.
            pending = metadata["n_slices"] * metadata["side"] ** 2 * 2 + size + 2**20
            if shutil.disk_usage(self.scratch).free < self.min_free_gib * 2**30 + pending:
                raise OSError("Streaming decode/write cannot preserve the scratch reserve")
            paths = [
                f"{relative.parts[0]}/{metadata['study_uid']}/{metadata['series_uid']}/{sop}.dcm"
                for sop in metadata["sop_uids"]
            ]
            generated = cache_series(
                self.source,
                paths,
                self.scratch,
                metadata["side"],
                metadata["source_revision"],
                min_free_gib=self.min_free_gib,
            )
            if generated != metadata:
                raise ValueError("Rebuilt original image/geometry/source/cache metadata differs")
            path = self.scratch / relative
            self.entries[key] = (path, retained_bytes, metadata_sha)
            self.used_bytes += retained_bytes
        payload = path.read_bytes()
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != metadata["cache_sha256"]:
            raise ValueError("Original volume bytes differ")
        if self.used_bytes > self.capacity_bytes:
            raise ValueError("Owned scratch cap exceeded")
        return payload

    def statistics(self) -> dict:
        return {
            "resident_volumes": len(self.entries),
            "resident_bytes": self.used_bytes,
            "capacity_bytes": self.capacity_bytes,
            "original_source_mutated": False,
        }
