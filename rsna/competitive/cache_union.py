"""Metadata-only union of already verified private native cache roots.

No pixel array is copied, re-encoded or quantized. CompetitiveStudyDataset keeps
its original payload SHA, dtype, geometry, normalization and sampling checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import pandas as pd

from rsna.schema import CACHE_VERSION

VERSION = "private_verified_native_metadata_union_v1"
MAX_FILES = 32000


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def metadata_sha(value: dict) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def relative_path(uid: str, sid: str) -> Path:
    value = PurePosixPath(f"train_series/{uid}/{sid}.npz")
    if len(value.parts) != 3 or ".." in value.parts or value.is_absolute():
        raise ValueError("Exact original native cache identity required")
    return Path(value.as_posix())


def signature(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Original regular cache payload required")
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


class MetadataUnion:
    def __init__(self, cache: Path, value: dict):
        self.cache, self.value = cache, value

    @classmethod
    def build(
        cls,
        destination: Path,
        roots: list[Path],
        series: pd.DataFrame,
        study_uids: set[str],
        *,
        max_bytes: int = 512 * 2**20,
    ):
        if destination.exists() or not roots or not study_uids:
            raise ValueError("Fresh metadata destination and actual source roots required")
        if len(set(p.resolve() for p in roots)) != len(roots) or any(p.is_symlink() for p in roots):
            raise ValueError("Distinct original nonsymlink cache roots required")
        rows = series.loc[series.StudyInstanceUID.isin(study_uids)]
        if (
            set(rows.StudyInstanceUID) != study_uids
            or rows.duplicated(["StudyInstanceUID", "SeriesInstanceUID"]).any()
            or not 0 < len(rows) <= MAX_FILES
        ):
            raise ValueError("Complete distinct native series membership required")
        overhead = 8192 * (len(rows) + len(study_uids) + 8)
        if overhead >= max_bytes:
            raise ValueError("Metadata union budget cannot cover files and directory overhead")
        destination.mkdir(mode=0o700)
        records, configurations = {}, set()
        used = 0
        for row in rows.itertuples(index=False):
            relative = relative_path(row.StudyInstanceUID, row.SeriesInstanceUID)
            matches = [p for root in roots if (p := root / relative).with_suffix(".json").is_file()]
            if len(matches) != 1:
                raise ValueError("Each source series must have exactly one owning cache")
            original = matches[0]
            raw = original.with_suffix(".json").read_bytes()
            meta = json.loads(raw)
            if original.with_suffix(".json").is_symlink() or (
                meta["cache_path"] != relative.as_posix()
                or meta["study_uid"] != row.StudyInstanceUID
                or meta["series_uid"] != row.SeriesInstanceUID
                or meta["cache_version"] != CACHE_VERSION
                or meta["split"] != "train_series"
            ):
                raise ValueError("Original native metadata/identity differs")
            state = signature(original)
            if state["bytes"] != meta["cache_bytes"]:
                raise ValueError("Original native payload size differs")
            configurations.add((meta["side"], meta["source_revision"]))
            if len(configurations) != 1:
                raise ValueError("Union must retain one original resolution/source")
            used += len(raw)
            # Also reserve blocks/directories/index overhead, not just JSON text.
            if used > max_bytes - overhead:
                raise ValueError("Metadata union exceeds its explicit byte budget")
            target = destination / relative.with_suffix(".json")
            target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(raw)
            records[relative.as_posix()] = {
                "source": str(original.resolve()),
                "payload_stat": state,
                "payload_sha256": meta["cache_sha256"],
                "metadata_bytes": len(raw),
                "metadata_sha256": sha(raw),
                "parsed_metadata_sha256": metadata_sha(meta),
            }
        value = {
            "version": VERSION,
            "roots": [str(p.resolve()) for p in roots],
            "studies": len(study_uids),
            "series": len(records),
            "side": configurations.pop()[0],
            "source_pixels_copied": 0,
            "metadata_text_bytes": used,
            "max_bytes": max_bytes,
            "records": records,
        }
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if used + len(payload) > max_bytes - overhead:
            raise ValueError("Union index exceeds explicit byte budget")
        with (destination / "index.json").open("xb") as handle:
            handle.write(payload)
        return cls.load(destination / "index.json", sha(payload))

    @classmethod
    def load(cls, path: Path, expected_sha256: str):
        raw = path.read_bytes()
        if path.is_symlink() or len(raw) > 64 * 2**20 or sha(raw) != expected_sha256:
            raise ValueError("Exact private metadata-union index required")
        value = json.loads(raw)
        if (
            value.get("version") != VERSION
            or value.get("source_pixels_copied") != 0
            or not 0 < value.get("series", 0) <= MAX_FILES
            or len(value.get("records", {})) != value["series"]
        ):
            raise ValueError("Admitted metadata-only union required")
        roots = [Path(p) for p in value["roots"]]
        for relative, record in value["records"].items():
            parts = PurePosixPath(relative).parts
            if len(parts) != 3 or parts[0] != "train_series" or ".." in parts:
                raise ValueError("Unsafe source cache path")
            source = Path(record["source"])
            if not any(source.resolve().is_relative_to(p) for p in roots):
                raise ValueError("Union payload escaped its original owning roots")
            signature_now = signature(source)
            if signature_now != record["payload_stat"]:
                raise ValueError("Original source payload changed after union creation")
            meta_path = path.parent / Path(relative).with_suffix(".json")
            if meta_path.is_symlink() or sha(meta_path.read_bytes()) != record["metadata_sha256"]:
                raise ValueError("Copied original metadata changed")
        return cls(path.parent, value)

    def __call__(self, metadata: dict) -> bytes:
        record = self.value["records"].get(metadata.get("cache_path"))
        if (
            record is None
            or metadata_sha(metadata) != record["parsed_metadata_sha256"]
            or metadata.get("cache_sha256") != record["payload_sha256"]
        ):
            raise ValueError("Loader metadata differs from admitted source member")
        source = Path(record["source"])
        if signature(source) != record["payload_stat"]:
            raise ValueError("Original payload stat changed")
        # The unchanged canonical loader hashes the returned bytes ONCE, before
        # decompressing, and checks shape/dtype/source-slice membership as before.
        return source.read_bytes()
