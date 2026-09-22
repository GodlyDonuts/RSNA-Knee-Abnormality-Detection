from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from rsna.schema import CACHE_VERSION


def inventory_fingerprint(inventory: pd.DataFrame) -> str:
    """Bind an audit to canonical inventory content, independent of Parquet encoding."""
    columns = ["split", "study_uid", "series_uid", "sop_uid", "path", "sha256", "bytes"]
    digest = hashlib.sha256()
    for row in inventory.sort_values("path")[columns].itertuples(index=False, name=None):
        values = [str(value) for value in row[:-1]] + [int(row[-1])]
        digest.update((json.dumps(values, separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def validate_inventory(inventory: pd.DataFrame) -> None:
    columns = {"split", "study_uid", "series_uid", "sop_uid", "path", "sha256", "bytes"}
    if not columns <= set(inventory.columns) or inventory.empty:
        raise ValueError("Audit requires a nonempty complete source inventory")
    if inventory[list(columns)].isna().any().any():
        raise ValueError("Source inventory contains missing values")
    if inventory.path.duplicated().any() or inventory.sop_uid.duplicated().any():
        raise ValueError("Source inventory contains duplicate paths/SOP UIDs")
    if (inventory.groupby("study_uid").split.nunique() != 1).any():
        raise ValueError("Study UID appears in multiple source splits")
    if not set(inventory.split) <= {"train_series", "test_series"}:
        raise ValueError("Unexpected source split")
    if not pd.api.types.is_integer_dtype(inventory.bytes) or (inventory.bytes <= 0).any():
        raise ValueError("Source inventory has invalid byte sizes")
    for row in inventory.itertuples(index=False):
        expected_path = f"{row.split}/{row.study_uid}/{row.series_uid}/{row.sop_uid}.dcm"
        if row.path != expected_path:
            raise ValueError("Source inventory path identity mismatch")


def audit_cache(inventory: pd.DataFrame, cache: Path, revision: str, output: Path) -> dict:
    validate_inventory(inventory)
    expected = set(
        inventory[["split", "study_uid", "series_uid"]].itertuples(index=False, name=None)
    )
    found = {
        tuple(file.relative_to(cache).parts[:2]) + (file.stem,) for file in cache.glob("*/*/*.json")
    }
    volumes = {
        tuple(file.relative_to(cache).parts[:2]) + (file.stem,) for file in cache.glob("*/*/*.npz")
    }
    if volumes != found:
        raise ValueError("Orphaned cache files require investigation")
    if found != expected:
        raise ValueError(
            f"Cache coverage: {len(expected - found)} missing, {len(found - expected)} extra"
        )
    warnings: Counter = Counter()
    syntaxes: Counter = Counter()
    owners: dict[str, str] = {}
    duplicates: set[tuple[str, str]] = set()
    series_rows = []
    total_slices, total_bytes, constant_slices = 0, 0, 0
    cache_sides: set[int] = set()
    for index, (keys, frame) in enumerate(inventory.groupby(["split", "study_uid", "series_uid"])):
        split, study, series = keys
        file = cache / split / study / f"{series}.npz"
        meta = json.loads(file.with_suffix(".json").read_text())
        if meta["source_revision"] != revision or meta["cache_version"] != CACHE_VERSION:
            raise ValueError(f"Cache provenance mismatch: {file}")
        if (meta["split"], meta["study_uid"], meta["series_uid"]) != keys:
            raise ValueError(f"Metadata identity mismatch: {file}")
        if meta["cache_path"] != str(file.relative_to(cache)):
            raise ValueError(f"Metadata cache path mismatch: {file}")
        if len(set(meta["sop_uids"])) != len(frame) or len(meta["sop_uids"]) != len(frame):
            raise ValueError(f"Missing/duplicate SOPs: {file}")
        if meta["n_slices"] != len(frame) or len(meta["slice_sha256"]) != len(frame):
            raise ValueError(f"Slice metadata coverage mismatch: {file}")
        if len(meta["positions"]) != len(frame):
            raise ValueError(f"Position metadata coverage mismatch: {file}")
        fingerprint = hashlib.sha256("\n".join(sorted(frame.path)).encode()).hexdigest()
        if meta["path_fingerprint"] != fingerprint:
            raise ValueError(f"Source path fingerprint mismatch: {file}")
        if meta["cache_bytes"] != file.stat().st_size:
            raise ValueError(f"Cache byte size mismatch: {file}")
        if type(meta["side"]) is not int or meta["side"] < 16:
            raise ValueError(f"Invalid cache side: {file}")
        cache_sides.add(meta["side"])
        if len(cache_sides) != 1:
            raise ValueError("Mixed cache dimensions across the release")
        if (
            len(meta["original_shape"]) != 3
            or meta["original_shape"][0] != len(frame)
            or any(type(size) is not int or size <= 0 for size in meta["original_shape"])
        ):
            raise ValueError(f"Invalid original volume dimensions: {file}")
        if len(meta["resized_shape"]) != 2 or any(
            type(size) is not int or size < 1 or size > meta["side"]
            for size in meta["resized_shape"]
        ):
            raise ValueError(f"Invalid resized volume dimensions: {file}")
        if not np.isfinite([meta["intensity_min"], meta["intensity_max"]]).all() or not (
            meta["intensity_min"] < meta["intensity_max"]
        ):
            raise ValueError(f"Invalid volume intensity range: {file}")
        geometry = [meta["orientation"], meta["pixel_spacing"], *meta["positions"]]
        if any(not np.isfinite(values).all() for values in geometry):
            raise ValueError(f"Non-finite DICOM geometry: {file}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != meta["cache_sha256"]:
            raise ValueError(f"Cache checksum mismatch: {file}")
        recorded = dict(zip(meta["sop_uids"], meta["slice_sha256"]))
        if recorded != dict(zip(frame.sop_uid, frame.sha256)):
            raise ValueError(f"Source content differs from pinned inventory: {file}")
        with np.load(file, allow_pickle=False) as archive:
            volume = archive["volume"]
        if volume.dtype != np.uint16 or volume.shape != (len(frame), meta["side"], meta["side"]):
            raise ValueError(f"Volume shape/type mismatch: {file}")
        if len(meta["pixel_sha256"]) != len(frame) or any(
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in meta["pixel_sha256"]
        ):
            raise ValueError(f"Pixel hash coverage mismatch: {file}")
        if (
            meta.get("pixel_hash_definition")
            != "shape_i64le_and_decoded_samples_f64le_before_rescale"
            or len(meta.get("pixel_nonconstant", [])) != len(frame)
            or any(type(value) is not bool for value in meta["pixel_nonconstant"])
        ):
            raise ValueError(f"Source pixel constancy/hash definition missing: {file}")
        for i, digest in enumerate(meta["pixel_sha256"]):
            if not meta["pixel_nonconstant"][i]:
                constant_slices += 1
                continue
            prior = owners.setdefault(digest, study)
            if prior != study:
                duplicates.add(tuple(sorted((prior, study))))
        warnings.update(meta["warnings"])
        syntaxes.update(meta["transfer_syntaxes"])
        total_slices += len(frame)
        total_bytes += file.stat().st_size
        series_rows.append(
            {
                "study_uid": study,
                "series_uid": series,
                "raw_split": split,
                "cache_path": meta["cache_path"],
                "n_slices": len(frame),
                "side": meta["side"],
                "cache_sha256": meta["cache_sha256"],
                "cache_bytes": meta["cache_bytes"],
                "warnings": meta["warnings"],
            }
        )
        if index % 1000 == 0:
            print(f"Rechecked {index + 1}/{len(expected)} series", flush=True)
    cross_split = defaultdict(set)
    for row in inventory[["study_uid", "split"]].drop_duplicates().itertuples(index=False):
        cross_split[row.study_uid].add(row.split)
    train_test_duplicates = [
        pair for pair in duplicates if len(cross_split[pair[0]] | cross_split[pair[1]]) > 1
    ]
    report = {
        "audit_schema_version": 2,
        "complete": True,
        "inventory_fingerprint_sha256": inventory_fingerprint(inventory),
        "source_revision": revision,
        "cache_version": CACHE_VERSION,
        "studies": int(inventory.study_uid.nunique()),
        "series": len(expected),
        "decoded_slices": total_slices,
        "source_bytes": int(inventory.bytes.sum()),
        "cache_side": next(iter(cache_sides)),
        "cache_bytes": total_bytes,
        "series_warning_counts": dict(warnings),
        "series_transfer_syntax_counts": dict(syntaxes),
        "constant_source_slices_excluded_from_duplicate_detection": constant_slices,
        "pixel_hash_definition": "shape_i64le_and_decoded_samples_f64le_before_rescale",
        "cross_study_pixel_duplicate_pairs": sorted(duplicates),
        "train_test_duplicate_pairs": sorted(train_test_duplicates),
        "study_uids_by_split": {
            str(split): sorted(frame.study_uid.unique().tolist())
            for split, frame in inventory.groupby("split")
        },
        "image_interpretation": "Technical decode/geometry/content audit; no diagnostic image rereading.",
        "patient_identity": "Study UIDs do not establish that different studies belong to different patients.",
    }
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "series_cache_index.parquet"
    pd.DataFrame(series_rows).to_parquet(index_path, index=False)
    report["series_cache_index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    (output / "dicom_audit.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_cache(pd.read_parquet(args.inventory), args.cache, args.revision, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
