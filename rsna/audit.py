from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from rsna.schema import CACHE_VERSION


def audit_cache(inventory: pd.DataFrame, cache: Path, revision: str, output: Path) -> dict:
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
    for index, (keys, frame) in enumerate(inventory.groupby(["split", "study_uid", "series_uid"])):
        split, study, series = keys
        file = cache / split / study / f"{series}.npz"
        meta = json.loads(file.with_suffix(".json").read_text())
        if meta["source_revision"] != revision or meta["cache_version"] != CACHE_VERSION:
            raise ValueError(f"Cache provenance mismatch: {file}")
        if (meta["split"], meta["study_uid"], meta["series_uid"]) != keys:
            raise ValueError(f"Metadata identity mismatch: {file}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != meta["cache_sha256"]:
            raise ValueError(f"Cache checksum mismatch: {file}")
        if len(set(meta["sop_uids"])) != len(frame):
            raise ValueError(f"Missing/duplicate SOPs: {file}")
        recorded = dict(zip(meta["sop_uids"], meta["slice_sha256"]))
        if recorded != dict(zip(frame.sop_uid, frame.sha256)):
            raise ValueError(f"Source content differs from pinned inventory: {file}")
        with np.load(file, allow_pickle=False) as archive:
            volume = archive["volume"]
        if volume.dtype != np.uint16 or volume.shape != (len(frame), meta["side"], meta["side"]):
            raise ValueError(f"Volume shape/type mismatch: {file}")
        if len(meta["pixel_sha256"]) != len(frame):
            raise ValueError(f"Pixel hash coverage mismatch: {file}")
        height, width = meta["resized_shape"]
        top, left = (meta["side"] - height) // 2, (meta["side"] - width) // 2
        for i, digest in enumerate(meta["pixel_sha256"]):
            if np.ptp(volume[i, top : top + height, left : left + width]) == 0:
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
        "source_revision": revision,
        "cache_version": CACHE_VERSION,
        "studies": int(inventory.study_uid.nunique()),
        "series": len(expected),
        "decoded_slices": total_slices,
        "source_bytes": int(inventory.bytes.sum()),
        "cache_bytes": total_bytes,
        "series_warning_counts": dict(warnings),
        "series_transfer_syntax_counts": dict(syntaxes),
        "constant_resized_slices_excluded_from_duplicate_detection": constant_slices,
        "cross_study_pixel_duplicate_pairs": sorted(duplicates),
        "train_test_duplicate_pairs": train_test_duplicates,
        "image_interpretation": "Technical decode/geometry/content audit; no diagnostic image rereading.",
        "patient_identity": "Study UIDs do not establish that different studies belong to different patients.",
    }
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(series_rows).to_parquet(output / "series_cache_index.parquet", index=False)
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
