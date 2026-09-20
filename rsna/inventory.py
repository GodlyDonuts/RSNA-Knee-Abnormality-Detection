from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


def validate_metadata(inventory: pd.DataFrame, root: Path) -> dict:
    if set(inventory.split) != {"train_series", "test_series"}:
        raise ValueError("Unexpected source splits")
    if inventory.path.duplicated().any() or inventory.sop_uid.duplicated().any():
        raise ValueError("Duplicate source paths or SOP identities")
    if inventory.groupby("series_uid").study_uid.nunique().gt(1).any():
        raise ValueError("A series belongs to multiple studies")
    summary = {}
    for split in ("train", "test"):
        studies = pd.read_csv(root / f"{split}.csv")
        series = pd.read_csv(root / f"{split}_series.csv")
        rows = inventory[inventory.split.eq(f"{split}_series")]
        if (
            studies.StudyInstanceUID.duplicated().any()
            or series.SeriesInstanceUID.duplicated().any()
        ):
            raise ValueError(f"Duplicate metadata identities: {split}")
        expected = set(
            series[["StudyInstanceUID", "SeriesInstanceUID"]].itertuples(index=False, name=None)
        )
        actual = set(rows[["study_uid", "series_uid"]].itertuples(index=False, name=None))
        if expected != actual or set(studies.StudyInstanceUID) != set(rows.study_uid):
            raise ValueError(
                f"Source inventory differs from original study/series metadata: {split}"
            )
        summary[split] = {"studies": len(studies), "series": len(series), "dicoms": len(rows)}
    return summary


def build_inventory(repo: Path, revision: str) -> pd.DataFrame:
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", revision], cwd=repo).split(b"\0")
    entries = []
    for entry in tree:
        if not entry:
            continue
        attributes, path = entry.decode().split("\t", 1)
        if path.endswith(".dcm"):
            entries.append((attributes.split()[2], path))
    rows = []
    for start in range(0, len(entries), 10000):
        batch = entries[start : start + 10000]
        data = subprocess.check_output(
            ["git", "cat-file", "--batch"],
            cwd=repo,
            input="".join(f"{oid}\n" for oid, _ in batch).encode(),
        )
        offset = 0
        for expected_oid, path in batch:
            end = data.index(b"\n", offset)
            oid, kind, length = data[offset:end].decode().split()
            if oid != expected_oid or kind != "blob":
                raise ValueError("Unexpected git object")
            offset = end + 1
            blob = data[offset : offset + int(length)]
            offset += int(length) + 1
            storage = "git"
            sha256 = hashlib.sha256(blob).hexdigest()
            size = len(blob)
            if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                values = dict(line.split(" ", 1) for line in blob.decode().strip().splitlines())
                storage = "lfs"
                sha256 = values["oid"].removeprefix("sha256:")
                size = int(values["size"])
            parts = Path(path).parts
            if len(parts) != 4 or parts[0] not in ("train_series", "test_series"):
                raise ValueError(f"Unexpected DICOM path: {path}")
            rows.append(
                {
                    "split": parts[0],
                    "study_uid": parts[1],
                    "series_uid": parts[2],
                    "sop_uid": Path(path).stem,
                    "path": path,
                    "sha256": sha256,
                    "bytes": size,
                    "storage": storage,
                }
            )
        print(f"Read {len(rows)}/{len(entries)} LFS pointers", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--original-paths", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = build_inventory(args.repo, args.revision)
    expected = {p for p in args.original_paths.read_text().splitlines() if p.endswith(".dcm")}
    actual = set(inventory.path)
    if expected != actual or inventory.path.duplicated().any():
        raise ValueError("Inventory does not match original archive DICOM paths")
    split_counts = validate_metadata(inventory, args.metadata_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(args.output, index=False)
    summary = {
        "revision": args.revision,
        "files": len(inventory),
        "bytes": int(inventory.bytes.sum()),
        "storage": inventory.storage.value_counts().to_dict(),
        "unique_studies": int(inventory.study_uid.nunique()),
        "unique_series": int(inventory.series_uid.nunique()),
        "splits": split_counts,
        "duplicate_content_files": int(inventory.sha256.duplicated(keep=False).sum()),
        "cross_study_duplicate_hashes": int(
            (inventory.groupby("sha256").study_uid.nunique() > 1).sum()
        ),
        "path_inventory_sha256": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
