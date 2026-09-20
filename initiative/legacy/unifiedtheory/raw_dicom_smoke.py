#!/usr/bin/env python3
"""Decode a few mounted studies through the raw-DICOM training path."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

try:
    from .constants import TARGETS
    from .raw_mil import RawStudyDataset, build_study_manifest, collate_raw_studies
except ImportError:
    from constants import TARGETS
    from raw_mil import RawStudyDataset, build_study_manifest, collate_raw_studies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--studies", type=int, default=2)
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args()

    started = time.monotonic()
    manifest = build_study_manifest(
        args.data_root / "train_series.csv", args.data_root, "train"
    )
    print(f"manifest studies={len(manifest)} seconds={time.monotonic() - started:.2f}", flush=True)
    labels = pd.read_csv(args.labels_csv, dtype={"StudyInstanceUID": str})
    labels = labels[labels["StudyInstanceUID"].isin(manifest)].head(args.studies)
    dataset = RawStudyDataset(
        labels, manifest, TARGETS, args.image_size, args.slices, 160.0, False
    )
    items = []
    for index in range(len(dataset)):
        started = time.monotonic()
        item = dataset[index]
        items.append(item)
        print(
            f"{item['uid']} pixels={tuple(item['pixels'].shape)} "
            f"range=({item['pixels'].min():.3f},{item['pixels'].max():.3f}) "
            f"seconds={time.monotonic() - started:.2f}",
            flush=True,
        )
    batch = collate_raw_studies(items)
    assert batch["num_studies"] == len(items)
    assert batch["pixels"].shape[0] == batch["study_index"].shape[0]
    print("raw DICOM smoke test passed", flush=True)


if __name__ == "__main__":
    main()
