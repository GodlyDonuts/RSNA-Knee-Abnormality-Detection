#!/usr/bin/env python3
"""Derive a compact, exactly paired summary cache from patch-grid caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

try:
    from .data import load_feature_cache
except ImportError:
    from data import load_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.cache_index)
    if source["StudyInstanceUID"].duplicated().any():
        raise ValueError("source cache index contains duplicate studies")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for number, row in enumerate(source.to_dict("records"), start=1):
        source_path = Path(row["cache_file"])
        target_path = args.output / source_path.name
        if not target_path.exists():
            payload = load_feature_cache(source_path)
            payload.pop("patch_features", None)
            payload.pop("patch_mask", None)
            torch.save(payload, target_path)
        row["cache_file"] = str(target_path.resolve())
        row["patch_dim"] = 0
        row["patches_per_slice"] = 0
        records.append(row)
        if number % 250 == 0 or number == len(source):
            print(f"summary cache {number}/{len(source)}", flush=True)
    output_index = args.output / args.cache_index.name
    pd.DataFrame(records).to_csv(output_index, index=False)
    print(f"wrote {output_index}")


if __name__ == "__main__":
    main()
