#!/usr/bin/env python3
"""Run the preregistered mean/index/physical aggregation comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--group-column", default="scanner_group")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    for seed in args.seeds:
        seed_output = args.output / f"seed{seed}"
        for aggregator in ("mean", "index_alibi", "physical_alibi"):
            for fold in range(args.folds):
                checkpoint = seed_output / f"{aggregator}_fold{fold}.pt"
                if checkpoint.exists():
                    print(f"skip existing {checkpoint}", flush=True)
                    continue
                command = [
                    sys.executable,
                    str(here / "train.py"),
                    "--cache-index",
                    str(args.cache_index),
                    "--labels-csv",
                    str(args.labels_csv),
                    "--output",
                    str(seed_output),
                    "--aggregator",
                    aggregator,
                    "--fold",
                    str(fold),
                    "--folds",
                    str(args.folds),
                    "--seed",
                    str(seed),
                    "--group-column",
                    args.group_column,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--workers",
                    str(args.workers),
                    *args.extra,
                ]
                print(" ".join(command), flush=True)
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
