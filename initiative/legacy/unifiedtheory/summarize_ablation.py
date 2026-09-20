#!/usr/bin/env python3
"""Summarize paired fold/seed results and evaluate the promotion rule."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


AGGREGATORS = ("mean", "index_alibi", "physical_alibi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--minimum-gain", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs: dict[tuple[str, int, int], float] = {}
    pattern = re.compile(r"(mean|index_alibi|physical_alibi)_fold(\d+)_history\.json$")
    for path in args.results.rglob("*_history.json"):
        match = pattern.search(path.name)
        if not match:
            continue
        seed_match = re.search(r"seed(\d+)", str(path.parent))
        seed = int(seed_match.group(1)) if seed_match else -1
        history = json.loads(path.read_text())
        finite = [float(row["macro_auc"]) for row in history if np.isfinite(row["macro_auc"])]
        if finite:
            runs[(match.group(1), seed, int(match.group(2)))] = max(finite)
    paired = []
    keys = sorted({(seed, fold) for _, seed, fold in runs})
    for seed, fold in keys:
        if all((aggregator, seed, fold) in runs for aggregator in AGGREGATORS):
            row = {aggregator: runs[(aggregator, seed, fold)] for aggregator in AGGREGATORS}
            row.update({"seed": seed, "fold": fold})
            paired.append(row)
    if not paired:
        raise SystemExit("no complete mean/index/physical fold triplets found")
    means = {
        aggregator: float(np.mean([row[aggregator] for row in paired]))
        for aggregator in AGGREGATORS
    }
    strongest_control = max(means["mean"], means["index_alibi"])
    gain = means["physical_alibi"] - strongest_control
    result = {
        "complete_paired_runs": len(paired),
        "mean_macro_auc": means,
        "physical_gain_over_strongest_control": gain,
        "minimum_preregistered_gain": args.minimum_gain,
        "promote_physical_alibi": bool(gain >= args.minimum_gain),
        "runs": paired,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
