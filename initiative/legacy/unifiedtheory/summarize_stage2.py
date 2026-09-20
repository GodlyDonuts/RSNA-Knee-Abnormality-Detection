#!/usr/bin/env python3
"""Compare one fixed patch-stage run with its paired summary baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="mean")
    parser.add_argument("--candidate-name", default="patch_mean")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-gain", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_oof(directory: Path, name: str, fold: int) -> pd.DataFrame:
    path = directory / f"{name}_fold{fold}_oof.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if missing := {"StudyInstanceUID", *TARGETS}.difference(frame.columns):
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return frame[["StudyInstanceUID", *TARGETS]]


def auc(y: pd.Series, score: pd.Series) -> float:
    valid = y.notna() & score.notna()
    values = y[valid]
    if values.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(values, score[valid]))


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels_csv)
    rows = []
    for fold in range(args.folds):
        held_out = labels[labels["fold"] == fold].copy()
        baseline = load_oof(args.baseline_dir, args.baseline_name, fold)
        candidate = load_oof(args.candidate_dir, args.candidate_name, fold)
        joined = held_out.merge(
            baseline,
            on="StudyInstanceUID",
            how="left",
            suffixes=("", "__baseline"),
            validate="one_to_one",
        ).merge(
            candidate.rename(columns={target: f"{target}__candidate" for target in TARGETS}),
            on="StudyInstanceUID",
            how="left",
            validate="one_to_one",
        )
        if len(joined) != len(held_out):
            raise ValueError(f"fold {fold} OOF coverage mismatch")
        for target in TARGETS:
            marker = pd.to_numeric(joined[f"{target}__gold"], errors="coerce").fillna(0)
            y = pd.to_numeric(joined[target], errors="coerce").where(marker > 0)
            baseline_auc = auc(y, joined[f"{target}__baseline"])
            candidate_auc = auc(y, joined[f"{target}__candidate"])
            # This is a fixed diversity test, not a fitted stack: both members
            # receive equal weight after ranking over the complete held-out
            # fold.  Expert labels are used only below for evaluation.
            baseline_rank = joined[f"{target}__baseline"].rank(
                method="average", pct=True
            )
            candidate_rank = joined[f"{target}__candidate"].rank(
                method="average", pct=True
            )
            blend_auc = auc(y, (baseline_rank + candidate_rank) / 2)
            rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "baseline_auc": baseline_auc,
                    "candidate_auc": candidate_auc,
                    "gain": candidate_auc - baseline_auc,
                    "fixed_rank_blend_auc": blend_auc,
                    "fixed_rank_blend_gain": blend_auc - baseline_auc,
                }
            )
    cells = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output.with_suffix(".csv"), index=False)
    baseline_macro = float(cells["baseline_auc"].mean())
    candidate_macro = float(cells["candidate_auc"].mean())
    gain = candidate_macro - baseline_macro
    blend_macro = float(cells["fixed_rank_blend_auc"].mean())
    blend_gain = blend_macro - baseline_macro
    result = {
        "paired_fold_target_cells": int(cells["gain"].notna().sum()),
        "baseline_macro_auc": baseline_macro,
        "candidate_macro_auc": candidate_macro,
        "candidate_gain": gain,
        "minimum_preregistered_gain": args.minimum_gain,
        "promote_candidate": bool(gain >= args.minimum_gain),
        "better_equal_worse_cells": [
            int((cells["gain"] > 0).sum()),
            int((cells["gain"] == 0).sum()),
            int((cells["gain"] < 0).sum()),
        ],
        "fixed_rank_blend_macro_auc": blend_macro,
        "fixed_rank_blend_gain": blend_gain,
        "promote_fixed_rank_blend": bool(blend_gain >= args.minimum_gain),
        "fixed_rank_blend_better_equal_worse_cells": [
            int((cells["fixed_rank_blend_gain"] > 0).sum()),
            int((cells["fixed_rank_blend_gain"] == 0).sum()),
            int((cells["fixed_rank_blend_gain"] < 0).sum()),
        ],
        "target_auc": {
            target: {
                "baseline": float(group["baseline_auc"].mean()),
                "candidate": float(group["candidate_auc"].mean()),
                "gain": float(group["gain"].mean()),
                "fixed_rank_blend": float(group["fixed_rank_blend_auc"].mean()),
                "fixed_rank_blend_gain": float(group["fixed_rank_blend_gain"].mean()),
            }
            for target, group in cells.groupby("target", sort=False)
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
