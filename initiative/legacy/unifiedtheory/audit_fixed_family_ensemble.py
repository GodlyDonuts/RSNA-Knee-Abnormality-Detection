#!/usr/bin/env python3
"""Audit a fixed equal-rank summary + patch ensemble across seeds and folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--summary-runs", type=Path, required=True)
    parser.add_argument("--patch-runs", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-gain", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_member(path: Path, expected: pd.Series) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if missing := {"StudyInstanceUID", *TARGETS}.difference(frame.columns):
        raise ValueError(f"{path} is missing {sorted(missing)}")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if set(frame["StudyInstanceUID"].astype(str)) != set(expected.astype(str)):
        raise ValueError(f"{path} does not exactly cover its held-out fold")
    return frame.set_index("StudyInstanceUID").loc[expected.astype(str)].reset_index()


def finite_auc(labels: pd.Series, scores: pd.Series) -> float:
    valid = labels.notna() & scores.notna()
    y = labels[valid]
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, scores[valid]))


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels_csv)
    rows = []
    combined_oof = []
    for fold in range(args.folds):
        held = labels[labels["fold"] == fold].copy().reset_index(drop=True)
        summary = [
            load_member(
                args.summary_runs / f"seed{seed}" / f"mean_fold{fold}_oof.csv",
                held["StudyInstanceUID"],
            )
            for seed in args.seeds
        ]
        patch = [
            load_member(
                args.patch_runs / f"seed{seed}" / f"patch_mean_fold{fold}_oof.csv",
                held["StudyInstanceUID"],
            )
            for seed in args.seeds
        ]
        fold_output = held[["StudyInstanceUID", "fold"]].copy()
        for target in TARGETS:
            summary_ranks = [member[target].rank(method="average", pct=True) for member in summary]
            patch_ranks = [member[target].rank(method="average", pct=True) for member in patch]
            summary_score = pd.concat(summary_ranks, axis=1).mean(axis=1)
            patch_score = pd.concat(patch_ranks, axis=1).mean(axis=1)
            combined_score = pd.concat(summary_ranks + patch_ranks, axis=1).mean(axis=1)
            marker = pd.to_numeric(held[f"{target}__gold"], errors="coerce").fillna(0)
            y = pd.to_numeric(held[target], errors="coerce").where(marker > 0)
            summary_auc = finite_auc(y, summary_score)
            patch_auc = finite_auc(y, patch_score)
            combined_auc = finite_auc(y, combined_score)
            rows.append(
                {
                    "fold": fold,
                    "target": target,
                    "summary_rank_auc": summary_auc,
                    "patch_rank_auc": patch_auc,
                    "combined_rank_auc": combined_auc,
                    "combined_gain": combined_auc - summary_auc,
                }
            )
            fold_output[target] = combined_score
            fold_output[f"{target}__gold"] = marker
        combined_oof.append(fold_output)

    cells = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output / "fold_target_auc.csv", index=False)
    pd.concat(combined_oof, ignore_index=True).to_csv(
        args.output / "fixed_family_rank_oof.csv", index=False
    )
    summary_macro = float(cells["summary_rank_auc"].mean())
    patch_macro = float(cells["patch_rank_auc"].mean())
    combined_macro = float(cells["combined_rank_auc"].mean())
    gain = combined_macro - summary_macro
    result = {
        "contract": "all seeds and both families receive equal rank weight; no fitted or target-wise weights",
        "paired_fold_target_cells": int(cells["combined_gain"].notna().sum()),
        "summary_rank_macro_auc": summary_macro,
        "patch_rank_macro_auc": patch_macro,
        "combined_rank_macro_auc": combined_macro,
        "combined_gain": gain,
        "minimum_preregistered_gain": args.minimum_gain,
        "promote_combined": bool(gain >= args.minimum_gain),
        "better_equal_worse_cells": [
            int((cells["combined_gain"] > 0).sum()),
            int((cells["combined_gain"] == 0).sum()),
            int((cells["combined_gain"] < 0).sum()),
        ],
        "target_auc": {
            target: {
                "summary": float(group["summary_rank_auc"].mean()),
                "patch": float(group["patch_rank_auc"].mean()),
                "combined": float(group["combined_rank_auc"].mean()),
                "gain": float(group["combined_gain"].mean()),
            }
            for target, group in cells.groupby("target", sort=False)
        },
    }
    (args.output / "audit_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
