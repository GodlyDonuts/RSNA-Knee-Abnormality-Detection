#!/usr/bin/env python3
"""Audit the fixed all-seed mean-pooling OOF ensemble on expert labels.

Every requested seed is retained for every target. Scores are combined inside
each held-out fold, so no study is evaluated by a checkpoint trained on it.
"""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hard-cases-per-class", type=int, default=5)
    return parser.parse_args()


def rank01(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_seed_oof(runs_dir: Path, seed: int, folds: int) -> pd.DataFrame:
    frames = []
    for fold in range(folds):
        path = runs_dir / f"seed{seed}" / f"mean_fold{fold}_oof.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"StudyInstanceUID", *TARGETS}
        if missing := required.difference(frame.columns):
            raise ValueError(f"{path} is missing {sorted(missing)}")
        frame = frame[["StudyInstanceUID", *TARGETS]].copy()
        frame["fold"] = fold
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"seed {seed} contains duplicate OOF studies")
    return result


def finite_auc(labels: pd.Series, scores: pd.Series) -> float:
    valid = labels.notna() & scores.notna()
    y = labels[valid].astype(float)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, scores[valid].astype(float)))


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels_csv)
    cache = pd.read_csv(args.cache_index)[["StudyInstanceUID", "scanner_group"]]
    required = {"StudyInstanceUID", "fold", *TARGETS}
    if missing := required.difference(labels.columns):
        raise ValueError(f"labels table is missing {sorted(missing)}")
    base = labels.merge(cache, on="StudyInstanceUID", how="left", validate="one_to_one")
    if base["scanner_group"].isna().any():
        raise ValueError("some labeled studies have no scanner group")

    seed_columns: dict[int, dict[str, str]] = {}
    for seed in args.seeds:
        oof = load_seed_oof(args.runs_dir, seed, args.folds)
        check = base[["StudyInstanceUID", "fold"]].merge(
            oof[["StudyInstanceUID", "fold"]],
            on="StudyInstanceUID",
            how="outer",
            suffixes=("_label", "_oof"),
            indicator=True,
            validate="one_to_one",
        )
        if not (check["_merge"] == "both").all() or not (
            check["fold_label"] == check["fold_oof"]
        ).all():
            raise ValueError(f"seed {seed} OOF coverage/fold assignment mismatch")
        renamed = {target: f"{target}__seed{seed}" for target in TARGETS}
        seed_columns[seed] = renamed
        base = base.merge(
            oof[["StudyInstanceUID", *TARGETS]].rename(columns=renamed),
            on="StudyInstanceUID",
            how="left",
            validate="one_to_one",
        )

    for target in TARGETS:
        member_columns = [seed_columns[seed][target] for seed in args.seeds]
        base[f"{target}__probability_mean"] = base[member_columns].mean(axis=1)
        ranked_columns = []
        for seed, column in zip(args.seeds, member_columns):
            ranked = f"{target}__seed{seed}__fold_rank"
            base[ranked] = base.groupby("fold", sort=False)[column].transform(rank01)
            ranked_columns.append(ranked)
        base[f"{target}__rank_mean"] = base[ranked_columns].mean(axis=1)

    methods = [f"seed{seed}" for seed in args.seeds] + ["probability_mean", "rank_mean"]
    cells = []
    for fold in range(args.folds):
        selected = base[base["fold"] == fold]
        for target in TARGETS:
            marker = pd.to_numeric(
                selected.get(f"{target}__gold", pd.Series(1, index=selected.index)),
                errors="coerce",
            ).fillna(0)
            expert_labels = pd.to_numeric(selected[target], errors="coerce").where(marker > 0)
            for method in methods:
                score_column = (
                    seed_columns[int(method.removeprefix("seed"))][target]
                    if method.startswith("seed")
                    else f"{target}__{method}"
                )
                cells.append(
                    {
                        "method": method,
                        "fold": fold,
                        "target": target,
                        "auc": finite_auc(expert_labels, selected[score_column]),
                        "gold_positive": int((expert_labels > 0.5).sum()),
                        "gold_negative": int((expert_labels <= 0.5).sum()),
                    }
                )
    cells_frame = pd.DataFrame(cells)
    cells_frame.to_csv(args.output / "fold_target_auc.csv", index=False)

    method_summary = {}
    for method in methods:
        chosen = cells_frame[cells_frame["method"] == method]
        method_summary[method] = {
            "macro_auc": float(chosen["auc"].mean()),
            "fold_macro_auc": {
                str(int(fold)): float(group["auc"].mean())
                for fold, group in chosen.groupby("fold")
            },
            "target_auc": {
                target: float(group["auc"].mean())
                for target, group in chosen.groupby("target", sort=False)
            },
            "finite_fold_target_cells": int(chosen["auc"].notna().sum()),
        }

    rank_cells = cells_frame[cells_frame["method"] == "rank_mean"].set_index(
        ["fold", "target"]
    )["auc"]
    single_cells = (
        cells_frame[cells_frame["method"].str.startswith("seed")]
        .groupby(["fold", "target"])["auc"]
        .mean()
    )
    paired_gain = rank_cells - single_cells

    hard_cases = []
    for target in TARGETS:
        marker = pd.to_numeric(
            base.get(f"{target}__gold", pd.Series(1, index=base.index)), errors="coerce"
        ).fillna(0)
        expert = base[marker > 0].copy()
        expert["label"] = pd.to_numeric(expert[target], errors="coerce")
        expert["score"] = expert[f"{target}__rank_mean"]
        negative = expert[expert["label"] <= 0.5].nlargest(
            args.hard_cases_per_class, "score"
        )
        positive = expert[expert["label"] > 0.5].nsmallest(
            args.hard_cases_per_class, "score"
        )
        for kind, cases in (("high_scoring_negative", negative), ("low_scoring_positive", positive)):
            for _, row in cases.iterrows():
                hard_cases.append(
                    {
                        "target": target,
                        "error_type": kind,
                        "StudyInstanceUID": row["StudyInstanceUID"],
                        "fold": int(row["fold"]),
                        "scanner_group": row["scanner_group"],
                        "label": float(row["label"]),
                        "rank_mean_score": float(row["score"]),
                    }
                )
    pd.DataFrame(hard_cases).to_csv(args.output / "hard_cases.csv", index=False)

    ensemble = base[["StudyInstanceUID", "fold", "scanner_group"]].copy()
    for target in TARGETS:
        ensemble[target] = base[f"{target}__rank_mean"]
        ensemble[f"{target}__gold"] = base.get(f"{target}__gold", 1)
    ensemble.to_csv(args.output / "mean_rank_ensemble_oof.csv", index=False)

    checkpoints = []
    for seed in args.seeds:
        for fold in range(args.folds):
            path = args.runs_dir / f"seed{seed}" / f"mean_fold{fold}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoints.append(
                {"seed": seed, "fold": fold, "path": str(path), "sha256": sha256(path)}
            )
    (args.output / "mean_checkpoint_manifest.json").write_text(
        json.dumps({"ensemble": "all-member rank mean", "checkpoints": checkpoints}, indent=2)
        + "\n"
    )

    result = {
        "contract": "all three seeds; no target-wise member selection; fold-local rank mean primary",
        "studies": len(base),
        "expert_studies": int(
            base[[f"{target}__gold" for target in TARGETS]].any(axis=1).sum()
        ),
        "fold_target_cells": args.folds * len(TARGETS),
        "methods": method_summary,
        "rank_ensemble_gain_over_mean_single_seed_cell_auc": float(paired_gain.mean()),
        "rank_ensemble_better_cells": int((paired_gain > 0).sum()),
        "rank_ensemble_equal_cells": int((paired_gain == 0).sum()),
        "rank_ensemble_worse_cells": int((paired_gain < 0).sum()),
    }
    (args.output / "audit_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
