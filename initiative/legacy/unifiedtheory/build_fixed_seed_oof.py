#!/usr/bin/env python3
"""Build a fixed equal-rank OOF ensemble across repeated training seeds."""

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
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rank01(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def finite_auc(y: pd.Series, prediction: pd.Series) -> float:
    valid = y.notna() & prediction.notna()
    return (
        float(roc_auc_score(y[valid], prediction[valid]))
        if y[valid].nunique() == 2
        else float("nan")
    )


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    sources: dict[str, list[str]] = {}
    checkpoint_sources = []
    ensemble_frames = []
    seed_frames: dict[int, list[pd.DataFrame]] = {seed: [] for seed in args.seeds}
    for fold in range(args.folds):
        members = []
        for seed in args.seeds:
            path = args.runs / f"seed{seed}" / f"{args.prefix}_fold{fold}_oof.csv"
            checkpoint = args.runs / f"seed{seed}" / f"{args.prefix}_fold{fold}.pt"
            if not path.is_file() or not checkpoint.is_file():
                raise FileNotFoundError(path if not path.is_file() else checkpoint)
            frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
            if missing := {"StudyInstanceUID", *TARGETS}.difference(frame.columns):
                raise ValueError(f"{path} is missing {sorted(missing)}")
            if frame["StudyInstanceUID"].duplicated().any():
                raise ValueError(f"{path} contains duplicate studies")
            members.append((seed, path, frame[["StudyInstanceUID", *TARGETS]]))
            checkpoint_sources.append(str(checkpoint))
        reference = members[0][2]["StudyInstanceUID"].tolist()
        for seed, path, frame in members[1:]:
            if frame["StudyInstanceUID"].tolist() != reference:
                raise ValueError(f"seed {seed} fold {fold} study order differs from the anchor seed")
        ensemble = pd.DataFrame({"StudyInstanceUID": reference})
        for target in TARGETS:
            ranks = [rank01(frame[target]) for _, _, frame in members]
            ensemble[target] = pd.concat(ranks, axis=1).mean(axis=1)
        output_path = args.output / f"{args.name}_fold{fold}_oof.csv"
        ensemble.to_csv(output_path, index=False)
        ensemble_frames.append(ensemble.assign(fold=fold))
        for seed, path, frame in members:
            seed_frames[seed].append(frame.assign(fold=fold))
            sources.setdefault(str(seed), []).append(str(path))

    audit: dict[str, object] = {
        "schema_version": 1,
        "method": "fixed equal-rank seed ensemble",
        "name": args.name,
        "prefix": args.prefix,
        "seeds": args.seeds,
        "folds": args.folds,
        "source_files": sources,
        "checkpoint_source_files": checkpoint_sources,
    }
    if args.labels_csv:
        labels = pd.read_csv(args.labels_csv, dtype={"StudyInstanceUID": str})
        predictions = {
            args.name: pd.concat(ensemble_frames, ignore_index=True),
            **{
                f"seed{seed}": pd.concat(frames, ignore_index=True)
                for seed, frames in seed_frames.items()
            },
        }
        model_scores = {}
        target_scores = {}
        for model, prediction in predictions.items():
            joined = labels.merge(
                prediction.drop(columns="fold"),
                on="StudyInstanceUID",
                suffixes=("", "__prediction"),
                validate="one_to_one",
            )
            scores = {}
            for target in TARGETS:
                marker = f"{target}__gold"
                y = pd.to_numeric(joined[target], errors="coerce")
                if marker in joined:
                    y = y.where(pd.to_numeric(joined[marker], errors="coerce").fillna(0) > 0)
                scores[target] = finite_auc(
                    y, pd.to_numeric(joined[f"{target}__prediction"], errors="coerce")
                )
            finite = [value for value in scores.values() if np.isfinite(value)]
            model_scores[model] = float(np.mean(finite)) if finite else float("nan")
            target_scores[model] = scores
        audit["macro_auc"] = model_scores
        audit["target_auc"] = target_scores
        best_seed = max(model_scores[f"seed{seed}"] for seed in args.seeds)
        audit["ensemble_gain_over_best_seed"] = model_scores[args.name] - best_seed
    (args.output / "manifest.json").write_text(json.dumps(audit, indent=2, allow_nan=True) + "\n")
    print(json.dumps(audit, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
