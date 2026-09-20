#!/usr/bin/env python3
"""Fit target-wise convex rank blends using only out-of-fold predictions."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def load_member(spec: str) -> tuple[str, pd.DataFrame, list[str]]:
    if "=" not in spec:
        raise ValueError("member must be NAME=GLOB")
    name, pattern = spec.split("=", 1)
    paths = sorted(glob.glob(pattern))
    if not name or not paths:
        raise ValueError(f"member {spec!r} resolved to no prediction files")
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    required = {"StudyInstanceUID", *TARGETS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"member {name!r} missing columns {sorted(missing)}")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"member {name!r} contains duplicate OOF studies")
    return name, frame, paths


def candidate_weights(members: int, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidates = [np.full(members, 1 / members)]
    candidates.extend(np.eye(members))
    candidates.extend(rng.dirichlet(np.full(members, 0.5), size=samples))
    return np.asarray(candidates, dtype=float)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, score)) if np.unique(y).size == 2 else float("nan")


def auc_many_no_ties(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Vectorized Mann-Whitney AUC for continuously blended score columns."""

    positive = int(y.sum())
    negative = len(y) - positive
    order = np.argsort(score, axis=0, kind="stable")
    sorted_y = y[order]
    ranks = np.arange(1, len(y) + 1, dtype=float)[:, None]
    positive_rank_sum = (sorted_y * ranks).sum(axis=0)
    return (positive_rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def best_weight(predictions: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, float]:
    # Equal and one-hot candidates can contain tied ranks, so evaluate that
    # small exact prefix with sklearn. Random convex blends are continuous and
    # use a vectorized Mann-Whitney calculation.
    exact = min(predictions.shape[1] + 1, len(candidates))
    exact_scores = np.asarray([auc(y, predictions @ weight) for weight in candidates[:exact]])
    best_index = int(np.nanargmax(exact_scores))
    best = candidates[best_index]
    best_score = float(exact_scores[best_index])
    for start in range(exact, len(candidates), 512):
        block = candidates[start : start + 512]
        scores = predictions @ block.T
        values = auc_many_no_ties(y, scores)
        offset = int(np.nanargmax(values))
        value = float(values[offset])
        if value > best_score + 1e-12:
            best_score = value
            best = block[offset]
    return best, best_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--member", action="append", required=True, help="NAME=OOF_GLOB")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = [load_member(spec) for spec in args.member]
    names = [row[0] for row in loaded]
    labels = pd.read_csv(args.labels_csv)
    labels["StudyInstanceUID"] = labels["StudyInstanceUID"].astype(str)
    if args.fold_column not in labels:
        raise ValueError("labels CSV must contain precomputed folds for honest stacking")
    gold_columns = [f"{target}__gold" for target in TARGETS if f"{target}__gold" in labels]
    merged = labels[["StudyInstanceUID", args.fold_column, *TARGETS, *gold_columns]].copy()
    for name, frame, _ in loaded:
        renamed = frame[["StudyInstanceUID", *TARGETS]].rename(
            columns={target: f"{name}::{target}" for target in TARGETS}
        )
        merged = merged.merge(renamed, on="StudyInstanceUID", how="inner", validate="one_to_one")
    if not len(merged):
        raise ValueError("members and labels have no shared OOF studies")
    candidates = candidate_weights(len(names), args.samples, args.seed)
    final_weights: dict[str, list[float]] = {}
    full_scores: dict[str, float] = {}
    nested_scores: dict[str, float] = {}
    for target_index, target in enumerate(TARGETS):
        y_all = pd.to_numeric(merged[target], errors="coerce").to_numpy(float)
        raw = np.column_stack(
            [pd.to_numeric(merged[f"{name}::{target}"], errors="coerce") for name in names]
        )
        valid = np.isfinite(y_all) & np.isfinite(raw).all(axis=1)
        gold_column = f"{target}__gold"
        if gold_column in merged:
            valid &= pd.to_numeric(merged[gold_column], errors="coerce").fillna(0).to_numpy() > 0
        y = y_all[valid]
        prediction = raw[valid]
        if np.unique(y).size < 2:
            final_weights[target] = (np.ones(len(names)) / len(names)).tolist()
            full_scores[target] = float("nan")
            nested_scores[target] = float("nan")
            continue
        prediction = np.column_stack([rank01(prediction[:, member]) for member in range(len(names))])
        weight, score = best_weight(prediction, y, candidates)
        final_weights[target] = weight.tolist()
        full_scores[target] = score

        fold_values = merged.loc[valid, args.fold_column].to_numpy()
        held_out = np.full(len(y), np.nan)
        for fold in np.unique(fold_values):
            train = fold_values != fold
            test = ~train
            if np.unique(y[train]).size < 2:
                fold_weight = np.ones(len(names)) / len(names)
            else:
                fold_weight, _ = best_weight(prediction[train], y[train], candidates)
            held_out[test] = prediction[test] @ fold_weight
        nested_scores[target] = auc(y, held_out)
    finite_nested = [value for value in nested_scores.values() if np.isfinite(value)]
    artifact = {
        "schema_version": 1,
        "method": "target-wise convex rank blend",
        "members": names,
        "weights": final_weights,
        "oof_auc_refit": full_scores,
        "nested_oof_auc": nested_scores,
        "nested_macro_auc": float(np.mean(finite_nested)) if finite_nested else float("nan"),
        "studies": len(merged),
        "search_samples": args.samples,
        "seed": args.seed,
        "source_files": {name: paths for name, _, paths in loaded},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=True) + "\n")
    print(json.dumps({"output": str(args.output), "nested_macro_auc": artifact["nested_macro_auc"]}, indent=2))


if __name__ == "__main__":
    main()
