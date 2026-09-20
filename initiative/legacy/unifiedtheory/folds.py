#!/usr/bin/env python3
"""Create deterministic scanner-grouped, multilabel-balanced fold assignments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


def grouped_multilabel_folds(
    frame: pd.DataFrame,
    group_column: str,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Greedily allocate whole scanner groups while balancing label mass.

    Missing labels contribute neither positive nor negative mass.  The
    objective balances positives, observed examples, and study count.  This is
    deliberately deterministic after the seeded group tie-break.
    """

    if group_column not in frame:
        raise ValueError(f"missing group column {group_column!r}")
    if frame[group_column].isna().any():
        raise ValueError("scanner groups must not be missing")
    if frame[group_column].nunique() < n_folds:
        raise ValueError("fewer scanner groups than folds")
    labels = frame.reindex(columns=TARGETS).apply(pd.to_numeric, errors="coerce")
    gold = pd.DataFrame(
        {
            target: pd.to_numeric(
                frame.get(f"{target}__gold", labels[target].notna().astype(int)),
                errors="coerce",
            )
            .fillna(0)
            .astype(bool)
            for target in TARGETS
        },
        index=frame.index,
    )
    groups = frame[group_column].astype(str)
    unique = groups.unique().tolist()
    rng = np.random.default_rng(seed)
    tie = {group: float(rng.random()) for group in unique}

    records = []
    for group in unique:
        rows = groups == group
        y = labels.loc[rows]
        is_gold = gold.loc[rows].to_numpy(dtype=bool)
        values = y.to_numpy(dtype=float)
        records.append(
            {
                "group": group,
                "count": int(rows.sum()),
                "positive": np.nansum(values, axis=0),
                "observed": np.isfinite(values).sum(axis=0),
                "gold_positive": (is_gold & (values > 0.5)).sum(axis=0),
                "gold_negative": (is_gold & (values <= 0.5)).sum(axis=0),
                "rarity": float(
                    np.nansum(values / np.maximum(labels.sum(axis=0).to_numpy(), 1))
                    + 4.0 * is_gold.sum()
                ),
            }
        )
    records.sort(key=lambda row: (-row["rarity"], -row["count"], tie[row["group"]]))

    global_positive = sum((row["positive"] for row in records), np.zeros(len(TARGETS)))
    global_observed = sum((row["observed"] for row in records), np.zeros(len(TARGETS)))
    target_positive = global_positive / n_folds
    target_observed = global_observed / n_folds
    global_gold_positive = sum(
        (row["gold_positive"] for row in records), np.zeros(len(TARGETS))
    )
    global_gold_negative = sum(
        (row["gold_negative"] for row in records), np.zeros(len(TARGETS))
    )
    target_gold_positive = global_gold_positive / n_folds
    target_gold_negative = global_gold_negative / n_folds
    target_count = len(frame) / n_folds
    fold_positive = np.zeros((n_folds, len(TARGETS)), dtype=float)
    fold_observed = np.zeros((n_folds, len(TARGETS)), dtype=float)
    fold_gold_positive = np.zeros((n_folds, len(TARGETS)), dtype=float)
    fold_gold_negative = np.zeros((n_folds, len(TARGETS)), dtype=float)
    fold_count = np.zeros(n_folds, dtype=float)
    allocation: dict[str, int] = {}

    for order, record in enumerate(records):
        # Seed every fold before optimizing, preventing an early zero-fold
        # degeneracy when group label vectors are similar.
        candidates = [order] if order < n_folds else list(range(n_folds))
        scores = []
        for fold in candidates:
            candidate_positive = fold_positive.copy()
            candidate_observed = fold_observed.copy()
            candidate_gold_positive = fold_gold_positive.copy()
            candidate_gold_negative = fold_gold_negative.copy()
            candidate_count = fold_count.copy()
            candidate_positive[fold] += record["positive"]
            candidate_observed[fold] += record["observed"]
            candidate_gold_positive[fold] += record["gold_positive"]
            candidate_gold_negative[fold] += record["gold_negative"]
            candidate_count[fold] += record["count"]
            positive_error = np.mean(
                ((candidate_positive - target_positive) / np.maximum(target_positive, 1)) ** 2
            )
            observed_error = np.mean(
                ((candidate_observed - target_observed) / np.maximum(target_observed, 1)) ** 2
            )
            count_error = np.mean(((candidate_count - target_count) / max(target_count, 1)) ** 2)
            gold_positive_error = np.mean(
                (
                    (candidate_gold_positive - target_gold_positive)
                    / np.maximum(target_gold_positive, 1)
                )
                ** 2
            )
            gold_negative_error = np.mean(
                (
                    (candidate_gold_negative - target_gold_negative)
                    / np.maximum(target_gold_negative, 1)
                )
                ** 2
            )
            scores.append(
                (
                    positive_error
                    + 0.25 * observed_error
                    + 0.5 * count_error
                    + 4.0 * (gold_positive_error + gold_negative_error),
                    fold,
                )
            )
        _, chosen = min(scores, key=lambda pair: (pair[0], pair[1]))
        allocation[record["group"]] = chosen
        fold_positive[chosen] += record["positive"]
        fold_observed[chosen] += record["observed"]
        fold_gold_positive[chosen] += record["gold_positive"]
        fold_gold_negative[chosen] += record["gold_negative"]
        fold_count[chosen] += record["count"]
    return groups.map(allocation).to_numpy(dtype=np.int64)


def gold_fold_quality(frame: pd.DataFrame, folds: np.ndarray) -> tuple[int, int, float]:
    """Score how many fold/target AUCs are defined on expert labels.

    The lexicographic score rewards, in order: fold/target cells containing
    both expert classes, total expert minority-class support, and balanced fold
    sizes.  It never uses validation predictions and therefore cannot tune on
    the competition metric.
    """

    valid_auc_cells = 0
    minority_support = 0
    for fold in np.unique(folds):
        selected = frame.iloc[np.flatnonzero(folds == fold)]
        for target in TARGETS:
            values = pd.to_numeric(selected[target], errors="coerce")
            marker = pd.to_numeric(
                selected.get(f"{target}__gold", values.notna().astype(int)),
                errors="coerce",
            ).fillna(0).astype(bool)
            expert = values[marker & values.notna()]
            positive = int((expert > 0.5).sum())
            negative = int((expert <= 0.5).sum())
            if positive and negative:
                valid_auc_cells += 1
            minority_support += min(positive, negative)
    counts = np.bincount(folds, minlength=int(folds.max()) + 1).astype(float)
    return valid_auc_cells, minority_support, -float(np.std(counts))


def select_grouped_multilabel_folds(
    frame: pd.DataFrame,
    group_column: str,
    n_folds: int,
    seed: int,
    candidate_seeds: int,
) -> tuple[np.ndarray, int, tuple[int, int, float]]:
    """Choose the deterministic grouped allocation with best gold coverage."""

    if candidate_seeds < 1:
        raise ValueError("candidate_seeds must be positive")
    best: tuple[tuple[int, int, float], int, np.ndarray] | None = None
    for candidate in range(seed, seed + candidate_seeds):
        folds = grouped_multilabel_folds(frame, group_column, n_folds, candidate)
        quality = gold_fold_quality(frame, folds)
        record = (quality, -candidate, folds)
        if best is None or record[:2] > best[:2]:
            best = record
    assert best is not None
    quality, negative_seed, folds = best
    return folds, -negative_seed, quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-column", default="scanner_group")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--candidate-seeds",
        type=int,
        default=64,
        help="grouped allocations to audit for expert-label AUC coverage",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = pd.read_csv(args.cache_index)
    labels = pd.read_csv(args.labels_csv)
    frame = cache.merge(labels, on="StudyInstanceUID", how="inner", validate="one_to_one")
    frame["fold"], selected_seed, quality = select_grouped_multilabel_folds(
        frame,
        args.group_column,
        args.folds,
        args.seed,
        args.candidate_seeds,
    )
    output_columns = ["StudyInstanceUID", "fold"]
    labels = labels.merge(frame[output_columns], on="StudyInstanceUID", how="left", validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output, index=False)
    summary = frame.groupby("fold").agg(
        studies=("StudyInstanceUID", "size"),
        scanner_groups=(args.group_column, "nunique"),
    )
    print(summary.to_string())
    print(
        "selected_seed="
        f"{selected_seed}; gold_auc_cells={quality[0]}/{args.folds * len(TARGETS)}; "
        f"gold_minority_support={quality[1]}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
