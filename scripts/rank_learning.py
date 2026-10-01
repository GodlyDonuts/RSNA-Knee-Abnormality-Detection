"""Paired component-bootstrap audit of completed rank32/rank64 development runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "initiative/audits/dinov3-adapter-a32-fold0-result-v1"
CHALLENGER = ROOT / "initiative/audits/dinov3-rank64-fold0-result-v1"
LABELS = ROOT / "initiative/releases/report-supervision-v1/study_labels.csv"


def auc_plan(truth, scores):
    """Cache score ties/order; later weights represent whole-component draws."""
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=float)
    if (
        truth.ndim != 1
        or scores.shape != truth.shape
        or not len(truth)
        or not np.isin(truth, (0, 1)).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("AUC requires aligned finite binary observations")
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    return truth.astype(float)[order], order, starts


def weighted_auc(plan, weights):
    truth, order, starts = plan
    weights = np.asarray(weights, dtype=float)
    if weights.shape != truth.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("AUC weights must be aligned, finite and nonnegative")
    ordered = weights[order]
    positives = np.add.reduceat(ordered * truth, starts)
    negatives = np.add.reduceat(ordered * (1 - truth), starts)
    total_positive, total_negative = positives.sum(), negatives.sum()
    if not total_positive or not total_negative:
        return None
    negatives_below = np.cumsum(negatives) - negatives
    return float(
        np.sum(positives * (negatives_below + 0.5 * negatives)) / (total_positive * total_negative)
    )


def component_weights(groups, rng):
    """Sample unique leakage groups once; every member receives its group count."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    count = int(inverse.max()) + 1
    draws = rng.integers(0, count, size=count)
    return np.bincount(draws, minlength=count)[inverse]


def interval(values, attempted):
    finite = [float(v) for v in values if v is not None]
    return {
        "defined_draws": len(finite),
        "undefined_draws": attempted - len(finite),
        "percentile_95": np.quantile(finite, (0.025, 0.975)).tolist() if finite else None,
    }


def inputs():
    # Replay all original source/result/prediction/label/membership bindings first.
    # Its existing immutable comparison file must match exactly; no new labels.
    original = runpy.run_path(str(CHALLENGER / "recompute.py"))
    original["main"]()
    comparison = json.loads((CHALLENGER / "comparison-v2.json").read_text())
    with (REFERENCE / "outer_predictions.csv").open(newline="") as stream:
        baseline = {row["study_uid"]: row for row in csv.DictReader(stream)}
    with (CHALLENGER / "outer_predictions.csv").open(newline="") as stream:
        challenger = {row["study_uid"]: row for row in csv.DictReader(stream)}
    uids = sorted(baseline)
    assert len(uids) == 1156 and baseline.keys() == challenger.keys()
    allowed = set(uids)
    with LABELS.open(newline="") as stream:
        labels = {r["study_uid"]: r for r in csv.DictReader(stream) if r["study_uid"] in allowed}
    assert labels.keys() == baseline.keys() and all(r["split"] == "train" for r in labels.values())
    groups = [labels[u]["leakage_group"] for u in uids]
    assert all(groups)
    plans = []
    for target, observed in zip(TARGETS, comparison["targets"], strict=True):
        assert observed["target"] == target
        mask = np.array([float(labels[u][target + "__mask"]) == 1 for u in uids])
        selected = [u for u, take in zip(uids, mask, strict=True) if take]
        truth = [int(float(labels[u][target])) for u in selected]
        pair = [
            auc_plan(truth, [float(model[u][target]) for u in selected])
            for model in (baseline, challenger)
        ]
        for plan, name in zip(pair, ("rank32", "rank64"), strict=True):
            assert abs(weighted_auc(plan, np.ones(len(selected))) - observed[name]) < 1e-10
        plans.append((mask, pair))
    return groups, plans, comparison


def audit(draws=4000, seed=20261001):
    if type(draws) is not int or draws < 100:
        raise ValueError("Use at least 100 paired component draws")
    groups, plans, comparison = inputs()
    rng = np.random.default_rng(seed)
    per_target = [[] for _ in TARGETS]
    macro = []
    for _ in range(draws):
        weights = component_weights(groups, rng)
        deltas = []
        for index, (mask, pair) in enumerate(plans):
            first, second = (weighted_auc(plan, weights[mask]) for plan in pair)
            delta = None if first is None or second is None else second - first
            per_target[index].append(delta)
            deltas.append(delta)
        # Retain all twelve target weights. Never repair missing classes by
        # dropping a target, stratifying on it or silently drawing again.
        macro.append(float(np.mean(deltas)) if all(x is not None for x in deltas) else None)
    return {
        "status": "paired_component_bootstrap_fixed_final_development_only",
        "draws": draws,
        "seed": seed,
        "studies": len(groups),
        "components": len(set(groups)),
        "comparison_sha256": hashlib.sha256(
            (CHALLENGER / "comparison-v2.json").read_bytes()
        ).hexdigest(),
        "labels_sha256": comparison["labels_sha256"],
        "rank32_predictions_sha256": comparison["rank32_predictions_sha256"],
        "rank64_predictions_sha256": comparison["rank64_predictions_sha256"],
        "rank32_macro_auc": comparison["rank32_macro_auc"],
        "rank64_macro_auc": comparison["rank64_macro_auc"],
        "rank64_minus_rank32": comparison["macro_delta"],
        "macro_uncertainty": interval(macro, draws),
        "targets": [
            {**row, "paired_delta_uncertainty": interval(values, draws)}
            for row, values in zip(comparison["targets"], per_target, strict=True)
        ],
        "method": "unstratified paired resampling of whole leakage components with replacement",
        "undefined_macro_policy": "record undefined whenever any of all twelve target AUCs is undefined",
        "interval_caveat": "Percentiles condition on defined draws and this one weak-label development pool; sparse classes and cross-seed/fold uncertainty are not resolved.",
        "causal_rank_isolation": False,
        "gold_or_confirmation_used": False,
        "kaggle_score": None,
        "training_parameters_or_masks_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path; preserve previous audits")
    result = audit(args.draws, args.seed)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {k: result[k] for k in ("components", "rank64_minus_rank32", "macro_uncertainty")}
        )
    )


if __name__ == "__main__":
    main()
