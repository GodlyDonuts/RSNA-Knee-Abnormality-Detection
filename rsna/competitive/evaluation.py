"""Seed-separated OOF evaluation with within-fold pair weighting and paired bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna.competitive.folds import _digest, load_folds, training_rows, validate_overlay
from rsna.release import load_verified_release
from rsna.schema import TARGETS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Competitive evaluation: " + message)


def _aligned(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    seed: int,
    membership: str,
) -> dict:
    _require(type(seed) is int and 0 <= seed < 2**32, "invalid model seed")
    _require(membership in {"all_train", "observed_train"}, "explicit membership contract required")
    overlay = validate_overlay(manifest, folds)
    train = training_rows(manifest)
    if membership == "observed_train":
        train = train.loc[train[[t + "__mask" for t in TARGETS]].eq(1).any(axis=1)]
    train = train.sort_values("study_uid")
    required = {"study_uid", "fold", "seed", *TARGETS}
    _require(predictions.columns.is_unique and required <= set(predictions), "prediction columns")
    _require(not predictions.study_uid.duplicated().any(), "duplicate prediction UID")
    _require(
        set(predictions.study_uid) == set(train.study_uid),
        "predictions require exact selected OOF membership",
    )
    pred = predictions.set_index("study_uid").loc[train.study_uid]
    expected = overlay.set_index("study_uid").loc[train.study_uid]
    for column in ("fold", "seed"):
        _require(
            bool(
                pred[column]
                .map(lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, bool))
                .all()
            ),
            "integer fold and seed required",
        )
    _require(bool(pred.seed.eq(seed).all()), "mixed or unexpected model seeds")
    _require(bool(pred.fold.eq(expected.fold).all()), "prediction fold differs from frozen overlay")
    if "leakage_group" in pred:
        _require(
            bool(pred.leakage_group.eq(expected.leakage_group).all()), "prediction group changed"
        )
    try:
        scores = pred[list(TARGETS)].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Competitive evaluation: nonnumeric probabilities") from error
    _require(
        bool(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()),
        "invalid probabilities",
    )
    masks = train[[t + "__mask" for t in TARGETS]].to_numpy(dtype=bool)
    labels = np.zeros(masks.shape, dtype=np.int64)
    for i, target in enumerate(TARGETS):
        labels[masks[:, i], i] = train.loc[masks[:, i], target].to_numpy(dtype=np.int64)
    return {
        "uids": train.study_uid.to_numpy(),
        "groups": train.leakage_group.to_numpy(),
        "folds": expected.fold.to_numpy(dtype=np.int64),
        "n_folds": int(overlay.fold.nunique()),
        "labels": labels,
        "masks": masks,
        "scores": scores,
    }


def _auc(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> tuple[float | None, int, int]:
    positive = int(weights[labels == 1].sum())
    negative = int(weights[labels == 0].sum())
    auc = (
        float(roc_auc_score(labels, scores, sample_weight=weights))
        if positive and negative
        else None
    )
    return auc, positive, negative


def _metrics(data: dict, weights: np.ndarray | None = None, *, pooled: bool = True) -> dict:
    if weights is None:
        weights = np.ones(len(data["uids"]), dtype=np.int64)
    per_fold = []
    weighted: dict[str, float | None] = {}
    pooled_auc: dict[str, float | None] = {}
    pair_totals: dict[str, int] = {}
    for j, target in enumerate(TARGETS):
        numerator, denominator = 0.0, 0
        for fold in range(data["n_folds"]):
            keep = data["masks"][:, j] & (data["folds"] == fold) & (weights > 0)
            auc, positive, negative = _auc(
                data["labels"][keep, j], data["scores"][keep, j], weights[keep]
            )
            pairs = positive * negative
            per_fold.append(
                {
                    "fold": fold,
                    "target": target,
                    "auc": auc,
                    "positive": positive,
                    "negative": negative,
                    "pairs": pairs,
                }
            )
            if auc is not None:
                numerator += auc * pairs
                denominator += pairs
        weighted[target] = numerator / denominator if denominator else None
        pair_totals[target] = denominator
        if pooled:
            keep = data["masks"][:, j] & (weights > 0)
            pooled_auc[target] = _auc(
                data["labels"][keep, j], data["scores"][keep, j], weights[keep]
            )[0]
    full_macro = (
        float(np.mean([x for x in weighted.values() if x is not None]))
        if all(x is not None for x in weighted.values())
        else None
    )
    result = {
        "within_fold": per_fold,
        "pair_weighted_target_auc": weighted,
        "within_fold_pair_totals": pair_totals,
        "full_12_target_macro_auc": full_macro,
    }
    if pooled:
        result["pooled_oof_secondary"] = {
            "target_auc": pooled_auc,
            "full_12_target_macro_auc": float(
                np.mean([x for x in pooled_auc.values() if x is not None])
            )
            if all(x is not None for x in pooled_auc.values())
            else None,
            "limitation": "Separately trained fold score scales can distort cross-fold rankings; never use held-out labels to normalize them.",
        }
    return result


def evaluate_oof(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    seed: int,
    membership: str = "all_train",
) -> dict:
    """One model seed, exact complete OOF UIDs; diagnostic subsets are rejected.

    Callers must load the full verified release and validated frozen fold overlay
    first. This function never selects a checkpoint or reads nontrain labels.
    Undefined folds have zero positive-negative pairs, not an invented AUC of .5.
    """
    data = _aligned(manifest, folds, predictions, seed, membership)
    return {
        "schema_version": 1,
        "seed": seed,
        "membership": membership,
        "n_studies": len(data["uids"]),
        "n_components": len(set(data["groups"])),
        "prediction_sha256": _digest(
            [
                [str(uid), int(fold), seed, *[float(x) for x in scores]]
                for uid, fold, scores in zip(data["uids"], data["folds"], data["scores"])
            ]
        ),
        "target_order": list(TARGETS),
        "primary_rule": "Within each target, weight fold AUC by observed positive-negative study pairs; average all12 targets only if all defined.",
        **_metrics(data),
    }


def paired_group_bootstrap(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    *,
    seed: int,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 20260922,
    membership: str = "all_train",
    confidence: float = 0.95,
) -> dict:
    """Paired A-minus-B percentile intervals, resampling whole groups within folds.

    Group multiplicities become study weights for both models identically. Folds
    and model seeds never mix. Undefined draws are counted and omitted rather than
    assigned .5; intervals are conditional on the draws where the metric exists.
    """
    _require(type(n_bootstrap) is int and n_bootstrap > 0, "positive bootstrap count required")
    _require(type(bootstrap_seed) is int and 0 <= bootstrap_seed < 2**32, "invalid bootstrap seed")
    _require(0 < confidence < 1, "invalid confidence")
    a = _aligned(manifest, folds, predictions_a, seed, membership)
    b = _aligned(manifest, folds, predictions_b, seed, membership)
    _require(np.array_equal(a["uids"], b["uids"]), "paired membership mismatch")
    rng = np.random.default_rng(bootstrap_seed)
    group_codes = []
    for fold in range(a["n_folds"]):
        row_indices = np.flatnonzero(a["folds"] == fold)
        groups, codes = np.unique(a["groups"][row_indices], return_inverse=True)
        group_codes.append((row_indices, len(groups), codes))
    values: dict[str, list[float]] = {t: [] for t in TARGETS}
    macros = []
    draw_hash = hashlib.sha256()
    for _ in range(n_bootstrap):
        weights = np.zeros(len(a["uids"]), dtype=np.int64)
        for row_indices, n_groups, codes in group_codes:
            if n_groups:
                sampled = np.bincount(rng.integers(0, n_groups, size=n_groups), minlength=n_groups)
                weights[row_indices] = sampled[codes]
        draw_hash.update(weights.astype("<i8").tobytes())
        ma, mb = _metrics(a, weights, pooled=False), _metrics(b, weights, pooled=False)
        for target in TARGETS:
            va, vb = ma["pair_weighted_target_auc"][target], mb["pair_weighted_target_auc"][target]
            if va is not None and vb is not None:
                values[target].append(va - vb)
        va, vb = ma["full_12_target_macro_auc"], mb["full_12_target_macro_auc"]
        if va is not None and vb is not None:
            macros.append(va - vb)

    def interval(samples: list[float]) -> dict:
        bounds = (
            np.quantile(samples, [(1 - confidence) / 2, (1 + confidence) / 2]).tolist()
            if samples
            else None
        )
        return {
            "interval": bounds,
            "defined_resamples": len(samples),
            "undefined_resamples": n_bootstrap - len(samples),
        }

    point_a, point_b = _metrics(a), _metrics(b)
    return {
        "schema_version": 1,
        "seed": seed,
        "bootstrap_seed": bootstrap_seed,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "membership": membership,
        "comparison": "a_minus_b",
        "paired_group_draws_sha256": draw_hash.hexdigest(),
        "target_differences": {t: interval(values[t]) for t in TARGETS},
        "full_12_target_macro_difference": interval(macros),
        "point_a": point_a,
        "point_b": point_b,
        "scope": "Conditional resampling uncertainty for these fitted models and this one seed; not model-search or training-seed uncertainty. Intervals omit explicitly counted undefined draws.",
    }


def leave_one_positive_component_out(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    seed: int,
    targets: tuple[str, ...] = ("Fracture", "MCL", "Baker's"),
    membership: str = "all_train",
) -> dict:
    """Target-specific fragility diagnostic, not extra confirmation or a refit."""
    _require(
        bool(targets) and len(set(targets)) == len(targets) and set(targets) <= set(TARGETS),
        "invalid sensitivity targets",
    )
    data = _aligned(manifest, folds, predictions, seed, membership)
    results = []
    for target in targets:
        j = TARGETS.index(target)
        groups = sorted(set(data["groups"][data["masks"][:, j] & (data["labels"][:, j] == 1)]))
        for group in groups:
            weights = (data["groups"] != group).astype(np.int64)
            metrics = _metrics(data, weights, pooled=False)
            results.append(
                {
                    "target": target,
                    "excluded_component": group,
                    "auc": metrics["pair_weighted_target_auc"][target],
                    "pairs": metrics["within_fold_pair_totals"][target],
                }
            )
    return {
        "seed": seed,
        "membership": membership,
        "scope": "Remove one observed-positive leakage component from evaluation, retain model parameters unchanged.",
        "results": results,
    }


def main() -> None:
    """Combine only explicitly named OOF files; no checkpoint/model selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--membership", choices=["observed_train", "all_train"], default="observed_train"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Validation precedes any training selection or prediction evaluation.
    manifest, binding = load_verified_release(args.studies, args.artifact_root)
    directory = args.folds if args.folds.is_dir() else args.folds.parent
    fold_bytes = (directory / "fold.csv").read_bytes()
    fold_metadata_bytes = (directory / "folds.json").read_bytes()
    overlay = load_folds(args.folds, manifest, binding)
    resolved = [path.resolve() for path in args.predictions]
    _require(len(set(resolved)) == len(resolved), "duplicate prediction input path")
    tables, inputs = [], []
    for index, path in enumerate(args.predictions):
        raw = path.read_bytes()
        table = pd.read_csv(io.BytesIO(raw), dtype={"study_uid": str})
        tables.append(table)
        # Content identities are portable; never archive machine-specific absolute paths.
        inputs.append(
            {
                "input_index": index,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "rows": len(table),
            }
        )
    predictions = pd.concat(tables, ignore_index=True)
    metrics = evaluate_oof(
        manifest, overlay, predictions, seed=args.seed, membership=args.membership
    )
    canonical = predictions.sort_values("study_uid")[["study_uid", "fold", "seed", *TARGETS]]
    prediction_bytes = canonical.to_csv(index=False, lineterminator="\n").encode()
    metrics_bytes = (json.dumps(metrics, indent=2, allow_nan=False) + "\n").encode()
    _require(
        (directory / "fold.csv").read_bytes() == fold_bytes
        and (directory / "folds.json").read_bytes() == fold_metadata_bytes,
        "fold artifacts changed during evaluation",
    )
    import sklearn

    record = {
        "schema_version": 1,
        "status": "complete_training_oof_evaluation",
        "seed": args.seed,
        "membership": args.membership,
        "selection_performed": False,
        "reserved_confirmation_or_gold_evaluated": False,
        "release_binding": binding,
        "input_prediction_files": inputs,
        "fold_csv_sha256": hashlib.sha256(fold_bytes).hexdigest(),
        "fold_metadata_sha256": hashlib.sha256(fold_metadata_bytes).hexdigest(),
        "prediction_table_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("evaluation.py", "folds.py")
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "protocol": "Exact UID OOF coverage for one seed; primary within-fold P*N weighting; pooled secondary; no outer-fold checkpoint selection.",
    }
    outputs = {
        "oof_predictions.csv": prediction_bytes,
        "metrics.json": metrics_bytes,
        "evaluation.json": (json.dumps(record, indent=2, allow_nan=False) + "\n").encode(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _require(
        all(not (args.output / name).exists() for name in outputs),
        "evaluation output already exists",
    )
    for name, raw in outputs.items():
        with (args.output / name).open("xb") as stream:
            stream.write(raw)
    print(
        json.dumps(
            {
                "status": record["status"],
                "seed": args.seed,
                "n_studies": metrics["n_studies"],
                "full_12_target_macro_auc": metrics["full_12_target_macro_auc"],
            }
        )
    )


if __name__ == "__main__":
    main()
