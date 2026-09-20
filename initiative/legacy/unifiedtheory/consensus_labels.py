#!/usr/bin/env python3
"""Build an equal-source report-label consensus with official gold overrides.

The consensus is intentionally fixed: every supplied public source receives
equal weight for every target.  No source or target weight is fitted on the 58
officially labelled studies.  Source disagreement controls only the loss
confidence; an official label always replaces the soft target at confidence 1.
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


UID = "StudyInstanceUID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--folds-csv", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        required=True,
        help="public report-label CSV; repeat once per fixed equal-weight source",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(path: Path, ids: pd.Index) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={UID: str})
    missing = {UID, *TARGETS}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    if frame[UID].duplicated().any():
        raise ValueError(f"{path} contains duplicate study identifiers")
    values = frame.set_index(UID)[TARGETS].apply(pd.to_numeric, errors="coerce")
    values = values.reindex(ids)
    finite = values.to_numpy(dtype=float)
    if np.isfinite(finite).any() and (
        np.nanmin(finite) < 0.0 or np.nanmax(finite) > 1.0
    ):
        raise ValueError(f"{path} contains labels outside [0, 1]")
    return values


def finite_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    valid = np.isfinite(labels) & np.isfinite(scores)
    if np.unique(labels[valid]).size < 2:
        return float("nan")
    return float(roc_auc_score(labels[valid], scores[valid]))


def build_consensus(
    train: pd.DataFrame, folds: pd.DataFrame, sources: list[pd.DataFrame]
) -> tuple[pd.DataFrame, dict]:
    if len(sources) < 2:
        raise ValueError("consensus requires at least two independent label sources")
    ids = pd.Index(train[UID].astype(str), name=UID)
    if ids.duplicated().any():
        raise ValueError("train.csv contains duplicate study identifiers")
    fold_frame = folds.copy()
    fold_frame[UID] = fold_frame[UID].astype(str)
    if fold_frame[UID].duplicated().any():
        raise ValueError("folds CSV contains duplicate study identifiers")
    if set(fold_frame[UID]) != set(ids):
        raise ValueError("folds CSV does not exactly cover train.csv")

    stacked = np.stack([source.loc[ids, TARGETS].to_numpy(float) for source in sources])
    valid = np.isfinite(stacked)
    count = valid.sum(axis=0)
    with np.errstate(invalid="ignore"):
        consensus = np.nanmean(stacked, axis=0)
        source_range = np.nanmax(stacked, axis=0) - np.nanmin(stacked, axis=0)
    consensus[count == 0] = np.nan
    source_range[count == 0] = 1.0
    coverage = count / len(sources)
    confidence = coverage * (0.5 + 0.5 * (1.0 - source_range))
    confidence[count == 0] = 0.0

    metadata = [
        column
        for column in fold_frame.columns
        if column != UID
        and column not in TARGETS
        and not any(column == f"{target}__{suffix}" for target in TARGETS for suffix in ("conf", "gold"))
    ]
    output = fold_frame.set_index(UID).loc[ids, metadata].reset_index()
    gold_auc: dict[str, dict[str, float]] = {"consensus": {}}
    for source_index in range(len(sources)):
        gold_auc[f"source_{source_index}"] = {}

    for target_index, target in enumerate(TARGETS):
        gold = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
        target_values = consensus[:, target_index].copy()
        target_confidence = confidence[:, target_index].copy()
        gold_mask = np.isfinite(gold)
        target_values[gold_mask] = gold[gold_mask]
        target_confidence[gold_mask] = 1.0
        output[target] = target_values
        output[f"{target}__conf"] = target_confidence
        output[f"{target}__gold"] = gold_mask.astype(int)
        gold_auc["consensus"][target] = finite_auc(gold, consensus[:, target_index])
        for source_index, source in enumerate(sources):
            gold_auc[f"source_{source_index}"][target] = finite_auc(
                gold, source.loc[ids, target].to_numpy(float)
            )

    audit = {
        "contract": "unfitted equal arithmetic mean across every source and target; official labels override at confidence one",
        "studies": len(output),
        "sources": len(sources),
        "gold_studies": int(
            pd.to_numeric(train[TARGETS[0]], errors="coerce").notna().sum()
        ),
        "missing_consensus_cells": int(np.isnan(consensus).sum()),
        "mean_training_confidence_before_gold_override": float(np.nanmean(confidence)),
        "gold_auc": gold_auc,
        "gold_macro_auc": {
            name: float(np.nanmean(list(values.values())))
            for name, values in gold_auc.items()
        },
    }
    return output, audit


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_csv, dtype={UID: str})
    folds = pd.read_csv(args.folds_csv, dtype={UID: str})
    ids = pd.Index(train[UID].astype(str), name=UID)
    sources = [load_source(path, ids) for path in args.source]
    output, audit = build_consensus(train, folds, sources)
    audit["source_files"] = [
        {"path": str(path), "sha256": sha256(path)} for path in args.source
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    audit_path = args.audit_json or args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **audit["gold_macro_auc"]}, indent=2))


if __name__ == "__main__":
    main()
