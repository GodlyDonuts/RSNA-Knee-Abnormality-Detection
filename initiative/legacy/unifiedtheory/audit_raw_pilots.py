#!/usr/bin/env python3
"""Apply fixed promotion gates to held-out raw-image pilot predictions."""

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


def parse_member(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("member must be NAME=OOF_CSV")
    name, path = value.split("=", 1)
    return name, Path(path)


def rank01(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def score_target(y: pd.Series, prediction: pd.Series) -> float:
    valid = y.notna() & prediction.notna()
    return (
        float(roc_auc_score(y[valid], prediction[valid]))
        if y[valid].nunique() == 2
        else float("nan")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--baseline", required=True, help="NAME=OOF_CSV")
    parser.add_argument("--candidate", action="append", required=True, help="NAME=OOF_CSV")
    parser.add_argument("--minimum-candidate-gain", type=float, default=0.02)
    parser.add_argument("--minimum-blend-gain", type=float, default=0.015)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if missing := {"StudyInstanceUID", *TARGETS}.difference(frame.columns):
        raise ValueError(f"{name} is missing {sorted(missing)}")
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{name} contains duplicate studies")
    return frame[["StudyInstanceUID", *TARGETS]]


def main() -> None:
    args = parse_args()
    baseline_name, baseline_path = parse_member(args.baseline)
    candidate_specs = [parse_member(value) for value in args.candidate]
    labels = pd.read_csv(args.labels_csv, dtype={"StudyInstanceUID": str})
    held_out = labels[labels["fold"] == args.fold].copy()
    baseline = load(baseline_path, baseline_name)
    joined = held_out.merge(
        baseline.rename(columns={target: f"{baseline_name}::{target}" for target in TARGETS}),
        on="StudyInstanceUID",
        how="left",
        validate="one_to_one",
    )
    rows = []
    promoted = []
    for name, path in candidate_specs:
        if not path.is_file():
            rows.append({"candidate": name, "status": "missing", "path": str(path)})
            continue
        candidate = load(path, name)
        data = joined.merge(
            candidate.rename(columns={target: f"{name}::{target}" for target in TARGETS}),
            on="StudyInstanceUID",
            how="left",
            validate="one_to_one",
        )
        target_rows = []
        for target in TARGETS:
            gold = pd.to_numeric(data[f"{target}__gold"], errors="coerce").fillna(0) > 0
            y = pd.to_numeric(data[target], errors="coerce").where(gold)
            baseline_prediction = pd.to_numeric(data[f"{baseline_name}::{target}"], errors="coerce")
            candidate_prediction = pd.to_numeric(data[f"{name}::{target}"], errors="coerce")
            baseline_auc = score_target(y, baseline_prediction)
            candidate_auc = score_target(y, candidate_prediction)
            blend_auc = score_target(
                y, (rank01(baseline_prediction) + rank01(candidate_prediction)) / 2
            )
            target_rows.append(
                {
                    "target": target,
                    "baseline_auc": baseline_auc,
                    "candidate_auc": candidate_auc,
                    "candidate_gain": candidate_auc - baseline_auc,
                    "fixed_blend_auc": blend_auc,
                    "fixed_blend_gain": blend_auc - baseline_auc,
                }
            )
        cells = pd.DataFrame(target_rows)
        baseline_macro = float(cells["baseline_auc"].mean())
        candidate_macro = float(cells["candidate_auc"].mean())
        blend_macro = float(cells["fixed_blend_auc"].mean())
        candidate_gain = candidate_macro - baseline_macro
        blend_gain = blend_macro - baseline_macro
        promote = bool(
            candidate_gain >= args.minimum_candidate_gain
            or blend_gain >= args.minimum_blend_gain
        )
        if promote:
            promoted.append(name)
        rows.append(
            {
                "candidate": name,
                "status": "evaluated",
                "path": str(path),
                "baseline_macro_auc": baseline_macro,
                "candidate_macro_auc": candidate_macro,
                "candidate_gain": candidate_gain,
                "fixed_blend_macro_auc": blend_macro,
                "fixed_blend_gain": blend_gain,
                "improved_targets": int((cells["candidate_gain"] > 0).sum()),
                "promote": promote,
                "target_results": target_rows,
            }
        )
    artifact = {
        "schema_version": 1,
        "fold": args.fold,
        "baseline": baseline_name,
        "minimum_candidate_gain": args.minimum_candidate_gain,
        "minimum_blend_gain": args.minimum_blend_gain,
        "promoted": promoted,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=True) + "\n")
    print(json.dumps(artifact, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
