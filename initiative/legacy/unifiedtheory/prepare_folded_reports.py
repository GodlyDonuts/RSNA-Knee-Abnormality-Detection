#!/usr/bin/env python3
"""Attach audited scanner-group folds to the original report table.

The fold optimizer consumes preliminary targets, while the final report
teacher must see the original multilingual reports and expert annotations.
This small join keeps those two stages explicit and rejects partial or
duplicated study sets instead of silently dropping cases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def attach_folds(train: pd.DataFrame, folded: pd.DataFrame) -> pd.DataFrame:
    required_train = {"StudyInstanceUID", "Report"}
    required_folded = {"StudyInstanceUID", "fold"}
    if missing := required_train.difference(train.columns):
        raise ValueError(f"training table is missing {sorted(missing)}")
    if missing := required_folded.difference(folded.columns):
        raise ValueError(f"fold table is missing {sorted(missing)}")
    if train["StudyInstanceUID"].duplicated().any():
        raise ValueError("training table contains duplicate StudyInstanceUID values")
    if folded["StudyInstanceUID"].duplicated().any():
        raise ValueError("fold table contains duplicate StudyInstanceUID values")

    assignments = folded[["StudyInstanceUID", "fold"]].copy()
    result = train.merge(assignments, on="StudyInstanceUID", how="left", validate="one_to_one")
    if result["fold"].isna().any():
        missing = int(result["fold"].isna().sum())
        raise ValueError(f"{missing} training studies have no scanner-group fold")
    if len(result) != len(train) or len(assignments) != len(train):
        raise ValueError(
            "fold assignments and training reports must cover the same complete study set"
        )
    result["fold"] = result["fold"].astype(int)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--folds-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = attach_folds(pd.read_csv(args.train_csv), pd.read_csv(args.folds_csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(
        f"wrote {args.output}: {len(result)} studies, "
        f"{result['fold'].nunique()} scanner-grouped folds"
    )


if __name__ == "__main__":
    main()
