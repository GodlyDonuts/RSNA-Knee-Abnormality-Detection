#!/usr/bin/env python3
"""CPU-only checks for the competition-compliant orchestration layer."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .constants import CACHE_SCHEMA_VERSION, TARGETS
    from .data import FeatureStudyDataset, load_feature_cache
    from .folds import gold_fold_quality, select_grouped_multilabel_folds
    from .kaggle_offline_infer import validate_submission
    from .prepare_folded_reports import attach_folds
except ImportError:
    from constants import CACHE_SCHEMA_VERSION, TARGETS
    from data import FeatureStudyDataset, load_feature_cache
    from folds import gold_fold_quality, select_grouped_multilabel_folds
    from kaggle_offline_infer import validate_submission
    from prepare_folded_reports import attach_folds


def synthetic_fold_frame() -> pd.DataFrame:
    rows = []
    for group in range(20):
        for member in range(2):
            row = {
                "StudyInstanceUID": f"study-{group}-{member}",
                "scanner_group": f"scanner-{group}",
            }
            for target_index, target in enumerate(TARGETS):
                row[target] = float((group + member + target_index) % 2)
                # Sparse expert labels deliberately exercise the gold-aware
                # objective without using any model prediction.
                row[f"{target}__gold"] = int((group + target_index) % 5 == 0)
            rows.append(row)
    return pd.DataFrame(rows)


def cache_payload() -> dict[str, torch.Tensor]:
    return {
        "schema_version": torch.tensor(CACHE_SCHEMA_VERSION),
        "features": torch.randn(2, 3, 8).half(),
        "patch_features": torch.randn(2, 3, 4, 4).half(),
        "patch_mask": torch.ones(2, 3, 4, dtype=torch.bool),
        "positions_mm": torch.arange(3).float()[None, :].expand(2, -1),
        "slice_mask": torch.ones(2, 3, dtype=torch.bool),
        "series_mask": torch.ones(2, dtype=torch.bool),
        "plane": torch.tensor([1, 2]),
        "fluid": torch.tensor([1, 2]),
        "fatsat": torch.tensor([2, 1]),
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    frame = synthetic_fold_frame()
    folds, seed, quality = select_grouped_multilabel_folds(
        frame, "scanner_group", n_folds=5, seed=2026, candidate_seeds=8
    )
    repeated, repeated_seed, repeated_quality = select_grouped_multilabel_folds(
        frame, "scanner_group", n_folds=5, seed=2026, candidate_seeds=8
    )
    if not np.array_equal(folds, repeated) or (seed, quality) != (
        repeated_seed,
        repeated_quality,
    ):
        raise AssertionError("fold search is not deterministic")
    assignment = frame.assign(fold=folds).groupby("scanner_group")["fold"].nunique()
    if int(assignment.max()) != 1:
        raise AssertionError("a scanner group crossed validation folds")
    if gold_fold_quality(frame, folds) != quality:
        raise AssertionError("reported fold quality does not match the selected folds")

    reports = frame[["StudyInstanceUID"]].assign(Report="synthetic report")
    folded = frame[["StudyInstanceUID"]].assign(fold=folds)
    attached = attach_folds(reports, folded)
    if attached["fold"].isna().any() or len(attached) != len(reports):
        raise AssertionError("fold/report join lost studies")
    try:
        attach_folds(reports, folded.iloc[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("partial fold coverage was silently accepted")

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        combined = root / "combined"
        summary = root / "summary"
        combined.mkdir()
        source_cache = combined / "study.pt"
        torch.save(cache_payload(), source_cache)
        index = pd.DataFrame(
            [
                {
                    "StudyInstanceUID": "study",
                    "cache_file": str(source_cache),
                    "scanner_group": "scanner",
                    "patch_dim": 4,
                    "patches_per_slice": 4,
                }
            ]
        )
        index_path = combined / "train_cache_index.csv"
        index.to_csv(index_path, index=False)

        dataset = FeatureStudyDataset(index, include_patch_features=False)
        item = dataset[0]
        if "patch_features" in item or "patch_mask" in item:
            raise AssertionError("summary dataset retained optional patch tensors")

        subprocess.run(
            [
                sys.executable,
                str(here / "derive_summary_cache.py"),
                "--cache-index",
                str(index_path),
                "--output",
                str(summary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        derived_index = pd.read_csv(summary / "train_cache_index.csv")
        derived = load_feature_cache(derived_index.iloc[0]["cache_file"])
        if "patch_features" in derived or int(derived_index.iloc[0]["patch_dim"]) != 0:
            raise AssertionError("derived summary cache still contains patch-grid data")

        sample = pd.DataFrame({"StudyInstanceUID": ["a", "b"]})
        for target in TARGETS:
            sample[target] = 0.5
        submission = sample.copy()
        submission[TARGETS] = 0.25
        sample_path = root / "sample_submission.csv"
        submission_path = root / "submission.csv"
        sample.to_csv(sample_path, index=False)
        submission.to_csv(submission_path, index=False)
        validate_submission(sample_path, submission_path)
        reversed_path = root / "reversed.csv"
        submission.iloc[::-1].to_csv(reversed_path, index=False)
        try:
            validate_submission(sample_path, reversed_path)
        except ValueError:
            pass
        else:
            raise AssertionError("submission UID order mismatch was accepted")

    print(
        json.dumps(
            {
                "status": "pass",
                "fold_seed": seed,
                "gold_auc_cells": quality[0],
                "scanner_groups": int(frame["scanner_group"].nunique()),
                "summary_cache": "patch tensors removed",
                "submission_contract": "exact schema, identifiers, and probabilities",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
