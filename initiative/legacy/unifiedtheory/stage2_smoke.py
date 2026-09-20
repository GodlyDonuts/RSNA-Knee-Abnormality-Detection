#!/usr/bin/env python3
"""Validate the report teacher, grouped folds, and patch hierarchy."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from .constants import TARGETS
    from .folds import grouped_multilabel_folds
    from .patch_model import PatchKneeAlibiModel, PatchKneeModelConfig
    from .report_teacher import build_soft_labels, rule_probability
except ImportError:
    from constants import TARGETS
    from folds import grouped_multilabel_folds
    from patch_model import PatchKneeAlibiModel, PatchKneeModelConfig
    from report_teacher import build_soft_labels, rule_probability


def make_patch_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    b, r, s, p = 4, 3, 5, 4
    return {
        "features": torch.randn(b, r, s, 12, generator=generator),
        "patch_features": torch.randn(b, r, s, p, 6, generator=generator),
        "patch_mask": torch.ones(b, r, s, p, dtype=torch.bool),
        "positions_mm": torch.tensor([0.0, 1.2, 2.4, 4.8, 6.0])[None, None, :]
        .expand(b, r, -1)
        .clone(),
        "slice_mask": torch.ones(b, r, s, dtype=torch.bool),
        "series_mask": torch.ones(b, r, dtype=torch.bool),
        "plane": torch.tensor([[1, 2, 3]]).expand(b, -1).clone(),
        "fluid": torch.tensor([[2, 2, 1]]).expand(b, -1).clone(),
        "fatsat": torch.tensor([[2, 1, 1]]).expand(b, -1).clone(),
    }


def main() -> None:
    positive, _, _ = rule_probability("complete acl tear with joint effusion", "ACL")
    negative, _, _ = rule_probability("no evidence of acl tear", "ACL")
    uncertain, _, _ = rule_probability("possible acl injury", "ACL")
    if not positive > uncertain > negative:
        raise AssertionError("report rule polarity is wrong")

    rows = []
    for index in range(30):
        row = {
            "StudyInstanceUID": f"report-{index}",
            "Report": "ACL tear and joint effusion" if index % 2 else "No ACL tear or effusion",
            "fold": index % 3,
        }
        for target_index, target in enumerate(TARGETS):
            row[target] = (index + target_index) % 2 if index < 12 else np.nan
        rows.append(row)
    report_frame = pd.DataFrame(rows)
    soft = build_soft_labels(report_frame, nli=None, fold_column="fold", nli_weight=0.45)
    if soft[TARGETS].isna().any().any():
        raise AssertionError("report teacher emitted missing probabilities")
    if not np.array_equal(
        soft.loc[:11, TARGETS].to_numpy(), report_frame.loc[:11, TARGETS].to_numpy()
    ):
        raise AssertionError("gold labels did not override report targets")

    fold_frame = soft.copy()
    fold_frame["scanner_group"] = [f"scanner-{i // 3}" for i in range(len(soft))]
    folds = grouped_multilabel_folds(fold_frame, "scanner_group", 3, seed=2026)
    for group in fold_frame["scanner_group"].unique():
        if np.unique(folds[fold_frame["scanner_group"].to_numpy() == group]).size != 1:
            raise AssertionError("one scanner group was split across folds")
    if set(folds.tolist()) != {0, 1, 2}:
        raise AssertionError("not every fold received a scanner group")

    torch.manual_seed(11)
    config = PatchKneeModelConfig(
        feature_dim=12,
        patch_dim=6,
        hidden_dim=16,
        n_heads=4,
        series_depth=1,
        study_depth=1,
        dropout=0.0,
        num_targets=3,
        series_dropout=0.0,
        token_adapter_bottleneck=4,
    )
    model = PatchKneeAlibiModel(config).eval()
    batch = make_patch_batch()
    with torch.no_grad():
        reference = model(**batch)

    patch_permutation = torch.tensor([2, 0, 3, 1])
    permuted_patch = dict(batch)
    permuted_patch["patch_features"] = batch["patch_features"].index_select(3, patch_permutation)
    permuted_patch["patch_mask"] = batch["patch_mask"].index_select(3, patch_permutation)
    with torch.no_grad():
        patch_output = model(**permuted_patch)
    patch_error = float((reference - patch_output).abs().max())
    if patch_error > 2e-5:
        raise AssertionError(f"patch permutation changed logits by {patch_error}")

    slice_permutation = torch.tensor([4, 1, 3, 0, 2])
    permuted_slice = dict(batch)
    for key in ("features", "patch_features", "patch_mask", "positions_mm", "slice_mask"):
        permuted_slice[key] = batch[key].index_select(2, slice_permutation)
    with torch.no_grad():
        slice_output = model(**permuted_slice)
    slice_error = float((reference - slice_output).abs().max())
    if slice_error > 3e-5:
        raise AssertionError(f"slice permutation changed logits by {slice_error}")

    mean_config = PatchKneeModelConfig(
        feature_dim=12,
        patch_dim=6,
        hidden_dim=16,
        n_heads=4,
        series_depth=1,
        study_depth=1,
        dropout=0.0,
        aggregator="mean",
        num_targets=3,
        series_dropout=0.0,
        token_adapter_bottleneck=0,
    )
    mean_model = PatchKneeAlibiModel(mean_config).eval()
    with torch.no_grad():
        mean_reference = mean_model(**batch)
        mean_slice_output = mean_model(**permuted_slice)
    mean_slice_error = float((mean_reference - mean_slice_output).abs().max())
    if mean_slice_error > 3e-5:
        raise AssertionError(f"patch mean changed under slice permutation by {mean_slice_error}")
    shifted_positions = dict(batch)
    shifted_positions["positions_mm"] = batch["positions_mm"] * 11.0 + 37.0
    with torch.no_grad():
        mean_position_output = mean_model(**shifted_positions)
    mean_position_error = float((mean_reference - mean_position_output).abs().max())
    if mean_position_error > 1e-6:
        raise AssertionError(f"patch mean used physical positions by {mean_position_error}")

    train_model = PatchKneeAlibiModel(config).train()
    labels = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.02)
    losses = []
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(train_model(**batch), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if losses[-1] >= 0.6 * losses[0]:
        raise AssertionError("patch hierarchy failed the tiny overfit")

    print(
        json.dumps(
            {
                "status": "pass",
                "report_probabilities": {
                    "positive": positive,
                    "uncertain": uncertain,
                    "negative": negative,
                },
                "patch_permutation_error": patch_error,
                "slice_permutation_error": slice_error,
                "mean_slice_permutation_error": mean_slice_error,
                "mean_position_invariance_error": mean_position_error,
                "overfit_initial": losses[0],
                "overfit_final": losses[-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
