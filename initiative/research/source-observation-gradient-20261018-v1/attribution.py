"""Attribute dense-parent initial source gradients by declared observation state."""

from __future__ import annotations

import numpy as np


def assess(labels, mask, teacher_logits, state_codes, preserve, *, source_weight=0.25):
    n = len(labels)
    if (
        labels.shape != (n, 12)
        or labels.dtype != np.float32
        or mask.shape != labels.shape
        or mask.dtype != np.bool_
        or teacher_logits.shape != (n, 2, 12)
        or teacher_logits.dtype != np.float32
        or state_codes.shape != (n, 12, 2)
        or state_codes.dtype != np.uint8
        or preserve.shape != labels.shape
        or preserve.dtype != np.bool_
        or not n
        or not np.isfinite(labels[mask]).all()
        or ((labels[mask] < 0) | (labels[mask] > 1)).any()
        or not np.isfinite(teacher_logits).all()
        or not np.isin(state_codes, [0, 1, 2]).all()
        or source_weight != 0.25
    ):
        raise ValueError(
            "Exact fitting-only two-view target/source states and unchanged weight required"
        )
    p = np.exp(-np.logaddexp(0, -teacher_logits.astype(np.float64)))
    gradient = np.where(mask[:, None, :], source_weight * (p - labels[:, None, :]), 0)
    energy = (gradient**2).mean(1)
    incompatible = preserve | (state_codes == 2).any(2)
    groups = {
        "both_declared_unmentioned": (state_codes == 0).all(2) & ~incompatible,
        "baseline_only_observed": (state_codes[:, :, 0] == 1)
        & (state_codes[:, :, 1] == 0)
        & ~incompatible,
        "Gemini_only_observed": (state_codes[:, :, 0] == 0)
        & (state_codes[:, :, 1] == 1)
        & ~incompatible,
        "both_observed": (state_codes == 1).all(2) & ~incompatible,
        "incompatible_preserved": incompatible,
    }
    if not np.all(np.stack(list(groups.values())).sum(0) == 1):
        raise ValueError("Every fitting cell must have one declared source-state category")
    total = energy.sum(0)
    summaries = {}
    for name, cells in groups.items():
        active = cells & mask
        value = (energy * active).sum(0)
        summaries[name] = {
            "cells_all12": active.sum(0).tolist(),
            "source_gradient_energy_fraction_all12": np.divide(
                value, total, out=np.zeros(12), where=total > 0
            ).tolist(),
            "macro_cell_energy_fraction": float(value.sum() / total.sum())
            if total.sum() > 0
            else None,
            "mean_label_minus_teacher_probability": float(
                (labels[:, None, :] - p)[np.broadcast_to(active[:, None, :], p.shape)].mean()
            )
            if active.any()
            else None,
        }
    return {
        "groups": summaries,
        "total_unnormalized_source_logit_gradient_RMS_all12": np.sqrt(energy.mean(0)).tolist(),
        "zero_energy_targets_all12": (total == 0).tolist(),
        "cells": n * 12,
        "population": n,
        "two_views_equally_weighted": True,
        "not_actual_sampled_student_trajectory": True,
        "not_clinical_error_or_accuracy": True,
        "source_state_not_inferred_by_probability_threshold": True,
    }
