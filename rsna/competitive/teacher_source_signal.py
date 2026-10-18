"""Fitting-only supervision envelope; neither accuracy nor learned-gradient evidence."""

from __future__ import annotations

import numpy as np


def mixture_arrays(parent_logits, labels, mask, weight):
    if (
        not isinstance(parent_logits, np.ndarray)
        or parent_logits.ndim != 3
        or parent_logits.shape[1:] != (2, 12)
        or len(parent_logits) == 0
        or labels.shape != mask.shape
        or labels.shape != (len(parent_logits), 12)
        or mask.dtype != np.bool_
        or not np.isfinite(parent_logits).all()
        or not np.isfinite(labels[mask]).all()
        or ((labels[mask] < 0) | (labels[mask] > 1)).any()
        or type(weight) not in (int, float)
        or not np.isfinite(weight)
        or not 0 < weight <= 1
    ):
        raise ValueError(
            "Finite complete corresponding-view teacher and masked probabilities required"
        )
    p = np.exp(-np.logaddexp(0, -parent_logits.astype(np.float64)))
    m = np.broadcast_to(mask[:, None, :], p.shape)
    y = np.where(m, labels[:, None, :], p)
    target = (1 - weight) * p + weight * y
    # This is the logit-loss derivative at the teacher, not a parameter gradient.
    parent_logit_derivative = (p - target) / p.size
    return p, target, parent_logit_derivative


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def audit(parent_logits, labels, mask):
    p, target, gradient = mixture_arrays(parent_logits, labels, mask, 0.25)
    y = labels[:, None, :]
    observed = mask[:, None, :]
    binary = observed & ((y == 0) | (y == 1))
    weak_positive = y == 1
    teacher_discordance = binary & ((p >= 0.5) != weak_positive)
    kept = teacher_discordance & ((target >= 0.5) != weak_positive)
    both_discordant = teacher_discordance.all(axis=1)
    both_kept = kept.all(axis=1)
    confidence = np.where(weak_positive, 1 - p, p)
    strong_both = both_discordant & (confidence >= 0.9).all(axis=1)
    rows = []
    for t in range(12):
        n = int(mask[:, t].sum())
        b = int(binary[:, :, t].sum()) * 2  # binary has one view until broadcast.
        views = int(teacher_discordance[:, :, t].sum())
        retained = int(kept[:, :, t].sum())
        rows.append(
            {
                "target_index": t,
                "observed_studies": n,
                "positive_studies": int((mask[:, t] & (labels[:, t] == 1)).sum()),
                "negative_studies": int((mask[:, t] & (labels[:, t] == 0)).sum()),
                "observed_soft_label_studies": int(
                    (mask[:, t] & (labels[:, t] > 0) & (labels[:, t] < 1)).sum()
                ),
                "masked_studies": int((~mask[:, t]).sum()),
                "binary_observed_view_cells": b,
                "teacher_weak_label_discordant_view_cells": views,
                "quarter_mixture_target_still_teacher_side_view_cells": retained,
                "fraction_discordant_view_targets_still_teacher_side": ratio(retained, views),
                "studies_teacher_discordant_in_both_views": int(both_discordant[:, t].sum()),
                "studies_quarter_target_still_teacher_side_in_both_views": int(
                    both_kept[:, t].sum()
                ),
                "studies_teacher_discordant_both_views_with_uncalibrated_probability_ge_0_9": int(
                    strong_both[:, t].sum()
                ),
                "parent_logit_loss_derivative_L2": float(np.linalg.norm(gradient[:, :, t])),
            }
        )
    sensitivity = []
    for w in (0.25, 0.5, 0.75, 1.0):
        _, q, _ = mixture_arrays(parent_logits, labels, mask, w)
        count = int((teacher_discordance & ((q >= 0.5) != weak_positive)).sum())
        sensitivity.append(
            {
                "source_weight": w,
                "discordant_view_targets_still_teacher_side": count,
                "fraction": ratio(count, int(teacher_discordance.sum())),
            }
        )
    return {
        "fitting_studies": len(labels),
        "views": 2,
        "targets": 12,
        "observed_study_target_cells": int(mask.sum()),
        "masked_study_target_cells": int((~mask).sum()),
        "teacher_weak_label_discordant_view_cells": int(teacher_discordance.sum()),
        "quarter_mixture_target_still_teacher_side_view_cells": int(kept.sum()),
        "per_target": rows,
        "predeclared_source_weight_sensitivity_not_a_recipe_selection": sensitivity,
        "masked_targets_equal_teacher_exactly": bool(
            np.array_equal(
                target[~np.broadcast_to(observed, p.shape)], p[~np.broadcast_to(observed, p.shape)]
            )
        ),
        "masked_target_maximum_FP64_roundoff_from_teacher": float(
            np.abs(
                target[~np.broadcast_to(observed, p.shape)] - p[~np.broadcast_to(observed, p.shape)]
            ).max(initial=0)
        ),
        "property_of_loss_target_not_a_bound_on_actual_network_predictions": True,
        "weak_label_discordance_is_not_a_teacher_error_or_clinical_diagnosis": True,
        "no_heldout_accuracy_or_actual_encoder_gradient_claim": True,
    }
