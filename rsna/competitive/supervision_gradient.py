"""Fitting-only BCE logit-derivative attribution for a changed teacher.

This evaluates a full-bag parent-logit reference, not actual sampled-window
student logits, encoder parameter gradients, label correctness or held-out AUC.
"""

from __future__ import annotations

import numpy as np


def audit(parent_logits, cross_logits, labels, mask):
    if (
        not isinstance(parent_logits, np.ndarray)
        or parent_logits.ndim != 3
        or parent_logits.shape[1:] != (2, 12)
        or not len(parent_logits)
        or cross_logits.shape != parent_logits.shape
        or labels.shape != (len(parent_logits), 12)
        or mask.shape != labels.shape
        or mask.dtype != np.bool_
        or not np.isfinite(parent_logits).all()
        or not np.isfinite(cross_logits).all()
        or not np.isfinite(labels[mask]).all()
        or ((labels[mask] < 0) | (labels[mask] > 1)).any()
    ):
        raise ValueError(
            "Finite aligned fitting-only parent/cross-teacher logits and masked soft targets required"
        )
    p = np.exp(-np.logaddexp(0, -parent_logits.astype(np.float64)))
    q = np.exp(-np.logaddexp(0, -cross_logits.astype(np.float64)))
    observed = np.broadcast_to(mask[:, None, :], p.shape)
    y = np.where(observed, labels[:, None, :], p)
    source = 0.25 * (p - y)
    teacher = 0.75 * (p - q)
    total = source + teacher

    def summary(a, b):
        x, y = a.reshape(-1), b.reshape(-1)
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        energy = nx * nx + ny * ny
        active = (np.abs(x) > 1e-12) & (np.abs(y) > 1e-12)
        return {
            "source_L2": nx,
            "teacher_L2": ny,
            "source_to_teacher_L2_ratio": nx / ny if ny else None,
            "source_fraction_of_separate_squared_norms": nx * nx / energy if energy else None,
            "source_teacher_cosine": float(np.dot(x, y) / (nx * ny)) if nx and ny else None,
            "opposing_nonzero_derivative_cells": int(((x * y < 0) & active).sum()),
            "both_nonzero_derivative_cells": int(active.sum()),
        }

    return {
        "fitting_studies": len(p),
        "views": 2,
        "targets": 12,
        "observed_cells": int(mask.sum()),
        "soft_observed_cells": int((mask & (labels > 0) & (labels < 1)).sum()),
        "aggregate": summary(source, teacher),
        "per_target": [
            {
                "target_index": t,
                **summary(source[:, :, t], teacher[:, :, t]),
                "mixture_derivative_L2": float(np.linalg.norm(total[:, :, t])),
                "teacher_only_derivative_L2": float(np.linalg.norm((p - q)[:, :, t])),
            }
            for t in range(12)
        ],
        "reference_is_original_full94_parent_logits_not_sampled_student": True,
        "unnormalized_BCE_logit_derivatives_not_parameter_gradients": True,
        "source_labels_masks_or_targets_changed": False,
        "clinical_label_quality_or_accuracy_claim": False,
    }
