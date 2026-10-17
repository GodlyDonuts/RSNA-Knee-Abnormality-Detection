"""Fitting-only response diagnostics, never an estimate of clinical accuracy.

Source values are weak labels. Teacher, initial and learned probabilities must
refer to exactly the same admitted cases and view. No threshold or recipe is
selected here; the strong-source thresholds are fixed at 0.1 and 0.9.
"""

from __future__ import annotations

import numpy as np


def _probability(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 1
        or not len(result)
        or not np.isfinite(result).all()
        or (result < 0).any()
        or (result > 1).any()
    ):
        raise ValueError("Nonempty finite one-dimensional probabilities required")
    return result


def pair_response(
    labels: np.ndarray,
    mask: np.ndarray,
    teacher: np.ndarray,
    candidate: np.ndarray,
    pointwise_target: np.ndarray,
) -> dict:
    """Count dependent strong-source pairs in bounded blocks, including ties.

    A teacher-opposed pair is strictly inverted. A correction must be strictly
    ordered; a tie is not a correction. Damage is a new strict inversion of a
    previously strictly ordered pair. Counts are not independent observations.
    """
    labels, teacher, candidate, pointwise_target = map(
        _probability, (labels, teacher, candidate, pointwise_target)
    )
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or any(
        x.shape != labels.shape for x in (mask, teacher, candidate, pointwise_target)
    ):
        raise ValueError("Matching population and boolean accepted mask required")
    positive = np.flatnonzero(mask & (labels >= 0.9))
    negative = np.flatnonzero(mask & (labels <= 0.1))
    counts = dict.fromkeys(
        (
            "teacher_opposed",
            "teacher_tied",
            "teacher_ordered",
            "candidate_ordered",
            "candidate_tied",
            "opposed_corrected",
            "opposed_candidate_tied",
            "ordered_newly_inverted",
            "pointwise_correctable_opposed",
            "correctable_opposed_actually_corrected",
        ),
        0,
    )
    for offset in range(0, len(positive), 128):
        selected = positive[offset : offset + 128]
        before = teacher[selected, None] - teacher[negative][None, :]
        after = candidate[selected, None] - candidate[negative][None, :]
        desired = pointwise_target[selected, None] - pointwise_target[negative][None, :]
        predicates = {
            "teacher_opposed": before < 0,
            "teacher_tied": before == 0,
            "teacher_ordered": before > 0,
            "candidate_ordered": after > 0,
            "candidate_tied": after == 0,
            "opposed_corrected": (before < 0) & (after > 0),
            "opposed_candidate_tied": (before < 0) & (after == 0),
            "ordered_newly_inverted": (before > 0) & (after < 0),
            "pointwise_correctable_opposed": (before < 0) & (desired > 0),
            "correctable_opposed_actually_corrected": (before < 0) & (desired > 0) & (after > 0),
        }
        for key, predicate in predicates.items():
            counts[key] += int(predicate.sum())
    total = len(positive) * len(negative)
    opposed = counts["teacher_opposed"]
    correctable = counts["pointwise_correctable_opposed"]
    return {
        "strong_source_positive_studies": len(positive),
        "strong_source_negative_studies": len(negative),
        "dependent_source_pairs": total,
        **counts,
        "opposed_correction_fraction": counts["opposed_corrected"] / opposed if opposed else None,
        "correctable_opposed_achieved_fraction": (
            counts["correctable_opposed_actually_corrected"] / correctable if correctable else None
        ),
        "strong_source_pair_agreement_before": (
            (counts["teacher_ordered"] + 0.5 * counts["teacher_tied"]) / total if total else None
        ),
        "strong_source_pair_agreement_after": (
            (counts["candidate_ordered"] + 0.5 * counts["candidate_tied"]) / total
            if total
            else None
        ),
    }


def response_summary(
    labels: np.ndarray,
    mask: np.ndarray,
    teacher: np.ndarray,
    initial: np.ndarray,
    candidate: np.ndarray,
    *,
    source_weight: float = 0.25,
) -> dict:
    """Summarize achieved response on fitting cases without exporting rows.

    The pseudo-target is an unconstrained pointwise optimum, not a realizability
    guarantee for the shared model. Cross-entropy is a full-bag diagnostic; it
    does not reconstruct the sampled training-batch objective.
    """
    labels, teacher, initial, candidate = map(_probability, (labels, teacher, initial, candidate))
    mask = np.asarray(mask)
    if (
        type(source_weight) not in (int, float)
        or not 0 < source_weight <= 1
        or mask.dtype != np.bool_
        or any(x.shape != labels.shape for x in (mask, teacher, initial, candidate))
        or not mask.any()
    ):
        raise ValueError("Matching observed cases and a finite positive source weight required")
    pointwise_target = source_weight * labels + (1 - source_weight) * teacher
    wanted = pointwise_target[mask]

    def scores(probability):
        observed = probability[mask]
        bounded = np.clip(observed, 1e-12, 1 - 1e-12)
        return {
            "pointwise_target_RMSE": float(np.sqrt(np.mean((observed - wanted) ** 2))),
            "full_bag_pseudo_target_cross_entropy": float(
                np.mean(-wanted * np.log(bounded) - (1 - wanted) * np.log1p(-bounded))
            ),
        }

    change = candidate[mask] - initial[mask]
    desired_change = wanted - initial[mask]
    nonzero = desired_change != 0
    return {
        "observed_fitting_studies": int(mask.sum()),
        "initial": scores(initial),
        "candidate": scores(candidate),
        "absolute_probability_change_quantiles": {
            name: float(value)
            for name, value in zip(
                ("median", "q90", "q99", "maximum"),
                np.quantile(np.abs(change), (0.5, 0.9, 0.99, 1)),
                strict=True,
            )
        },
        "nonzero_desired_changes": int(nonzero.sum()),
        "changes_in_pointwise_desired_direction_fraction": (
            float(np.mean(change[nonzero] * desired_change[nonzero] > 0)) if nonzero.any() else None
        ),
        "teacher_pairs": pair_response(labels, mask, teacher, candidate, pointwise_target),
        "initial_pairs": pair_response(labels, mask, initial, candidate, pointwise_target),
        "clinical_or_held_out_accuracy_claim": False,
    }
