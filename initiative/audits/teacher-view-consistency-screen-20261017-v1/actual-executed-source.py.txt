"""Fitting-only, label-free screen of the existing teacher's mirror disagreement.

Read only the approved UID roster and cached teacher logits. Source labels,
masks, reports, pixels, weights and protected clinical values are not accessed.
Aggregate inconsistency is not disease accuracy or an accepted training recipe.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

UID_SHA = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
TEACHER_SHA = "d9853c7745161807705087bef505f1ceb488f2138264dea3902388662fe37566"
TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)


def summarize(logits):
    if (
        not isinstance(logits, np.ndarray)
        or logits.ndim != 3
        or logits.shape[1:] != (2, 12)
        or logits.shape[0] < 2
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Finite paired-view teacher logits for twelve targets required")
    p = 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
    n = len(p)
    rows = []
    for target, name in enumerate(TARGETS):
        gap = np.abs(p[:, 0, target] - p[:, 1, target])
        strict = discordant = 0
        for start in range(0, n, 64):
            stop = min(start + 64, n)
            upper = np.arange(n)[None, :] > np.arange(start, stop)[:, None]
            a = p[start:stop, 0, target, None] - p[None, :, 0, target]
            b = p[start:stop, 1, target, None] - p[None, :, 1, target]
            valid = upper & (a != 0) & (b != 0)
            strict += int(valid.sum())
            discordant += int((valid & (np.signbit(a) != np.signbit(b))).sum())
        rows.append(
            {
                "target": name,
                "mean_absolute_view_probability_gap": float(gap.mean()),
                "probability_gap_p50_p90_p95_p99_max": [
                    float(x) for x in np.quantile(gap, [0.5, 0.9, 0.95, 0.99, 1])
                ],
                "fraction_gap_over_005": float((gap > 0.05).mean()),
                "fraction_gap_over_010": float((gap > 0.1).mean()),
            "fraction_opposite_sides_of_05": float(
                (((p[:, 0, target] - 0.5) * (p[:, 1, target] - 0.5)) < 0).mean()
            ),
                "all_study_pairs": n * (n - 1) // 2,
                "strict_pairs_in_both_views": strict,
                "discordant_view_pairs": discordant,
                "discordance_fraction_conditional_on_strict_pairs": discordant / strict
                if strict
                else None,
                "per_view_MSE_to_probability_consensus": float(np.mean(gap**2) / 4),
            }
        )
    return {
        "status": "fitting_only_teacher_mirror_inconsistency_screen",
        "fitting_studies": n,
        "targets": rows,
        "source_labels_masks_reports_pixels_weights_or_protected_values_read": False,
        "new_encoder_optimizer_or_accuracy_evaluation": False,
        "clinical_or_competition_accuracy_claim": False,
        "GPU_experiment_admitted_by_screen": False,
    }


def run(root):
    root = Path(root)
    proof = json.loads((root / "verified-aggregate.json").read_text())
    path = root / "targets.private.npz"
    if proof["original_UID_sha256"] != UID_SHA or path.stat().st_size != 483656:
        raise ValueError("Original admitted fitting input and existing byte proof required")
    with np.load(path, allow_pickle=False) as arrays:
        # Deliberately do not access arrays['labels'] or arrays['masks'].
        uids = arrays["uids"].tolist()
        logits = arrays["parent_logits"]
    if (
        len(uids) != 3887
        or len(set(uids)) != 3887
        or hashlib.sha256(json.dumps(uids, separators=(",", ":")).encode()).hexdigest() != UID_SHA
        or logits.shape != (3887, 2, 12)
        or logits.dtype != np.float32
        or hashlib.sha256(logits.tobytes()).hexdigest() != TEACHER_SHA
        or proof["targets"]["teacher_array_sha256"] != TEACHER_SHA
    ):
        raise ValueError("Exact previously admitted ordered fitting teacher array required")
    result = summarize(logits)
    result.update(teacher_array_sha256=TEACHER_SHA, fitting_UID_sha256=UID_SHA)
    return result
