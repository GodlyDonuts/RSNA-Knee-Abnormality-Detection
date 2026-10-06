"""Measure fitting-only logical-union support before proposing shared anatomy queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rsna.schema import TARGETS
from scripts.research.audit_ranking_exposure import fitting_inputs

GROUPS = {
    "ACL_or_MCL": ("ACL", "MCL"),
    "Medial_or_Lateral_Meniscus": ("Medial Meniscus", "Lateral Meniscus"),
    "Contusion_or_Fracture": ("Contusion", "Fracture"),
}


def logical_union(labels, masks):
    """Positive if any observed member is positive; negative only if all are negative.

    Unknown member labels never become negatives. These derived logical tasks
    are not new clinical diagnoses and do not supply extra positives to a rare
    disease. They could supervise a separately tested shared representation.
    """
    labels, masks = np.asarray(labels), np.asarray(masks)
    if labels.ndim != 2 or labels.shape != masks.shape or labels.shape[1] < 2:
        raise ValueError("Aligned study/member matrices required")
    observed = masks == 1
    if (
        not np.isin(masks, (0, 1)).all()
        or not np.isfinite(labels[observed]).all()
        or not np.isin(labels[observed], (0, 1)).all()
    ):
        raise ValueError("Original strict binary labels and masks required")
    positive = (observed & (labels == 1)).any(axis=1)
    negative = (observed & (labels == 0)).all(axis=1)
    return positive.astype(np.uint8), (positive | negative).astype(np.uint8)


def summarize(frame):
    rows = []
    for name, members in GROUPS.items():
        values = frame[list(members)].to_numpy()
        masks = frame[[f"{target}__mask" for target in members]].to_numpy()
        label, mask = logical_union(values, masks)
        row = {
            "logical_task": name,
            "members": list(members),
            "positive_studies": int((mask & label).sum()),
            "negative_studies": int((mask & (label == 0)).sum()),
            "unknown_studies": int((mask == 0).sum()),
            "positive_components": int(
                frame.loc[(mask & label).astype(bool), "leakage_group"].nunique()
            ),
            "negative_components": int(
                frame.loc[(mask & (label == 0)).astype(bool), "leakage_group"].nunique()
            ),
            "members_support": [],
        }
        for i, target in enumerate(members):
            positive = (masks[:, i] == 1) & (values[:, i] == 1)
            negative = (masks[:, i] == 1) & (values[:, i] == 0)
            sibling_positive = ((masks == 1) & (values == 1))[:, 1 - i]
            row["members_support"].append(
                {
                    "target": target,
                    "positive_studies": int(positive.sum()),
                    "negative_studies": int(negative.sum()),
                    "observed_negative_with_positive_sibling_studies": int(
                        (negative & sibling_positive).sum()
                    ),
                    "positive_with_positive_sibling_studies": int(
                        (positive & sibling_positive).sum()
                    ),
                }
            )
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ("studies", "folds", "original-run", "live-CPU", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the original aggregate; never overwrite an audit")
    frame, binding = fitting_inputs(args.studies, args.folds, args.original_run)
    live = json.loads(args.live_CPU.read_text())
    if (
        live["training_studies"] != binding["fitting_studies"]
        or live["outer_studies"] != binding["outer_roster_studies"]
        or live["training_uids_sha256"] != binding["training_uids_sha256"]
        or live["outer_uids_sha256"] != binding["outer_uids_sha256"]
        or len(TARGETS) != 12
    ):
        raise ValueError("Current live experiment and original fitting membership differ")
    result = {
        "status": "actual_original_fitting_only_logical_union_support_measured",
        "binding": binding,
        "groups": summarize(frame),
        "clinical_labels_masks_or_folds_changed": False,
        "new_rare_disease_labels_created": False,
        "MRI_encoder_optimizer_or_GPU_executed": False,
        "model_mechanism_implemented_or_accuracy_gain_established": False,
        "interpretation": "Observed logical unions support testing shared anatomy representations; they are not additional rare-disease positives or patient-level clinical truth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
