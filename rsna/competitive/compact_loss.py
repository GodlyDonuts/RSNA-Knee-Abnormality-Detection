"""Fitting-only positive-class weighting, separate from missing-label confidence.

Source probabilities are not clinical prevalence. No label is rounded, dropped,
replaced or made observed by this policy. The mean/clipping formula follows the
public Raptor recipe, evaluated on our own admitted fitting rows only.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch

from rsna.schema import TARGETS

VERSION = "compact_fitting_probability_positive_weight_v1"
ENGINE = "native_coatnet_bag12_N3_positive_weighted_v1"


def derive_policy(frame: pd.DataFrame) -> dict:
    identifiers = frame.study_uid.tolist()
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or any(not isinstance(x, str) or not x for x in identifiers)
    ):
        raise ValueError("Distinct admitted fitting identities required")
    ordered = frame.set_index("study_uid").loc[sorted(identifiers)]
    labels = ordered[list(TARGETS)].to_numpy(dtype=np.float64)
    masks = ordered[[t + "__mask" for t in TARGETS]].to_numpy(dtype=np.float64)
    if not np.isfinite(masks).all() or not np.isin(masks, [0, 1]).all():
        raise ValueError("Original binary observation masks required")
    if (
        not np.isfinite(labels[masks.astype(bool)]).all()
        or ((labels[masks.astype(bool)] < 0) | (labels[masks.astype(bool)] > 1)).any()
    ):
        raise ValueError("Observed source probabilities must be finite in [0,1]")
    counts = masks.sum(0)
    if (counts == 0).any():
        raise ValueError("Each target needs observed fitting cells for positive weighting")
    safe = np.where(masks.astype(bool), labels, 0)
    means = safe.sum(0) / counts
    clipped = np.clip(means, 0.03, 0.7)
    weights = np.clip((1 - clipped) / clipped, 1, 10)
    roster = json.dumps(sorted(identifiers)).encode()
    digest = hashlib.sha256(
        roster + safe.astype("<f8").tobytes() + masks.astype("<f8").tobytes()
    ).hexdigest()
    return {
        "version": VERSION,
        "targets": list(TARGETS),
        "fitting_count": len(frame),
        "fitting_uids_sha256": hashlib.sha256(roster).hexdigest(),
        "observed_targets_and_masks_sha256": digest,
        "observed_cells": counts.astype(np.int64).tolist(),
        "source_probability_means": means.tolist(),
        "prevalence_clip": [0.03, 0.7],
        "positive_weight_clip": [1.0, 10.0],
        "positive_weights": weights.tolist(),
        "loss_denominator": "unchanged_effective_batch_observed_cells",
        "source_probabilities_are_not_clinical_prevalence": True,
    }


def validate_policy(policy: dict) -> None:
    keys = {
        "version",
        "targets",
        "fitting_count",
        "fitting_uids_sha256",
        "observed_targets_and_masks_sha256",
        "observed_cells",
        "source_probability_means",
        "prevalence_clip",
        "positive_weight_clip",
        "positive_weights",
        "loss_denominator",
        "source_probabilities_are_not_clinical_prevalence",
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != keys
        or policy["version"] != VERSION
        or policy["targets"] != list(TARGETS)
    ):
        raise ValueError("Exact positive-weight policy schema required")
    n = policy["fitting_count"]
    counts = policy["observed_cells"]
    means = np.asarray(policy["source_probability_means"], dtype=np.float64)
    weights = np.asarray(policy["positive_weights"], dtype=np.float64)
    if (
        type(n) is not int
        or n < 1
        or not isinstance(counts, list)
        or len(counts) != 12
        or any(type(x) is not int or not 0 < x <= n for x in counts)
        or means.shape != (12,)
        or not np.isfinite(means).all()
        or ((means < 0) | (means > 1)).any()
        or weights.shape != (12,)
        or not np.isfinite(weights).all()
        or policy["prevalence_clip"] != [0.03, 0.7]
        or policy["positive_weight_clip"] != [1.0, 10.0]
        or policy["loss_denominator"] != "unchanged_effective_batch_observed_cells"
        or policy["source_probabilities_are_not_clinical_prevalence"] is not True
    ):
        raise ValueError("Positive-weight fitting scope or formula differs")
    p = np.clip(means, 0.03, 0.7)
    if not np.array_equal(weights, np.clip((1 - p) / p, 1, 10)):
        raise ValueError("Weights differ from bound fitting-only probability formula")
    for name in ["fitting_uids_sha256", "observed_targets_and_masks_sha256"]:
        digest = policy[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(x not in "0123456789abcdef" for x in digest)
        ):
            raise ValueError("Fitting roster/target fingerprint required")


def loss_weights(policy: dict, data, binding: dict, device: torch.device) -> torch.Tensor:
    validate_policy(policy)
    if len(data) == policy["fitting_count"]:
        if derive_policy(data.studies) != policy:
            raise ValueError("Actual fitting rows/targets differ from the frozen loss policy")
    elif not (
        binding.get("diagnostic_only") is True
        and len(data) == 16
        and policy["fitting_count"] == 2247
    ):
        raise ValueError(
            "Production uses full fitting weights; only the bound16-study diagnostic may subset"
        )
    if (
        binding.get("diagnostic_only") is False
        and binding.get("fitting_uids_sha256") != policy["fitting_uids_sha256"]
    ):
        raise ValueError("Loss policy and actual production fitting rosters differ")
    return torch.tensor(policy["positive_weights"], dtype=torch.float32, device=device)
