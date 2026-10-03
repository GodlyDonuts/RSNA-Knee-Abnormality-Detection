"""Apply an explicitly pinned experimental overlay to fitting cells only.

This loader validates engineering boundaries, not the clinical correctness of
report decisions. The accepted release and evaluation masks remain immutable.
The optional trainer contrast binds this admission; competition deployment
remains explicitly blocked pending separate acceptance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def roster_sha(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        json.dumps(sorted(frame.study_uid.tolist()), separators=(",", ":")).encode()
    ).hexdigest()


def _sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_fitting_supervision(policy: object) -> None:
    """Only the separately reviewed single-target paired experiment is supported."""
    if (
        not isinstance(policy, dict)
        or set(policy) != {"kind", "sha256", "enabled"}
        or policy.get("kind") != "effusion_review_v1"
        or not _sha(policy.get("sha256"))
        or type(policy.get("enabled")) is not bool
    ):
        raise ValueError("Invalid experimental fitting-supervision policy")


def apply_fitting_supervision(
    train: pd.DataFrame,
    outer: pd.DataFrame,
    path: Path | None,
    policy: dict | None,
    *,
    release_sha256: str,
    folds_sha256: str,
    fold: int,
) -> tuple[pd.DataFrame, dict | None]:
    """Validate both arms against the full cohort before diagnostic subsetting.

    The control admits the same reviewed proposal but keeps the original masks.
    The candidate changes only its in-memory fitting copy. Neither arm alters
    the accepted release or evaluation frame; artifact paths are not bound.
    """
    if policy is None:
        if path is not None:
            raise ValueError("Fitting-supervision artifact lacks an explicit bound policy")
        return train, None
    validate_fitting_supervision(policy)
    if path is None:
        raise ValueError("Bound fitting-supervision policy requires its exact artifact")
    candidate, receipt = load_fitting_effusion_overlay(
        train,
        outer,
        path,
        policy["sha256"],
        release_sha256=release_sha256,
        folds_sha256=folds_sha256,
        fold=fold,
    )
    receipt["enabled"] = policy["enabled"]
    receipt["applied_cells"] = (
        receipt["recovered_masked_positive_cells"] if policy["enabled"] else 0
    )
    return (candidate if policy["enabled"] else train), receipt


def load_fitting_effusion_overlay(
    train: pd.DataFrame,
    outer: pd.DataFrame,
    path: Path,
    expected_sha256: str,
    *,
    release_sha256: str,
    folds_sha256: str,
    fold: int,
) -> tuple[pd.DataFrame, dict]:
    """Recover only individually accepted, previously masked positive cells.

    Caller must first select the verified original release and complete outer
    fold. Hashes bind that selection before any diagnostic subset is taken.
    No new studies, target changes, confidence weights or evaluation changes
    are permitted. Exact artifact SHA is the explicit admission boundary.
    """
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError("Invalid fitting-overlay artifact")
    raw = path.read_bytes()
    if not _sha(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Fitting-overlay SHA differs from explicit admission")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported fitting-overlay schema")
    if (
        value.get("status") != "source_review_accepted_experiment_only"
        or value.get("target") != "Effusion"
        or value.get("fit_scope") != "outer_fold"
        or type(value.get("fold")) is not int
        or value["fold"] != fold
        or value.get("release_sha256") != release_sha256
        or value.get("folds_sha256") != folds_sha256
        or not _sha(release_sha256)
        or not _sha(folds_sha256)
    ):
        raise ValueError("Fitting-overlay scope/release/folds differ")
    if train.empty or outer.empty or train.study_uid.duplicated().any():
        raise ValueError("Overlay requires complete distinct fitting and outer rosters")
    if outer.study_uid.duplicated().any() or set(train.study_uid) & set(outer.study_uid):
        raise ValueError("Fitting and outer studies overlap")
    if set(train.leakage_group) & set(outer.leakage_group):
        raise ValueError("Fitting and outer leakage components overlap")
    if (
        not train.split.eq("train").all()
        or not train.report_supervision_allowed.eq(True).all()  # noqa: E712
        or train.fold.eq(fold).any()
        or not outer.fold.eq(fold).all()
        or value.get("fitting_uid_sha256") != roster_sha(train)
        or value.get("outer_uid_sha256") != roster_sha(outer)
    ):
        raise ValueError("Overlay membership is not the exact accepted fitting fold")
    reviews = value.get("reviews")
    if not isinstance(reviews, dict) or any(
        not _sha(reviews.get(k))
        for k in ("first_reader", "independent_reader", "original_context", "root_assessment")
    ):
        raise ValueError("Missing frozen original/source review bindings")
    cells = value.get("cells")
    if (
        not isinstance(cells, list)
        or not cells
        or type(value.get("cell_count")) is not int
        or value["cell_count"] != len(cells)
    ):
        raise ValueError("Invalid finite fitting-overlay membership")
    by_uid = dict(zip(train.study_uid, train.index, strict=True))
    if not train.index.is_unique:
        raise ValueError("Fitting overlay needs unique frame indices")
    result = train.copy(deep=True)
    seen: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Invalid overlay target cell")
        uid = cell.get("study_uid")
        if not isinstance(uid, str) or uid not in by_uid or uid in seen:
            raise ValueError("Duplicate or non-fitting overlay study")
        seen.add(uid)
        if (
            cell.get("target") != "Effusion"
            or cell.get("disposition") != "conditional_training_only_reuse_recommendation"
            or cell.get("first_reader_state") != "positive"
            or cell.get("independent_reader_state") != "positive"
            or any(
                not _sha(cell.get(k))
                for k in ("source_report_sha256", "current_annotation_sha256", "assessment_sha256")
            )
            or not isinstance(cell.get("original_hold_id"), str)
            or not cell["original_hold_id"].startswith("hold-")
            or not isinstance(cell.get("original_obligation_id"), str)
            or not cell["original_obligation_id"].startswith("ob-")
        ):
            raise ValueError("Overlay cell lacks its exact accepted issue disposition")
        index = by_uid[uid]
        if result.at[index, "Effusion__mask"] != 0 or result.at[index, "Effusion"] != 1:
            raise ValueError("Overlay can only recover an original masked positive")
        result.at[index, "Effusion__mask"] = 1
    binding = {
        "schema_version": 1,
        "overlay_sha256": expected_sha256,
        "release_sha256": release_sha256,
        "folds_sha256": folds_sha256,
        "fitting_uid_sha256": roster_sha(train),
        "outer_uid_sha256": roster_sha(outer),
        "fold": fold,
        "target": "Effusion",
        "recovered_masked_positive_cells": len(cells),
        "reviews": reviews,
        "experimental_only": True,
        "evaluation_masks_changed": False,
    }
    return result, binding
