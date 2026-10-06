"""Swap the complete fitting label table for an explicitly pinned external source.

This is the label-source ablation admission: identical release membership,
identical folds, identical evaluation frame — only the in-memory fitting label
values and masks change. The accepted release and all evaluation masks remain
immutable. The overlay covers every fitting study so no cell can silently keep
its original supervision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from rsna.competitive.supervision import roster_sha
from rsna.schema import TARGETS

MAX_OVERLAY_BYTES = 64 * 1024**2


def _sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def load_fitting_label_source(
    train: pd.DataFrame,
    outer: pd.DataFrame,
    path: Path,
    expected_sha256: str,
    *,
    release_sha256: str,
    folds_sha256: str,
    fold: int,
) -> tuple[pd.DataFrame, dict]:
    """Replace every fitting label and mask with the bound external source.

    The caller selects the verified release and completes the outer fold first;
    hashes bind that selection before any label is swapped. Overlay studies must
    cover the fitting set exactly. Evaluation frames are never touched.
    """
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_OVERLAY_BYTES:
        raise ValueError("Invalid label-source overlay artifact")
    raw = path.read_bytes()
    if not _sha(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Label-source overlay SHA differs from explicit admission")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported label-source overlay schema")
    if (
        value.get("status") != "label_source_experiment_only"
        or value.get("fit_scope") != "outer_fold"
        or type(value.get("fold")) is not int
        or value["fold"] != fold
        or value.get("release_sha256") != release_sha256
        or value.get("folds_sha256") != folds_sha256
        or not _sha(release_sha256)
        or not _sha(folds_sha256)
    ):
        raise ValueError("Label-source overlay scope/release/folds differ")
    source = value.get("source")
    if not isinstance(source, dict) or not _sha(source.get("csv_sha256")):
        raise ValueError("Label-source overlay lacks its pinned source identity")
    if train.empty or outer.empty or train.study_uid.duplicated().any():
        raise ValueError("Label-source overlay requires complete distinct rosters")
    if outer.study_uid.duplicated().any() or set(train.study_uid) & set(outer.study_uid):
        raise ValueError("Fitting and outer studies overlap")
    if set(train.leakage_group) & set(outer.leakage_group):
        raise ValueError("Fitting and outer leakage components overlap")
    if (
        not train.split.eq("train").all()
        or train.fold.eq(fold).any()
        or not outer.fold.eq(fold).all()
        or value.get("fitting_uid_sha256") != roster_sha(train)
        or value.get("outer_uid_sha256") != roster_sha(outer)
    ):
        raise ValueError("Overlay membership is not the exact accepted fitting fold")
    studies = value.get("studies")
    if (
        not isinstance(studies, list)
        or not studies
        or type(value.get("study_count")) is not int
        or value["study_count"] != len(studies)
    ):
        raise ValueError("Invalid label-source study table")
    by_uid = dict(zip(train.study_uid, train.index, strict=True))
    if not train.index.is_unique:
        raise ValueError("Label-source overlay needs unique frame indices")
    result = train.copy(deep=True)
    seen: set[str] = set()
    changed_cells = 0
    for study in studies:
        if not isinstance(study, dict):
            raise ValueError("Invalid label-source study entry")
        uid = study.get("study_uid")
        if not isinstance(uid, str) or uid not in by_uid or uid in seen:
            raise ValueError("Duplicate, missing or non-fitting overlay study")
        seen.add(uid)
        labels, masks = study.get("labels"), study.get("mask")
        if (
            not isinstance(labels, dict)
            or not isinstance(masks, dict)
            or set(labels) != set(TARGETS)
            or set(masks) != set(TARGETS)
        ):
            raise ValueError("Overlay study must cover every target exactly once")
        index = by_uid[uid]
        for target in TARGETS:
            label, mask = labels[target], masks[target]
            if mask not in (0, 1, 0.0, 1.0, False, True):
                raise ValueError("Label-source masks must be binary")
            if mask in (1, 1.0, True) and (
                not isinstance(label, (int, float))
                or isinstance(label, bool)
                or not (0.0 <= float(label) <= 1.0)
            ):
                raise ValueError("Observed labels must be finite values in [0,1]")
            if mask in (0, 0.0, False):
                label = 0.0
            if result.at[index, target] != float(label) or result.at[
                index, f"{target}__mask"
            ] != int(mask):
                changed_cells += 1
            result.at[index, target] = float(label)
            result.at[index, f"{target}__mask"] = int(mask)
    if seen != set(train.study_uid):
        raise ValueError("Overlay must cover the complete fitting set")
    binding = {
        "schema_version": 1,
        "overlay_sha256": expected_sha256,
        "release_sha256": release_sha256,
        "folds_sha256": folds_sha256,
        "fitting_uid_sha256": roster_sha(train),
        "outer_uid_sha256": roster_sha(outer),
        "fold": fold,
        "source": source,
        "fitting_studies": len(studies),
        "swapped_cells": changed_cells,
        "experimental_only": True,
        "evaluation_masks_changed": False,
    }
    return result, binding
