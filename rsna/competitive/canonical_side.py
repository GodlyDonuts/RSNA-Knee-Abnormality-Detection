"""Verified canonical-side (laterality) binding for fitting and evaluation.

The artifact is experiment-bound like a fitting-supervision overlay: it pins
the release, folds, fold, and exact fitting/outer rosters, then supplies a
deterministic horizontal-mirror flag per study. Studies whose side is unknown,
ambiguous, or bilateral carry ``side: null`` and are never mirrored. This is a
canonicalization applied identically to fitting, evaluation, and inference —
not an augmentation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from rsna.competitive.supervision import roster_sha

MAX_ARTIFACT_BYTES = 64 * 1024**2


def _sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_canonical_side_policy(policy: dict) -> None:
    if not isinstance(policy, dict):
        raise ValueError("Canonical-side policy must be a mapping")
    if policy.get("kind") != "canonical_side_v1":
        raise ValueError("Unknown canonical-side policy kind")
    if not _sha(policy.get("sha256")):
        raise ValueError("Canonical-side policy needs an artifact sha256")
    if type(policy.get("enabled")) is not bool:
        raise ValueError("Canonical-side policy needs an explicit enabled flag")


def load_canonical_side_table(
    train: pd.DataFrame,
    outer: pd.DataFrame,
    path: Path,
    expected_sha256: str,
    *,
    release_sha256: str,
    folds_sha256: str,
    fold: int,
) -> tuple[pd.DataFrame, dict]:
    """Return the study_uid/flip frame for the exact accepted fold rosters."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Invalid canonical-side artifact")
    raw = path.read_bytes()
    if not _sha(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Canonical-side artifact SHA differs from explicit admission")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Unsupported canonical-side artifact schema")
    if (
        value.get("status") != "canonical_side_experiment_only"
        or value.get("fit_scope") != "outer_fold"
        or type(value.get("fold")) is not int
        or value["fold"] != fold
        or value.get("release_sha256") != release_sha256
        or value.get("folds_sha256") != folds_sha256
        or not _sha(release_sha256)
        or not _sha(folds_sha256)
    ):
        raise ValueError("Canonical-side artifact scope/release/folds differ")
    if train.empty or outer.empty or train.study_uid.duplicated().any():
        raise ValueError("Canonical-side binding requires complete distinct rosters")
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
        raise ValueError("Canonical-side artifact is not bound to the accepted fold rosters")
    canonical = value.get("canonical_side")
    if canonical not in ("L", "R"):
        raise ValueError("Canonical-side artifact must name canonical_side L or R")
    sides = value.get("sides")
    if not isinstance(sides, dict) or not sides:
        raise ValueError("Canonical-side artifact has no study sides")
    if any(entry not in ("L", "R", None) for entry in sides.values()):
        raise ValueError("Canonical-side entries must be L, R, or null")
    roster = set(train.study_uid) | set(outer.study_uid)
    if set(sides) != roster:
        raise ValueError("Canonical-side artifact must cover exactly the train+outer roster")
    frame = pd.DataFrame(
        {
            "study_uid": list(sides),
            "flip": [entry is not None and entry != canonical for entry in sides.values()],
        }
    )
    receipt = {
        "kind": "canonical_side_v1",
        "artifact_sha256": expected_sha256,
        "canonical_side": canonical,
        "fitting_uid_sha256": roster_sha(train),
        "outer_uid_sha256": roster_sha(outer),
        "fold": fold,
        "studies": len(sides),
        "mirrored_studies": int(frame["flip"].sum()),
        "unresolved_side_studies": sum(1 for entry in sides.values() if entry is None),
        "experimental_only": True,
    }
    return frame, receipt
