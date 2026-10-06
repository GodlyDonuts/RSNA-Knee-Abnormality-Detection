"""Re-use an exact private FP32 fitting prediction artifact, without labels or rerunning it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsna.schema import TARGETS


def load_reference(path: Path, receipt: dict, roster: list[str], checkpoint_sha256: str):
    if (
        receipt.get("status") != "passed"
        or receipt.get("inference_precision") != "fp32"
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or receipt.get("original_exact_source_SHA_verified") is not True
        or receipt.get("original_kernel_version") != 1
        or receipt.get("studies") != len(roster)
        or len(set(roster)) != len(roster)
        or not roster
    ):
        raise ValueError("Exact original FP32 model/version/fitting roster proof required")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 2**20:
        raise ValueError("Bounded original private prediction CSV required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["submission_sha256"]:
        raise ValueError("Original FP32 whole prediction bytes differ")
    import io

    frame = pd.read_csv(io.BytesIO(raw), dtype={"StudyInstanceUID": str})
    if (
        list(frame.columns) != ["StudyInstanceUID", *TARGETS]
        or len(frame) != len(roster)
        or frame.StudyInstanceUID.duplicated().any()
        or frame.StudyInstanceUID.tolist() != roster
    ):
        raise ValueError("Exact original ordered fitting prediction rows required")
    scores = frame[list(TARGETS)].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Finite original FP32 probabilities required")
    return frame.set_index("StudyInstanceUID")


def compare_predictions(reference, candidate, *, max_absolute_error=0.003):
    if max_absolute_error != 0.003:
        raise ValueError("Predeclared numerical-agreement tolerance required")
    if (
        not reference.index.is_unique
        or not candidate.index.is_unique
        or set(reference.index) != set(candidate.index)
    ):
        raise ValueError("Identical complete original prediction study sets required")
    left = reference[list(TARGETS)].to_numpy(dtype=float)
    right = candidate.loc[reference.index, list(TARGETS)].to_numpy(dtype=float)
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or (right < 0).any()
        or (right > 1).any()
    ):
        raise ValueError("Finite probabilities required for comparison")
    error = np.abs(left - right)
    if error.size == 0 or error.max() > max_absolute_error:
        raise ValueError("FP16 prediction agreement exceeds fixed tolerance")
    return {
        "studies": len(reference),
        "targets": list(TARGETS),
        "max_absolute_probability_error": float(error.max()),
        "mean_absolute_probability_error": float(error.mean()),
        "per_target_max_abs_error": error.max(axis=0).tolist(),
        "predeclared_max_absolute_error": max_absolute_error,
        "model_accuracy_or_clinical_equivalence_established": False,
    }


def roster_digest(roster):
    return hashlib.sha256(json.dumps(sorted(roster), separators=(",", ":")).encode()).hexdigest()
