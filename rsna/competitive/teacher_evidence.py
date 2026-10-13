"""Fitting-only frozen-teacher evidence: immutable storage and bounded auxiliary loss.

Teacher outputs are stored apart from labels as soft evidence. They never replace
labels, masks or folds, never reach evaluation studies and are never required at student
inference. The teacher's own training roster is unknown, so nothing here supports a clean
held-out teacher/student comparison, competition eligibility or clinical truth.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from rsna.competitive.loss import effective_batch_loss
from rsna.competitive.public_input import canonical_sha256, roster_sha256
from rsna.competitive.public_source_table import ACCEPTED_SOURCE_TABLE
from rsna.schema import TARGETS

KIND = "public_teacher_evidence_v1"
TEACHER = "pilkwang/rsna-knee-baseline-v1/v15/m_e5427d6c21.pt"
FITTING_STUDIES = ACCEPTED_SOURCE_TABLE["fitting_studies"]
FITTING_ROSTER_SHA256 = ACCEPTED_SOURCE_TABLE["fitting_roster_sha256"]
EVALUATION_ROSTER_SHA256 = ACCEPTED_SOURCE_TABLE["development_roster_sha256"]
ACCEPTANCE_STUDIES = 8
WINDOWS, SLOTS = 10, 6
MAX_EVIDENCE_BYTES = 8 << 20
MAX_AUXILIARY_WEIGHT = 1.0
EVIDENCE_FILE, RECEIPT_FILE = "evidence.npz", "receipt.json"
ARRAYS = ("study_uid", "window_logits", "probabilities", "slot_mask", "input_sha256")
BINDING_KEYS = {
    "kind",
    "teacher",
    "source_sha256",
    "renderer_node_text_sha256",
    "teacher_node_text_sha256",
    "teacher_review_sha256",
    "checkpoint_sha256",
    "member_manifest_sha256",
    "member_config",
    "pixel_group_sha256",
    "backbone_config_sha256",
    "runtime",
    "fitting_roster_sha256",
    "requested_roster_sha256",
    "requested_studies",
    "input",
    "targets",
    "windows",
}
CLAIMS = {
    "labels_read": False,
    "reports_read": False,
    "gold_or_confirmation_read": False,
    "evaluation_pixels_read": False,
    "replacement_labels": False,
    "clinical_truth": False,
    "teacher_training_roster": "unknown",
    "clean_heldout_teacher_comparison": False,
    "competition_eligibility": False,
    "accuracy_claim": False,
    "attention_or_slot_evidence": "not_exposed_by_original_source",
    "student_inference_requires_teacher": False,
    "public_release": False,
}


class EvidenceError(ValueError):
    """Evidence differs from its immutable binding, roster or storage contract."""


def fitting_request(
    fitting: Sequence[str],
    requested: Sequence[str],
    *,
    max_studies: int,
    fitting_sha256: str = FITTING_ROSTER_SHA256,
    fitting_studies: int = FITTING_STUDIES,
) -> list[str]:
    """Requested UIDs, in order, only if all belong to the pinned complete fitting roster."""
    population = [str(uid) for uid in fitting]
    if (
        len(population) != fitting_studies
        or len(set(population)) != fitting_studies
        or roster_sha256(population) != fitting_sha256
    ):
        raise EvidenceError("The complete accepted fitting roster is required")
    chosen = [str(uid) for uid in requested]
    if not 0 < len(chosen) <= min(max_studies, fitting_studies) or len(set(chosen)) != len(chosen):
        raise EvidenceError("Requested studies must be unique and within the storage cap")
    if set(chosen) - set(population):
        raise EvidenceError("Teacher collection refuses studies outside the fitting roster")
    return chosen


def validate_binding(binding: object) -> dict:
    if (
        not isinstance(binding, dict)
        or set(binding) != BINDING_KEYS
        or binding["kind"] != KIND
        or binding["teacher"] != TEACHER
        or binding["targets"] != list(TARGETS)
        or binding["windows"] != WINDOWS
        or binding["requested_roster_sha256"] != roster_sha256(binding["requested_studies"])
        or not isinstance(binding["runtime"], dict)
        or not isinstance(binding["input"], dict)
    ):
        raise EvidenceError("Invalid teacher evidence binding")
    return binding


def _arrays(uids: Sequence[str], rows: Sequence[dict]) -> dict[str, np.ndarray]:
    if len(uids) != len(rows) or not rows:
        raise EvidenceError("One teacher row per requested study is required")
    arrays = {
        "study_uid": np.asarray(list(uids), dtype=np.str_),
        "window_logits": np.stack([r["window_logits"] for r in rows]).astype(np.float32),
        "probabilities": np.stack([r["probabilities"] for r in rows]).astype(np.float32),
        "slot_mask": np.stack([r["slot_mask"] for r in rows]).astype(np.bool_),
        "input_sha256": np.asarray([str(r["input_sha256"]) for r in rows], dtype=np.str_),
    }
    _check_arrays(arrays, list(uids))
    return arrays


def _check_arrays(arrays: dict[str, np.ndarray], uids: list[str]) -> None:
    n, t = len(uids), len(TARGETS)
    probabilities, logits = arrays["probabilities"], arrays["window_logits"]
    if (
        set(arrays) != set(ARRAYS)
        or arrays["study_uid"].tolist() != uids
        or logits.dtype != np.float32
        or logits.shape != (n, WINDOWS, t)
        or probabilities.dtype != np.float32
        or probabilities.shape != (n, t)
        or arrays["slot_mask"].dtype != np.bool_
        or arrays["slot_mask"].shape != (n, SLOTS)
        or not arrays["slot_mask"].any(1).all()
        or arrays["input_sha256"].shape != (n,)
        or any(len(h) != 64 for h in arrays["input_sha256"].tolist())
    ):
        raise EvidenceError("Teacher evidence arrays differ from the contract")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise EvidenceError("Non-finite teacher evidence")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise EvidenceError("Teacher probabilities outside [0,1]")


def _serialize(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{name: arrays[name] for name in ARRAYS})
    return buffer.getvalue()


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def summary(arrays: dict[str, np.ndarray]) -> dict:
    """Image-level teacher uncertainty only; says nothing about ground truth."""
    weight = confidence(
        torch.from_numpy(arrays["probabilities"]), torch.from_numpy(arrays["window_logits"])
    )
    return {
        "studies": int(arrays["probabilities"].shape[0]),
        "uncertain_cells_confidence_below_0.5": {
            t: int((weight[:, i] < 0.5).sum()) for i, t in enumerate(TARGETS)
        },
        "zero_weight_cells": int((weight == 0).sum()),
    }


def write_evidence(out: Path, binding: dict, rows: Sequence[dict]) -> dict:
    """Create the immutable evidence directory once; never overwrite or append."""
    validate_binding(binding)
    arrays = _arrays(binding["requested_studies"], rows)
    payload = _serialize(arrays)
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise EvidenceError("Teacher evidence exceeds its storage cap")
    out = Path(out)
    out.mkdir(parents=False, exist_ok=False)
    _write_exclusive(out / EVIDENCE_FILE, payload)
    receipt = {
        "kind": KIND,
        "status": "collected",
        "binding": binding,
        "binding_sha256": canonical_sha256(binding),
        "evidence": {
            "file": EVIDENCE_FILE,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "summary": summary(arrays),
        "claims": CLAIMS,
    }
    _write_exclusive(out / RECEIPT_FILE, json.dumps(receipt, indent=2, allow_nan=False).encode())
    return receipt


def load_evidence(out: Path, binding: dict | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """Verify receipt, bytes and arrays; `binding` must match exactly when given."""
    out = Path(out)
    if not (out / RECEIPT_FILE).is_file():
        raise EvidenceError("Teacher evidence is incomplete; preserved for inspection")
    receipt = json.loads((out / RECEIPT_FILE).read_text())
    if (
        not isinstance(receipt, dict)
        or receipt.get("kind") != KIND
        or receipt.get("claims") != CLAIMS
        or receipt.get("binding_sha256") != canonical_sha256(receipt.get("binding"))
        or (binding is not None and receipt["binding"] != binding)
    ):
        raise EvidenceError("Existing teacher evidence has a different binding")
    validate_binding(receipt["binding"])
    payload = (out / EVIDENCE_FILE).read_bytes()
    record = receipt["evidence"]
    if (
        record.get("file") != EVIDENCE_FILE
        or len(payload) != record.get("bytes")
        or len(payload) > MAX_EVIDENCE_BYTES
        or hashlib.sha256(payload).hexdigest() != record.get("sha256")
    ):
        raise EvidenceError("Teacher evidence bytes differ from the receipt")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAYS):
            raise EvidenceError("Teacher evidence arrays differ from the contract")
        arrays = {name: archive[name] for name in ARRAYS}
    _check_arrays(arrays, list(receipt["binding"]["requested_studies"]))
    return receipt, arrays


def reconcile(out: Path, binding: dict) -> dict | None:
    """Existing identical evidence -> its receipt; absent -> None; anything else refuses."""
    if not Path(out).exists():
        return None
    receipt, _ = load_evidence(out, validate_binding(binding))
    return receipt


def confidence(probabilities: Tensor, window_logits: Tensor) -> Tensor:
    """Bounded [0,1] weight: teacher decisiveness minus its own across-window spread."""
    if (
        probabilities.ndim != 2
        or window_logits.shape != (probabilities.shape[0], WINDOWS, probabilities.shape[1])
        or not torch.isfinite(probabilities).all()
        or not torch.isfinite(window_logits).all()
    ):
        raise EvidenceError("Aligned finite teacher evidence required")
    windows = torch.sigmoid(window_logits.float())
    spread = windows.amax(1) - windows.amin(1)
    return ((2 * probabilities.float() - 1).abs() - spread).clamp(0, 1)


def teacher_targets(
    receipt: dict, arrays: dict[str, np.ndarray], study_uids: Sequence[str]
) -> tuple[Tensor, Tensor]:
    """Soft targets and weights for a fitting batch; absent studies get zero weight."""
    fitting = receipt["binding"]["fitting_roster_sha256"]
    if fitting != FITTING_ROSTER_SHA256 or receipt["claims"] != CLAIMS:
        raise EvidenceError("Teacher evidence is not bound to the accepted fitting roster")
    index = {uid: i for i, uid in enumerate(arrays["study_uid"].tolist())}
    probabilities = torch.from_numpy(arrays["probabilities"])
    weights = confidence(probabilities, torch.from_numpy(arrays["window_logits"]))
    soft = torch.zeros(len(study_uids), len(TARGETS))
    weight = torch.zeros(len(study_uids), len(TARGETS))
    for row, uid in enumerate(study_uids):
        if uid in index:
            soft[row], weight[row] = probabilities[index[uid]], weights[index[uid]]
    return soft, weight


def auxiliary_weight(value: object) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))  # type: ignore[arg-type]
        or not 0 <= float(value) <= MAX_AUXILIARY_WEIGHT  # type: ignore[arg-type]
    ):
        raise EvidenceError("Teacher auxiliary weight must be finite in [0, 1]")
    return float(value)  # type: ignore[arg-type]


def teacher_auxiliary_loss(logits: Tensor, soft: Tensor, weight: Tensor, counts: Tensor) -> Tensor:
    """Confidence-weighted soft BCE with the same global normalization as the label loss."""
    return effective_batch_loss(logits, soft, weight, counts)


def combined_loss(label_loss: Tensor, auxiliary: Callable[[], Tensor], weight: object) -> Tensor:
    """Weight zero returns the ordinary label loss itself; the teacher term is not evaluated."""
    value = auxiliary_weight(weight)
    if value == 0:
        return label_loss
    return label_loss + value * auxiliary()


def disagreement(
    probabilities: Tensor, weight: Tensor, labels: Tensor, label_mask: Tensor
) -> dict[str, dict[str, int]]:
    """Research-only teacher-vs-accepted-label counts on fitting rows; labels stay authoritative.

    Confident cells (weight >= 0.5) are split into agree/disagree with observed labels;
    uncertain and unlabeled cells are counted separately and never resolved.
    """
    if not (probabilities.shape == weight.shape == labels.shape == label_mask.shape):
        raise EvidenceError("Aligned teacher and label tensors required")
    observed, confident = label_mask > 0, weight >= 0.5
    agree = (probabilities >= 0.5) == (labels >= 0.5)
    out = {}
    for i, target in enumerate(TARGETS):
        o, c, a = observed[:, i], confident[:, i], agree[:, i]
        out[target] = {
            "observed": int(o.sum()),
            "confident_agree": int((o & c & a).sum()),
            "confident_disagree": int((o & c & ~a).sum()),
            "uncertain_observed": int((o & ~c).sum()),
            "unlabeled": int((~o).sum()),
        }
    return out
