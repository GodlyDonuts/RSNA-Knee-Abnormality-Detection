"""Private fitting-only evidence from the twenty-member public teacher committee.

Extends the single-teacher contract in `teacher_evidence` (same fitting roster, claims,
immutable write-once directory and reconciliation) with per-member original window logits
and study probabilities, the uniform committee mean and BETWEEN-MODEL disagreement. The
proposed 64 MiB cap is root-funded only: writing refuses without a funding receipt SHA.
Across-window spread is anatomical, not epistemic, and is not used as committee confidence.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rsna.competitive import teacher_evidence as te
from rsna.competitive.public_input import canonical_sha256, roster_sha256
from rsna.schema import TARGETS

KIND = "public_teacher_committee_evidence_v1"
SOURCE = "pilkwang/rsna-knee-baseline-v1/v15"
MEMBERS = 20
MAX_EVIDENCE_BYTES = 64 << 20
FUNDING = "proposed_root_funding_required"
ARRAYS = (
    "study_uid",
    "member_window_logits",
    "member_probabilities",
    "committee_probabilities",
    "between_model_std",
    "slot_mask",
    "input_sha256",
)
BINDING_KEYS = {
    "kind",
    "source",
    "members",
    "committee_receipt_sha256",
    "member_manifest_sha256",
    "prefix",
    "forward",
    "source_sha256",
    "renderer_node_text_sha256",
    "teacher_node_text_sha256",
    "teacher_review_sha256",
    "pixel_group_sha256",
    "backbone_config_sha256",
    "runtime",
    "fitting_roster_sha256",
    "requested_roster_sha256",
    "requested_studies",
    "input",
    "targets",
    "windows",
    "storage",
}
MEMBER_KEYS = {"id", "file", "checkpoint_sha256", "config", "config_sha256"}
CLAIMS = {**te.CLAIMS, "committee_members_independent": False, "focal_suppression": "unproven"}


class CommitteeEvidenceError(te.EvidenceError):
    """Committee evidence differs from its binding, member list, funding or array contract."""


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def validate_binding(binding: object) -> dict:
    if (
        not isinstance(binding, dict)
        or set(binding) != BINDING_KEYS
        or binding["kind"] != KIND
        or binding["source"] != SOURCE
        or binding["targets"] != list(TARGETS)
        or binding["windows"] != te.WINDOWS
        or binding["fitting_roster_sha256"] != te.FITTING_ROSTER_SHA256
        or binding["requested_roster_sha256"] != roster_sha256(binding["requested_studies"])
        or binding["forward"] not in ("shared_prefix_bitwise_checked", "full_graph")
        or not isinstance(binding["members"], list)
        or len(binding["members"]) != MEMBERS
        or not _sha(binding["committee_receipt_sha256"])
        or not isinstance(binding["runtime"], dict)
        or not isinstance(binding["input"], dict)
        or not isinstance(binding["prefix"], dict)
        or not isinstance(binding["storage"], dict)
        or binding["storage"].get("cap_bytes") != MAX_EVIDENCE_BYTES
    ):
        raise CommitteeEvidenceError("Invalid committee evidence binding")
    ids = [m.get("id") if isinstance(m, dict) else None for m in binding["members"]]
    for member in binding["members"]:
        if (
            not isinstance(member, dict)
            or set(member) != MEMBER_KEYS
            or not _sha(member["checkpoint_sha256"])
            or member["config_sha256"] != canonical_sha256(member["config"])
        ):
            raise CommitteeEvidenceError("Committee member identity is incomplete")
    if len(set(ids)) != MEMBERS:
        raise CommitteeEvidenceError("Committee member identities are duplicated")
    return binding


def committee_statistics(member_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniform member mean and population std across members (float64, stored float32)."""
    values = np.asarray(member_probabilities, dtype=np.float64)
    return values.mean(1).astype(np.float32), values.std(1).astype(np.float32)


def _arrays(uids: Sequence[str], rows: Sequence[dict]) -> dict[str, np.ndarray]:
    if len(uids) != len(rows) or not rows:
        raise CommitteeEvidenceError("One committee row per requested study is required")
    members = np.stack([r["member_probabilities"] for r in rows]).astype(np.float32)
    mean, std = committee_statistics(members)
    arrays = {
        "study_uid": np.asarray(list(uids), dtype=np.str_),
        "member_window_logits": np.stack([r["member_window_logits"] for r in rows]).astype(
            np.float32
        ),
        "member_probabilities": members,
        "committee_probabilities": mean,
        "between_model_std": std,
        "slot_mask": np.stack([r["slot_mask"] for r in rows]).astype(np.bool_),
        "input_sha256": np.asarray([str(r["input_sha256"]) for r in rows], dtype=np.str_),
    }
    _check_arrays(arrays, list(uids))
    return arrays


def _check_arrays(arrays: dict[str, np.ndarray], uids: list[str]) -> None:
    n, m, t = len(uids), MEMBERS, len(TARGETS)
    logits, members = arrays["member_window_logits"], arrays["member_probabilities"]
    if (
        set(arrays) != set(ARRAYS)
        or arrays["study_uid"].tolist() != uids
        or logits.dtype != np.float32
        or logits.shape != (n, m, te.WINDOWS, t)
        or members.dtype != np.float32
        or members.shape != (n, m, t)
        or arrays["slot_mask"].dtype != np.bool_
        or arrays["slot_mask"].shape != (n, te.SLOTS)
        or not arrays["slot_mask"].any(1).all()
        or arrays["input_sha256"].shape != (n,)
        or any(len(h) != 64 for h in arrays["input_sha256"].tolist())
    ):
        raise CommitteeEvidenceError("Committee evidence arrays differ from the contract")
    if not np.isfinite(logits).all() or not np.isfinite(members).all():
        raise CommitteeEvidenceError("Non-finite committee evidence")
    if ((members < 0) | (members > 1)).any():
        raise CommitteeEvidenceError("Committee probabilities outside [0,1]")
    mean, std = committee_statistics(members)
    if not (
        np.array_equal(arrays["committee_probabilities"], mean)
        and np.array_equal(arrays["between_model_std"], std)
    ):
        raise CommitteeEvidenceError("Committee mean/disagreement differ from member values")


def summary(arrays: dict[str, np.ndarray]) -> dict:
    """Between-model disagreement counts only; says nothing about ground truth."""
    weight = committee_confidence(
        torch.from_numpy(arrays["committee_probabilities"]),
        torch.from_numpy(arrays["between_model_std"]),
    )
    return {
        "studies": int(weight.shape[0]),
        "members": MEMBERS,
        "uncertain_cells_confidence_below_0.5": {
            t: int((weight[:, i] < 0.5).sum()) for i, t in enumerate(TARGETS)
        },
        "zero_weight_cells": int((weight == 0).sum()),
    }


def _serialize(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    savez: Any = np.savez
    savez(buffer, **{name: arrays[name] for name in ARRAYS})
    return buffer.getvalue()


def write_evidence(out: Path, binding: dict, rows: Sequence[dict]) -> dict:
    """Create the immutable committee directory once, only under a root funding receipt."""
    validate_binding(binding)
    if not _sha(binding["storage"].get("funding_receipt_sha256")):
        raise CommitteeEvidenceError("The 64 MiB committee cap is proposed, not funded")
    arrays = _arrays(binding["requested_studies"], rows)
    payload = _serialize(arrays)
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise CommitteeEvidenceError("Committee evidence exceeds its storage cap")
    out = Path(out)
    out.mkdir(parents=False, exist_ok=False)
    te._write_exclusive(out / te.EVIDENCE_FILE, payload)
    receipt = {
        "kind": KIND,
        "status": "collected",
        "binding": binding,
        "binding_sha256": canonical_sha256(binding),
        "evidence": {
            "file": te.EVIDENCE_FILE,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "summary": summary(arrays),
        "claims": CLAIMS,
    }
    te._write_exclusive(
        out / te.RECEIPT_FILE, json.dumps(receipt, indent=2, allow_nan=False).encode()
    )
    return receipt


def load_evidence(out: Path, binding: dict | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    out = Path(out)
    if not (out / te.RECEIPT_FILE).is_file():
        raise CommitteeEvidenceError("Committee evidence is incomplete; preserved for inspection")
    receipt = json.loads((out / te.RECEIPT_FILE).read_text())
    if (
        not isinstance(receipt, dict)
        or receipt.get("kind") != KIND
        or receipt.get("claims") != CLAIMS
        or receipt.get("binding_sha256") != canonical_sha256(receipt.get("binding"))
        or (binding is not None and receipt["binding"] != binding)
    ):
        raise CommitteeEvidenceError("Existing committee evidence has a different binding")
    validate_binding(receipt["binding"])
    payload = (out / te.EVIDENCE_FILE).read_bytes()
    record = receipt["evidence"]
    if (
        record.get("file") != te.EVIDENCE_FILE
        or len(payload) != record.get("bytes")
        or len(payload) > MAX_EVIDENCE_BYTES
        or hashlib.sha256(payload).hexdigest() != record.get("sha256")
    ):
        raise CommitteeEvidenceError("Committee evidence bytes differ from the receipt")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAYS):
            raise CommitteeEvidenceError("Committee evidence arrays differ from the contract")
        arrays = {name: archive[name] for name in ARRAYS}
    _check_arrays(arrays, list(receipt["binding"]["requested_studies"]))
    return receipt, arrays


def reconcile(out: Path, binding: dict) -> dict | None:
    """Existing identical evidence -> its receipt; absent -> None; anything else refuses."""
    if not Path(out).exists():
        return None
    receipt, _ = load_evidence(out, validate_binding(binding))
    return receipt


def committee_confidence(mean: Tensor, between_std: Tensor) -> Tensor:
    """Bounded [0,1]: committee decisiveness minus twice the between-model std."""
    if mean.shape != between_std.shape or mean.ndim != 2:
        raise CommitteeEvidenceError("Aligned committee mean and disagreement required")
    if not torch.isfinite(mean).all() or not torch.isfinite(between_std).all():
        raise CommitteeEvidenceError("Non-finite committee evidence")
    return ((2 * mean.float() - 1).abs() - 2 * between_std.float()).clamp(0, 1)


def committee_targets(
    receipt: dict, arrays: dict[str, np.ndarray], study_uids: Sequence[str]
) -> tuple[Tensor, Tensor]:
    """Committee-mean soft targets and weights for a fitting batch; absent studies weigh 0."""
    if receipt["binding"]["fitting_roster_sha256"] != te.FITTING_ROSTER_SHA256 or (
        receipt["claims"] != CLAIMS
    ):
        raise CommitteeEvidenceError("Committee evidence is not bound to the fitting roster")
    index = {uid: i for i, uid in enumerate(arrays["study_uid"].tolist())}
    mean = torch.from_numpy(arrays["committee_probabilities"])
    weights = committee_confidence(mean, torch.from_numpy(arrays["between_model_std"]))
    soft = torch.zeros(len(study_uids), len(TARGETS))
    weight = torch.zeros(len(study_uids), len(TARGETS))
    for row, uid in enumerate(study_uids):
        if uid in index:
            soft[row], weight[row] = mean[index[uid]], weights[index[uid]]
    return soft, weight
