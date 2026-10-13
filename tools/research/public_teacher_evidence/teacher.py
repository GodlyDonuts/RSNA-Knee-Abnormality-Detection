"""Frozen Pilkwang v15 teacher graph, safe checkpoint loading and logits-only evidence.

The teacher is compiled from reviewed original source segments (teacher_review.json) on
top of the already reviewed renderer namespace, so crop, slots, laterality, triplets,
TTA windows and `predict_member` are the original code. Two original calls are replaced
because they are unsafe here: `build_model` (Hub/file `from_pretrained`) by an explicit
hash-bound DINOv2-small constructor whose weights come only from the checkpoint's strict
complete state, and `torch.load(weights_only=False)` by `weights_only=True` on
hash-verified bytes. The original source exposes no attention or per-slot output; only
per-window `Model.forward` logits (read by a hook) and `predict_member` probabilities are
collected, and the probabilities must be reproduced bitwise from those logits.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rsna.competitive import public_input
from rsna.schema import TARGETS
from tools.research.public_dinov2_input_reference import provenance, reference

REVIEW_PATH = Path(__file__).with_name("teacher_review.json")
TEACHER_NODES = (
    "POOL_PARTS",
    "SLOT_PRIOR_TABLE",
    "SLOT_PRIOR_STRENGTH",
    "SEED",
    "SlotHead",
    "Model",
    "FINGERPRINT_TOL",
    "fingerprint",
    "check_fingerprint",
)
# DINOv2-small, the only variant `m_e5427d6c21.pt` was fitted with (fingerprint receipt).
BACKBONE_FIELDS = {
    "model_type": "dinov2",
    "hidden_size": 384,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "patch_size": 14,
    "num_channels": 3,
}
WINDOWS = 10


class TeacherMismatch(ValueError):
    """The teacher, its source, state or inputs differ from the reviewed identity."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_teacher_review(path: Path = REVIEW_PATH) -> dict:
    review = json.loads(Path(path).read_text())
    renderer = provenance.load_review()
    if (
        review.get("source_sha256") != provenance.SOURCE_SHA256
        or review.get("status") != "reviewed"
        or review.get("nodes") != list(TEACHER_NODES)
        or review.get("node_text_format") != "original_source_segments_utf8"
        or review.get("renderer_node_text_sha256") != renderer["node_text_sha256"]
        or review.get("renderer_node_text_sha256") != public_input.NODE_TEXT_SHA256
    ):
        raise TeacherMismatch("Teacher review is not bound to the reviewed source/renderer")
    return review


def load_namespace(
    raw: bytes, pixel_group: dict, review: dict | None = None, messages: list[str] | None = None
) -> tuple[dict[str, Any], dict]:
    """Exact source bytes -> reviewed renderer + teacher namespace; no other cell runs."""
    review = load_teacher_review() if review is None else review
    if public_input.canonical_sha256(pixel_group) != public_input.PIXEL_GROUP_SHA256:
        raise TeacherMismatch("Member pixel group differs from the admitted public group")
    renderer, receipt = reference.load(raw, messages)
    if receipt["node_text_sha256"] != review["renderer_node_text_sha256"]:
        raise TeacherMismatch("Renderer node text differs from the teacher review binding")
    reference.configure(renderer, pixel_group)
    exceptions = review.get("reviewed_exceptions", {})
    text, digest = provenance.extract_nodes(
        provenance.verify_source(raw), list(review["nodes"]), exceptions
    )
    if digest != review["node_text_sha256"]:
        raise TeacherMismatch("Teacher node text digest changed")
    preamble = {k: v for k, v in renderer.items() if k not in {"__builtins__", "__name__"}}
    namespace = provenance.compile_nodes(text, preamble, exceptions)
    if list(namespace["TARGETS"]) != list(TARGETS):
        raise TeacherMismatch("Original teacher target order differs from rsna.schema.TARGETS")
    if (namespace["IMG"], namespace["N_SLOT"], namespace["GROUP"]) != (336, 6, 3):
        raise TeacherMismatch("Teacher namespace is not the admitted 336px/6-slot/3-slice graph")
    return namespace, {
        **receipt,
        "teacher_nodes": list(review["nodes"]),
        "teacher_node_text_sha256": digest,
        "teacher_review_sha256": sha256(Path(REVIEW_PATH).read_bytes()),
    }


def backbone_config(raw: bytes, expected_sha256: str) -> dict:
    if sha256(raw) != expected_sha256:
        raise TeacherMismatch("Backbone config bytes differ from their bound hash")
    config = json.loads(raw)
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in BACKBONE_FIELDS.items()):
        raise TeacherMismatch("Backbone config is not the DINOv2-small architecture")
    return config


def dinov2_backbone(config: dict) -> nn.Module:
    """Random-initialized architecture only; never the Hub, remote code or from_pretrained."""
    from transformers import Dinov2Config, Dinov2Model

    return Dinov2Model(Dinov2Config(**config))


def load_checkpoint(raw: bytes, expected_sha256: str = provenance.CHECKPOINT_SHA256) -> dict:
    """Weights-only load of already hash-verified bytes; no pickle code can run."""
    if sha256(raw) != expected_sha256:
        raise TeacherMismatch("Checkpoint bytes differ from the audited public member")
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("model"), dict)
        or not checkpoint["model"]
        or "fingerprint" not in checkpoint
    ):
        raise TeacherMismatch("Checkpoint lacks the original model state or fingerprint")
    return checkpoint


def build_teacher(
    namespace: dict[str, Any],
    member_config: dict,
    checkpoint: dict,
    backbone: nn.Module,
    device: torch.device,
) -> nn.Module:
    """Original `Model`, strict complete state, frozen, then the original fingerprint check."""
    pool = member_config.get("pool", "cls_mean")
    if member_config.get("variant") != "small" or pool not in namespace["POOL_PARTS"]:
        raise TeacherMismatch("Member config is not the reviewed DINOv2-small pooling")
    dim = int(backbone.config.hidden_size)
    model = namespace["Model"](backbone, dim, pool=pool, prior=bool(member_config.get("prior")))
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise TeacherMismatch("Checkpoint state does not strictly match the teacher") from error
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    namespace["check_fingerprint"](
        model, device, namespace["IMG"], checkpoint["fingerprint"], tag="teacher: "
    )
    return model


def collect_study(
    namespace: dict[str, Any],
    model: nn.Module,
    images: torch.Tensor,
    slot_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """One source-rendered study through the original `predict_member`, without gradients."""
    if (
        images.dtype != torch.uint8
        or tuple(images.shape) != (6, 12, 336, 336)
        or slot_mask.dtype != torch.bool
        or tuple(slot_mask.shape) != (6,)
    ):
        raise TeacherMismatch("Exact source-rendered uint8 study row required")
    if not slot_mask.any():
        raise TeacherMismatch("No acquired source MRI slot")
    cache = np.ascontiguousarray(images.numpy()[None])
    mask = slot_mask.numpy().astype(np.float32)[None]
    captured: list[torch.Tensor] = []
    hook: Callable[..., None] = lambda _m, _i, out: captured.append(out.detach())  # noqa: E731
    handle = model.register_forward_hook(hook)
    try:
        with torch.no_grad():
            predicted = namespace["predict_member"](
                model, cache, mask, [0], device, namespace["IMG"]
            )
    finally:
        handle.remove()
    starts = list(namespace["window_starts"](cache.shape[2], namespace["GROUP"]))
    if len(starts) != WINDOWS or len(captured) != WINDOWS:
        raise TeacherMismatch("Original TTA windows differ from the ten admitted windows")
    if any(z.requires_grad for z in captured):
        raise TeacherMismatch("Teacher forward recorded gradients")
    total: torch.Tensor | None = None
    for z in captured:
        value = torch.sigmoid(z.float())
        total = value if total is None else total + value
    assert total is not None
    if not np.array_equal((total / WINDOWS).cpu().numpy(), predicted):
        raise TeacherMismatch("Window logits do not reproduce the original probabilities")
    logits = torch.cat([z.float() for z in captured]).cpu().numpy().astype(np.float32)
    probabilities = np.asarray(predicted[0], dtype=np.float32)
    if (
        logits.shape != (WINDOWS, len(TARGETS))
        or probabilities.shape != (len(TARGETS),)
        or not np.isfinite(logits).all()
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise TeacherMismatch("Non-finite or invalid teacher output")
    return {
        "window_logits": logits,
        "probabilities": probabilities,
        "slot_mask": slot_mask.numpy().copy(),
        "input_sha256": np.asarray(sha256(cache.tobytes())),
    }
