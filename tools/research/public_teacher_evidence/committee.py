"""Exact shared-prefix forward for the twenty retained Pilkwang v15 committee members.

Root's inspection receipt (`initiative/audits/public-teacher-committee-prefix-20261013-v1`)
reports that all twenty members carry byte-identical DINOv2 embeddings and transformer
blocks 0-5. This module re-proves that on the loaded members, then evaluates each member
through its own original `Model.forward`, `predict_member` and `collect_study`; only the
backbone object is swapped for a view that takes blocks 0-5 from one verified prefix and
blocks 6-11, final layernorm, pooling and head from the member itself. The prefix output is
reused only for a bitwise-identical normalized input under the identical dtype, device,
autocast and grad state, never across studies, and refused if any prefix tensor changed.
Every member keeps its ordinary full graph; the shared view is admitted only if its
original fingerprint and its window logits/probabilities equal the full graph's bitwise.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from tools.research.public_teacher_evidence import teacher
from tools.research.public_teacher_evidence.teacher import TeacherMismatch

MEMBERS = 20
PREFIX_LAYERS = 6
PREFIX_TENSORS = 113
PREFIX_ELEMENTS = 11_404_416
# Root's inspection digest; its canonicalization is root's, so it is bound, not recomputed.
INSPECTED_PREFIX_DIGEST = "f127a88be3d764d063b630173017219002a17d3a1ae26ae915c6c7248fd7d78f"
PREFIX_DIGEST_METHOD = "sha256(name\\0dtype\\0shape\\0raw little-endian bytes), sorted names"


def prefix_names(model: nn.Module, layers: int = PREFIX_LAYERS) -> list[str]:
    """State entries feeding block `layers`: embeddings and encoder blocks 0..layers-1."""
    heads = ("backbone.embeddings.",) + tuple(f"backbone.encoder.layer.{i}." for i in range(layers))
    return sorted(name for name in model.state_dict(keep_vars=True) if name.startswith(heads))


def _bytes(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.device == b.device
        and torch.equal(_bytes(a), _bytes(b))
    )


def prefix_digest(model: nn.Module, layers: int = PREFIX_LAYERS) -> dict:
    state = model.state_dict()
    digest = hashlib.sha256()
    elements = 0
    names = prefix_names(model, layers)
    for name in names:
        value = state[name].detach().cpu().contiguous()
        elements += value.numel()
        digest.update(f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode())
        digest.update(_bytes(value).numpy().tobytes())
    return {
        "layers": layers,
        "tensors": len(names),
        "elements": elements,
        "sha256": digest.hexdigest(),
        "method": PREFIX_DIGEST_METHOD,
    }


def inspected_prefix_digest(model: nn.Module) -> str:
    """Exact canonicalization of the authoritative owning-node inspection receipt."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in prefix_names(model):
        value = state[name].detach().cpu().contiguous()
        digest.update(
            json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        )
        digest.update(_bytes(value).numpy().tobytes())
    return digest.hexdigest()


def _stamp(model: nn.Module, names: Sequence[str]) -> tuple:
    state = model.state_dict(keep_vars=True)
    return tuple((state[n]._version, state[n].data_ptr(), state[n].requires_grad) for n in names)


def _has_hooks(module: nn.Module) -> bool:
    return any(
        getattr(m, attr, None)
        for m in module.modules()
        for attr in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
    )


def _out(value: Any) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


class SharedPrefix(nn.Module):
    """Embeddings + blocks 0..5 of one verified member, cached per identical normalized input.

    Bounded to one study's windows. An entry is reused only when shape, dtype, device,
    layout, autocast and grad state and every byte of the normalized input are equal; the
    whole cache is refused if any prefix tensor of any member was modified in place.
    """

    def __init__(self, members: Sequence[nn.Module], names: Sequence[str], windows: int):
        super().__init__()
        lead = members[0].backbone
        self.embeddings = lead.embeddings
        self.layers = nn.ModuleList(lead.encoder.layer[:PREFIX_LAYERS])
        self._members = list(members)
        self._names = list(names)
        self._stamps = [_stamp(m, names) for m in members]
        lead_state = members[0].state_dict()
        self._snapshot = {n: lead_state[n].detach().clone() for n in names}
        self._windows = windows
        self._entries: list[tuple[tuple, torch.Tensor, torch.Tensor]] = []
        self.computed = 0
        self.reused = 0
        self.eval()

    def clear(self) -> None:
        self._entries.clear()

    def verify_unchanged(self) -> None:
        """Every member's prefix still bitwise equals the verified snapshot (also via .data)."""
        for model in self._members:
            state = model.state_dict()
            if not all(_bitwise_equal(state[n], self._snapshot[n]) for n in self._names):
                self.clear()
                raise TeacherMismatch("A shared prefix tensor changed after verification")

    def _state(self, x: torch.Tensor) -> tuple:
        kind = x.device.type
        autocast = torch.is_autocast_enabled(kind)
        return (
            tuple(x.shape),
            x.dtype,
            x.device,
            x.stride(),
            autocast,
            torch.get_autocast_dtype(kind) if autocast else None,
            torch.is_grad_enabled(),
            torch.is_inference_mode_enabled(),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if any(_stamp(m, self._names) != s for m, s in zip(self._members, self._stamps)):
            self.clear()
            raise TeacherMismatch("A shared prefix tensor changed after verification")
        if torch.is_grad_enabled() or self.training:
            raise TeacherMismatch("Shared prefix is evaluation-only without gradients")
        key = self._state(pixel_values)
        for entry_key, entry_input, hidden in self._entries:
            if entry_key == key and torch.equal(_bytes(entry_input), _bytes(pixel_values)):
                self.reused += 1
                return hidden
        if len(self._entries) >= self._windows:
            raise TeacherMismatch("Shared prefix cache exceeded one study's windows")
        hidden = self.embeddings(pixel_values, bool_masked_pos=None)
        for layer in self.layers:
            hidden = _out(layer(hidden))
        self._entries.append((key, pixel_values.detach().clone(), hidden))
        self.computed += 1
        return hidden


class SuffixBackbone(nn.Module):
    """Member blocks 6..11 and final layernorm on top of the shared prefix output."""

    def __init__(self, prefix: SharedPrefix, backbone: nn.Module):
        super().__init__()
        self.config = backbone.config
        self.prefix = [prefix]  # not a submodule: the prefix is owned by the committee
        self.suffix = nn.ModuleList(backbone.encoder.layer[PREFIX_LAYERS:])
        self.layernorm = backbone.layernorm

    def forward(self, pixel_values: torch.Tensor, **unexpected: Any) -> SimpleNamespace:
        if unexpected:
            raise TeacherMismatch("Shared-prefix backbone accepts only pixel_values")
        hidden = self.prefix[0](pixel_values)
        for layer in self.suffix:
            hidden = _out(layer(hidden))
        return SimpleNamespace(last_hidden_state=self.layernorm(hidden))


def shared_view(model: nn.Module, prefix: SharedPrefix) -> nn.Module:
    """The member's original `Model` object graph with only `backbone` replaced."""
    view = copy.copy(model)
    for attr, value in list(vars(model).items()):
        if isinstance(value, dict):
            fresh = OrderedDict() if "hook" in attr else OrderedDict(value)
            object.__setattr__(view, attr, fresh)
    view._modules["backbone"] = SuffixBackbone(prefix, model.backbone)
    return view.eval()


def verify_prefix(
    members: Sequence[nn.Module], expected: tuple[int, int] = (PREFIX_TENSORS, PREFIX_ELEMENTS)
) -> tuple[list[str], dict]:
    """Prove identical config, normalization buffers and prefix state across all members."""
    if len(members) != MEMBERS:
        raise TeacherMismatch(f"Committee requires exactly {MEMBERS} members")
    lead = members[0]
    names = prefix_names(lead)
    lead_state = lead.state_dict()
    config = lead.backbone.config.to_dict()
    for model in members:
        if type(model) is not type(lead) or type(model.backbone) is not type(lead.backbone):
            raise TeacherMismatch("Committee members differ in graph class")
        if model.backbone.config.to_dict() != config:
            raise TeacherMismatch("Committee members differ in backbone config")
        if model.training or _has_hooks(model.backbone):
            raise TeacherMismatch("Committee member is not a clean evaluation graph")
        if any(p.requires_grad for p in model.parameters()):
            raise TeacherMismatch("Committee member is not frozen")
        state = model.state_dict()
        if prefix_names(model) != names:
            raise TeacherMismatch("Committee members differ in prefix tensor names")
        for name in [*names, "mean", "std"]:
            if not _bitwise_equal(state[name], lead_state[name]):
                raise TeacherMismatch(f"Committee prefix differs at {name}")
    summary = prefix_digest(lead)
    if (summary["tensors"], summary["elements"]) != expected:
        raise TeacherMismatch("Prefix size differs from the inspected six-block prefix")
    if expected == (PREFIX_TENSORS, PREFIX_ELEMENTS):
        actual_inspected = inspected_prefix_digest(lead)
        if actual_inspected != INSPECTED_PREFIX_DIGEST:
            raise TeacherMismatch("Shared prefix differs from the original inspection digest")
        summary["original_inspection_sha256"] = actual_inspected
    return names, summary


class Committee:
    """Twenty verified full graphs plus their shared-prefix views, in member order."""

    def __init__(
        self,
        namespace: dict[str, Any],
        members: Sequence[nn.Module],
        ids: Sequence[str],
        expected: tuple[int, int] = (PREFIX_TENSORS, PREFIX_ELEMENTS),
    ):
        if len(ids) != len(members) or len(set(ids)) != len(ids):
            raise TeacherMismatch("Committee member identities are missing or duplicated")
        names, self.prefix_summary = verify_prefix(members, expected)
        self.namespace = namespace
        self.ids = list(ids)
        self.full = list(members)
        windows = len(namespace["window_starts"](12, namespace["GROUP"]))
        self.prefix = SharedPrefix(members, names, windows)
        self.views = [shared_view(m, self.prefix) for m in members]

    def verify_fingerprints(self, device: torch.device, references: Sequence[torch.Tensor]) -> list:
        """Original fingerprint for each shared view, bitwise equal to its full graph."""
        errors = []
        self.prefix.verify_unchanged()
        for full, view, ref in zip(self.full, self.views, references, strict=True):
            self.prefix.clear()
            img = self.namespace["IMG"]
            shared = self.namespace["fingerprint"](view, device, img)
            if not np.array_equal(shared, self.namespace["fingerprint"](full, device, img)):
                raise TeacherMismatch("Shared-prefix fingerprint differs from the full graph")
            self.namespace["check_fingerprint"](view, device, img, ref, tag="committee: ")
            errors.append(float(np.abs(shared - ref.numpy()).max()))
        self.prefix.clear()
        return errors

    def collect_study(
        self,
        images: torch.Tensor,
        slot_mask: torch.Tensor,
        device: torch.device,
        mode: str = "shared",
        check_full: bool = True,
    ) -> dict[str, np.ndarray]:
        """Per-member original window logits/probabilities, member order, one study."""
        if mode not in ("shared", "full"):
            raise TeacherMismatch("Committee forward mode must be shared or full")
        graphs = self.views if mode == "shared" else self.full
        rows = []
        self.prefix.clear()
        self.prefix.verify_unchanged()
        try:
            for index, graph in enumerate(graphs):
                row = teacher.collect_study(self.namespace, graph, images, slot_mask, device)
                if mode == "shared" and check_full:
                    full = teacher.collect_study(
                        self.namespace, self.full[index], images, slot_mask, device
                    )
                    for key in ("window_logits", "probabilities", "input_sha256"):
                        if not np.array_equal(row[key], full[key]):
                            raise TeacherMismatch(
                                f"Shared-prefix {key} differs from the full graph"
                            )
                rows.append(row)
            self.prefix.verify_unchanged()
        finally:
            self.prefix.clear()
        if len({str(r["input_sha256"]) for r in rows}) != 1:
            raise TeacherMismatch("Committee members saw different inputs")
        return {
            "member_window_logits": np.stack([r["window_logits"] for r in rows]),
            "member_probabilities": np.stack([r["probabilities"] for r in rows]),
            "slot_mask": rows[0]["slot_mask"],
            "input_sha256": rows[0]["input_sha256"],
        }
