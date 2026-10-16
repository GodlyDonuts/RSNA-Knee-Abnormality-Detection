"""Bounded, zero-start sequence context over a licensed frozen MRI encoder.

This module contains no external implementation, weights or clinical records.
The parent has seen our internal cohort: this is an adaptation for competition
testing, never independent OOF. Slot/depth codes are not physical registration.
"""

from __future__ import annotations

import math

import torch
from torch import nn

VERSION = "licensed_parent_bounded_sequence_context_v1"
SIZES = (26, 22, 18, 12, 18)


def window_metadata(centers, *, mirrored=False):
    """Encode original selected-slot position, reflecting sagittal depth for TTA.

    Parent mirror reverses sagittal donor channels and coronal/axial width. A
    positional readout must also reflect sagittal depth; the old MIL was unordered.
    Output is normalized sampling position, not patient millimeters or anatomy.
    """
    if (
        not isinstance(centers, torch.Tensor)
        or centers.ndim != 1
        or centers.dtype != torch.long
        or not len(centers)
        or (centers < 1).any()
        or (centers > 94).any()
        or type(mirrored) is not bool
    ):
        raise ValueError("Integer original 1..94 centers and explicit mirror flag required")
    bounds = centers.new_tensor([26, 48, 66, 78])
    slots = torch.bucketize(centers.contiguous(), bounds, right=True)
    starts = centers.new_tensor([0, 26, 48, 66, 78])[slots]
    lengths = centers.new_tensor(SIZES)[slots]
    depth = (centers - starts).float() / (lengths - 1)
    if mirrored:
        depth = torch.where(slots < 2, 1 - depth, depth)
    return slots, depth


class ContextResidual(nn.Module):
    """Finding queries pool contextualized windows; final correction starts at zero.

    The frozen parent logits are supplied, not recalculated or fit here. The
    all-token mask may preserve the parent's zero-image centers. `present` marks
    actual center pixels without silently changing the parent's original bag.
    """

    def __init__(
        self, embedding_dim=1024, width=128, targets=12, layers=2, limit=0.5, seed=20261016
    ):
        super().__init__()
        if (
            type(embedding_dim) is not int
            or embedding_dim < 1
            or type(width) is not int
            or width < 4
            or width % 4
            or type(targets) is not int
            or targets < 1
            or type(layers) is not int
            or not 0 <= layers <= 2
            or type(seed) is not int
            or not 0 <= seed < 2**63
            or type(limit) not in (int, float)
            or not math.isfinite(limit)
            or not 0 < limit <= 0.5
        ):
            raise ValueError("Positive dimensions, up to two layers and bounded limit required")
        self.embedding_dim, self.targets, self.limit = embedding_dim, targets, float(limit)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.project = nn.Sequential(
                nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, width), nn.GELU()
            )
            self.slots = nn.Embedding(5, width)
            self.position = nn.Linear(4, width, bias=False)
            self.present = nn.Embedding(2, width)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        width, 4, width * 2, dropout=0, batch_first=True, norm_first=True
                    )
                    for _ in range(layers)
                ]
            )
            self.norm = nn.LayerNorm(width)
            self.queries = nn.Parameter(torch.randn(targets, width) / math.sqrt(width))
            self.output = nn.Parameter(torch.zeros(targets, width))
            self.bias = nn.Parameter(torch.zeros(targets))

    def forward(self, features, parent_logits, slots, depth, present, keep):
        if features.ndim != 3:
            raise ValueError("B,K,D frozen encoder features required")
        b, k, d = features.shape
        if (
            min(b, k) < 1
            or d != self.embedding_dim
            or features.dtype != torch.float32
            or parent_logits.shape != (b, self.targets)
            or parent_logits.dtype != torch.float32
            or any(x.shape != (b, k) for x in (slots, depth, present, keep))
            or slots.dtype != torch.long
            or depth.dtype != torch.float32
            or present.dtype != torch.bool
            or keep.dtype != torch.bool
            or not keep.any(dim=1).all()
            or not torch.isfinite(parent_logits).all()
            or not torch.isfinite(features[keep]).all()
            or not torch.isfinite(depth[keep]).all()
            or ((depth[keep] < 0) | (depth[keep] > 1)).any()
            or ((slots[keep] < 0) | (slots[keep] > 4)).any()
            or any(
                x.device != features.device for x in (parent_logits, slots, depth, present, keep)
            )
        ):
            raise ValueError(
                "Aligned finite FP32 data, valid slot/depth and boolean masks required"
            )
        # Padded values are removed before projections, never multiplied by zero
        # after a NaN/Inf linear operation. Real missing centers remain in keep.
        f = features.masked_fill(~keep[..., None], 0)
        r = depth.masked_fill(~keep, 0)
        pos = torch.stack((r, r.square(), (math.pi * r).sin(), (math.pi * r).cos()), -1)
        h = (
            self.project(f)
            + self.slots(slots.masked_fill(~keep, 0))
            + self.position(pos)
            + self.present(present.masked_fill(~keep, False).long())
        )
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=~keep)
        h = self.norm(h).masked_fill(~keep[..., None], 0)
        scores = torch.einsum("bkh,th->bkt", h, self.queries) / math.sqrt(h.shape[-1])
        attention = scores.masked_fill(~keep[..., None], -torch.inf).softmax(dim=1)
        pooled = torch.einsum("bkt,bkh->bth", attention, h)
        delta = self.limit * torch.tanh((pooled * self.output).sum(-1) + self.bias)
        return parent_logits + delta, delta


def anchored_loss(logits, parent_logits, labels, mask, *, source_weight=0.25):
    """Fixed mixture of observed public targets and the unmodified parent teacher.

    Masked/absent supervision cannot become an invented negative. The teacher is
    detached. This is label anchoring, not an assurance of improved accuracy.
    """
    if (
        logits.ndim != 2
        or any(x.shape != logits.shape for x in (parent_logits, labels, mask))
        or mask.dtype != torch.bool
        or not torch.isfinite(logits).all()
        or not torch.isfinite(parent_logits).all()
        or not torch.isfinite(labels[mask]).all()
        or ((labels[mask] < 0) | (labels[mask] > 1)).any()
        or type(source_weight) not in (int, float)
        or not math.isfinite(source_weight)
        or not 0 < source_weight <= 1
    ):
        raise ValueError("Finite aligned source/teacher logits and observed probabilities required")
    teacher = parent_logits.detach().sigmoid()
    observed = torch.where(mask, labels, teacher)
    target = source_weight * observed + (1 - source_weight) * teacher
    return nn.functional.binary_cross_entropy_with_logits(logits, target.detach())


def tta_probabilities(logits):
    """Preserve parent mean-of-two-probabilities, never mean-of-logits TTA."""
    if logits.ndim != 3 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise ValueError("B,2,target finite view logits required")
    return logits.sigmoid().mean(dim=1)
