"""Masked neighboring-slice messages within each MRI series."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class NeighborSliceContext(nn.Module):
    """Mix previous/current/next features without crossing a series boundary.

    Positions retain the loader's signed distance/100 mm and known flag. Unknown
    geometry uses ordinal order only; it never fabricates physical distances.
    A zero-initialized output starts as an exact identity residual.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("Context width must be a positive integer")
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.geometry = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features: Tensor, mask: Tensor, positions: Tensor) -> Tensor:
        if features.ndim != 4 or mask.shape != features.shape[:3]:
            raise ValueError("Context expects [B,S,K,H] features and [B,S,K] masks")
        if positions.shape != (*mask.shape, 2) or mask.dtype != torch.bool:
            raise ValueError("Context needs two position fields and boolean masks")
        if min(features.shape) < 1 or features.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError("Invalid context dimensions")
        if positions.device != features.device or mask.device != features.device:
            raise ValueError("Context inputs must share a device")
        if not torch.isfinite(features[mask]).all() or not torch.isfinite(positions[mask]).all():
            raise ValueError("Valid slice context must be finite")
        flags = positions[mask][:, 1]
        if not ((flags == 0) | (flags == 1)).all():
            raise ValueError("Geometry flags must be binary")
        if (positions[mask][flags == 0, 0] != 0).any():
            raise ValueError("Unknown geometry cannot have a physical distance")
        paired = mask[:, :, 1:] & mask[:, :, :-1]
        known_pair = paired & positions[:, :, 1:, 1].bool() & positions[:, :, :-1, 1].bool()
        gaps = positions[:, :, 1:, 0] - positions[:, :, :-1, 0]
        if (gaps[known_pair] <= 0).any():
            raise ValueError("Known neighboring slice positions must be strictly ordered")
        clean = torch.where(mask[..., None], features, torch.zeros_like(features))
        physical = torch.where(mask[..., None], positions, torch.zeros_like(positions))
        normalized = self.norm(clean)
        empty = normalized.new_zeros((*normalized.shape[:2], 1, normalized.shape[-1]))
        previous = torch.cat((empty, normalized[:, :, :-1]), dim=2)
        following = torch.cat((normalized[:, :, 1:], empty), dim=2)
        neighbors = torch.stack((previous, normalized, following), dim=3)
        absent = torch.zeros_like(mask[:, :, :1])
        neighbor_mask = (
            torch.stack(
                (
                    torch.cat((absent, mask[:, :, :-1]), dim=2),
                    mask,
                    torch.cat((mask[:, :, 1:], absent), dim=2),
                ),
                dim=3,
            )
            & mask[..., None]
        )
        no_position = torch.zeros_like(physical[:, :, :1])
        neighbor_positions = torch.stack(
            (
                torch.cat((no_position, physical[:, :, :-1]), dim=2),
                physical,
                torch.cat((physical[:, :, 1:], no_position), dim=2),
            ),
            dim=3,
        )
        known = neighbor_positions[..., 1] * physical[..., None, 1]
        delta = (neighbor_positions[..., 0] - physical[..., None, 0]) * known
        ordinal = delta.new_tensor((-1, 0, 1)).expand_as(delta)
        geometry = torch.stack((delta, known, ordinal), dim=-1).to(normalized.dtype)
        scores = (self.query(normalized)[..., None, :] * self.key(neighbors)).sum(-1) / math.sqrt(
            features.shape[-1]
        ) + self.geometry(geometry).squeeze(-1)
        scores = scores.masked_fill(~neighbor_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=3) * neighbor_mask.to(scores.dtype)
        weights = weights / weights.sum(3, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
        message = (weights[..., None] * self.value(neighbors)).sum(3)
        mixed = clean + self.output(message).to(clean.dtype)
        return torch.where(mask[..., None], mixed, torch.zeros_like(mixed))
