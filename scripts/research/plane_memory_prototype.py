"""Isolated query-memory prototype; no active trainer/model integration."""

from __future__ import annotations

import math

import torch
from torch import nn


class PlaneMemory(nn.Module):
    """Pool each sequence, then let target latents exchange evidence across sequences.

    This models co-occurring evidence, not physical registration across unknown
    FrameOfReference identifiers. Encoder images/weights are outside this module.
    """

    def __init__(self, embedding_dim=1024, hidden_dim=128, targets=12, seed=20261006):
        super().__init__()
        if min(embedding_dim, hidden_dim, targets) < 1 or hidden_dim % 4:
            raise ValueError("Positive dimensions and four-head divisible hidden width required")
        self.dim, self.targets = hidden_dim, targets
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.project = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
            self.queries = nn.Parameter(torch.randn(targets, hidden_dim) / math.sqrt(hidden_dim))
            self.plane_embedding = nn.Embedding(4, hidden_dim)
            self.depth_embedding = nn.Linear(2, hidden_dim, bias=False)
            self.cross_sequence = nn.MultiheadAttention(hidden_dim, 4, batch_first=True, dropout=0)
            self.norm = nn.LayerNorm(hidden_dim)
            self.score = nn.Parameter(torch.zeros(targets, hidden_dim))
            self.bias = nn.Parameter(torch.zeros(targets))

    def evidence(self, tokens, mask, planes, positions):
        if tokens.ndim != 5 or tokens.shape[:3] != mask.shape or mask.dtype != torch.bool:
            raise ValueError("Exact B,S,K,P,D tokens and boolean genuine-slice masks required")
        b, s, k, p, _ = tokens.shape
        if (
            min(b, s, k, p) < 1
            or planes.shape != (b, s)
            or positions.shape != (b, s, k, 2)
            or not mask.any(dim=(1, 2)).all()
        ):
            raise ValueError("Aligned sequence/depth metadata and nonempty studies required")
        valid = mask.any(dim=2)
        if planes.dtype != torch.long or (planes[valid] < 0).any() or (planes[valid] > 3).any():
            raise ValueError("Known three-plane codes or explicit unknown3 required")
        if not torch.isfinite(tokens[mask]).all() or not torch.isfinite(positions[mask]).all():
            raise ValueError("Genuine embeddings and positions must be finite")
        # Padding is removed before all linear projections and numerical reductions.
        features = self.project(tokens.masked_fill(~mask[..., None, None], 0))
        features = (
            features
            + self.depth_embedding(positions.masked_fill(~mask[..., None], 0))[:, :, :, None, :]
        )
        scores = torch.einsum("bskph,th->bsktp", features, self.queries) / math.sqrt(self.dim)
        scores = scores.permute(0, 1, 3, 2, 4).flatten(-2).float()
        keep = mask[:, :, :, None].expand(b, s, k, p).flatten(-2)
        safe = scores.masked_fill(~keep[:, :, None, :], -torch.inf)
        safe = torch.where(valid[:, :, None, None], safe, torch.zeros_like(safe))
        weights = (
            torch.softmax(safe, dim=-1).to(features.dtype).masked_fill(~keep[:, :, None, :], 0)
        )
        sequence = torch.einsum("bstn,bsnh->bsth", weights, features.flatten(2, 3))
        sequence = sequence + self.plane_embedding(planes.masked_fill(~valid, 3))[:, :, None, :]
        sequence = sequence.permute(0, 2, 1, 3).reshape(b * self.targets, s, self.dim)
        query = self.queries[None].expand(b, -1, -1).reshape(b * self.targets, 1, self.dim)
        mixed, _ = self.cross_sequence(
            query,
            sequence,
            sequence,
            key_padding_mask=~valid[:, None, :]
            .expand(b, self.targets, s)
            .reshape(b * self.targets, s),
            need_weights=False,
        )
        return self.norm(query + mixed).reshape(b, self.targets, self.dim)

    def forward(self, tokens, mask, planes, positions):
        latent = self.evidence(tokens, mask, planes, positions)
        return torch.einsum("bth,th->bt", latent, self.score) + self.bias
