"""Isolated geometry-supervision prototype; not enabled in production training.

Predict unsigned physical slice separation from raw encoder features, before
position/protocol embeddings. Geometry supplies targets only. No disease label
is assigned to an individual slice, and no feature-equality loss is imposed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SliceRelations:
    indices: Tensor  # N x 4: study, series, first slice, second slice
    targets: Tensor  # log1p(unsigned millimeters) / log1p(100)
    weights: Tensor  # equal contribution per represented study
    shape: tuple[int, int, int]


def slice_relations(
    positions: Tensor,
    slice_mask: Tensor,
    series_mask: Tensor,
    sample_mask: Tensor,
    *,
    seed: int,
    max_pairs: int = 32,
    shuffle_targets: bool = False,
) -> SliceRelations:
    """Sample within-series pairs without consuming the trainer's random stream.

    Positions follow data._positions: centered millimeters / 100 and a binary
    physical-geometry flag. Ordinal fallbacks, padding and distributed tail
    duplicates cannot generate supervision. The caller must admit training
    membership and derive seed from the persisted epoch/batch identity.
    """
    if positions.ndim != 4 or positions.shape[-1] != 2 or min(positions.shape) < 1:
        raise ValueError("Expected nonempty B,S,K,2 positions")
    shape = (positions.shape[0], positions.shape[1], positions.shape[2])
    if (
        slice_mask.shape != shape
        or series_mask.shape != shape[:2]
        or sample_mask.shape != shape[:1]
        or any(mask.dtype != torch.bool for mask in (slice_mask, series_mask, sample_mask))
    ):
        raise ValueError("Relation masks must be boolean with matching dimensions")
    if (
        type(seed) is not int
        or not 0 <= seed < 2**63
        or type(max_pairs) is not int
        or max_pairs < 1
    ):
        raise ValueError("Invalid relation sampling budget or seed")
    if not positions.is_floating_point() or type(shuffle_targets) is not bool:
        raise ValueError("Floating geometry and boolean shuffle policy required")
    valid = (
        slice_mask.detach().cpu()
        & series_mask.detach().cpu()[..., None]
        & sample_mask.detach().cpu()[:, None, None]
    )
    physical = positions.detach().to(device="cpu", dtype=torch.float64)
    values = physical[valid]
    if not torch.isfinite(values).all() or not ((values[:, 1] == 0) | (values[:, 1] == 1)).all():
        raise ValueError("Valid geometry must be finite with binary flags")
    if (values[values[:, 1] == 0, 0] != 0).any():
        raise ValueError("Unknown physical geometry must have zero coordinates")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    target_generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x52454C4154494F4E)
    rows: list[list[int]] = []
    targets: list[float] = []
    for study in range(shape[0]):
        for series in range(shape[1]):
            known = torch.where(valid[study, series] & (physical[study, series, :, 1] == 1))[0]
            if len(known) < 2:
                continue
            coords = physical[study, series, known, 0]
            gaps = coords.diff()
            if not ((gaps > 0).all() or (gaps < 0).all()):
                raise ValueError("Known slice positions must be strictly monotone")
            pairs = torch.combinations(known, r=2)
            if len(pairs) > max_pairs:
                pairs = pairs[torch.randperm(len(pairs), generator=generator)[:max_pairs]]
            separation = (
                physical[study, series, pairs[:, 0], 0] - physical[study, series, pairs[:, 1], 0]
            ).abs() * 100
            labels = torch.log1p(separation) / math.log1p(100)
            if shuffle_targets and len(labels) > 1:
                labels = labels[torch.randperm(len(labels), generator=target_generator)]
            rows.extend([study, series, int(i), int(j)] for i, j in pairs)
            targets.extend(labels.tolist())
    indices = torch.tensor(rows, dtype=torch.long).reshape(-1, 4)
    weights = torch.empty(len(rows), dtype=torch.float32)
    if rows:
        counts = torch.bincount(indices[:, 0], minlength=shape[0])
        studies = (counts > 0).sum()
        weights = 1 / (counts[indices[:, 0]].float() * studies)
    return SliceRelations(indices, torch.tensor(targets, dtype=torch.float32), weights, shape)


class SliceDistanceHead(nn.Module):
    """Symmetric image-feature-only head, discarded for classification inference."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128, *, seed: int = 0):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (embedding_dim, hidden_dim)):
            raise ValueError("Positive feature/head dimensions required")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Invalid auxiliary initialization seed")
        # Auxiliary initialization must not perturb matched classification heads.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(2 * embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
            )
        self.embedding_dim = embedding_dim

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        if first.shape != second.shape or first.ndim != 2 or first.shape[-1] != self.embedding_dim:
            raise ValueError("Expected matching N,D raw encoder features")
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError("Relation features must be finite")
        return self.network(torch.cat(((first - second).abs(), (first + second) * 0.5), -1))[:, 0]

    def loss(self, features: Tensor, relations: SliceRelations) -> Tensor:
        if features.shape != (*relations.shape, self.embedding_dim):
            raise ValueError("Raw encoder feature layout differs from relation layout")
        if not len(relations.indices):
            # Padding may be NaN; an empty selection remains differentiable zero.
            return (
                features.reshape(-1)[:0].sum() + cast(nn.Linear, self.network[0]).weight.sum() * 0
            )
        study, series, first, second = relations.indices.to(features.device).unbind(1)
        prediction = self(features[study, series, first], features[study, series, second])
        loss = F.smooth_l1_loss(
            prediction.float(), relations.targets.to(features.device), reduction="none"
        )
        return (loss * relations.weights.to(features.device)).sum()
