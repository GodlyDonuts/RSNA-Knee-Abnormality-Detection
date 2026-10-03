"""Experimental signed local evidence; study labels never become patch labels."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from rsna.schema import TARGETS


def validate_witness(policy: dict) -> None:
    if not isinstance(policy, dict) or set(policy) != {"patches", "slices", "seed", "weight"}:
        raise ValueError("Witness policy requires exact patches/slices/seed/weight fields")
    for name in ("patches", "slices", "seed"):
        if type(policy[name]) is not int or policy[name] < (0 if name == "seed" else 1):
            raise ValueError("Invalid witness integer policy")
    if policy["seed"] >= 2**63:
        raise ValueError("Witness seed exceeds independent generator range")
    weight = policy["weight"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight not in (0, 1):
        raise ValueError("Initial witness experiment accepts only matched zero/one arms")


def pool_witness_evidence(scores: Tensor, slice_mask: Tensor, patches: int, slices: int) -> Tensor:
    """Top patch evidence, then top slice evidence, then strongest genuine series.

    No noisy-OR/product over bag length, positive pseudo-labels or padded tokens.
    Fixed top-k reduces background dilution but may amplify artifacts; accuracy
    and cross-acquisition consistency need an actual matched experiment.
    """
    if scores.ndim != 5 or scores.shape[:3] != slice_mask.shape:
        raise ValueError("Witness scores must be [B,S,K,P,T] with an exact slice mask")
    if slice_mask.dtype != torch.bool or not slice_mask.any(dim=(1, 2)).all():
        raise ValueError("Each witness study requires genuine boolean-masked slices")
    if type(patches) is not int or type(slices) is not int or min(patches, slices) < 1:
        raise ValueError("Witness budgets must be positive integers")
    if scores.shape[3] < 1 or not torch.isfinite(scores[slice_mask]).all():
        raise ValueError("Genuine witness tokens must be finite and nonempty")
    # Replace padding before any arithmetic; even NaN padding is irrelevant.
    safe = scores.masked_fill(~slice_mask[..., None, None], 0)
    per_slice = safe.topk(min(patches, scores.shape[3]), dim=3).values.mean(dim=3)
    per_slice = per_slice.masked_fill(~slice_mask[..., None], -torch.inf)
    count = slice_mask.sum(dim=2).clamp(max=slices)
    selected = per_slice.topk(min(slices, scores.shape[2]), dim=2).values
    rank = torch.arange(selected.shape[2], device=scores.device)[None, None, :, None]
    selected = selected.masked_fill(rank >= count[..., None, None], 0)
    per_series = selected.sum(dim=2) / count.clamp_min(1)[..., None]
    return per_series.masked_fill(~slice_mask.any(dim=2)[..., None], -torch.inf).amax(dim=1)


class PatchWitness(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, policy: dict):
        super().__init__()
        validate_witness(policy)
        self.policy = dict(policy)
        # A new branch must not move the shared classifier/data-order RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(policy["seed"])
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
            self.score = nn.Linear(hidden_dim, len(TARGETS))
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)

    def forward(self, patches: Tensor, slice_mask: Tensor) -> Tensor:
        if patches.ndim != 4 or patches.shape[0] != int(slice_mask.sum()):
            raise ValueError("Witness patches must correspond to genuine slices only")
        local = self.score(self.projection(patches)).flatten(1, 2)
        dense = local.new_zeros((*slice_mask.shape, local.shape[1], len(TARGETS)))
        dense[slice_mask] = local
        return (
            pool_witness_evidence(dense, slice_mask, self.policy["patches"], self.policy["slices"])
            * self.policy["weight"]
        )
