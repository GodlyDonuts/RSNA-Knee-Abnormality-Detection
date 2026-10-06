"""Conventional CLS plus spatial-mean readout, with a common zero-start control."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def validate_patch_summary(policy: object) -> dict:
    if (
        not isinstance(policy, dict)
        or set(policy) != {"weight"}
        or type(policy["weight"]) not in (int, float)
        or policy["weight"] not in (0, 1)
    ):
        raise ValueError("Patch-summary comparison requires an explicit zero/one weight")
    return dict(policy)


class PatchMeanProjection(nn.Module):
    """Separate linear block of the concatenated [CLS, mean-patch] projection.

    W_cls CLS + W_patch mean(patches) equals a linear projection of concatenated
    features. W_patch starts at zero, keeping initial predictions and caller RNG
    identical. Control/candidate share every parameter; only the gate differs.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, policy: dict):
        super().__init__()
        self.ratio = validate_patch_summary(policy)["weight"]
        self.weight = nn.Parameter(torch.zeros(hidden_dim, embedding_dim))

    def forward(self, patches: Tensor) -> Tensor:
        if (
            patches.ndim != 4
            or min(patches.shape[1:3]) < 1
            or patches.shape[-1] != self.weight.shape[1]
            or not torch.isfinite(patches).all()
        ):
            raise ValueError("Finite register-excluded spatial patches required")
        return F.linear(patches.mean(dim=(1, 2)), self.weight) * self.ratio
