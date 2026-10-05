"""Bound, matched neighbor-channel input policy; no accuracy admission implied."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from rsna.competitive.neighbor_channels import neighbor_delta_channels
from rsna.competitive.neighbor_geometry import matched_neighbor_indices


def neighbor_input_channels(policy: dict | None) -> int:
    if policy is None:
        return 1
    if not isinstance(policy, dict) or set(policy) != {"mode", "strength"}:
        raise ValueError("Exact neighbor mode/strength policy required")
    mode, strength = policy["mode"], policy["strength"]
    if (
        not isinstance(mode, str)
        or mode not in {"gray", "physical", "distant"}
        or isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
        or not 0 <= strength <= 0.5
        or (mode == "gray" and strength != 0)
    ):
        raise ValueError("Invalid bounded neighbor input policy")
    return 1 if strength == 0 else 3


def sample_neighbor_images(
    volume: np.ndarray, metadata: dict, centers: np.ndarray, side: int, policy: dict | None
) -> Tensor:
    """Normalize/resize original slices, then compose; zero follows legacy bytes.

    Donors are original full-volume indices, not adjacent sampled slots. Physical
    and distant arms use the identical donor multiset and common fallback policy.
    No per-slice normalization, spatial alignment, learned interpolation or new
    clinical labels. The dataset separately verifies original bytes/metadata.
    """
    channels = neighbor_input_channels(policy)
    sampled = torch.from_numpy(volume[centers].astype(np.float32))[:, None] / 65535
    gray = F.interpolate(
        sampled, (side, side), mode="bilinear", align_corners=False, antialias=True
    ).clamp_(0, 1)
    if channels == 1:
        return gray
    assert policy is not None
    p, n, dp, dn = matched_neighbor_indices(metadata, centers)
    before, after = (p, n) if policy["mode"] == "physical" else (dp, dn)
    unique, inverse = np.unique(np.concatenate((before, after)), return_inverse=True)
    raw = torch.from_numpy(volume[unique].astype(np.float32))[:, None] / 65535
    context = (
        F.interpolate(raw, (side, side), mode="bilinear", align_corners=False, antialias=True)
        .clamp_(0, 1)[:, 0]
        .numpy()
    )
    c = gray[:, 0].numpy()
    return torch.from_numpy(
        np.stack(
            [
                neighbor_delta_channels(
                    center,
                    context[inverse[i]],
                    context[inverse[i + len(centers)]],
                    strength=policy["strength"],
                )
                for i, center in enumerate(c)
            ]
        )
    )
