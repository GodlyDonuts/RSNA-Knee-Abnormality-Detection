"""Source-pixel masking and fixed descriptors with no target-derived scaling.

Normalizers come only from DICOM encoding metadata. A canonical source-frame
alias is masked once before every overlapping triplet/resize. Array aliases and
region identities are private bookkeeping, never neural-network input features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

VERSION = "raw_alias_closed_grid4_halo8_DCT32_grad16_std16_v1"


def fixed_metadata_scale(raw, *, bits_stored, signed=False, photometric="MONOCHROME2"):
    """Scale stored samples by encoding bounds, never by image min/max/percentiles.

    Rescale slope/intercept cancel when using their fixed transformed encoding
    endpoints; require positive slope in the first raw-source admission. The
    actual source adapter must inspect those tags, rather than infer from pixels.
    """
    if (
        not isinstance(raw, torch.Tensor)
        or raw.ndim != 2
        or raw.is_floating_point()
        or raw.dtype == torch.bool
        or raw.numel() == 0
        or type(bits_stored) is not int
        or not 1 <= bits_stored <= 16
        or type(signed) is not bool
        or photometric not in ("MONOCHROME1", "MONOCHROME2")
    ):
        raise ValueError("One original integer frame and explicit DICOM encoding metadata required")
    lo = -(2 ** (bits_stored - 1)) if signed else 0
    hi = lo + 2**bits_stored - 1
    if (raw < lo).any() or (raw > hi).any():
        raise ValueError("Stored pixel values outside metadata encoding range")
    normalized = (raw.float() - lo) / (hi - lo)
    return 1 - normalized if photometric == "MONOCHROME1" else normalized


def descriptor64(patch):
    """Fixed low-frequency shape, full-resolution gradient and local texture.

    Every coordinate has a fixed analytic scale; no learned target encoder,
    cohort statistics or label information. Sufficiency for lesions is unproven.
    """
    if (
        not isinstance(patch, torch.Tensor)
        or patch.ndim != 2
        or min(patch.shape) < 8
        or patch.requires_grad
        or patch.dtype != torch.float32
        or not torch.isfinite(patch).all()
        or (patch < 0).any()
        or (patch > 1).any()
    ):
        raise ValueError("Detached metadata-normalized FP32 region, at least8x8, required")
    small = F.adaptive_avg_pool2d(patch[None, None], (8, 8))[0, 0]
    k = torch.arange(8, device=patch.device, dtype=patch.dtype)[:, None]
    n = torch.arange(8, device=patch.device, dtype=patch.dtype)[None, :]
    basis = torch.cos(math.pi * (n + 0.5) * k / 8) * math.sqrt(2 / 8)
    basis[0] *= 1 / math.sqrt(2)
    coeff = basis @ small @ basis.T / 8
    order = sorted(((i, j) for i in range(8) for j in range(8)), key=lambda x: (sum(x), x[0]))[:32]
    low = torch.stack([coeff[i, j] for i, j in order])
    dx = F.pad((patch[:, 1:] - patch[:, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((patch[1:] - patch[:-1]).abs(), (0, 0, 0, 1))
    gradient = F.adaptive_avg_pool2d(((dx + dy) / 2)[None, None], (4, 4)).flatten()
    mean = F.adaptive_avg_pool2d(patch[None, None], (4, 4))
    square = F.adaptive_avg_pool2d(patch.square()[None, None], (4, 4))
    std = 2 * (square - mean.square()).clamp_min(0).sqrt().flatten()
    return torch.cat((low, gradient, std)).detach()


@dataclass(frozen=True)
class MaskedStudy:
    context_frames: torch.Tensor
    descriptors: torch.Tensor
    observed: torch.Tensor
    region_frame: torch.Tensor
    region_tile: torch.Tensor
    raw_hidden: torch.Tensor
    acquisition: torch.Tensor


def masked_study(frames, aliases, acquisition, ordinal, *, pattern, halo=8, fill=0.5):
    """Non-overlapping4x4 regions; mask raw-grid targets and pixel halo globally.

    Input frames are metadata-scaled raw matrices, not old percentile-normalized
    cache arrays. Shapes must match within this bounded engineering call. The
    production adapter groups variable matrix sizes by source acquisition.
    Complete source roster/alias IDs must precede descriptors or encoder calls.
    """
    if (
        not isinstance(frames, torch.Tensor)
        or frames.ndim != 3
        or frames.dtype != torch.float32
        or frames.requires_grad
        or min(frames.shape[1:]) < 32
        or not torch.isfinite(frames).all()
        or (frames < 0).any()
        or (frames > 1).any()
    ):
        raise ValueError("Detached finite metadata-scaled raw frames required")
    count, height, width = frames.shape
    for value in (aliases, acquisition, ordinal):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (count,)
            or value.dtype != torch.int64
            or value.device != frames.device
            or (value < 0).any()
        ):
            raise ValueError("Exact private source aliases/acquisition/ordinal required")
    if (
        pattern not in (0, 1)
        or type(pattern) is not int
        or type(halo) is not int
        or halo < 0
        or min(height, width) // 4 <= 2 * halo
        or not math.isfinite(fill)
        or not 0 <= fill <= 1
    ):
        raise ValueError("Pinned complementary mask and finite raw fill required")
    representatives = {}
    for i, key in enumerate(aliases.tolist()):
        if key in representatives:
            j = representatives[key]
            if not torch.equal(frames[i], frames[j]):
                raise ValueError("Alias frames disagree; never guess a duplicate transformation")
        else:
            representatives[key] = i
    hidden = torch.zeros_like(frames, dtype=torch.bool)
    descriptors = []
    observed = []
    region_frame = []
    region_tile = []
    for alias, i in representatives.items():
        peers = (aliases == alias).nonzero().flatten()
        for row in range(4):
            for col in range(4):
                y0, y1 = row * height // 4, (row + 1) * height // 4
                x0, x1 = col * width // 4, (col + 1) * width // 4
                # Metadata only, shared for all alias occurrences and all arms.
                held = (row + col + int(ordinal[i]) // 4 + int(acquisition[i])) % 2 == pattern
                descriptors.append(descriptor64(frames[i, y0:y1, x0:x1]))
                observed.append(held)
                region_frame.append(i)
                region_tile.append((row, col))
                if held:
                    hidden[
                        peers,
                        max(0, y0 - halo) : min(height, y1 + halo),
                        max(0, x0 - halo) : min(width, x1 + halo),
                    ] = True
    context = frames.clone().masked_fill(hidden, fill)
    return MaskedStudy(
        context,
        torch.stack(descriptors),
        torch.tensor(observed, device=frames.device, dtype=torch.bool),
        torch.tensor(region_frame, device=frames.device, dtype=torch.int64),
        torch.tensor(region_tile, device=frames.device, dtype=torch.int64),
        hidden,
        acquisition.clone(),
    )


def triplets(study, centers, neighbors, *, side=320):
    """Render ONLY already-masked frames; source aliases remain masked everywhere.

    centers/neighbor indices are metadata-bound, never sampled using target values.
    Caller must forbid cross-acquisition neighbors and bind complete roster.
    """
    if not isinstance(study, MaskedStudy):
        raise ValueError("Only source-masked study data may enter the renderer")
    context_frames = study.context_frames
    if (
        context_frames.ndim != 3
        or context_frames.dtype != torch.float32
        or not torch.isfinite(context_frames).all()
        or centers.dtype != torch.int64
        or centers.ndim != 1
        or neighbors.dtype != torch.int64
        or neighbors.shape != (len(centers), 3)
        or min(side, len(centers)) < 1
        or (neighbors < 0).any()
        or (neighbors >= len(context_frames)).any()
        or not torch.equal(neighbors[:, 1], centers)
        or not torch.equal(
            study.acquisition[neighbors], study.acquisition[centers, None].expand_as(neighbors)
        )
    ):
        raise ValueError("Metadata-bound original within-acquisition triplets required")
    return F.interpolate(
        context_frames[neighbors], size=(side, side), mode="bilinear", align_corners=False
    ).contiguous()
