"""Bounded batching of exact CPU descriptors for both complementary views.

Serving-only preparation for fresh RawGroups. No source pixels, masks, context
selection, support/prior construction, density arithmetic or model is changed.
The original assembler still validates and consumes both complete contexts.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache

import torch
from torch.nn import functional as F

from rsna.competitive.predictive import inputs
from rsna.competitive.predictive.cases import RawGroup


def cache_descriptors_once(groups, *, max_descriptor_bytes=32 * 2**20):
    """Return groups sharing original pixels/metadata plus exact FP32 descriptors.

    Preserve canonical alias insertion order and the original row-major 4x4
    regions. Bound the total new descriptor storage before allocating it; never
    truncate a study. Already cached groups are rejected rather than trusting an
    unbound cache supplied by a caller.
    """
    if (
        not isinstance(groups, (list, tuple))
        or not groups
        or type(max_descriptor_bytes) is not int
        or max_descriptor_bytes < 1
    ):
        raise ValueError("Fresh original groups and a positive descriptor byte cap required")
    plans = []
    all_aliases = set()
    total_bytes = 0
    for group in groups:
        if not isinstance(group, RawGroup) or group.descriptors is not None:
            raise ValueError("Fresh uncached RawGroups required")
        frames = group.frames
        if (
            frames.device.type != "cpu"
            or frames.dtype != torch.float32
            or frames.ndim != 3
            or not len(frames)
            or min(frames.shape[1:]) < 32
            or frames.requires_grad
            or not torch.isfinite(frames).all()
            or (frames < 0).any()
            or (frames > 1).any()
        ):
            raise ValueError("Detached finite original metadata-scaled CPU frames required")
        for value in (group.aliases, group.acquisition, group.ordinal):
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.shape != (len(frames),)
                or value.dtype != torch.int64
                or (value < 0).any()
            ):
                raise ValueError("Complete original integer frame metadata required")
        representatives = {}
        for index, alias in enumerate(group.aliases.tolist()):
            if alias in representatives:
                if not torch.equal(frames[index], frames[representatives[alias]]):
                    raise ValueError("Alias frames disagree")
            else:
                representatives[alias] = index
        if all_aliases.intersection(representatives):
            raise ValueError("Exact aliases must stay within one matrix group")
        all_aliases.update(representatives)
        total_bytes += len(representatives) * 16 * 64 * 4
        if total_bytes > max_descriptor_bytes:
            raise ValueError("Complete descriptor storage exceeds cap; no truncation")
        plans.append((group, tuple(representatives.values())))

    result = []
    for group, representatives in plans:
        _, height, width = group.frames.shape
        descriptors = torch.empty((len(representatives) * 16, 64), dtype=torch.float32)
        # Equal-size patches per tile, preserving canonical output order.
        # Four MiB of stacked pixels bounds temporary batched arrays. Extremely
        # large patches use the unchanged scalar implementation.
        for row in range(4):
            for col in range(4):
                y0, y1 = row * height // 4, (row + 1) * height // 4
                x0, x1 = col * width // 4, (col + 1) * width // 4
                pixel_bytes = (y1 - y0) * (x1 - x0) * 4
                chunk = min(16, max(1, 4 * 2**20 // pixel_bytes))
                for start in range(0, len(representatives), chunk):
                    selected = representatives[start : start + chunk]
                    if pixel_bytes > 4 * 2**20:
                        value = inputs.descriptor64(group.frames[selected[0], y0:y1, x0:x1])[None]
                    else:
                        patches = torch.stack([group.frames[i, y0:y1, x0:x1] for i in selected])
                        value = descriptor_batch(patches)
                    for offset in range(len(selected)):
                        descriptors[(start + offset) * 16 + row * 4 + col].copy_(value[offset])
        result.append(replace(group, descriptors=descriptors))
    return result


@lru_cache(maxsize=1)
def _basis():
    k = torch.arange(8, dtype=torch.float32)[:, None]
    n = torch.arange(8, dtype=torch.float32)[None, :]
    basis = torch.cos(math.pi * (n + 0.5) * k / 8) * math.sqrt(2 / 8)
    basis[0] *= 1 / math.sqrt(2)
    order = sorted(((i, j) for i in range(8) for j in range(8)), key=lambda x: (sum(x), x[0]))[:32]
    return basis, tuple(torch.tensor(x) for x in zip(*order))


def descriptor_batch(patches):
    """CPU FP32 pooling in batches; original per-patch matmul/reduction order."""
    if (
        not isinstance(patches, torch.Tensor)
        or patches.ndim != 3
        or patches.dtype != torch.float32
        or patches.device.type != "cpu"
        or patches.requires_grad
        or len(patches) < 1
        or min(patches.shape[1:]) < 8
        or not torch.isfinite(patches).all()
        or (patches < 0).any()
        or (patches > 1).any()
    ):
        raise ValueError("Detached finite CPU FP32 patch batch required")
    basis, idx = _basis()
    small = F.adaptive_avg_pool2d(patches[:, None], (8, 8))[:, 0]
    # Batched GEMM can change rounding. Preserve original 8x8 matmuls exactly.
    coeff = torch.stack([basis @ s @ basis.T / 8 for s in small])
    low = coeff[:, idx[0], idx[1]]
    dx = F.pad((patches[:, :, 1:] - patches[:, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((patches[:, 1:] - patches[:, :-1]).abs(), (0, 0, 0, 1))
    gradient = F.adaptive_avg_pool2d(((dx + dy) / 2)[:, None], (4, 4)).flatten(1)
    mean = F.adaptive_avg_pool2d(patches[:, None], (4, 4))
    square = F.adaptive_avg_pool2d(patches.square()[:, None], (4, 4))
    std = 2 * (square - mean.square()).clamp_min(0).sqrt().flatten(1)
    return torch.cat((low, gradient, std), 1).detach()
