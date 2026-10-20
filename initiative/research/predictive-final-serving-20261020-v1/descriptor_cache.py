"""Compute the unchanged source descriptor once for both complementary views.

Serving-only preparation for fresh RawGroups. No source pixels, masks, context
selection, support/prior construction, density arithmetic or model is changed.
The original assembler still validates and consumes both complete contexts.
"""

from __future__ import annotations

from dataclasses import replace

import torch

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
        offset = 0
        for index in representatives:
            for row in range(4):
                for col in range(4):
                    patch = group.frames[
                        index,
                        row * height // 4 : (row + 1) * height // 4,
                        col * width // 4 : (col + 1) * width // 4,
                    ]
                    descriptors[offset].copy_(inputs.descriptor64(patch))
                    offset += 1
        result.append(replace(group, descriptors=descriptors))
    return result
