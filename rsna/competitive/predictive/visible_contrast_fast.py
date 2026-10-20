"""Explicit equivalent preparation path for the separate contrast experiment.

Visible sampling positions depend only on matrix dimensions, mask parity and
halo. Reuse that geometry rather than assembling full masked studies merely to
extract their visible pixels. The original assembler still validates all source
pixels/aliases and produces the unchanged targets, supports and rendered images.
No existing training or serving entrypoint activates this helper automatically.
"""

from __future__ import annotations

from types import FunctionType

import numpy as np
import torch

from rsna.competitive.predictive import visible_contrast


def visible_positions(height, width, *, parity, halo):
    """Exact row-major capped positions used by the original boolean selection."""
    if (
        any(type(v) is not int for v in (height, width, parity, halo))
        or min(height, width) < 32
        or parity not in (0, 1)
        or halo < 0
        or min(height, width) // 4 <= 2 * halo
    ):
        raise ValueError("Original complementary grid geometry required")
    hidden = torch.zeros((height, width), dtype=torch.bool)
    for row in range(4):
        for col in range(4):
            if (row + col) % 2 == parity:
                y0, y1 = row * height // 4, (row + 1) * height // 4
                x0, x1 = col * width // 4, (col + 1) * width // 4
                hidden[max(0, y0 - halo) : min(height, y1 + halo),
                       max(0, x0 - halo) : min(width, x1 + halo)] = True
    visible = (~hidden).reshape(-1).nonzero().flatten()
    if not len(visible):
        raise ValueError("A canonical source frame has no visible pixels")
    chosen = torch.linspace(0, len(visible) - 1, min(4096, len(visible))).round().long()
    return visible[chosen]


def acquisition_scales(groups, *, pattern, halo=8):
    """Same samples, order, quantiles and floor as the frozen candidate.

    Groups must be the admitted CPU FP32 raw source groups. Full source and
    alias consistency remains checked by the original masked renderer. The
    bounded geometry cache exists only within this call; no study state leaks
    into another study or complementary mask.
    """
    if type(pattern) is not int or pattern not in (0, 1):
        raise ValueError("An explicit original mask pattern is required")
    positions, samples = {}, {}
    for group in groups:
        frames = group.frames
        if (frames.ndim != 3 or frames.dtype != torch.float32 or frames.requires_grad
                or frames.device.type != "cpu"):
            raise ValueError("Original detached CPU FP32 frames required")
        for field in (group.aliases, group.acquisition, group.ordinal):
            if (field.shape != (len(frames),) or field.dtype != torch.int64
                    or field.device.type != "cpu" or (field < 0).any()):
                raise ValueError("Original frame metadata required")
        first = {}
        for i, alias in enumerate(group.aliases.tolist()):
            first.setdefault(alias, i)
        for i in first.values():
            acquisition = int(group.acquisition[i])
            parity = (pattern - int(group.ordinal[i]) // 4 - acquisition) % 2
            key = (frames.shape[1], frames.shape[2], parity, halo)
            if key not in positions:
                positions[key] = visible_positions(key[0], key[1], parity=parity, halo=halo)
            selected = frames[i].reshape(-1)[positions[key]].numpy().copy()
            samples.setdefault(acquisition, []).append(selected)
    result = {}
    for acquisition, parts in samples.items():
        values = np.concatenate(parts)
        low, high = np.quantile(values, [.01, .99], method="linear")
        result[acquisition] = visible_contrast.VisibleScale(
            float(low), max(float(high - low), .01), len(values))
    if not result:
        raise ValueError("Visible canonical source acquisitions required")
    return result


def assemble(groups, *, side=320, halo=8, context_cap=24, classes=12, normalize_images=True):
    """Call the unchanged contrast renderer with equivalent geometry-only sampling."""
    original = visible_contrast.assemble
    namespace = {**original.__globals__, "acquisition_scales": acquisition_scales}
    renderer = FunctionType(original.__code__, namespace, "fast_visible_contrast_assemble",
                            original.__defaults__, original.__closure__)
    renderer.__kwdefaults__ = original.__kwdefaults__
    return renderer(groups, side=side, halo=halo, context_cap=context_cap, classes=classes,
                    normalize_images=normalize_images)
