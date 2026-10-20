"""Explicit experimental contrast branch; the original renderer is unchanged.

Each complementary mask has its own normalization, fitted exclusively on visible
canonical source pixels within each acquisition. Original descriptor targets and
support priors stay in their original units. Offset and log-range accompany the
image/region metadata so the predictor retains the original intensity scale.
No existing trainer or inference entrypoint calls this experiment automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import FunctionType

import numpy as np
import torch

from rsna.competitive.predictive import cases
from rsna.competitive.predictive.inputs import masked_study

VERSION = "visible_acquisition_q01_q99_cap4096_floor01_metadata10_v1"


@dataclass(frozen=True)
class VisibleScale:
    offset: float
    span: float
    sampled_visible_pixels: int


def acquisition_scales(groups, *, pattern, halo=8):
    """Deterministic capped sampling; neither withheld pixels nor labels enter.

    Sample at most4,096 evenly spaced visible positions from each canonical frame.
    Alias copies never receive extra weight. Sampling positions depend only on
    mask geometry, not image values. Positive range floor caps gain at100.
    """
    samples = {}
    for group in groups:
        view = masked_study(
            group.frames, group.aliases, group.acquisition, group.ordinal,
            pattern=pattern, halo=halo, cached_descriptors=group.descriptors,
        )
        first = {}
        for i, alias in enumerate(group.aliases.tolist()):
            first.setdefault(alias, i)
        for i in first.values():
            visible = group.frames[i][~view.raw_hidden[i]]
            if not visible.numel():
                raise ValueError("A canonical source frame has no visible pixels")
            positions = torch.linspace(0, len(visible) - 1, min(4096, len(visible))).round().long()
            selected = visible[positions].numpy().copy()
            samples.setdefault(int(group.acquisition[i]), []).append(selected)
    result = {}
    for acquisition, parts in samples.items():
        values = np.concatenate(parts)
        low, high = np.quantile(values, [.01, .99], method="linear")
        result[acquisition] = VisibleScale(float(low), max(float(high - low), .01), len(values))
    if not result:
        raise ValueError("Visible canonical source acquisitions required")
    return result


def scale_view(view, aliases, scales):
    """Normalize only the context, refill hidden pixels and preserve targets."""
    first = {}
    for i, alias in enumerate(aliases.tolist()):
        first.setdefault(alias, i)
    # Alias-only acquisitions inherit their representative's coordinates/scale,
    # matching the canonical alias treatment in the original assembler.
    keys = [int(view.acquisition[first[alias]]) for alias in aliases.tolist()]
    offset = torch.tensor([scales[k].offset for k in keys], dtype=torch.float32)[:, None, None]
    span = torch.tensor([scales[k].span for k in keys], dtype=torch.float32)[:, None, None]
    context = ((view.context_frames - offset) / span).clamp_(0, 1)
    context.masked_fill_(view.raw_hidden, .5)
    return replace(view, context_frames=context)


def append_scale_metadata(metadata, scales):
    keys = sorted(scales)
    columns = torch.empty((len(metadata), 2), dtype=torch.float32)
    assigned = torch.zeros(len(metadata), dtype=torch.bool)
    for i, key in enumerate(keys):
        rank = torch.tensor(i / max(1, len(keys) - 1), dtype=torch.float32)
        rows = metadata[:, 0] == rank
        if (assigned & rows).any():
            raise ValueError("Acquisition rank ambiguity")
        columns[rows, 0] = scales[key].offset
        columns[rows, 1] = math.log(scales[key].span)
        assigned |= rows
    if not assigned.all():
        raise ValueError("Original acquisition coordinates do not match visible scales")
    return torch.cat((metadata, columns), dim=1)


def assemble(groups, *, side=320, halo=8, context_cap=24, classes=12, normalize_images=True):
    """Two source-masked contexts with ten metadata coordinates.

    normalize_images=False is a matched graph/control: it exposes the same
    visible-only scale coordinates while retaining original image tensors.
    Both versions preserve original targets/priors. Ordinary8-coordinate
    B/C/C0 checkpoints cannot be loaded into this separate10-coordinate graph.
    """
    if type(normalize_images) is not bool:
        raise ValueError("Explicit experimental image-normalization choice required")
    scales = {}

    def experimental_masked_study(frames, aliases, acquisition, ordinal, **kwargs):
        pattern = kwargs["pattern"]
        if pattern not in scales:
            scales[pattern] = acquisition_scales(groups, pattern=pattern, halo=halo)
        original = masked_study(frames, aliases, acquisition, ordinal, **kwargs)
        return scale_view(original, aliases, scales[pattern]) if normalize_images else original

    # Isolated globals keep the original function/code intact and avoid global
    # monkeypatching when independent readers run in parallel.
    namespace = {**cases.assemble.__globals__, "masked_study": experimental_masked_study}
    renderer = FunctionType(cases.assemble.__code__, namespace, "visible_contrast_assemble",
                            cases.assemble.__defaults__, cases.assemble.__closure__)
    renderer.__kwdefaults__ = cases.assemble.__kwdefaults__
    contexts = renderer(groups, side=side, halo=halo, context_cap=context_cap, classes=classes)
    for pattern, context in enumerate(contexts):
        for field in ("token_metadata", "region_metadata"):
            context[field] = append_scale_metadata(context[field], scales[pattern])
    return contexts, scales
