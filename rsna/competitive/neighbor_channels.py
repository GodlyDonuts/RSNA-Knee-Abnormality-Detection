"""Standalone CPU prototype; no trainer, loader or inference integration.

Encode supplied neighboring-slice differences as bounded color perturbations.
The caller must separately admit original-image/geometry/input bindings. This
function cannot tell whether supplied arrays are physical neighbors.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def neighbor_delta_channels(
    center: NDArray[np.float32],
    previous: NDArray[np.float32],
    following: NDArray[np.float32],
    *,
    strength: float,
) -> NDArray[np.float32]:
    """Return [3,H,W] FP32, preserving center mean up to FP32 rounding.

    Inputs are equally normalized, aligned 2D FP32 slices in [0,1]. For
    0 <= strength <= 1/2, the real-arithmetic output remains in [0,1]
    without clipping. Float64 intermediates also handle FP32 subnormals.
    Strength zero returns the exact repeated center bytes, including signed
    zeros. No RNG, spatial smoothing, parameters or global state is used.
    """
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
        or not 0 <= strength <= 0.5
    ):
        raise ValueError("Finite neighbor strength in [0,0.5] required")
    for image in (center, previous, following):
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.float32
            or image.ndim != 2
            or not image.size
            or not np.isfinite(image).all()
            or (image < 0).any()
            or (image > 1).any()
        ):
            raise ValueError("Nonempty 2D FP32 slices in [0,1] required")
    if center.shape != previous.shape or center.shape != following.shape:
        raise ValueError("Aligned slice shapes required; geometry admission is separate")
    if strength == 0:
        return np.stack((center, center, center))
    c = center.astype(np.float64)
    before = previous.astype(np.float64) - c
    after = following.astype(np.float64) - c
    amplitude = strength * np.minimum(c, 1 - c)
    return np.stack(
        (c + amplitude * before, c - amplitude * (before + after), c + amplitude * after)
    ).astype(np.float32)
