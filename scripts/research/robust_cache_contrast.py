"""Standalone selected-stack contrast candidate; not used by the live trainer."""

from __future__ import annotations

import numpy as np


def normalize_selected_cache(volume: np.ndarray, resized_shape, *, policy: str) -> np.ndarray:
    """Keep the cached physical field and zero padding; normalize valid content only.

    ``minmax`` reproduces the existing uint16-to-float conversion. ``p01_p99``
    estimates one range for the entire selected stack, preserving inter-slice
    intensity relationships. This is not the public raw cropped-pixel renderer.
    """
    if policy not in {"minmax", "p01_p99"}:
        raise ValueError("Unknown standalone contrast policy")
    if volume.dtype != np.uint16 or volume.ndim != 3 or not volume.shape[0]:
        raise ValueError("A nonempty selected uint16 stack is required")
    side = volume.shape[1]
    if side < 1 or volume.shape[2] != side or len(resized_shape) != 2:
        raise ValueError("Original square cache and content dimensions are required")
    rh, rw = resized_shape
    if any(type(v) is not int or not 1 <= v <= side for v in (rh, rw)):
        raise ValueError("Invalid original content rectangle")
    top, left = (side - rh) // 2, (side - rw) // 2
    mask = np.ones((side, side), dtype=bool)
    mask[top : top + rh, left : left + rw] = False
    if np.any(volume[:, mask]):
        raise ValueError("Declared artificial padding must remain zero")
    baseline = volume.astype(np.float32) / np.float32(65535)
    if policy == "minmax":
        return baseline
    content = volume[:, top : top + rh, left : left + rw]
    low, high = np.percentile(content, [1, 99], method="linear")
    if high <= low:
        raise ValueError("Selected content has no robust intensity range")
    result = np.zeros_like(baseline)
    values = (content.astype(np.float32) - np.float32(low)) / np.float32(high - low)
    result[:, top : top + rh, left : left + rw] = np.clip(values, 0, 1)
    return result
