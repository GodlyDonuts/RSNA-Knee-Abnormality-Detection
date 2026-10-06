"""Original letterbox content and patient coordinates; never assume shared frames."""

from __future__ import annotations

import numpy as np


def patch_geometry(metadata: dict, indices, *, patch_size: int = 16) -> dict:
    side = metadata["side"]
    original = metadata["original_shape"]
    resized = metadata["resized_shape"]
    if (
        type(side) is not int
        or type(patch_size) is not int
        or patch_size < 1
        or side % patch_size
        or len(original) != 3
        or len(resized) != 2
    ):
        raise ValueError("Exact original/cache/patch dimensions required")
    count, height, width = original
    rh, rw = resized
    if min(count, height, width, rh, rw) < 1 or max(rh, rw) > side:
        raise ValueError("Invalid original content rectangle")
    selected = np.asarray(indices)
    if (
        selected.ndim != 1
        or not len(selected)
        or not np.issubdtype(selected.dtype, np.integer)
        or ((selected < 0) | (selected >= count)).any()
    ):
        raise ValueError("Actual ordered source slice indices required")
    top, left = (side - rh) // 2, (side - rw) // 2
    edge = np.arange(side // patch_size) * patch_size
    y = np.maximum(0, np.minimum(edge + patch_size, top + rh) - np.maximum(edge, top))
    x = np.maximum(0, np.minimum(edge + patch_size, left + rw) - np.maximum(edge, left))
    content_fraction = np.outer(y, x) / (patch_size**2)
    centers = (np.arange(side // patch_size) + 0.5) * patch_size - 0.5
    cache_y, cache_x = np.meshgrid(centers, centers, indexing="ij")
    # align_corners=False resampling of the physical-aspect-preserving content.
    original_y = (cache_y - top + 0.5) * height / rh - 0.5
    original_x = (cache_x - left + 0.5) * width / rw - 0.5
    inside = (cache_y >= top) & (cache_y < top + rh) & (cache_x >= left) & (cache_x < left + rw)
    valid = inside & (content_fraction >= 0.5)
    if not valid.any():
        raise ValueError("No patch center retains at least half original MRI content")
    orientation = np.asarray(metadata.get("orientation", []), dtype=float)
    positions = np.asarray(metadata.get("positions", []), dtype=float)
    spacing = np.asarray(metadata.get("pixel_spacing", []), dtype=float)
    known = (
        metadata.get("ordering") == "patient_geometry"
        and orientation.shape == (6,)
        and positions.shape == (count, 3)
        and spacing.shape == (2,)
    )
    coordinates = None
    if known:
        if (
            not np.isfinite(np.r_[orientation, positions.ravel(), spacing]).all()
            or (spacing <= 0).any()
        ):
            raise ValueError("Nonfinite or nonpositive patient geometry")
        column_increment, row_increment = orientation[:3], orientation[3:]
        if not np.allclose(
            [
                np.linalg.norm(column_increment),
                np.linalg.norm(row_increment),
                np.dot(column_increment, row_increment),
            ],
            [1, 1, 0],
            atol=1e-3,
        ):
            raise ValueError("Patient basis must be orthonormal")
        coordinates = (
            positions[selected, None, None, :]
            + original_x[None, :, :, None] * spacing[1] * column_increment
            + original_y[None, :, :, None] * spacing[0] * row_increment
        )
    return {
        "content_fraction": content_fraction,
        "valid_content_patch": valid,
        "patient_centers_mm": coordinates,
        "patient_geometry_known": known,
        "cross_series_registration_verified": False,
    }
