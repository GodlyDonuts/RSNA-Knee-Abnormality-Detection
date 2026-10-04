"""Standalone neighbor indices; no loader/trainer/inference integration.

The caller must bind metadata to original verified source pixels. These indices
cannot establish original per-slice orientation from an untrusted metadata file.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def physical_neighbor_indices(metadata: dict) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Replicate center at boundaries, missing geometry, gaps or in-plane shifts.

    The fixed protocol admits an edge only at positive spacing <=1.5 times the
    series median and in-plane origin displacement <=0.001 mm. No sorting,
    interpolation, inferred positions or cross-series donors are introduced.
    A claimed geometry record with malformed fields raises instead of silently
    being admitted. Valid but unusable edges fall back to the center.
    """
    count = metadata.get("n_slices")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Positive original slice count required")
    center = np.arange(count, dtype=np.int64)
    previous, following = center.copy(), center.copy()
    if metadata.get("ordering") != "patient_geometry":
        return previous, following
    orientation = np.asarray(metadata.get("orientation", []), dtype=np.float64)
    positions = np.asarray(metadata.get("positions", []), dtype=np.float64)
    spacing = np.asarray(metadata.get("pixel_spacing", []), dtype=np.float64)
    if (
        orientation.shape != (6,)
        or positions.shape != (count, 3)
        or spacing.shape != (2,)
        or not np.isfinite(orientation).all()
        or not np.isfinite(positions).all()
        or not np.isfinite(spacing).all()
        or (spacing <= 0).any()
    ):
        raise ValueError("Complete finite original geometry/grid required")
    row, column = orientation[:3], orientation[3:]
    if not np.allclose(
        [np.linalg.norm(row), np.linalg.norm(column), np.dot(row, column)],
        [1, 1, 0],
        atol=1e-3,
        rtol=0,
    ):
        raise ValueError("Orthonormal original orientation required")
    if count == 1:
        return previous, following
    normal = np.cross(row, column)
    normal /= np.linalg.norm(normal)
    differences = np.diff(positions, axis=0)
    gaps = differences @ normal
    # Non-monotone or repeated positions cannot establish adjacent slice order.
    if (gaps <= 0).any():
        return previous, following
    in_plane = np.sqrt((differences @ row) ** 2 + (differences @ column) ** 2)
    admitted = (gaps <= 1.5 * np.median(gaps)) & (in_plane <= 0.001)
    edges = np.flatnonzero(admitted)
    following[edges] = edges + 1
    previous[edges + 1] = edges
    return previous, following


def distant_neighbor_indices(count: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Two deterministic same-volume donors >=quarter-depth and never adjacent.

    Short volumes without two distinct sufficiently distant donors replicate the
    center. This is an unlabelled control, not physical neighbor admission.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Positive original slice count required")
    center = np.arange(count, dtype=np.int64)
    previous, following = center.copy(), center.copy()
    minimum = max(2, (count + 3) // 4)
    for index in range(count):
        distance = np.abs(center - index)
        eligible = center[distance >= minimum]
        if len(eligible) >= 2:
            chosen = sorted(eligible.tolist(), key=lambda donor: (-abs(donor - index), donor))
            previous[index], following[index] = chosen[:2]
    return previous, following
