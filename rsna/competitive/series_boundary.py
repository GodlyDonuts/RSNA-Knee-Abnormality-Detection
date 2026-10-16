"""Label-free series-boundary policies for a fixed concatenated MRI volume.

The historical reference permits triples across series joins. These alternatives
either retain only complete same-series triples or repeat the center at an edge.
No spatial interpolation, learned pixels, finding labels or new evidence are added.
"""

from __future__ import annotations

import numpy as np


def series_donor_indices(centers, slot_sizes, *, mode="legacy") -> np.ndarray:
    if mode not in {"legacy", "edge-repeat"}:
        raise ValueError("Explicit legacy or center-preserving edge-repeat policy required")
    centers = np.asarray(centers)
    if (
        centers.ndim != 1
        or not len(centers)
        or not np.issubdtype(centers.dtype, np.integer)
        or not isinstance(slot_sizes, (list, tuple))
        or not slot_sizes
        or any(type(n) is not int or n < 1 for n in slot_sizes)
    ):
        raise ValueError("Nonempty integer centers and positive fixed series lengths required")
    bounds = np.cumsum([0, *slot_sizes], dtype=np.int64)
    if bounds[-1] < 3 or (centers < 1).any() or (centers >= bounds[-1] - 1).any():
        raise ValueError("Triples must remain inside the complete original volume")
    centers = centers.astype(np.int64, copy=False)
    indices = np.stack([centers - 1, centers, centers + 1], axis=1)
    if mode == "legacy":
        return indices
    slots = np.searchsorted(bounds[1:], centers, side="right")
    return np.clip(indices, bounds[slots, None], bounds[slots + 1, None] - 1)


def same_series_windows(centers, slot_sizes) -> np.ndarray:
    legacy = series_donor_indices(centers, slot_sizes)
    repaired = series_donor_indices(centers, slot_sizes, mode="edge-repeat")
    return np.all(legacy == repaired, axis=1)
