"""Advisory cross-source MRI overlap candidates, never clinical or label inference.

Coarse image signatures cannot prove independent patients or establish absence
of overlap. Positive candidates require subsequent original-pixel review.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image


def slice_signatures(image: np.ndarray) -> list[str]:
    """Affine-intensity/D4 invariant full-frame and foreground difference hashes.

    Ignore empty/low-contrast slices. Foreground cropping only removes dark image
    margins for candidate generation; it never changes a training input.
    """
    if image.ndim != 2 or min(image.shape) < 16 or not np.isfinite(image).all():
        raise ValueError("A finite original gray slice is required")
    value = image.astype(np.float32)
    low, high = np.quantile(value, [0.01, 0.99])
    if high <= low:
        return []
    value = np.clip((value - low) / (high - low), 0, 1).astype(np.float32)
    views = [value]
    y, x = np.where(value > 0.03)
    if len(x):
        crop = value[y.min() : y.max() + 1, x.min() : x.max() + 1]
        if min(crop.shape) >= 16 and crop.shape != value.shape:
            views.append(crop)
    hashes = set()
    for view in views:
        # Canonicalize orientation, rather than guessing patient laterality or
        # treating array coordinates as physical geometry.
        candidates = []
        for turns in range(4):
            rotated = np.rot90(view, turns)
            for framed in (rotated, np.fliplr(rotated)):
                thumb = np.asarray(
                    Image.fromarray(np.ascontiguousarray(framed)).resize(
                        (9, 8), Image.Resampling.BOX
                    )
                )
                bits = (thumb[:, 1:] > thumb[:, :-1]).flatten()
                # Suppress near-constant coarse patterns with little identity
                # information. They would mostly match anatomy/background.
                if not 8 <= int(bits.sum()) <= 56:
                    continue
                candidates.append(int.from_bytes(np.packbits(bits).tobytes(), "big"))
        if candidates:
            hashes.add(f"{min(candidates):016x}")
    return sorted(hashes)


def volume_signature(volume: np.ndarray, count: int = 8) -> list[dict[str, Any]]:
    if volume.ndim != 3 or not len(volume) or not 1 <= count <= 64:
        raise ValueError("Original nonempty slice stack and bounded sampling required")
    indices = np.linspace(0, len(volume) - 1, min(count, len(volume))).round().astype(int)
    return [{"slice": int(index), "hashes": slice_signatures(volume[index])} for index in indices]


def overlap_candidates(
    left: list[dict[str, Any]], right: list[dict[str, Any]], *, max_hamming: int = 3
) -> list[dict[str, Any]]:
    """Return pairs with >=3 distinct slice indices AND patterns in BOTH sources.

    Four disjoint sixteen-bit lookup chunks are exhaustive for <=3 bit errors:
    at least one chunk must be identical. False positives remain possible;
    common signatures are retained rather than silently claiming independence.
    """
    if not 0 <= max_hamming <= 3:
        raise ValueError("Four-chunk exact candidate coverage is bounded to three bit errors")
    index: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for row_index, row in enumerate(left):
        for sampled in row["samples"]:
            for value in sampled["hashes"]:
                number = int(value, 16)
                for chunk in range(4):
                    index[(chunk, (number >> (16 * chunk)) & 65535)].append(
                        (row_index, sampled["slice"], number)
                    )
    pairs: dict[tuple[int, int], set[tuple[int, int, int, int, int]]] = defaultdict(set)
    for row_index, row in enumerate(right):
        for sampled in row["samples"]:
            for value in sampled["hashes"]:
                number = int(value, 16)
                possible = set()
                for chunk in range(4):
                    possible.update(index.get((chunk, (number >> (16 * chunk)) & 65535), []))
                for other, source_slice, source_hash in possible:
                    distance = (number ^ source_hash).bit_count()
                    if distance <= max_hamming:
                        pairs[(other, row_index)].add(
                            (source_slice, sampled["slice"], distance, source_hash, number)
                        )
    result = []
    for (a, b), matches in sorted(pairs.items()):
        if any(len({x[position] for x in matches}) < 3 for position in (0, 1, 3, 4)):
            continue
        result.append(
            {
                "left": {k: v for k, v in left[a].items() if k != "samples"},
                "right": {k: v for k, v in right[b].items() if k != "samples"},
                "slice_matches": [list(x) for x in sorted({x[:3] for x in matches})],
                "status": "coarse_pixel_overlap_candidate_original_review_required",
            }
        )
    return result
