"""Stream original sparse or every-source triplets without a dense tensor cache.

This changes coverage AND RGB context. It does not assert sparse/dense parent
predictions agree. Native gaps block path edges, never delete image windows.
The caller separately binds original metadata/cache bytes and selected series.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from rsna.competitive.criterion_geometry import within_series_relations
from rsna.competitive.neighbor_input import sample_neighbor_images

VERSION = "original_native_streamed_sparse16_or_dense_raw_triplets_v1"


def source_plan(metadata: dict, *, sampling: str, slot: int, metadata_sha256: str):
    if sampling not in {"sparse16", "dense"} or type(slot) is not int or not 0 <= slot < 6:
        raise ValueError("Explicit source sampling and original series slot required")
    if not re.fullmatch(r"[0-9a-f]{64}", metadata_sha256):
        raise ValueError("Original whole metadata-byte binding required")
    count = metadata.get("n_slices")
    if type(count) is not int or count < 1:
        raise ValueError("Positive original slice membership required")
    selected = (
        np.arange(count, dtype=np.int64)
        if sampling == "dense"
        else np.linspace(0, count - 1, min(16, count)).round().astype(np.int64)
    )
    relations = within_series_relations(metadata, selected)
    graph = {
        "slot": slot,
        "series_uid": metadata["series_uid"],
        "source_metadata_sha256": metadata_sha256,
        "selected_source_indices": selected.tolist(),
        "supports": relations.supports.tolist(),
        "centers_mm": relations.centers_mm.tolist(),
        "continuity_edges": relations.continuity_edges.tolist(),
        "continuity_gaps_mm": relations.continuity_gaps_mm.tolist(),
        "repeat_edges": relations.repeat_edges.tolist(),
        "repeat_gaps_mm": relations.repeat_gaps_mm.tolist(),
        "geometry_known": relations.geometry_known,
        "native_original_shape": metadata["original_shape"],
        "native_resized_shape": metadata["resized_shape"],
        "native_cache_side": metadata["side"],
        "pixel_spacing": metadata["pixel_spacing"],
        "image_orientation_patient": metadata["orientation"],
        "selected_image_positions_patient": [metadata["positions"][int(j)] for j in selected],
        "full_frame_letterbox_align_corners_false": True,
        "input_augmentation_or_mirror": False,
    }
    windows = np.asarray([(slot, j) for j in range(len(relations.supports))], dtype=np.int64)
    return graph, windows.reshape(-1, 2)


def tensor_chunks(volume, metadata, *, sampling, side=384, chunk_size=12):
    """Canonical per-slice float32 normalization/resize, chunked by window.

    At most 3*chunk_size source slices are normalized at a time. Outputs own
    their storage, contain no padding, preserve original channel chronology,
    and never cross series. Full compressed volume decoding remains per-series.
    """
    count, cache_side = metadata.get("n_slices"), metadata.get("side")
    if (
        not isinstance(volume, np.ndarray)
        or volume.dtype != np.uint16
        or type(count) is not int
        or type(cache_side) is not int
        or count < 1
        or cache_side < 1
        or volume.shape != (count, cache_side, cache_side)
        or type(side) is not int
        or side < 1
        or type(chunk_size) is not int
        or chunk_size < 1
        or sampling not in {"sparse16", "dense"}
    ):
        raise ValueError("Exact native uint16 volume and explicit bounded chunk policy required")
    selected = (
        np.arange(count, dtype=np.int64)
        if sampling == "dense"
        else np.linspace(0, count - 1, min(16, count)).round().astype(np.int64)
    )
    for first in range(0, max(0, len(selected) - 2), chunk_size):
        last = min(len(selected) - 2, first + chunk_size)
        supports = np.stack([selected[j : j + 3] for j in range(first, last)])
        indices, inverse = np.unique(supports.reshape(-1), return_inverse=True)
        gray = sample_neighbor_images(volume, metadata, indices, side, None)[:, 0]
        images = gray[torch.from_numpy(inverse)].reshape(last - first, 3, side, side).contiguous()
        yield first, supports, images
