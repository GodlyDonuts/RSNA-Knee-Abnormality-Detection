"""Inference-only extra physical source coverage with exact original-window subset.

This never changes the trained model, original centers, donor policy, labels,
contrast or field of view. Every original matched-neighbor window survives.
"""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import torch

from rsna.competitive.compact_source_neighbors import (
    MatchedSourceNeighborDataset,
    channel_plan,
)
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.neighbor_geometry import physical_neighbor_indices
from rsna.competitive.neighbor_input import sample_neighbor_images
from rsna.data import select_series

POLICY = {
    "version": "compact_neighbor_dense30_original_subset_v1",
    "maximum_centers_per_series": 30,
    "maximum_series": 6,
    "original_centers_and_fallbacks": "retain_exact_original_matched_neighbor_windows",
    "extra_centers": "both_original_physical_edges_admitted",
    "selection": "farthest_physical_depth_from_retained_centers_ties_lower_source_index",
    "new_training_or_label_based_selection": False,
}


def dense_plan(metadata: dict, maximum: int = 30):
    """Return source triples, original-window indices and legacy fallback count."""
    if type(maximum) is not int or not 14 <= maximum <= 30:
        raise ValueError("Explicit cap14–30 must preserve every original center")
    original, fallback = channel_plan(metadata)
    if not len(original):
        return original.copy(), np.empty(0, dtype=np.int64), 0
    previous, following = physical_neighbor_indices(metadata)
    count = metadata["n_slices"]
    centers = np.arange(count, dtype=np.int64)
    candidates = centers[(previous == centers - 1) & (following == centers + 1)]
    chosen = {int(t[1]): t.copy() for t in original}
    if len(candidates) and len(chosen) < maximum:
        orientation = np.asarray(metadata["orientation"], dtype=np.float64)
        normal = np.cross(orientation[:3], orientation[3:])
        normal /= np.linalg.norm(normal)
        depth = np.asarray(metadata["positions"], dtype=np.float64) @ normal
        remaining = set(candidates.tolist()) - set(chosen)
        while remaining and len(chosen) < maximum:
            center = max(
                remaining,
                key=lambda j: (min(abs(depth[j] - depth[c]) for c in chosen), -j),
            )
            chosen[center] = np.asarray([center - 1, center, center + 1], dtype=np.int64)
            remaining.remove(center)
    ordered = sorted(chosen)
    lookup = {c: i for i, c in enumerate(ordered)}
    original_indices = np.asarray([lookup[int(t[1])] for t in original], dtype=np.int64)
    triples = np.stack([chosen[c] for c in ordered])
    if len(triples) > maximum or not np.array_equal(triples[original_indices], original):
        raise ValueError("Dense coverage lost or changed an original source window")
    return triples, original_indices, int(fallback.sum())


def render_series(volume: np.ndarray, metadata: dict, side: int):
    if volume.dtype != np.uint16 or volume.shape != (
        metadata["n_slices"],
        metadata["side"],
        metadata["side"],
    ):
        raise ValueError("Original admitted native uint16 source volume required")
    triples, original, fallbacks = dense_plan(metadata)
    if not len(triples):
        return torch.empty(0, 3, side, side), original, fallbacks
    unique, inverse = np.unique(triples.reshape(-1), return_inverse=True)
    image = sample_neighbor_images(volume, metadata, unique, side, None)[:, 0]
    result = image[inverse].reshape(len(triples), 3, side, side)
    if not torch.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError("Finite unchanged native intensity mapping required")
    return result, original, fallbacks


class DenseSourceNeighborDataset:
    """Image-only bounded view; target/mask fields are deliberately never read."""

    inference_policy = POLICY

    def __init__(self, base: CompetitiveStudyDataset):
        MatchedSourceNeighborDataset(base)  # Require the exact existing native384 dataset contract.
        self.base = base
        self.studies = base.studies

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        uid = self.base.studies.iloc[index].study_uid
        windows, original_indices = [], []
        offset, fallbacks = 0, 0
        digest = hashlib.sha256()
        for entry in select_series(self.base.series[uid], 6, False):
            sid = entry["SeriesInstanceUID"]
            expected, metadata = self.base._metadata[(uid, sid)]
            path = self.base.cache / self.base.raw_split / uid / f"{sid}.npz"
            if hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest() != expected:
                raise ValueError("Original source metadata changed")
            payload = (
                path.read_bytes()
                if self.base.volume_provider is None
                else self.base.volume_provider(metadata)
            )
            if (
                len(payload) != metadata["cache_bytes"]
                or hashlib.sha256(payload).hexdigest() != metadata["cache_sha256"]
            ):
                raise ValueError("Original source pixel bytes changed")
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                volume = archive["volume"]
            values, original, legacy = render_series(volume, metadata, 384)
            windows.append(values)
            original_indices.extend((original + offset).tolist())
            offset += len(values)
            fallbacks += legacy
            triples, _, _ = dense_plan(metadata)
            digest.update(json.dumps([sid, triples.tolist()], separators=(",", ":")).encode())
        values = torch.cat(windows) if windows else torch.empty(0, 3, 384, 384)
        if len(values) > 180:
            raise ValueError("Dense study exceeds exact six-series/30-center budget")
        return {
            "study_uid": uid,
            "dense_windows": values.mul(2).sub(1),
            "original_window_indices": torch.tensor(original_indices, dtype=torch.int64),
            "legacy_edge_fallback_windows": fallbacks,
            "source_plan_sha256": digest.hexdigest(),
            "inference_policy": dict(POLICY),
        }


@torch.inference_mode()
def prediction_rows(
    model, data: DenseSourceNeighborDataset, expected_ids, *, device, chunk_size=12
):
    if (
        not isinstance(data, DenseSourceNeighborDataset)
        or model.training
        or any(m.training for m in model.modules())
    ):
        raise ValueError("Explicit dense source view and saved-statistic evaluation mode required")
    if (
        type(chunk_size) is not int
        or not 1 <= chunk_size <= 12
        or not expected_ids
        or len(set(expected_ids)) != len(expected_ids)
    ):
        raise ValueError("Complete distinct roster and bounded encoder chunk1–12 required")
    allowed = set(expected_ids)
    rows = {}
    count = 0
    model.to(device)
    for sample in data:
        uid = sample["study_uid"]
        windows = sample["dense_windows"]
        if uid not in allowed or uid in rows or sample["inference_policy"] != POLICY:
            raise ValueError("Unexpected/duplicate study or unadmitted coverage policy")
        if not len(windows):
            raise ValueError(
                "Study has no original or dense source windows; explicit serving fallback needed"
            )
        p = model(windows.to(device), chunk_size=chunk_size).sigmoid().float().cpu()
        if p.shape != (12,) or not torch.isfinite(p).all():
            raise ValueError("Finite twelve-target MRI probability vector required")
        rows[uid] = p.tolist()
        count += len(windows)
    if set(rows) != allowed:
        raise ValueError("Incomplete dense prediction roster")
    return {
        "rows": {uid: rows[uid] for uid in expected_ids},
        "encoded_windows": count,
        "inference_policy": dict(POLICY),
        "target_runtime_qualified": False,
    }
