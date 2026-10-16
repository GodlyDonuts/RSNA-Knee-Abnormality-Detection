"""Source-neighbor donor contrast at N3's EXACT original candidate centers.

This does not expand coverage, change bag RNG, or change the live N3 recipe.
Unusable physical edges retain the original selected-sample triple explicitly.
"""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import torch

from rsna.competitive.data import CompetitiveStudyDataset, _indices
from rsna.competitive.neighbor_geometry import physical_neighbor_indices
from rsna.competitive.neighbor_input import sample_neighbor_images
from rsna.data import select_series
from rsna.schema import TARGETS

POLICY = {
    "version": "compact_matched_centers_source_neighbors_v1",
    "centers": "original_uniform16_interior_centers",
    "physical_edges": "positive_le1.5median_inplane_le0.001mm",
    "unusable_edge": "retain_original_selected_sample_triple",
    "training_bag_seed": "unchanged_original_N3",
}
ENGINE = "native_coatnet_bag12_matched_source_neighbors_v1"


def channel_plan(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Keep every old middle channel; change ONLY the two donor slices."""
    count = metadata["n_slices"]
    if type(count) is not int or count < 1:
        raise ValueError("Positive original source slice count required")
    selected = _indices(count, 16, False)
    previous, following = physical_neighbor_indices(metadata)
    triples, fallback = [], []
    for index in range(max(0, len(selected) - 2)):
        center = int(selected[index + 1])
        admitted = previous[center] == center - 1 and following[center] == center + 1
        triples.append(
            [center - 1, center, center + 1] if admitted else selected[index : index + 3].tolist()
        )
        fallback.append(not admitted)
    return np.asarray(triples, dtype=np.int64).reshape(-1, 3), np.asarray(fallback, dtype=bool)


def source_window_tensor(volume: np.ndarray, metadata: dict, side: int) -> tuple[torch.Tensor, int]:
    """Use the original native grayscale loader/normalization, without new contrast."""
    if volume.dtype != np.uint16 or volume.shape != (
        metadata["n_slices"],
        metadata["side"],
        metadata["side"],
    ):
        raise ValueError("Original verified native uint16 volume required")
    triples, fallback = channel_plan(metadata)
    if not len(triples):
        return torch.empty(0, 3, side, side), 0
    unique, inverse = np.unique(triples.reshape(-1), return_inverse=True)
    images = sample_neighbor_images(volume, metadata, unique, side, None)[:, 0]
    return images[inverse].reshape(len(triples), 3, side, side), int(fallback.sum())


class MatchedSourceNeighborDataset:
    """A verified native dataset view; does not reread/decode the legacy view."""

    input_policy = POLICY

    def __init__(self, base: CompetitiveStudyDataset):
        if (
            not isinstance(base, CompetitiveStudyDataset)
            or base.augment
            or base.side != 384
            or base.slices != 16
            or base.max_series != 6
            or base.slice_sampling != "uniform"
            or base.series_sampling != "selected"
            or base.neighbor_channels is not None
            or base.cache_contrast is not None
        ):
            raise ValueError("Exactly the original fixed native384 grayscale N3 dataset required")
        self.base = base
        self.studies, self.series, self._metadata = base.studies, base.series, base._metadata

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.studies.iloc[index]
        uid = row.study_uid
        images = torch.zeros(6, 14, 3, 384, 384)
        series_mask = torch.zeros(6, dtype=torch.bool)
        slice_mask = torch.zeros(6, 16, dtype=torch.bool)
        fallback_count, plan_digest = 0, hashlib.sha256()
        for slot, entry in enumerate(select_series(self.series[uid], 6, False)):
            sid = entry["SeriesInstanceUID"]
            path = self.base.cache / self.base.raw_split / uid / f"{sid}.npz"
            expected, metadata = self._metadata[(uid, sid)]
            if hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest() != expected:
                raise ValueError("Original metadata changed")
            payload = (
                path.read_bytes()
                if self.base.volume_provider is None
                else self.base.volume_provider(metadata)
            )
            if len(payload) != metadata["cache_bytes"] or (
                hashlib.sha256(payload).hexdigest() != metadata["cache_sha256"]
            ):
                raise ValueError("Original cache volume bytes changed")
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                volume = archive["volume"]
            values, fallbacks = source_window_tensor(volume, metadata, 384)
            images[slot, : len(values)] = values
            series_mask[slot] = True
            slice_mask[slot, : min(metadata["n_slices"], 16)] = True
            fallback_count += fallbacks
            triples, _ = channel_plan(metadata)
            plan_digest.update(json.dumps([sid, triples.tolist()], separators=(",", ":")).encode())
        labels = torch.tensor([float(row[t]) for t in TARGETS])
        mask = torch.tensor([float(row[t + "__mask"]) for t in TARGETS])
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Original binary label masks required")
        observed = labels[mask.bool()]
        if not torch.isfinite(observed).all() or ((observed < 0) | (observed > 1)).any():
            raise ValueError("Original observed probabilities required")
        if not self.base.soft_targets and not ((observed == 0) | (observed == 1)).all():
            raise ValueError("Original binary evaluation targets required")
        return {
            "source_window_images": images,
            "series_mask": series_mask,
            "slice_mask": slice_mask,
            "study_uid": uid,
            "labels": torch.where(mask.bool(), labels, torch.zeros_like(labels)),
            "label_mask": mask,
            "source_window_plan_sha256": plan_digest.hexdigest(),
            "legacy_edge_fallback_windows": fallback_count,
        }


def select_window_tensor(sample: dict, windows: list[tuple[int, int]]) -> torch.Tensor:
    images = sample["source_window_images"]
    if (
        images.ndim != 5
        or images.shape[2] != 3
        or not images.is_floating_point()
        or not torch.isfinite(images).all()
        or images.min() < 0
        or images.max() > 1
        or not windows
        or len(set(windows)) != len(windows)
    ):
        raise ValueError("Finite unique native source windows required")
    from rsna.competitive.compact_bag import candidate_windows

    candidates = set(candidate_windows(sample["series_mask"], sample["slice_mask"]))
    if any(window not in candidates for window in windows):
        raise ValueError("Source window outside original N3 candidates")
    return torch.stack([images[s, j] for s, j in windows]).mul(2).sub(1)
