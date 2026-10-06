"""Verified cache sampling without repeated slices or invented physical positions."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import default_collate

from rsna.competitive.neighbor_input import neighbor_input_channels, sample_neighbor_images
from rsna.data import PLANES, StudyDataset, select_series
from rsna.schema import TARGETS


def _positions(metadata: dict, count: int) -> Tensor:
    result = torch.zeros(count, 2)
    if metadata.get("ordering") != "patient_geometry":
        return result
    orientation = np.asarray(metadata.get("orientation", []), dtype=np.float64)
    coordinates = np.asarray(metadata.get("positions", []), dtype=np.float64)
    if orientation.shape != (6,) or coordinates.shape != (count, 3):
        raise ValueError("Patient geometry has incomplete positions/orientation")
    if not np.isfinite(orientation).all() or not np.isfinite(coordinates).all():
        raise ValueError("Nonfinite patient geometry")
    row, col = orientation[:3], orientation[3:]
    if not np.allclose(
        [np.linalg.norm(row), np.linalg.norm(col), np.dot(row, col)], [1, 1, 0], atol=1e-3
    ):
        raise ValueError("Invalid patient orientation basis")
    normal = np.cross(row, col)
    normal /= np.linalg.norm(normal)
    projected = coordinates @ normal
    if count > 1 and (np.diff(projected) <= 0).any():
        raise ValueError("Patient-geometry cache slices must be strictly ordered")
    result[:, 0] = torch.from_numpy(
        ((projected - (projected[0] + projected[-1]) / 2) / 100).astype(np.float32)
    )
    result[:, 1] = 1
    return result


def _indices(count: int, requested: int, augment: bool) -> np.ndarray:
    selected = min(count, requested)
    if not augment:
        return np.linspace(0, count - 1, selected).round().astype(np.int64)
    # Disjoint integer bins ensure no repeated source slices, including short series.
    edges = np.linspace(0, count, selected + 1).astype(np.int64)
    return np.array(
        [torch.randint(int(a), int(b), ()).item() for a, b in zip(edges[:-1], edges[1:])]
    )


def evaluation_slice_sampling(policy: str) -> str:
    """Training-only stratification evaluates with the fixed uniform slice grid."""
    if policy not in {"uniform", "stratified", "all"}:
        raise ValueError("Unknown slice coverage policy")
    return "uniform" if policy == "stratified" else policy


class CompetitiveStudyDataset(StudyDataset):
    def __init__(
        self,
        studies: pd.DataFrame,
        series: pd.DataFrame,
        cache: Path,
        slices: int = 16,
        side: int = 224,
        max_series: int = 6,
        augment: bool = False,
        raw_split: str = "train_series",
        slice_sampling: str = "uniform",
        series_sampling: str = "selected",
        neighbor_channels: dict | None = None,
        volume_provider: Callable[[dict], bytes] | None = None,
        sides: pd.DataFrame | None = None,
    ):
        if raw_split not in {"train_series", "test_series"}:
            raise ValueError("Unknown raw split")
        if slice_sampling not in {"uniform", "stratified", "all"} or series_sampling not in {
            "selected",
            "all",
        }:
            raise ValueError("Unknown slice or series coverage policy")
        if slice_sampling == "stratified" and (augment or raw_split != "train_series"):
            raise ValueError("Stratified slices require training data with fixed series selection")
        self.input_channels = neighbor_input_channels(neighbor_channels)
        if neighbor_channels is not None and (
            augment or slice_sampling != "uniform" or series_sampling != "selected"
        ):
            raise ValueError("Neighbor input contrast requires fixed uniform selected-series views")
        self.neighbor_channels = None if neighbor_channels is None else dict(neighbor_channels)
        super().__init__(
            studies,
            series,
            Path(cache),
            slices,
            side,
            max_series,
            augment,
            raw_split,
            metadata_only=volume_provider is not None,
        )
        self.volume_provider = volume_provider
        self.slice_sampling, self.series_sampling = slice_sampling, series_sampling
        self._canonical_flip: dict[str, bool] = {}
        if sides is not None:
            if list(sides.columns) != ["study_uid", "flip"]:
                raise ValueError("Canonical-side table must be study_uid/flip only")
            if sides.study_uid.duplicated().any() or len(sides.study_uid) == 0:
                raise ValueError("Canonical-side studies must be unique and nonempty")
            if not pd.api.types.is_bool_dtype(sides["flip"]):
                raise ValueError("Canonical-side flip values must be boolean")
            requested = set(sides.study_uid)
            roster = set(self.studies.study_uid)
            if roster - requested:
                raise ValueError("Canonical-side table does not cover the study roster")
            self._canonical_flip = {
                uid: bool(flip) for uid, flip in zip(sides.study_uid, sides["flip"])
            }
        self._metadata: dict[tuple[str, str], tuple[str, dict]] = {}
        for uid in self.studies.study_uid:
            for entry in self.series[uid]:
                series_uid = entry["SeriesInstanceUID"]
                path = self.cache / raw_split / uid / f"{series_uid}.npz"
                payload = path.with_suffix(".json").read_bytes()
                meta = json.loads(payload)
                if (
                    meta.get("split") != raw_split
                    or meta.get("cache_path") != path.relative_to(self.cache).as_posix()
                ):
                    raise ValueError("Cache path/split binding mismatch")
                count = meta.get("n_slices")
                if (
                    not isinstance(count, int)
                    or count < 1
                    or meta.get("original_shape", [])[:1] != [count]
                ):
                    raise ValueError("Invalid source slice count")
                sops = meta.get("sop_uids", [])
                hashes = meta.get("slice_sha256", [])
                if len(sops) != count or len(set(sops)) != count or len(hashes) != count:
                    raise ValueError("Incomplete source SOP/hash membership")
                if any(
                    not isinstance(v, str)
                    or len(v) != 64
                    or any(c not in "0123456789abcdef" for c in v)
                    for v in hashes
                ):
                    raise ValueError("Invalid source SHA256 membership")
                paths = sorted(f"{raw_split}/{uid}/{series_uid}/{sop}.dcm" for sop in sops)
                fingerprint = hashlib.sha256("\n".join(paths).encode()).hexdigest()
                if meta.get("path_fingerprint") != fingerprint:
                    raise ValueError("Source path fingerprint mismatch")
                _positions(meta, count)
                for key in ["Fluid_Sensitive", "Fat_Suppression"]:
                    if entry[key] not in (0, 1):
                        raise ValueError("Series protocol flags must be binary")
                self._metadata[(uid, series_uid)] = (hashlib.sha256(payload).hexdigest(), meta)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.studies.iloc[index]
        uid = row.study_uid
        slots = len(self.series[uid]) if self.series_sampling == "all" else self.max_series
        selected = select_series(self.series[uid], slots, self.augment)
        depth = (
            max(
                self._metadata[(uid, entry["SeriesInstanceUID"])][1]["n_slices"]
                for entry in selected
            )
            if self.slice_sampling == "all"
            else self.slices
        )
        images = torch.zeros(slots, depth, self.input_channels, self.side, self.side)
        features = torch.zeros(slots, 5)
        series_mask = torch.zeros(slots, dtype=torch.bool)
        slice_mask = torch.zeros(slots, depth, dtype=torch.bool)
        positions = torch.zeros(slots, depth, 2)
        for slot, entry in enumerate(selected):
            sid = entry["SeriesInstanceUID"]
            path = self.cache / self.raw_split / uid / f"{sid}.npz"
            expected_meta_hash, meta = self._metadata[(uid, sid)]
            if (
                hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest()
                != expected_meta_hash
            ):
                raise ValueError("Cache metadata changed after dataset construction")
            payload = (
                path.read_bytes() if self.volume_provider is None else self.volume_provider(meta)
            )
            if len(payload) != meta.get("cache_bytes") or hashlib.sha256(
                payload
            ).hexdigest() != meta.get("cache_sha256"):
                raise ValueError("Cache volume checksum/size mismatch")
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                volume = archive["volume"]
            count = meta["n_slices"]
            if volume.dtype != np.uint16 or volume.shape != (count, meta["side"], meta["side"]):
                raise ValueError("Cache volume dtype/shape mismatch")
            indices = (
                np.arange(count)
                if self.slice_sampling == "all"
                else _indices(
                    count, self.slices, self.augment or self.slice_sampling == "stratified"
                )
            )
            valid = len(indices)
            sampled = sample_neighbor_images(
                volume, meta, indices, self.side, self.neighbor_channels
            )
            if self._canonical_flip.get(uid):
                sampled = sampled.flip(-1)
            images[slot, :valid] = sampled
            positions[slot, :valid] = _positions(meta, count)[indices]
            features[slot, PLANES[entry["Anatomical_Plane"]]] = 1
            features[slot, 3] = entry["Fluid_Sensitive"]
            features[slot, 4] = entry["Fat_Suppression"]
            series_mask[slot] = True
            slice_mask[slot, :valid] = True
        labels = torch.tensor(
            [float(row[t]) if t in row else 0 for t in TARGETS], dtype=torch.float32
        )
        label_mask = torch.tensor(
            [float(row[f"{t}__mask"]) if f"{t}__mask" in row else 0 for t in TARGETS],
            dtype=torch.float32,
        )
        if (
            not torch.isfinite(label_mask).all()
            or not ((label_mask == 0) | (label_mask == 1)).all()
        ):
            raise ValueError("Label masks must be finite binary values")
        supervised = labels[label_mask.bool()]
        if (
            not torch.isfinite(supervised).all()
            or ((supervised < 0) | (supervised > 1)).any()
        ):
            raise ValueError("Supervised targets must be finite probabilities in [0,1]")
        labels = torch.where(label_mask.bool(), labels, torch.zeros_like(labels))
        return dict(
            images=images,
            features=features,
            series_mask=series_mask,
            slice_mask=slice_mask,
            positions=positions,
            labels=labels,
            label_mask=label_mask,
            study_uid=uid,
        )


def collate_studies(rows: list[dict]) -> dict:
    """Pad variable series/depth for batching; no source slice is repeated or dropped."""
    if not rows:
        raise ValueError("Cannot collate an empty study batch")
    shapes = [row["images"].shape for row in rows]
    first = rows[0]["images"]
    if any(len(shape) != 5 or min(shape) < 1 or shape[2:] != shapes[0][2:] for shape in shapes):
        raise ValueError("Studies need compatible [S,K,C,H,W] images")
    if shapes[0][2] not in {1, 3} or not first.is_floating_point():
        raise ValueError("Study images must be floating grayscale or bound neighbor channels")
    variable = ("images", "features", "series_mask", "slice_mask", "positions")
    for row in rows:
        s, k = row["images"].shape[:2]
        expected = {
            "features": (s, 5),
            "series_mask": (s,),
            "slice_mask": (s, k),
            "positions": (s, k, 2),
        }
        if any(row[name].shape != shape for name, shape in expected.items()):
            raise ValueError("Study metadata shape differs from images")
        if row["images"].dtype != first.dtype or any(
            row[name].device != first.device for name in variable
        ):
            raise ValueError("Collated image dtype and input devices must match")
        if row["slice_mask"].dtype != torch.bool or row["series_mask"].dtype != torch.bool:
            raise ValueError("Study masks must be boolean")
    if len(set(shapes)) == 1:
        return default_collate(rows)
    b, s, k = len(rows), max(shape[0] for shape in shapes), max(shape[1] for shape in shapes)
    batch = default_collate(
        [{name: value for name, value in row.items() if name not in variable} for row in rows]
    )
    batch.update(
        images=first.new_zeros((b, s, k, *first.shape[2:])),
        features=rows[0]["features"].new_zeros((b, s, 5)),
        positions=rows[0]["positions"].new_zeros((b, s, k, 2)),
        series_mask=torch.zeros((b, s), dtype=torch.bool, device=first.device),
        slice_mask=torch.zeros((b, s, k), dtype=torch.bool, device=first.device),
    )
    for index, row in enumerate(rows):
        size, depth = row["images"].shape[:2]
        batch["images"][index, :size, :depth] = row["images"]
        batch["features"][index, :size] = row["features"]
        batch["positions"][index, :size, :depth] = row["positions"]
        batch["series_mask"][index, :size] = row["series_mask"]
        batch["slice_mask"][index, :size, :depth] = row["slice_mask"]
    return batch
