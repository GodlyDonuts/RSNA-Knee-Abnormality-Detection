"""Verified cache sampling without repeated slices or invented physical positions."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

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
    ):
        if raw_split not in {"train_series", "test_series"}:
            raise ValueError("Unknown raw split")
        super().__init__(studies, series, Path(cache), slices, side, max_series, augment, raw_split)
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
        selected = select_series(self.series[uid], self.max_series, self.augment)
        images = torch.zeros(self.max_series, self.slices, 1, self.side, self.side)
        features = torch.zeros(self.max_series, 5)
        series_mask = torch.zeros(self.max_series, dtype=torch.bool)
        slice_mask = torch.zeros(self.max_series, self.slices, dtype=torch.bool)
        positions = torch.zeros(self.max_series, self.slices, 2)
        for slot, entry in enumerate(selected):
            sid = entry["SeriesInstanceUID"]
            path = self.cache / self.raw_split / uid / f"{sid}.npz"
            expected_meta_hash, meta = self._metadata[(uid, sid)]
            if (
                hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest()
                != expected_meta_hash
            ):
                raise ValueError("Cache metadata changed after dataset construction")
            payload = path.read_bytes()
            if len(payload) != meta.get("cache_bytes") or hashlib.sha256(
                payload
            ).hexdigest() != meta.get("cache_sha256"):
                raise ValueError("Cache volume checksum/size mismatch")
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                volume = archive["volume"]
            count = meta["n_slices"]
            if volume.dtype != np.uint16 or volume.shape != (count, meta["side"], meta["side"]):
                raise ValueError("Cache volume dtype/shape mismatch")
            indices = _indices(count, self.slices, self.augment)
            valid = len(indices)
            sampled = torch.from_numpy(volume[indices].astype(np.float32))[:, None] / 65535
            # Bilinear antialiasing can round a source value of 1 slightly above
            # 1 in FP32. Keep the normalized image contract exact for the model.
            images[slot, :valid] = F.interpolate(
                sampled,
                (self.side, self.side),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).clamp_(0, 1)
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
            or not ((supervised == 0) | (supervised == 1)).all()
        ):
            raise ValueError("Supervised targets must be finite binary values")
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
