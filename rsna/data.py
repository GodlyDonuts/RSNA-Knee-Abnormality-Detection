from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from rsna.schema import CACHE_VERSION, TARGETS

PLANES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def select_series(candidates: list[dict], max_series: int, augment: bool) -> list[dict]:
    """Preserve plane coverage, with reviewed tier-1 alternatives selected last.

    Tiers affect sampling order only. They are not model features, source edits,
    or claims of equivalent clinical coverage. Stable sorting preserves the
    previous ordering and RNG consumption when every tier is zero or absent.
    """
    candidates = list(candidates)
    if augment:
        candidates = [candidates[i] for i in torch.randperm(len(candidates)).tolist()]
    else:
        candidates.sort(
            key=lambda r: (-r["Fluid_Sensitive"], -r["Fat_Suppression"], r["SeriesInstanceUID"])
        )
    candidates.sort(key=lambda r: r.get("Selection_Tier", 0))
    selected = []
    for plane in PLANES:
        match = next((r for r in candidates if r["Anatomical_Plane"] == plane), None)
        if match is not None:
            selected.append(match)
    selected.extend(r for r in candidates if r not in selected)
    return selected[:max_series]


class StudyDataset(Dataset):
    def __init__(
        self,
        studies: pd.DataFrame,
        series: pd.DataFrame,
        cache: Path,
        slices: int = 24,
        side: int = 224,
        max_series: int = 8,
        augment: bool = False,
        raw_split: str = "train_series",
        metadata_only: bool = False,
    ):
        if slices < 1 or side < 16 or max_series < 3:
            raise ValueError("Need positive slices, side >=16, max_series >=3")
        if studies.study_uid.duplicated().any():
            raise ValueError("Study sampling requires unique study IDs")
        if series.SeriesInstanceUID.duplicated().any():
            raise ValueError("Series IDs must be unique")
        if "Selection_Tier" in series and (
            not pd.api.types.is_integer_dtype(series.Selection_Tier.dtype)
            or not series.Selection_Tier.isin([0, 1]).all()
        ):
            raise ValueError("Selection_Tier must contain integer 0 or 1 for every series")
        self.studies = studies.reset_index(drop=True)
        self.cache = cache
        self.slices = slices
        self.side = side
        self.max_series = max_series
        self.augment = augment
        self.raw_split = raw_split
        self.series = {
            uid: frame.sort_values("SeriesInstanceUID").to_dict("records")
            for uid, frame in series.groupby("StudyInstanceUID")
        }
        configurations = set()
        for uid in self.studies.study_uid:
            if uid not in self.series:
                raise ValueError(f"No series: {uid}")
            for row in self.series[uid]:
                if row["Anatomical_Plane"] not in PLANES:
                    raise ValueError(f"Unknown plane: {row['Anatomical_Plane']}")
                path = cache / raw_split / uid / f"{row['SeriesInstanceUID']}.npz"
                if not path.with_suffix(".json").exists() or (
                    not metadata_only and not path.exists()
                ):
                    raise ValueError(f"Missing preprocessed series: {path}")
                metadata = json.loads(path.with_suffix(".json").read_text())
                if metadata["cache_version"] != CACHE_VERSION:
                    raise ValueError("Unsupported series cache version")
                if (
                    metadata["study_uid"] != uid
                    or metadata["series_uid"] != row["SeriesInstanceUID"]
                ):
                    raise ValueError("Series cache identity mismatch")
                configurations.add((metadata["side"], metadata["source_revision"]))
        if len(configurations) != 1:
            raise ValueError("Dataset must use one cache resolution and source revision")
        self.cache_side, self.source_revision = configurations.pop()

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.studies.iloc[index]
        uid = row.study_uid
        selected = select_series(self.series[uid], self.max_series, self.augment)
        images = torch.zeros(self.max_series, self.slices, 1, self.side, self.side)
        features = torch.zeros(self.max_series, 5)
        series_mask = torch.zeros(self.max_series, dtype=torch.bool)
        flip = self.augment and torch.rand(()).item() < 0.5
        for slot, entry in enumerate(selected):
            path = self.cache / self.raw_split / uid / f"{entry['SeriesInstanceUID']}.npz"
            meta = json.loads(path.with_suffix(".json").read_text())
            if meta["cache_version"] != CACHE_VERSION:
                raise ValueError("Unsupported series cache version")
            with np.load(path, allow_pickle=False) as archive:
                volume = archive["volume"]
            if volume.dtype != np.uint16 or volume.ndim != 3 or not len(volume):
                raise ValueError(f"Invalid volume: {path}")
            if self.augment:
                edges = torch.linspace(0, len(volume), self.slices + 1)
                locations = edges[:-1] + torch.rand(self.slices) * (edges[1:] - edges[:-1])
                indices = locations.long().clamp(max=len(volume) - 1).numpy()
            else:
                indices = np.linspace(0, len(volume) - 1, self.slices).round().astype(int)
            sampled = torch.from_numpy(volume[indices].astype(np.float32))[:, None] / 65535
            resized = F.interpolate(
                sampled,
                (self.side, self.side),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            if flip:
                resized = resized.flip(-1)
            images[slot] = resized
            features[slot, PLANES[entry["Anatomical_Plane"]]] = 1
            features[slot, 3] = entry["Fluid_Sensitive"]
            features[slot, 4] = entry["Fat_Suppression"]
            series_mask[slot] = True
        labels = torch.tensor([float(row[t]) if t in row else 0.0 for t in TARGETS])
        label_mask = torch.tensor(
            [float(row[f"{t}__mask"]) if f"{t}__mask" in row else 0.0 for t in TARGETS]
        )
        return {
            "images": images,
            "features": features,
            "series_mask": series_mask,
            "labels": labels,
            "label_mask": label_mask,
            "study_uid": uid,
        }
