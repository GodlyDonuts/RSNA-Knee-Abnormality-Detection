"""Cached feature dataset and dynamic padding for knee MRI studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from .constants import CACHE_SCHEMA_VERSION, TARGETS
except ImportError:  # Allow direct execution from this directory.
    from constants import CACHE_SCHEMA_VERSION, TARGETS


def load_feature_cache(path: str | Path) -> Dict[str, Tensor]:
    """Load a tensor-only cache produced by :mod:`extract_features`."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    required = {
        "features",
        "positions_mm",
        "slice_mask",
        "series_mask",
        "plane",
        "fluid",
        "fatsat",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path} is missing cache tensors: {sorted(missing)}")
    version = int(payload.get("schema_version", torch.tensor(-1)).item())
    if version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported cache schema {version} in {path}; expected {CACHE_SCHEMA_VERSION}"
        )
    return payload


class FeatureStudyDataset(Dataset[Dict[str, Any]]):
    """One cached variable-size series hierarchy per study.

    ``frame`` must contain ``StudyInstanceUID`` and ``cache_file``.  Labels and
    optional ``<target>__conf`` columns are read when present.  Missing labels
    remain masked rather than silently converted to negatives.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        targets: Sequence[str] = TARGETS,
        report_embedding_dir: Optional[str | Path] = None,
        include_patch_features: bool = True,
    ) -> None:
        if "StudyInstanceUID" not in frame or "cache_file" not in frame:
            raise ValueError("frame must contain StudyInstanceUID and cache_file")
        self.frame = frame.reset_index(drop=True).copy()
        self.targets = list(targets)
        self.report_embedding_dir = (
            Path(report_embedding_dir) if report_embedding_dir is not None else None
        )
        self.include_patch_features = include_patch_features

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.frame.iloc[index]
        item: Dict[str, Any] = dict(load_feature_cache(row["cache_file"]))
        if not self.include_patch_features:
            # Summary-model ablations share the same stage-two cache files but
            # should not pay the collation, retained host-RAM, or PCIe cost of
            # the optional spatial patch grid.  torch.load still deserializes
            # a combined cache; the full pipeline derives a compact summary
            # cache once to remove that disk-I/O cost as well.
            item.pop("patch_features", None)
            item.pop("patch_mask", None)
        uid = str(row["StudyInstanceUID"])
        item["uid"] = uid
        labels = []
        confidence = []
        label_mask = []
        gold_mask = []
        for target in self.targets:
            value = row.get(target, np.nan)
            valid = not pd.isna(value)
            labels.append(float(value) if valid else 0.0)
            label_mask.append(valid)
            conf_value = row.get(f"{target}__conf", 1.0)
            confidence.append(float(conf_value) if valid and not pd.isna(conf_value) else 0.0)
            gold_value = row.get(f"{target}__gold", 1.0 if valid else 0.0)
            gold_mask.append(bool(gold_value) if not pd.isna(gold_value) else False)
        item["labels"] = torch.tensor(labels, dtype=torch.float32)
        item["confidence"] = torch.tensor(confidence, dtype=torch.float32)
        item["label_mask"] = torch.tensor(label_mask, dtype=torch.bool)
        item["gold_mask"] = torch.tensor(gold_mask, dtype=torch.bool)

        if self.report_embedding_dir is not None:
            report_path = self.report_embedding_dir / f"{uid}.pt"
            if report_path.exists():
                try:
                    report = torch.load(report_path, map_location="cpu", weights_only=True)
                except TypeError:
                    report = torch.load(report_path, map_location="cpu")
                if isinstance(report, Mapping):
                    report = report["embedding"]
                item["report_embedding"] = torch.as_tensor(report, dtype=torch.float32)
                item["report_mask"] = torch.tensor(True)
            else:
                item["report_mask"] = torch.tensor(False)
        return item


def collate_studies(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad series and slices independently while retaining explicit masks."""

    if not items:
        raise ValueError("cannot collate an empty batch")
    batch = len(items)
    max_series = max(int(item["features"].shape[0]) for item in items)
    max_slices = max(int(item["features"].shape[1]) for item in items)
    feature_dim = int(items[0]["features"].shape[2])
    if any(int(item["features"].shape[2]) != feature_dim for item in items):
        raise ValueError("mixed DINO feature dimensions in one batch")

    features = torch.zeros(batch, max_series, max_slices, feature_dim, dtype=torch.float32)
    positions = torch.zeros(batch, max_series, max_slices, dtype=torch.float32)
    slice_mask = torch.zeros(batch, max_series, max_slices, dtype=torch.bool)
    series_mask = torch.zeros(batch, max_series, dtype=torch.bool)
    plane = torch.zeros(batch, max_series, dtype=torch.long)
    fluid = torch.zeros(batch, max_series, dtype=torch.long)
    fatsat = torch.zeros(batch, max_series, dtype=torch.long)
    has_patches = ["patch_features" in item for item in items]
    if any(has_patches) and not all(has_patches):
        raise ValueError("cannot mix summary-only and patch-token caches in one batch")
    patch_features = None
    patch_mask = None
    if all(has_patches):
        patches_per_slice = int(items[0]["patch_features"].shape[2])
        patch_dim = int(items[0]["patch_features"].shape[3])
        if any(
            item["patch_features"].shape[2:] != (patches_per_slice, patch_dim)
            for item in items
        ):
            raise ValueError("mixed patch grid or feature dimensions")
        patch_features = torch.zeros(
            batch,
            max_series,
            max_slices,
            patches_per_slice,
            patch_dim,
            dtype=torch.float32,
        )
        patch_mask = torch.zeros(
            batch, max_series, max_slices, patches_per_slice, dtype=torch.bool
        )

    for b, item in enumerate(items):
        r, s, _ = item["features"].shape
        features[b, :r, :s] = item["features"].float()
        positions[b, :r, :s] = item["positions_mm"].float()
        slice_mask[b, :r, :s] = item["slice_mask"].bool()
        series_mask[b, :r] = item["series_mask"].bool()
        plane[b, :r] = item["plane"].long()
        fluid[b, :r] = item["fluid"].long()
        fatsat[b, :r] = item["fatsat"].long()
        if patch_features is not None and patch_mask is not None:
            patch_features[b, :r, :s] = item["patch_features"].float()
            patch_mask[b, :r, :s] = item["patch_mask"].bool()

    out: Dict[str, Any] = {
        "features": features,
        "positions_mm": positions,
        "slice_mask": slice_mask,
        "series_mask": series_mask,
        "plane": plane,
        "fluid": fluid,
        "fatsat": fatsat,
        "labels": torch.stack([item["labels"] for item in items]),
        "confidence": torch.stack([item["confidence"] for item in items]),
        "label_mask": torch.stack([item["label_mask"] for item in items]),
        "gold_mask": torch.stack([item["gold_mask"] for item in items]),
        "uid": [item["uid"] for item in items],
    }
    if patch_features is not None and patch_mask is not None:
        out["patch_features"] = patch_features
        out["patch_mask"] = patch_mask
    if any("report_mask" in item for item in items):
        present = [bool(item.get("report_mask", False)) for item in items]
        if any(present):
            exemplar = next(item["report_embedding"] for item in items if "report_embedding" in item)
            report = torch.zeros(batch, exemplar.numel(), dtype=torch.float32)
            report_mask = torch.tensor(present, dtype=torch.bool)
            for b, item in enumerate(items):
                if "report_embedding" in item:
                    vector = item["report_embedding"].flatten().float()
                    if vector.numel() != report.shape[1]:
                        raise ValueError("mixed report embedding dimensions")
                    report[b] = vector
            out["report_embedding"] = report
            out["report_mask"] = report_mask
    return out


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def model_inputs(batch: Mapping[str, Any]) -> Dict[str, Tensor]:
    keys = (
        "features",
        "positions_mm",
        "slice_mask",
        "series_mask",
        "plane",
        "fluid",
        "fatsat",
    )
    return {key: batch[key] for key in keys}


def patch_model_inputs(batch: Mapping[str, Any]) -> Dict[str, Tensor]:
    values = model_inputs(batch)
    values["patch_features"] = batch["patch_features"]
    values["patch_mask"] = batch["patch_mask"]
    return values


def merge_cache_and_labels(
    cache_index: str | Path,
    labels_csv: Optional[str | Path],
) -> pd.DataFrame:
    cache = pd.read_csv(cache_index)
    if labels_csv is None:
        return cache
    labels = pd.read_csv(labels_csv)
    if "StudyInstanceUID" not in labels:
        raise ValueError("labels CSV must contain StudyInstanceUID")
    duplicate = [c for c in labels.columns if c in cache.columns and c != "StudyInstanceUID"]
    labels = labels.drop(columns=duplicate)
    merged = cache.merge(labels, on="StudyInstanceUID", how="inner", validate="one_to_one")
    if not len(merged):
        raise ValueError("cache index and labels CSV have no shared StudyInstanceUID")
    return merged
