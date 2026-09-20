"""Raw-DICOM, high-resolution multiple-instance models for knee MRI.

The cached-feature experiments establish a cheap baseline. This module is the
deliberately different family used for the expensive stage: one selected
series per anatomical plane, stratified physical slices, trainable image
features, and target-wise pooling over slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from .constants import PLANE_TO_ID
    from .dino_adapter import freeze_except_last_blocks
    from .extract_features import (
        _crop_physical,
        _laterality,
        _normal_and_position,
        _pixel_array,
        _plane_id,
        _tristate,
    )
except ImportError:
    from constants import PLANE_TO_ID
    from dino_adapter import freeze_except_last_blocks
    from extract_features import (
        _crop_physical,
        _laterality,
        _normal_and_position,
        _pixel_array,
        _plane_id,
        _tristate,
    )


@dataclass(frozen=True)
class SeriesRecord:
    uid: str
    plane: int
    fluid: int
    fatsat: int
    directory: str


@lru_cache(maxsize=16384)
def _directory_paths(directory: str) -> tuple[str, ...]:
    paths = tuple(str(path) for path in sorted(Path(directory).glob("*.dcm")))
    if not paths:
        raise FileNotFoundError(f"series directory contains no DICOMs: {directory}")
    return paths


def build_study_manifest(
    series_csv: Path, data_root: Path, split: str = "train"
) -> dict[str, tuple[SeriesRecord, ...]]:
    table = pd.read_csv(series_csv)
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"}
    if missing := required.difference(table.columns):
        raise ValueError(f"series table is missing columns: {sorted(missing)}")
    studies: dict[str, tuple[SeriesRecord, ...]] = {}
    for study_uid, rows in table.groupby("StudyInstanceUID", sort=True):
        records: list[SeriesRecord] = []
        for _, row in rows.iterrows():
            series_uid = str(row["SeriesInstanceUID"])
            directory = data_root / f"{split}_series" / str(study_uid) / series_uid
            records.append(
                SeriesRecord(
                    uid=series_uid,
                    plane=_plane_id(row.get("Anatomical_Plane")),
                    fluid=_tristate(row.get("Fluid_Sensitive")),
                    fatsat=_tristate(row.get("Fat_Suppression")),
                    directory=str(directory),
                )
            )
        if records:
            studies[str(study_uid)] = tuple(records)
    return studies


@lru_cache(maxsize=8192)
def _ordered_headers(directory: str) -> tuple[tuple[str, float], ...]:
    import pydicom

    rows: list[tuple[float, int, str]] = []
    physical = True
    for ordinal, path in enumerate(_directory_paths(directory)):
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        _, position = _normal_and_position(ds)
        if position is None:
            physical = False
            try:
                position = float(ds.InstanceNumber)
            except (AttributeError, TypeError, ValueError):
                position = float(ordinal)
        rows.append((float(position), ordinal, path))
    rows.sort(key=lambda row: (row[0], row[1]))
    if not physical:
        return tuple((path, float(index)) for index, (_, _, path) in enumerate(rows))
    return tuple((path, position) for position, _, path in rows)


def _stratified_indices(length: int, count: int, training: bool) -> np.ndarray:
    if length <= count:
        return np.arange(length, dtype=np.int64)
    if not training:
        return np.unique(np.rint(np.linspace(0, length - 1, count)).astype(np.int64))
    edges = np.linspace(0, length, count + 1)
    indices = []
    for left, right in zip(edges[:-1], edges[1:]):
        lo = int(np.floor(left))
        hi = max(lo + 1, int(np.ceil(right)))
        indices.append(int(torch.randint(lo, min(hi, length), (1,)).item()))
    return np.asarray(indices, dtype=np.int64)


def _resize(array: np.ndarray, image_size: int) -> Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
    return F.interpolate(
        tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
    )[0, 0]


def load_series_tensor(
    record: SeriesRecord,
    slices: int,
    crop_mm: float,
    image_size: int,
    training: bool,
) -> tuple[Tensor, Tensor]:
    import pydicom

    ordered = _ordered_headers(record.directory)
    centers = _stratified_indices(len(ordered), slices, training)
    needed = sorted(
        {
            max(0, min(len(ordered) - 1, int(index) + offset))
            for index in centers
            for offset in (-1, 0, 1)
        }
    )
    decoded: dict[int, tuple[Any, np.ndarray]] = {}
    for index in needed:
        try:
            ds = pydicom.dcmread(ordered[index][0], force=True)
            decoded[index] = (ds, _crop_physical(_pixel_array(ds), ds, crop_mm))
        except Exception as error:
            print(
                f"warning: skipping unreadable DICOM {ordered[index][0]}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
    if not decoded:
        raise RuntimeError(f"all selected frames are unreadable in {record.directory}")
    readable = np.asarray(sorted(decoded), dtype=np.int64)

    def nearest(index: int) -> int:
        return int(readable[np.abs(readable - index).argmin()])

    centers = np.asarray([nearest(int(index)) for index in centers], dtype=np.int64)
    arrays = [value[1] for value in decoded.values()]
    low = float(np.median([np.percentile(array, 1) for array in arrays]))
    high = float(np.median([np.percentile(array, 99) for array in arrays]))
    if not np.isfinite(high - low) or high - low < 1e-6:
        low = float(min(array.min() for array in arrays))
        high = float(max(array.max() for array in arrays))
    scale = max(high - low, 1e-6)

    def normalized(index: int) -> Tensor:
        array = np.clip((decoded[index][1] - low) / scale, 0.0, 1.0)
        return _resize(array, image_size)

    images = []
    for center in centers:
        triplet = [
            normalized(nearest(max(0, int(center) - 1))),
            normalized(int(center)),
            normalized(nearest(min(len(ordered) - 1, int(center) + 1))),
        ]
        images.append(torch.stack(triplet))
    pixels = torch.stack(images)
    positions = torch.tensor([ordered[int(index)][1] for index in centers], dtype=torch.float32)

    first_ds = decoded[int(centers[0])][0]
    if _laterality(first_ds) == "R":
        if record.plane in {PLANE_TO_ID["coronal"], PLANE_TO_ID["axial"]}:
            pixels = pixels.flip(-1)
        elif record.plane == PLANE_TO_ID["sagittal"]:
            pixels = pixels.flip(0)
            positions = positions.flip(0)
    if training:
        gain = 0.9 + 0.2 * torch.rand(1)
        bias = 0.04 * (2 * torch.rand(1) - 1)
        pixels = (pixels * gain + bias).clamp(0, 1)
    return pixels, positions


class RawStudyDataset(Dataset[dict[str, Any]]):
    """Decode one series per available plane without creating a pixel cache."""

    def __init__(
        self,
        frame: pd.DataFrame,
        manifest: dict[str, tuple[SeriesRecord, ...]],
        targets: Sequence[str],
        image_size: int,
        slices_per_plane: int,
        crop_mm: float,
        training: bool,
        max_series_per_plane: int = 1,
        report_embeddings: dict[str, Tensor] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.manifest = manifest
        self.targets = list(targets)
        self.image_size = image_size
        self.slices_per_plane = slices_per_plane
        self.crop_mm = crop_mm
        self.training = training
        if max_series_per_plane < 1:
            raise ValueError("max_series_per_plane must be positive")
        self.max_series_per_plane = max_series_per_plane
        self.report_embeddings = report_embeddings
        missing = [uid for uid in self.frame["StudyInstanceUID"].astype(str) if uid not in manifest]
        if missing:
            raise ValueError(f"{len(missing)} labeled studies have no readable series manifest")

    def __len__(self) -> int:
        return len(self.frame)

    def _choose_series(self, uid: str) -> list[SeriesRecord]:
        by_plane: dict[int, list[SeriesRecord]] = {}
        for record in self.manifest[uid]:
            by_plane.setdefault(record.plane, []).append(record)
        selected = []
        for plane in sorted(by_plane):
            candidates = sorted(
                by_plane[plane],
                key=lambda record: (
                    record.fluid + record.fatsat,
                    len(_directory_paths(record.directory)),
                    record.uid,
                ),
                reverse=True,
            )
            count = min(self.max_series_per_plane, len(candidates))
            if self.training and len(candidates) > count:
                extra = torch.randperm(len(candidates) - 1)[: count - 1].tolist()
                selected.extend(
                    [candidates[0], *[candidates[1 + index] for index in extra]]
                )
            else:
                selected.extend(candidates[:count])
        return selected

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        uid = str(row["StudyInstanceUID"])
        pixels: list[Tensor] = []
        planes: list[Tensor] = []
        fluids: list[Tensor] = []
        fatsats: list[Tensor] = []
        for record in self._choose_series(uid):
            images, _ = load_series_tensor(
                record,
                self.slices_per_plane,
                self.crop_mm,
                self.image_size,
                self.training,
            )
            pixels.append(images)
            planes.append(torch.full((images.shape[0],), record.plane, dtype=torch.long))
            fluids.append(torch.full((images.shape[0],), record.fluid, dtype=torch.long))
            fatsats.append(torch.full((images.shape[0],), record.fatsat, dtype=torch.long))
        labels = torch.tensor(
            [float(row[target]) if pd.notna(row[target]) else 0.0 for target in self.targets],
            dtype=torch.float32,
        )
        mask = torch.tensor([pd.notna(row[target]) for target in self.targets], dtype=torch.bool)
        confidence = torch.tensor(
            [float(row.get(f"{target}__conf", 1.0)) for target in self.targets], dtype=torch.float32
        )
        gold = torch.tensor(
            [bool(row.get(f"{target}__gold", pd.notna(row[target]))) for target in self.targets],
            dtype=torch.bool,
        )
        item = {
            "uid": uid,
            "pixels": torch.cat(pixels),
            "plane": torch.cat(planes),
            "fluid": torch.cat(fluids),
            "fatsat": torch.cat(fatsats),
            "labels": labels,
            "label_mask": mask,
            "confidence": confidence,
            "gold_mask": gold,
        }
        if self.report_embeddings is not None:
            embedding = self.report_embeddings.get(uid)
            if embedding is None:
                raise ValueError(f"missing report embedding for {uid}")
            item["report_embedding"] = embedding.float()
        return item


def collate_raw_studies(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    study_index = []
    for index, item in enumerate(items):
        study_index.append(torch.full((item["pixels"].shape[0],), index, dtype=torch.long))
    batch = {
        "uid": [item["uid"] for item in items],
        "pixels": torch.cat([item["pixels"] for item in items]),
        "plane": torch.cat([item["plane"] for item in items]),
        "fluid": torch.cat([item["fluid"] for item in items]),
        "fatsat": torch.cat([item["fatsat"] for item in items]),
        "study_index": torch.cat(study_index),
        "num_studies": len(items),
        "labels": torch.stack([item["labels"] for item in items]),
        "label_mask": torch.stack([item["label_mask"] for item in items]),
        "confidence": torch.stack([item["confidence"] for item in items]),
        "gold_mask": torch.stack([item["gold_mask"] for item in items]),
    }
    if all("report_embedding" in item for item in items):
        batch["report_embedding"] = torch.stack(
            [item["report_embedding"] for item in items]
        )
    return batch


def _load_external_state(module: nn.Module, path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
    cleaned = {
        str(key).removeprefix("module.").removeprefix("backbone."): value
        for key, value in state.items()
    }
    result = module.load_state_dict(cleaned, strict=False)
    loaded = len(cleaned) - len(result.unexpected_keys)
    fraction = loaded / max(1, len(module.state_dict()))
    if fraction < 0.5:
        raise ValueError(f"external backbone checkpoint matched only {fraction:.1%} of parameters")
    return {
        "loaded_fraction": fraction,
        "missing": result.missing_keys,
        "unexpected": result.unexpected_keys,
    }


class SliceFeatureBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        model_name: str | None,
        trainable_blocks: int,
        local_files_only: bool,
        pretrained: bool,
        checkpoint: Path | None,
    ) -> None:
        super().__init__()
        self.name = name
        self.load_report: dict[str, Any] | None = None
        if name == "dinov2":
            from transformers import AutoConfig, AutoModel

            if not model_name:
                raise ValueError("dinov2 requires --model-name")
            if pretrained:
                self.model = AutoModel.from_pretrained(
                    model_name, local_files_only=local_files_only
                )
            else:
                config = AutoConfig.from_pretrained(
                    model_name, local_files_only=local_files_only
                )
                self.model = AutoModel.from_config(config)
            hidden = int(self.model.config.hidden_size)
            self.output_dim = 2 * hidden
            freeze_except_last_blocks(self.model, trainable_blocks)
            self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        elif name == "efficientnet_b3":
            from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

            weights = EfficientNet_B3_Weights.DEFAULT if pretrained and checkpoint is None else None
            model = efficientnet_b3(weights=weights)
            self.output_dim = int(model.classifier[1].in_features)
            model.classifier = nn.Identity()
            self.model = model
            self.num_register_tokens = 0
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            stages = list(self.model.features)
            if trainable_blocks > len(stages):
                raise ValueError(f"EfficientNet exposes only {len(stages)} feature stages")
            for stage in stages[-trainable_blocks:] if trainable_blocks else []:
                for parameter in stage.parameters():
                    parameter.requires_grad = True
        elif name == "radimagenet_resnet50":
            from torchvision.models import resnet50

            resnet = resnet50(weights=None)
            self.model = nn.Sequential(*list(resnet.children())[:-1])
            self.output_dim = int(resnet.fc.in_features)
            self.num_register_tokens = 0
            if checkpoint is not None:
                self.load_report = _load_external_state(self.model, checkpoint)
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            stages = [stage for stage in self.model if any(True for _ in stage.parameters())]
            if trainable_blocks > len(stages):
                raise ValueError(f"RadImageNet ResNet exposes only {len(stages)} parameter stages")
            for stage in stages[-trainable_blocks:] if trainable_blocks else []:
                for parameter in stage.parameters():
                    parameter.requires_grad = True
        else:
            raise ValueError(f"unsupported backbone {name}")
        if checkpoint is not None and name != "radimagenet_resnet50":
            self.load_report = _load_external_state(self.model, checkpoint)

    def forward(self, pixels: Tensor) -> Tensor:
        if self.name == "dinov2":
            hidden = self.model(
                pixel_values=pixels, interpolate_pos_encoding=True
            ).last_hidden_state
            patches = hidden[:, 1 + self.num_register_tokens :]
            return torch.cat([hidden[:, 0], patches.mean(dim=1)], dim=-1)
        return self.model(pixels).flatten(1)


class RawStudyMILModel(nn.Module):
    def __init__(
        self,
        backbone: SliceFeatureBackbone,
        num_targets: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        pool: str = "max",
        encoder_batch_size: int = 12,
        topk: int = 3,
        report_dim: int = 0,
    ) -> None:
        super().__init__()
        if pool not in {"max", "topk", "logmeanexp"}:
            raise ValueError("pool must be max, topk, or logmeanexp")
        if topk < 1:
            raise ValueError("topk must be positive")
        self.backbone = backbone
        self.pool = pool
        self.topk = topk
        self.report_dim = report_dim
        self.encoder_batch_size = encoder_batch_size
        self.plane_embedding = nn.Embedding(len(PLANE_TO_ID), 16)
        self.plane_target_bias = nn.Embedding(len(PLANE_TO_ID), num_targets)
        self.fluid_target_bias = nn.Embedding(3, num_targets)
        self.fatsat_target_bias = nn.Embedding(3, num_targets)
        nn.init.zeros_(self.plane_target_bias.weight)
        nn.init.zeros_(self.fluid_target_bias.weight)
        nn.init.zeros_(self.fatsat_target_bias.weight)
        self.head = nn.Sequential(
            nn.LayerNorm(backbone.output_dim + 16),
            nn.Linear(backbone.output_dim + 16, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )
        self.report_projection = (
            nn.Sequential(
                nn.LayerNorm(backbone.output_dim),
                nn.Linear(backbone.output_dim, report_dim),
            )
            if report_dim > 0
            else None
        )

    def forward(
        self,
        pixels: Tensor,
        plane: Tensor,
        fluid: Tensor,
        fatsat: Tensor,
        study_index: Tensor,
        num_studies: int,
        return_aux: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        features = []
        for start in range(0, pixels.shape[0], self.encoder_batch_size):
            features.append(self.backbone(pixels[start : start + self.encoder_batch_size]))
        features_all = torch.cat(features)
        slice_logits = self.head(
            torch.cat([features_all, self.plane_embedding(plane)], dim=-1)
        )
        slice_logits = (
            slice_logits
            + self.plane_target_bias(plane)
            + self.fluid_target_bias(fluid)
            + self.fatsat_target_bias(fatsat)
        )
        studies = []
        for index in range(num_studies):
            values = slice_logits[study_index == index]
            if values.numel() == 0:
                raise ValueError("every study must contain at least one decoded slice")
            if self.pool == "max":
                studies.append(values.max(dim=0).values)
            elif self.pool == "topk":
                count = min(self.topk, values.shape[0])
                studies.append(values.topk(count, dim=0).values.mean(dim=0))
            else:
                studies.append(torch.logsumexp(values, dim=0) - np.log(values.shape[0]))
        logits = torch.stack(studies)
        if not return_aux:
            return logits
        if self.report_projection is None:
            raise ValueError("return_aux requires a positive report_dim")
        study_features = torch.stack(
            [features_all[study_index == index].mean(dim=0) for index in range(num_studies)]
        )
        return {
            "logits": logits,
            "report_embedding": self.report_projection(study_features),
        }

    def parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict[str, Any]]:
        backbone = [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]
        backbone_ids = {id(parameter) for parameter in backbone}
        head = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        return [{"params": backbone, "lr": backbone_lr}, {"params": head, "lr": head_lr}]


def normalize_pixels(pixels: Tensor) -> Tensor:
    mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = pixels.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (pixels - mean) / std
