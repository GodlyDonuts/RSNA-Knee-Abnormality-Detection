#!/usr/bin/env python3
"""Extract physically ordered 2.5-D DINO features from knee MRI DICOMs.

This script is intended to run in a Kaggle/RunPod environment where the user
has legitimately mounted the competition data and pretrained weights.  It
never downloads competition data or accepts competition rules on the user's
behalf.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from .constants import (
        CACHE_SCHEMA_VERSION,
        PLANE_TO_ID,
        TRISTATE_FALSE,
        TRISTATE_TRUE,
        TRISTATE_UNKNOWN,
    )
except ImportError:
    from constants import (
        CACHE_SCHEMA_VERSION,
        PLANE_TO_ID,
        TRISTATE_FALSE,
        TRISTATE_TRUE,
        TRISTATE_UNKNOWN,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-slices", type=int, default=64)
    parser.add_argument(
        "--patch-grid",
        type=int,
        default=0,
        help="also cache an NxN grid of DINO patch tokens per slice; 4 is a practical stage-two value",
    )
    parser.add_argument("--crop-mm", type=float, default=160.0)
    parser.add_argument("--limit-studies", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _safe_text(value: Any, default: str = "") -> str:
    return str(value).strip() if value is not None else default


def _tristate(value: Any) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return TRISTATE_UNKNOWN
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return TRISTATE_TRUE
        if lowered in {"false", "0", "no", "n"}:
            return TRISTATE_FALSE
        return TRISTATE_UNKNOWN
    return TRISTATE_TRUE if bool(value) else TRISTATE_FALSE


def _plane_id(value: Any) -> int:
    text = _safe_text(value).lower()
    for name, index in PLANE_TO_ID.items():
        if name != "unknown" and name in text:
            return index
    return PLANE_TO_ID["unknown"]


def _normal_and_position(ds: Any) -> tuple[np.ndarray | None, float | None]:
    try:
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        location = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        normal = np.cross(orientation[:3], orientation[3:6])
        norm = np.linalg.norm(normal)
        if norm <= 1e-10:
            return None, None
        normal /= norm
        return normal, float(np.dot(location, normal))
    except (AttributeError, TypeError, ValueError):
        return None, None


def _read_series(paths: Sequence[Path]) -> tuple[list[Any], np.ndarray]:
    import pydicom

    datasets = [pydicom.dcmread(str(path), force=True) for path in paths]
    sortable: list[tuple[float, int, Any]] = []
    have_physical = True
    for ordinal, ds in enumerate(datasets):
        _, position = _normal_and_position(ds)
        if position is None:
            have_physical = False
            try:
                position = float(ds.InstanceNumber)
            except (AttributeError, TypeError, ValueError):
                position = float(ordinal)
        sortable.append((position, ordinal, ds))
    sortable.sort(key=lambda triple: (triple[0], triple[1]))
    ordered = [triple[2] for triple in sortable]
    positions = np.asarray([triple[0] for triple in sortable], dtype=np.float32)
    # The numeric fallback retains correct ordering semantics but is not called
    # millimetres in metadata.  Physical ALiBi reduces to ordinal ALiBi here.
    if not have_physical:
        positions = np.arange(len(ordered), dtype=np.float32)
    return ordered, positions


def _select_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, length - 1, maximum)).astype(np.int64))


def _pixel_array(ds: Any) -> np.ndarray:
    array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return array * slope + intercept


def _crop_physical(array: np.ndarray, ds: Any, crop_mm: float) -> np.ndarray:
    try:
        spacing = [float(v) for v in ds.PixelSpacing]
        target_h = min(array.shape[0], max(16, int(round(crop_mm / spacing[0]))))
        target_w = min(array.shape[1], max(16, int(round(crop_mm / spacing[1]))))
    except (AttributeError, TypeError, ValueError, IndexError, ZeroDivisionError):
        target_h = min(array.shape[0], array.shape[1])
        target_w = target_h
    y0 = max(0, (array.shape[0] - target_h) // 2)
    x0 = max(0, (array.shape[1] - target_w) // 2)
    return array[y0 : y0 + target_h, x0 : x0 + target_w]


def _laterality(ds: Any) -> str:
    for attribute in ("ImageLaterality", "Laterality"):
        value = _safe_text(getattr(ds, attribute, "")).upper()
        if value in {"L", "R"}:
            return value
    description = " ".join(
        _safe_text(getattr(ds, name, ""))
        for name in ("SeriesDescription", "ProtocolName", "StudyDescription")
    ).upper()
    if "RIGHT" in description or " RT " in f" {description} ":
        return "R"
    if "LEFT" in description or " LT " in f" {description} ":
        return "L"
    return ""


def prepare_25d_images(
    datasets: Sequence[Any],
    positions: np.ndarray,
    plane: int,
    max_slices: int,
    crop_mm: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    indices = _select_indices(len(datasets), max_slices)
    selected = [datasets[int(i)] for i in indices]
    selected_positions = positions[indices]
    decoded: list[tuple[Any, float, np.ndarray]] = []
    for ds, position in zip(selected, selected_positions):
        try:
            decoded.append((ds, float(position), _crop_physical(_pixel_array(ds), ds, crop_mm)))
        except Exception as error:
            # Public medical-imaging corpora occasionally contain a truncated
            # frame or inconsistent pixel metadata.  One bad slice must not
            # discard an otherwise usable series, but an entirely unreadable
            # selected series remains a hard failure below.
            identity = _safe_text(
                getattr(ds, "SOPInstanceUID", getattr(ds, "InstanceNumber", "unknown")),
                "unknown",
            )
            print(
                f"warning: skipping unreadable DICOM slice {identity}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
    if not decoded:
        raise RuntimeError("all selected DICOM slices in a series were unreadable")
    selected = [row[0] for row in decoded]
    selected_positions = np.asarray([row[1] for row in decoded], dtype=np.float32)
    arrays = [row[2] for row in decoded]
    lows = np.asarray([np.percentile(a, 1) for a in arrays], dtype=np.float32)
    highs = np.asarray([np.percentile(a, 99) for a in arrays], dtype=np.float32)
    low = float(np.median(lows))
    high = float(np.median(highs))
    if not np.isfinite(high - low) or high - low < 1e-6:
        low, high = float(min(a.min() for a in arrays)), float(max(a.max() for a in arrays))
    scale = max(high - low, 1e-6)
    arrays = [np.clip((a - low) / scale, 0, 1) for a in arrays]

    if _laterality(selected[0]) == "R":
        if plane in {PLANE_TO_ID["coronal"], PLANE_TO_ID["axial"]}:
            arrays = [np.fliplr(a).copy() for a in arrays]
        elif plane == PLANE_TO_ID["sagittal"]:
            arrays = arrays[::-1]
            selected_positions = selected_positions[::-1].copy()

    images: list[np.ndarray] = []
    for i in range(len(arrays)):
        channels = [arrays[max(0, i - 1)], arrays[i], arrays[min(len(arrays) - 1, i + 1)]]
        # Processors accept uint8 HWC images.  The three channels are adjacent
        # physical slices, not duplicated grayscale.
        image = np.stack(channels, axis=-1)
        images.append(np.rint(255 * image).astype(np.uint8))
    return images, selected_positions.astype(np.float32)


class DinoFeatureExtractor:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        local_files_only: bool,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(
            model_name, local_files_only=local_files_only, use_fast=False
        )
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model.to(self.device).eval()
        self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0) or 0)

    @torch.inference_mode()
    def encode(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        summary, _ = self._encode(images, patch_grid=0)
        return summary

    @torch.inference_mode()
    def encode_with_patches(
        self, images: Sequence[np.ndarray], patch_grid: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_grid < 1:
            raise ValueError("patch_grid must be positive")
        summary, patches = self._encode(images, patch_grid=patch_grid)
        assert patches is not None
        return summary, patches

    def _encode(
        self, images: Sequence[np.ndarray], patch_grid: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features: list[torch.Tensor] = []
        patch_features: list[torch.Tensor] = []
        amp = self.device.type == "cuda"
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]
            inputs = self.processor(images=list(batch_images), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp):
                hidden = self.model(**inputs).last_hidden_state
            cls = hidden[:, 0]
            patch_start = 1 + self.num_register_tokens
            patches = hidden[:, patch_start:]
            pooled = patches.mean(dim=1) if patches.shape[1] else hidden[:, 1:].mean(dim=1)
            features.append(torch.cat([cls, pooled], dim=-1).float().cpu())
            if patch_grid:
                side = int(round(math.sqrt(patches.shape[1])))
                if side * side != patches.shape[1]:
                    raise ValueError(
                        f"cannot form a square DINO patch grid from {patches.shape[1]} tokens"
                    )
                spatial = patches.transpose(1, 2).reshape(
                    patches.shape[0], patches.shape[2], side, side
                )
                compact = torch.nn.functional.adaptive_avg_pool2d(
                    spatial.float(), (patch_grid, patch_grid)
                )
                compact = compact.flatten(2).transpose(1, 2).cpu()
                patch_features.append(compact)
        return torch.cat(features, dim=0), (
            torch.cat(patch_features, dim=0) if patch_features else None
        )


def _series_paths(root: Path, split: str, study_uid: str, series_uid: str) -> list[Path]:
    direct = root / f"{split}_series" / study_uid / series_uid
    if direct.exists():
        return sorted(direct.glob("*.dcm"))
    # Some mounted datasets add one outer directory.  Restrict the fallback to
    # this exact pair of UIDs so an accidental broad scan cannot mix patients.
    pattern = f"**/{split}_series/{study_uid}/{series_uid}/*.dcm"
    return sorted(root.glob(pattern))


def _cache_name(uid: str) -> str:
    digest = hashlib.sha1(uid.encode("utf-8")).hexdigest()[:12]
    return f"{uid.replace('/', '_')}__{digest}.pt"


def cached_metadata(
    rows: pd.DataFrame,
    root: Path,
    split: str,
    payload: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Recover scanner grouping when resuming from tensor-only cache files.

    Scanner grouping must never silently collapse to a generic ``cached``
    value after an interrupted extraction.  Reading one header per study is
    cheap compared with re-encoding its images and makes restart behavior
    equivalent to an uninterrupted run.
    """

    import pydicom

    header0: Any = None
    for _, row in rows.iterrows():
        paths = _series_paths(
            root,
            split,
            str(row["StudyInstanceUID"]),
            str(row["SeriesInstanceUID"]),
        )
        if paths:
            header0 = pydicom.dcmread(str(paths[0]), stop_before_pixels=True, force=True)
            break
    manufacturer = _safe_text(getattr(header0, "Manufacturer", "unknown"), "unknown")
    model = _safe_text(getattr(header0, "ManufacturerModelName", "unknown"), "unknown")
    field = _safe_text(getattr(header0, "MagneticFieldStrength", "unknown"), "unknown")
    patch = payload.get("patch_features")
    return {
        "n_series": int(payload["series_mask"].sum()),
        "feature_dim": int(payload["features"].shape[-1]),
        "patch_dim": int(patch.shape[-1]) if patch is not None else 0,
        "patches_per_slice": int(patch.shape[-2]) if patch is not None else 0,
        "manufacturer": manufacturer,
        "scanner_model": model,
        "field_strength": field,
        "scanner_group": f"{manufacturer}|{model}|{field}",
    }


def extract_study(
    rows: pd.DataFrame,
    root: Path,
    split: str,
    extractor: DinoFeatureExtractor,
    max_slices: int,
    crop_mm: float,
    patch_grid: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    series_features: list[torch.Tensor] = []
    series_positions: list[torch.Tensor] = []
    series_patch_features: list[torch.Tensor] = []
    planes: list[int] = []
    fluids: list[int] = []
    fatsats: list[int] = []
    header0: Any = None

    for _, row in rows.iterrows():
        study_uid = str(row["StudyInstanceUID"])
        series_uid = str(row["SeriesInstanceUID"])
        paths = _series_paths(root, split, study_uid, series_uid)
        if not paths:
            print(f"warning: no DICOMs for {study_uid}/{series_uid}", flush=True)
            continue
        datasets, positions = _read_series(paths)
        if header0 is None:
            header0 = datasets[0]
        plane = _plane_id(row.get("Anatomical_Plane"))
        images, selected_positions = prepare_25d_images(
            datasets, positions, plane, max_slices, crop_mm
        )
        if patch_grid:
            summary, patches = extractor.encode_with_patches(images, patch_grid)
            series_features.append(summary)
            series_patch_features.append(patches)
        else:
            series_features.append(extractor.encode(images))
        series_positions.append(torch.from_numpy(selected_positions))
        planes.append(plane)
        fluids.append(_tristate(row.get("Fluid_Sensitive")))
        fatsats.append(_tristate(row.get("Fat_Suppression")))

    if not series_features:
        raise RuntimeError(f"no readable series for study {rows.iloc[0]['StudyInstanceUID']}")
    n_series = len(series_features)
    max_length = max(t.shape[0] for t in series_features)
    feature_dim = series_features[0].shape[1]
    features = torch.zeros(n_series, max_length, feature_dim, dtype=torch.float16)
    positions_mm = torch.zeros(n_series, max_length, dtype=torch.float32)
    slice_mask = torch.zeros(n_series, max_length, dtype=torch.bool)
    for index, (feature, position) in enumerate(zip(series_features, series_positions)):
        length = feature.shape[0]
        features[index, :length] = feature.half()
        positions_mm[index, :length] = position.float()
        slice_mask[index, :length] = True
    payload = {
        "schema_version": torch.tensor(CACHE_SCHEMA_VERSION, dtype=torch.long),
        "features": features,
        "positions_mm": positions_mm,
        "slice_mask": slice_mask,
        "series_mask": torch.ones(n_series, dtype=torch.bool),
        "plane": torch.tensor(planes, dtype=torch.long),
        "fluid": torch.tensor(fluids, dtype=torch.long),
        "fatsat": torch.tensor(fatsats, dtype=torch.long),
    }
    patch_dim = 0
    patches_per_slice = 0
    if series_patch_features:
        patches_per_slice = int(series_patch_features[0].shape[1])
        patch_dim = int(series_patch_features[0].shape[2])
        patch_features = torch.zeros(
            n_series,
            max_length,
            patches_per_slice,
            patch_dim,
            dtype=torch.float16,
        )
        patch_mask = torch.zeros(
            n_series, max_length, patches_per_slice, dtype=torch.bool
        )
        for index, patch in enumerate(series_patch_features):
            length = patch.shape[0]
            patch_features[index, :length] = patch.half()
            patch_mask[index, :length] = True
        payload["patch_features"] = patch_features
        payload["patch_mask"] = patch_mask
    manufacturer = _safe_text(getattr(header0, "Manufacturer", "unknown"), "unknown")
    model = _safe_text(getattr(header0, "ManufacturerModelName", "unknown"), "unknown")
    field = _safe_text(getattr(header0, "MagneticFieldStrength", "unknown"), "unknown")
    metadata = {
        "n_series": n_series,
        "feature_dim": int(feature_dim),
        "patch_dim": patch_dim,
        "patches_per_slice": patches_per_slice,
        "manufacturer": manufacturer,
        "scanner_model": model,
        "field_strength": field,
        "scanner_group": f"{manufacturer}|{model}|{field}",
    }
    return payload, metadata


def main() -> None:
    args = parse_args()
    series_csv = args.data_root / f"{args.split}_series.csv"
    if not series_csv.exists():
        raise FileNotFoundError(f"missing {series_csv}")
    table = pd.read_csv(series_csv)
    required = {"StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"series table is missing columns: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    extractor = DinoFeatureExtractor(
        args.model_name, args.device, args.batch_size, args.local_files_only
    )
    grouped = list(table.groupby("StudyInstanceUID", sort=True))
    if args.limit_studies:
        grouped = grouped[: args.limit_studies]
    records: list[dict[str, Any]] = []
    for number, (uid, rows) in enumerate(grouped, start=1):
        cache_path = args.output / _cache_name(str(uid))
        if cache_path.exists() and not args.overwrite:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            metadata = cached_metadata(rows, args.data_root, args.split, payload)
        else:
            payload, metadata = extract_study(
                rows,
                args.data_root,
                args.split,
                extractor,
                args.max_slices,
                args.crop_mm,
                args.patch_grid,
            )
            torch.save(payload, cache_path)
        records.append(
            {
                "StudyInstanceUID": str(uid),
                "cache_file": str(cache_path.resolve()),
                **metadata,
            }
        )
        print(f"[{number}/{len(grouped)}] {uid}: {metadata['n_series']} series", flush=True)
    index_path = args.output / f"{args.split}_cache_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)
    print(f"wrote {index_path}")


if __name__ == "__main__":
    main()
