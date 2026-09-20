from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pydicom
import torch
import torch.nn.functional as F
from pydicom.dataset import Dataset


@dataclass
class Slice:
    sop_uid: str
    pixels: npt.NDArray[np.float32]
    position: tuple[float, ...]
    orientation: tuple[float, ...]
    spacing: tuple[float, ...]
    instance: int
    sha256: str
    transfer_syntax: str


def decode_slice(data: bytes, study_uid: str, series_uid: str, sop_uid: str) -> Slice:
    ds: Dataset = pydicom.dcmread(io.BytesIO(data))
    actual = (str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID), str(ds.SOPInstanceUID))
    if actual != (study_uid, series_uid, sop_uid):
        raise ValueError(f"DICOM UID mismatch: expected {(study_uid, series_uid, sop_uid)}")
    pixels = ds.pixel_array.astype(np.float32)
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError("Only single-frame grayscale slices are supported")
    photo = str(ds.get("PhotometricInterpretation", ""))
    if photo not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError(f"Unsupported photometric interpretation: {photo}")
    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    pixels = pixels * slope + intercept
    if not np.isfinite(pixels).all():
        raise ValueError("Non-finite pixel values")
    if photo == "MONOCHROME1":
        pixels = pixels.max() + pixels.min() - pixels
    return Slice(
        sop_uid=sop_uid,
        pixels=pixels,
        position=tuple(float(x) for x in ds.get("ImagePositionPatient", [])),
        orientation=tuple(float(x) for x in ds.get("ImageOrientationPatient", [])),
        spacing=tuple(float(x) for x in ds.get("PixelSpacing", [])),
        instance=int(ds.get("InstanceNumber", 0)),
        sha256=hashlib.sha256(data).hexdigest(),
        transfer_syntax=str(ds.file_meta.TransferSyntaxUID),
    )


def order_slices(slices: list[Slice]) -> tuple[list[Slice], str, list[str]]:
    if not slices:
        raise ValueError("Empty series")
    warnings = []
    reference = slices[0]
    geometric = len(reference.orientation) == 6 and all(
        len(s.position) == 3
        and len(s.orientation) == 6
        and np.allclose(s.orientation, reference.orientation, atol=1e-3)
        for s in slices
    )
    if geometric:
        normal = np.cross(reference.orientation[:3], reference.orientation[3:])
        if np.linalg.norm(normal) < 0.9:
            raise ValueError("Invalid image orientation vectors")
        ordered = sorted(slices, key=lambda s: (np.dot(s.position, normal), s.instance, s.sop_uid))
        positions = np.array([np.dot(s.position, normal) for s in ordered])
        gaps = np.diff(positions)
        if np.any(gaps < 1e-4):
            warnings.append("duplicate_slice_positions")
        positive_gaps = gaps[gaps > 1e-4]
        if len(positive_gaps) and positive_gaps.max() > 1.6 * np.median(positive_gaps):
            warnings.append("irregular_slice_spacing")
        return ordered, "patient_geometry", warnings
    warnings.append("missing_or_mixed_geometry")
    if len({s.instance for s in slices}) != len(slices):
        raise ValueError("Neither geometry nor unique InstanceNumber can order series")
    return sorted(slices, key=lambda s: s.instance), "instance_number", warnings


def make_volume(slices: list[Slice], side: int) -> tuple[npt.NDArray[np.uint16], dict]:
    if side < 16:
        raise ValueError("Cache side must be at least 16")
    ordered, method, warnings = order_slices(slices)
    shapes = {s.pixels.shape for s in ordered}
    if len(shapes) != 1:
        raise ValueError("Mixed pixel matrix dimensions within series")
    volume = np.stack([s.pixels for s in ordered])
    low, high = float(volume.min()), float(volume.max())
    if high <= low:
        raise ValueError("Constant-intensity series")
    normalized = (volume - low) / (high - low)
    height, width = volume.shape[1:]
    spacing = ordered[0].spacing
    if any(s.spacing != spacing for s in ordered):
        raise ValueError("Mixed pixel spacing within series")
    if len(spacing) == 2 and all(x > 0 for x in spacing):
        physical_height, physical_width = height * spacing[0], width * spacing[1]
    else:
        physical_height, physical_width = float(height), float(width)
        warnings.append("missing_pixel_spacing")
    scale = side / max(physical_height, physical_width)
    out_height = max(1, min(side, round(physical_height * scale)))
    out_width = max(1, min(side, round(physical_width * scale)))
    resized = F.interpolate(
        torch.from_numpy(normalized[:, None]),
        size=(out_height, out_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[:, 0]
    left, top = (side - out_width) // 2, (side - out_height) // 2
    padded = F.pad(resized, (left, side - out_width - left, top, side - out_height - top))
    encoded = np.rint(padded.numpy().clip(0, 1) * 65535).astype(np.uint16)
    metadata = {
        "ordering": method,
        "warnings": warnings,
        "intensity_min": low,
        "intensity_max": high,
        "original_shape": list(volume.shape),
        "resized_shape": [out_height, out_width],
        "pixel_spacing": list(spacing),
        "sop_uids": [s.sop_uid for s in ordered],
        "positions": [list(s.position) for s in ordered],
        "orientation": list(ordered[0].orientation),
        "slice_sha256": [s.sha256 for s in ordered],
        "pixel_sha256": [hashlib.sha256(s.pixels.tobytes()).hexdigest() for s in ordered],
        "transfer_syntaxes": sorted({s.transfer_syntax for s in ordered}),
        "constant_slices": sum(float(s.pixels.max()) == float(s.pixels.min()) for s in ordered),
    }
    return encoded, metadata
