"""Synthetic DICOM-like knee studies with known geometry; no MRI or labels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

PLANES = {
    "Sagittal": ((0.0, 1.0, 0.0, 0.0, 0.0, -1.0), "SAG"),
    "Coronal": ((1.0, 0.0, 0.0, 0.0, 0.0, -1.0), "COR"),
    "Axial": ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), "AX"),
}


@dataclass(frozen=True)
class SeriesSpec:
    plane: str
    description: str
    slices: int = 20
    rows: int = 64
    columns: int = 48
    spacing: tuple[float, float] = (0.4, 0.6)
    thickness: float = 3.0
    oblique_degrees: float = 0.0
    laterality: str = "R"
    duplicate_positions: int = 0
    constant: bool = False
    laterality_tag: bool = True
    seed: int | None = None


def orientation(plane: str, degrees: float) -> np.ndarray:
    row, col = np.asarray(PLANES[plane][0]).reshape(2, 3)
    if degrees:
        theta = np.deg2rad(degrees)
        normal = np.cross(row, col)
        # Rodrigues rotation of the in-plane row vector about the column axis.
        row = row * np.cos(theta) + np.cross(col, row) * np.sin(theta)
        row -= col * np.dot(row, col)
        row /= np.linalg.norm(row)
        assert np.linalg.norm(np.cross(row, col)) > 0.9 and normal.any()
    return np.concatenate([row, col])


def series_pixels(spec: SeriesSpec, seed: int) -> np.ndarray:
    if spec.constant:
        return np.full((spec.slices, spec.rows, spec.columns), 700, dtype=np.uint16)
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(
        np.arange(spec.slices), np.arange(spec.rows), np.arange(spec.columns), indexing="ij"
    )
    ramp = 100 * z + 7 * y + 3 * x
    return (ramp + rng.integers(0, 50, ramp.shape)).astype(np.uint16)


def write_series(
    root: Path, study: str, index: int, spec: SeriesSpec, scramble: bool = True
) -> list[Path]:
    """Write one series; filenames are hash-scrambled so order must come from headers."""
    series = f"{study}.{index}"
    folder = root / study / series
    folder.mkdir(parents=True, exist_ok=True)
    pixels = series_pixels(spec, seed=index if spec.seed is None else spec.seed)
    basis = orientation(spec.plane, spec.oblique_degrees)
    normal = np.cross(basis[:3], basis[3:])
    side = -1.0 if spec.laterality == "R" else 1.0
    origin = np.array([side * 60.0, -40.0, 30.0])
    paths = []
    for k in range(spec.slices):
        step = max(0, k - spec.duplicate_positions) if k < 2 * spec.duplicate_positions else k
        position = origin + normal * spec.thickness * step
        sop = f"{series}.{k + 1}"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = sop
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = study, series, sop
        ds.SOPClassUID, ds.Modality = MRImageStorage, "MR"
        ds.SeriesDescription = f"{PLANES[spec.plane][1]} {spec.description}"
        if spec.laterality_tag:
            ds.ImageLaterality = ds.Laterality = spec.laterality
        ds.Rows, ds.Columns = spec.rows, spec.columns
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelSpacing = [f"{v:g}" for v in spec.spacing]
        ds.SliceThickness = spec.thickness
        ds.ImageOrientationPatient = [f"{v:.6f}" for v in basis]
        ds.ImagePositionPatient = [f"{v:.4f}" for v in position]
        ds.SliceLocation = float(np.dot(position, normal))
        # Instance numbers are deliberately reversed relative to geometry.
        ds.InstanceNumber = spec.slices - k
        ds.PixelData = pixels[k].tobytes()
        name = "im" + hashlib.sha256(sop.encode()).hexdigest()[:12] if scramble else f"{k:04d}"
        path = folder / f"{name}.dcm"
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths


CASES: dict[str, list[SeriesSpec]] = {
    "complete_right": [
        SeriesSpec("Sagittal", "PD FS", slices=24),
        SeriesSpec("Sagittal", "T1", slices=22),
        SeriesSpec("Coronal", "PD FS", slices=18, spacing=(0.3, 0.3)),
        SeriesSpec("Coronal", "T1", slices=18),
        SeriesSpec("Axial", "T2 FS", slices=30, spacing=(0.35, 0.5)),
        SeriesSpec("Axial", "PD", slices=28),
    ],
    "left_oblique": [
        SeriesSpec("Sagittal", "PD FS", laterality="L", oblique_degrees=12.0),
        SeriesSpec("Coronal", "T1", laterality="L", oblique_degrees=-8.0),
        SeriesSpec("Axial", "T2 FS", laterality="L", oblique_degrees=5.0),
    ],
    "absent_slots": [SeriesSpec("Sagittal", "PD FS", slices=9)],
    "duplicates_and_constant": [
        SeriesSpec("Sagittal", "PD FS", duplicate_positions=2),
        SeriesSpec("Coronal", "PD FS", constant=True),
        SeriesSpec("Axial", "T2 FS", slices=3),
    ],
    "large_field": [SeriesSpec("Coronal", "PD FS", rows=96, columns=128, spacing=(1.8, 1.8))],
    "t1_fallback_untagged": [
        SeriesSpec("Sagittal", "PD", laterality_tag=False, laterality="L"),
        SeriesSpec("Coronal", "PD", slices=16, laterality_tag=False, laterality="L"),
        SeriesSpec("Coronal", "PD", slices=12, laterality_tag=False, laterality="L"),
    ],
}

MIRROR = {
    "left": [
        SeriesSpec("Coronal", "PD FS", laterality="L", seed=7),
        SeriesSpec("Sagittal", "PD FS", laterality="L", seed=8),
    ],
    "right": [
        SeriesSpec("Coronal", "PD FS", laterality="R", seed=7),
        SeriesSpec("Sagittal", "PD FS", laterality="R", seed=8),
    ],
}


def plane_map(study: Path, specs: list[SeriesSpec]) -> dict[str, str]:
    """Series-CSV-style `Anatomical_Plane` metadata for one written study."""
    return {f"{study.name}.{i}": spec.plane for i, spec in enumerate(specs, 1)}


def write_case(root: Path, name: str, cases: dict | None = None) -> Path:
    cases = CASES if cases is None else cases
    study = "1.2.826.0.1." + str(sorted(cases).index(name) + 1 + 100 * (cases is not CASES))
    for index, spec in enumerate(cases[name], 1):
        write_series(root, study, index, spec)
    return root / study
