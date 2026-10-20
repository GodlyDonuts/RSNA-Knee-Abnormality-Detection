"""Offline source-DICOM assembly for the predictive-evidence renderer.

This follows the admitted integer decoder and raw16 group conversion, including
their acquisition-wide ordinals across matrix groups. It does not use the
earlier pilot's within-matrix ordinals, percentile scaling, reports or labels.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.predictive.cases import RawGroup
from rsna.competitive.predictive.inputs import fixed_metadata_scale

VERSION = "original_integer_DICOM_global_acquisition_ordinal_v1"


def _uid(value):
    # Import lazily so numerical research modules do not acquire a DICOM runtime
    # dependency merely by importing this adapter.
    from pydicom.uid import UID

    text = str(value)
    if not text or len(text) > 64 or any(c not in "0123456789." for c in text):
        raise ValueError("Valid DICOM UID required")
    if not UID(text).is_valid:
        raise ValueError("Valid DICOM UID required")
    return text


def _integer(ds, name):
    try:
        value = ds[name].value
        integer = int(value)
        if not math.isfinite(float(value)) or float(value) != integer:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("Required DICOM integer metadata is missing or invalid") from None
    return integer


def _position(ds, instance):
    orientation = ds.get("ImageOrientationPatient")
    position = ds.get("ImagePositionPatient")
    if orientation is None and position is None:
        return float(instance), False
    try:
        orientation = np.asarray(orientation, dtype=float)
        position = np.asarray(position, dtype=float)
        if (
            orientation.shape != (6,)
            or position.shape != (3,)
            or not np.isfinite(orientation).all()
            or not np.isfinite(position).all()
        ):
            raise ValueError
        normal = np.cross(orientation[:3], orientation[3:])
        if not np.isfinite(normal).all() or not np.any(normal):
            raise ValueError
        projected = float(np.dot(position, normal))
        if not math.isfinite(projected):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("DICOM geometry is incomplete or invalid") from None
    return projected, True


def load_study(
    study_root: str | Path,
    study_uid: str,
    *,
    ordered_files: Iterable[str | Path] | None = None,
) -> tuple[list[RawGroup], dict]:
    """Decode a complete ``study_root/<series UID>/<SOP UID>.dcm`` study.

    By default files are read in sorted series/SOP path order. An original
    source-table order can be supplied as ``ordered_files`` (absolute paths or
    paths relative to ``study_root``). It must contain every enumerated DICOM
    exactly once. This preserves the original alias IDs and matrix insertion
    order for byte-exact source-adapter comparisons.

    The returned groups contain CPU FP32 encoding-normalized frames and int64
    aliases/acquisitions/ordinals. Aggregates contain no UIDs, paths or pixels.
    Malformed or unexpected inputs fail; there is no guessed image or constant
    prediction fallback. Equal geometric sort keys retain original read order,
    as in the admitted source decoder.
    """
    import pydicom

    uid = _uid(study_uid)
    root = Path(study_root)
    if root.is_symlink() or not root.is_dir() or root.name != uid:
        raise ValueError("Exact original study directory required")
    root = root.resolve()
    series = []
    files = []
    for directory in sorted(root.iterdir(), key=lambda p: p.name):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Study directory must contain original series directories only")
        series.append(_uid(directory.name))
        members = sorted(directory.iterdir(), key=lambda p: p.name)
        if not members:
            raise ValueError("Empty DICOM series is not accepted")
        for path in members:
            if path.is_symlink() or not path.is_file() or path.suffix != ".dcm":
                raise ValueError("Series directory must contain original DICOM files only")
            _uid(path.stem)
            files.append(path)
    if not files:
        raise ValueError("A complete nonempty original DICOM study is required")
    if ordered_files is not None:
        supplied = [Path(path) for path in ordered_files]
        supplied = [path if path.is_absolute() else root / path for path in supplied]
        # Compare lexical absolute paths; resolving a supplied symlink or path
        # traversal must not silently admit a different source path.
        if len(supplied) != len(files) or len(set(supplied)) != len(files):
            raise ValueError("Original source order must include every DICOM exactly once")
        if set(supplied) != set(files):
            raise ValueError("Original source order must match the complete study file roster")
        files = supplied

    acquisitions = {name: index for index, name in enumerate(series)}
    groups = {}
    aliases = {}
    alias_values = {}
    bit_counts = {}
    photometric_counts = {}
    file_bytes = 0
    geometry_frames = 0
    signed_frames = 0
    for path in files:
        try:
            ds = pydicom.dcmread(path)
        except Exception:
            raise ValueError("Original DICOM could not be read") from None
        actual = tuple(
            str(ds.get(name, ""))
            for name in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
        )
        if actual != (uid, path.parent.name, path.stem):
            raise ValueError("DICOM identity does not match its study/series/SOP path")
        bits = _integer(ds, "BitsStored")
        signed = _integer(ds, "PixelRepresentation")
        photometric = str(ds.get("PhotometricInterpretation", ""))
        if (
            not 1 <= bits <= 16
            or signed not in (0, 1)
            or photometric not in ("MONOCHROME1", "MONOCHROME2")
            or _integer(ds, "SamplesPerPixel") != 1
            or ("NumberOfFrames" in ds and _integer(ds, "NumberOfFrames") != 1)
        ):
            raise ValueError("One monochrome original integer DICOM frame is required")
        try:
            slope = float(ds.get("RescaleSlope", 1))
            intercept = float(ds.get("RescaleIntercept", 0))
            if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Finite DICOM rescale with positive slope required") from None
        instance = _integer(ds, "InstanceNumber")
        position, has_geometry = _position(ds, instance)
        try:
            pixel = ds.pixel_array
        except Exception:
            raise ValueError("Original DICOM pixel decoding failed") from None
        if (
            pixel.ndim != 2
            or not np.issubdtype(pixel.dtype, np.integer)
            or not pixel.size
            or tuple(pixel.shape) != (_integer(ds, "Rows"), _integer(ds, "Columns"))
        ):
            raise ValueError("Original two-dimensional integer DICOM pixels required")
        key = hashlib.sha256(
            np.array(pixel.shape, dtype="<i8").tobytes() + pixel.astype("<f8").tobytes()
        ).hexdigest()
        if key not in aliases:
            aliases[key] = len(aliases)
        frame = torch.from_numpy(pixel.astype(np.int32))
        normalized = fixed_metadata_scale(
            frame, bits_stored=bits, signed=bool(signed), photometric=photometric
        )
        # The original masking path rejects inconsistent transforms of an exact
        # raw-pixel alias. Reject it here too, before any encoder can be called.
        if key in alias_values and not torch.equal(alias_values[key], normalized):
            raise ValueError("Exact source aliases have inconsistent encoding transformations")
        alias_values.setdefault(key, normalized)
        groups.setdefault(tuple(pixel.shape), []).append(
            (acquisitions[path.parent.name], position, instance, normalized, aliases[key])
        )
        bit_counts[str(bits)] = bit_counts.get(str(bits), 0) + 1
        photometric_counts[photometric] = photometric_counts.get(photometric, 0) + 1
        file_bytes += path.stat().st_size
        geometry_frames += int(has_geometry)
        signed_frames += signed

    # Match the admitted raw16 conversion exactly: ordinals span ALL matrices
    # within an acquisition. The stable sort and overwrite behavior for exact
    # duplicate keys are intentional original-source semantics.
    items = sorted((item for group in groups.values() for item in group), key=lambda x: x[:3])
    order = {}
    for acquisition in sorted({item[0] for item in items}):
        for i, item in enumerate(x for x in items if x[0] == acquisition):
            order[(item[0], item[1], item[2], item[4])] = i
    result = []
    for group in groups.values():
        group.sort(key=lambda x: x[:3])
        result.append(
            RawGroup(
                torch.stack([x[3] for x in group]),
                torch.tensor([x[4] for x in group], dtype=torch.int64),
                torch.tensor([x[0] for x in group], dtype=torch.int64),
                torch.tensor([order[(x[0], x[1], x[2], x[4])] for x in group], dtype=torch.int64),
            )
        )
    stats = {
        "version": VERSION,
        "series": len(series),
        "dicoms": len(files),
        "source_bytes": file_bytes,
        "matrix_groups": len(groups),
        "matrix_shapes": [list(shape) for shape in groups],
        "unique_pixel_aliases": len(aliases),
        "duplicate_frames": len(files) - len(aliases),
        "geometry_frames": geometry_frames,
        "instance_fallback_frames": len(files) - geometry_frames,
        "signed_frames": signed_frames,
        "bits_stored_counts": bit_counts,
        "photometric_counts": photometric_counts,
        "original_source_order_supplied": ordered_files is not None,
    }
    return result, stats
