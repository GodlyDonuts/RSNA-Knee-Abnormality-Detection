"""Keep exact successful decoding; recover usable MRI from irregular metadata.

This is a prospective serving adapter, not a change to the trained numerical
source. Every output still comes from real decoded MRI. No label, constant
prediction or fabricated image is used. All rejected frames are counted.
"""

import collections
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.predictive.cases import RawGroup
from rsna.competitive.predictive.dicom import _integer, _position
from rsna.competitive.predictive.dicom import load_study as original_load_study
from rsna.competitive.predictive.inputs import fixed_metadata_scale

VERSION = "exact_first_usable_MRI_metadata_recovery_v1"


def load_study(study_root, study_uid):
    try:
        groups, stats = original_load_study(study_root, study_uid)
        # The downstream original renderer requires four tiles with a halo of8.
        if any(min(g.frames.shape[1:]) // 4 <= 16 for g in groups):
            raise ValueError("Source matrix is too small for the trained renderer")
        return groups, {**stats, "decoder_route": "original_exact", "decoder_version": VERSION}
    except ValueError:
        return recover(study_root, study_uid)


def recover(study_root, study_uid):
    import pydicom

    root = Path(study_root)
    if not root.is_dir() or root.name != study_uid:
        raise ValueError("Requested MRI study directory is unavailable")
    members = [
        (d, sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"))
        for d in sorted(root.iterdir())
        if d.is_dir()
    ]
    groups, aliases, alias_values = {}, {}, {}
    rejected = collections.Counter()
    repairs = collections.Counter()
    counts = collections.Counter()
    bits_counts, photos = collections.Counter(), collections.Counter()
    for acquisition, (directory, files) in enumerate(members):
        for file_index, path in enumerate(files):
            try:
                ds = pydicom.dcmread(path)
            except Exception:
                rejected["unreadable_DICOM"] += 1
                continue
            if str(ds.get("Modality", "MR")).upper() != "MR":
                rejected["non_MRI"] += 1
                continue
            try:
                bits, signed = _integer(ds, "BitsStored"), _integer(ds, "PixelRepresentation")
                photometric = str(ds.get("PhotometricInterpretation", ""))
                if (
                    not 1 <= bits <= 16
                    or signed not in (0, 1)
                    or photometric not in ("MONOCHROME1", "MONOCHROME2")
                    or _integer(ds, "SamplesPerPixel") != 1
                ):
                    raise ValueError
                pixel = ds.pixel_array
                if (
                    pixel.ndim != 2
                    or not np.issubdtype(pixel.dtype, np.integer)
                    or min(pixel.shape) // 4 <= 16
                ):
                    raise ValueError
                slope, intercept = (
                    float(ds.get("RescaleSlope", 1)),
                    float(ds.get("RescaleIntercept", 0)),
                )
                if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
                    raise ValueError
                normalized = fixed_metadata_scale(
                    torch.from_numpy(pixel.astype(np.int32)),
                    bits_stored=bits,
                    signed=bool(signed),
                    photometric=photometric,
                )
            except Exception:
                rejected["unsupported_or_invalid_pixel_encoding"] += 1
                continue
            expected = (study_uid, directory.name, path.stem)
            actual = tuple(
                str(ds.get(k, ""))
                for k in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
            )
            if actual != expected:
                repairs["path_identity_used"] += 1
            try:
                instance = _integer(ds, "InstanceNumber")
            except ValueError:
                instance = file_index
                repairs["missing_instance_source_order"] += 1
            try:
                position, has_geometry = _position(ds, instance)
            except ValueError:
                position, has_geometry = float(instance), False
                repairs["incomplete_geometry_instance_order"] += 1
            key = hashlib.sha256(
                np.array(pixel.shape, dtype="<i8").tobytes() + pixel.astype("<f8").tobytes()
            ).hexdigest()
            if key in alias_values and not torch.equal(alias_values[key], normalized):
                rejected["conflicting_encoding_of_duplicate_pixels"] += 1
                continue
            if key not in aliases:
                aliases[key] = len(aliases)
                alias_values[key] = normalized
            groups.setdefault(tuple(pixel.shape), []).append(
                (acquisition, position, instance, normalized, aliases[key])
            )
            counts["source_bytes"] += path.stat().st_size
            counts["geometry_frames"] += int(has_geometry)
            counts["signed_frames"] += signed
            counts["dicoms"] += 1
            bits_counts[str(bits)] += 1
            photos[photometric] += 1
    if not groups:
        raise ValueError("No usable original MRI frames; no fabricated prediction")
    items = sorted((x for group in groups.values() for x in group), key=lambda x: x[:3])
    order = {}
    for acquisition in sorted({x[0] for x in items}):
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
    return result, {
        "version": VERSION,
        "decoder_version": VERSION,
        "decoder_route": "usable_MRI_metadata_recovery",
        "series": len(members),
        "dicoms": counts["dicoms"],
        "source_bytes": counts["source_bytes"],
        "matrix_groups": len(groups),
        "matrix_shapes": [list(x) for x in groups],
        "unique_pixel_aliases": len(aliases),
        "duplicate_frames": counts["dicoms"] - len(aliases),
        "geometry_frames": counts["geometry_frames"],
        "instance_fallback_frames": counts["dicoms"] - counts["geometry_frames"],
        "signed_frames": counts["signed_frames"],
        "bits_stored_counts": dict(bits_counts),
        "photometric_counts": dict(photos),
        "rejected_frames": dict(rejected),
        "metadata_repairs": dict(repairs),
        "original_source_order_supplied": False,
    }
