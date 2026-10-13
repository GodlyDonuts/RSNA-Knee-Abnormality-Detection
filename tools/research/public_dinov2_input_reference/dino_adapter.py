"""Research adapter from source-exact public pixels to DINOv3 study inputs.

No source crop, intensity, ordering or laterality is reconstructed here. Pixel
arrays must come from the reviewed original renderer. Geometry and sequence
features are supplied explicitly; no physical positions are fabricated.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rsna.schema import TARGETS


def study_inputs(
    images: Tensor,
    series_mask: Tensor,
    features: Tensor,
    slice_positions: Tensor,
    labels: Tensor,
    label_mask: Tensor,
    *,
    study_uid: str,
) -> dict:
    """Keep every original overlapping triplet, in the original channel order.

    Returns float [6,10,3,336,336] in [0,1]. The official encoder, rather
    than this adapter, performs ImageNet normalization. Each triplet's physical
    position is its center slice, with that slice's original validity flag.
    """
    if images.shape != (6, 12, 336, 336) or images.dtype != torch.uint8:
        raise ValueError("Exact source-rendered uint8 six-slot twelve-slice336 cache required")
    if series_mask.shape != (6,) or series_mask.dtype != torch.bool or not series_mask.any():
        raise ValueError("Nonempty original acquired-slot mask required")
    if features.shape != (6, 5) or slice_positions.shape != (6, 12, 2):
        raise ValueError("Explicit source sequence features and original geometry required")
    if features.dtype != torch.float32 or slice_positions.dtype != torch.float32:
        raise ValueError("Source metadata must use the model's float32 contract")
    if not torch.isfinite(features).all() or not torch.isfinite(slice_positions).all():
        raise ValueError("Source metadata must be finite")
    validity = slice_positions[..., 1]
    if not ((validity == 0) | (validity == 1)).all():
        raise ValueError("Physical-position validity must be binary")
    if slice_positions[..., 0][validity == 0].count_nonzero():
        raise ValueError("Unknown physical positions must remain zero")
    if (
        images[~series_mask].count_nonzero()
        or features[~series_mask].count_nonzero()
        or slice_positions[~series_mask].count_nonzero()
    ):
        raise ValueError("Absent source slots must remain zero")
    if labels.shape != (len(TARGETS),) or label_mask.shape != labels.shape:
        raise ValueError("All twelve source targets and masks are required")
    if (
        labels.dtype != torch.float32
        or label_mask.dtype != torch.float32
        or any(
            t.device != images.device
            for t in (series_mask, features, slice_positions, labels, label_mask)
        )
    ):
        raise ValueError("Floating targets and source inputs must share one device")
    if not label_mask.is_floating_point() or not ((label_mask == 0) | (label_mask == 1)).all():
        raise ValueError("Target masks must be floating binary values")
    observed = labels[label_mask.bool()]
    if not torch.isfinite(observed).all() or not ((observed >= 0) & (observed <= 1)).all():
        raise ValueError("Observed soft targets must be finite values in [0,1]")
    if not isinstance(study_uid, str) or not study_uid:
        raise ValueError("Original study identity is required")
    windows = torch.stack([images[:, start : start + 3] for start in range(10)], dim=1)
    return {
        "images": windows.float().div(255),
        "series_mask": series_mask.clone(),
        "slice_mask": series_mask[:, None].expand(6, 10).clone(),
        "features": features.clone(),
        "positions": slice_positions[:, 1:11].clone(),
        "labels": torch.where(label_mask.bool(), labels, torch.zeros_like(labels)).clone(),
        "label_mask": label_mask.clone(),
        "study_uid": study_uid,
    }


def render_inputs(
    namespace: dict, study_dir, plane_map: dict, labels: Tensor, label_mask: Tensor
) -> dict:
    """Original renderer plus explicit original-header geometry for its samples.

    A triplet position is the center slice's coordinate, centered on the
    observed series extent and expressed in100mm units as in our current model.
    Missing or inconsistent geometry remains unknown. Right sagittal stacks
    reverse positions exactly when the original renderer reverses pixel slices.
    """
    from pathlib import Path

    import numpy as np
    import pydicom

    from rsna.data import PLANES

    from . import reference

    if (
        namespace["IMG"] != 336
        or namespace["CACHE_SLICES"] != 12
        or namespace["GROUP"] != 3
        or tuple(namespace["SLICE_BAND"]) != (0.2, 0.8)
        or namespace["N_SLOT"] != 6
    ):
        raise ValueError("Exact admitted public pixel group required")
    out = reference.render_original(namespace, [Path(study_dir)], plane_map)
    uid = Path(study_dir).name
    if out["studies"] != [uid]:
        raise ValueError("Original renderer produced no source study")
    chosen = out["slots"][uid]
    names = [slot[0] for slot in namespace["SLOTS"]]
    images = torch.from_numpy(out["cache"][0])
    mask = torch.from_numpy(out["mask"][0]).bool()
    features = torch.zeros(6, 5)
    positions = torch.zeros(6, 12, 2)
    for slot, name in enumerate(names):
        if not mask[slot]:
            continue
        record = chosen[name]
        plane = namespace["SLOTS"][slot][1]
        features[slot, PLANES[plane]] = 1
        features[slot, 3] = float(record["fluid"])
        features[slot, 4] = float(record["fatsat"])
        files = record.get("ordered") or record["files"]
        n = len(files)
        lo, hi = (int(bound * (n - 1)) for bound in namespace["SLICE_BAND"])
        indices = np.unique(np.linspace(lo, hi, 12).astype(int)) if hi > lo else np.array([n // 2])
        while len(indices) < 12:
            indices = np.append(indices, indices[-1])
        headers = []
        for file in files:
            try:
                h = pydicom.dcmread(Path(record["dir"]) / file, stop_before_pixels=True)
                iop = np.asarray(h.ImageOrientationPatient, dtype=np.float64)
                ipp = np.asarray(h.ImagePositionPatient, dtype=np.float64)
                if (
                    iop.shape != (6,)
                    or ipp.shape != (3,)
                    or not np.isfinite(iop).all()
                    or not np.isfinite(ipp).all()
                ):
                    raise ValueError("Missing geometry")
                row, col = iop[:3], iop[3:]
                if not np.allclose(
                    [np.linalg.norm(row), np.linalg.norm(col), np.dot(row, col)],
                    [1, 1, 0],
                    atol=1e-3,
                ):
                    raise ValueError("Invalid orientation")
                headers.append((iop, ipp))
            except (AttributeError, ValueError, OSError, pydicom.errors.InvalidDicomError):
                headers.append(None)
        valid = [h for h in headers if h is not None]
        if valid and all(np.allclose(h[0], valid[0][0], atol=1e-3) for h in valid):
            normal = np.cross(valid[0][0][:3], valid[0][0][3:])
            projected = [float(h[1] @ normal) for h in valid]
            center = (min(projected) + max(projected)) / 2
            for sample, index in enumerate(indices[:12]):
                h = headers[int(index)]
                if h is not None:
                    positions[slot, sample] = torch.tensor(
                        [(float(h[1] @ normal) - center) / 100, 1]
                    )
        if out["lat"].get(uid) == "R" and plane == "Sagittal":
            positions[slot] = positions[slot].flip(0)
    return study_inputs(images, mask, features, positions, labels, label_mask, study_uid=uid)
