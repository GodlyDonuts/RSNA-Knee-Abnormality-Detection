"""Separate tensor/mask/order comparisons and sanitized finite aggregates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor


def compare_renderings(
    reference: dict, candidate: dict, atol: float = 0.0, rtol: float = 0.0
) -> dict:
    """Report tensor, slot-mask and ordering equality independently; never merge them."""
    a, b = reference["images"], candidate["images"]
    shape_equal = tuple(a.shape) == tuple(b.shape)
    result: dict = {"shape_equal": shape_equal, "shapes": [list(a.shape), list(b.shape)]}
    if shape_equal:
        diff = (a.double() - b.double()).abs()
        result.update(
            tensor_exact=bool(torch.equal(a, b)),
            tensor_max_abs=float(diff.max()) if diff.numel() else 0.0,
            tensor_within_tolerance=bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        )
    result["slot_mask_equal"] = bool(torch.equal(reference["slot_mask"], candidate["slot_mask"]))
    for key in ("slot_series", "slice_order"):
        if key in reference or key in candidate:
            result[f"{key}_equal"] = reference.get(key) == candidate.get(key)
    return result


def tensor_stats(images: Tensor, mask: Tensor | None = None) -> dict:
    values = images if mask is None else images[mask]
    finite = bool(torch.isfinite(values).all()) if values.numel() else True
    out: dict = {"shape": list(images.shape), "finite": finite, "elements": values.numel()}
    if values.numel() and finite:
        v = values.double()
        out.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), std=float(v.std()))
    return out


def summarize(values: Iterable[float]) -> dict:
    """Finite quantile summary; non-finite inputs are counted, never exported."""
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    finite = array[np.isfinite(array)]
    out: dict = {"n": int(array.size), "nonfinite": int(array.size - finite.size)}
    if finite.size:
        q = np.quantile(finite, [0, 0.25, 0.5, 0.75, 1])
        out.update(dict(zip(["min", "q25", "median", "q75", "max"], map(float, q))))
    return out


def physical_field_mm(shape: Sequence[int], spacing: Sequence[float]) -> float:
    if len(spacing) != 2 or not all(math.isfinite(s) and s > 0 for s in spacing):
        return math.nan
    return float(max(shape[-2] * spacing[0], shape[-1] * spacing[1]))


SANITIZED_KEYS = frozenset(
    {
        "roster_size",
        "roster_rule",
        "provenance",
        "fov_mm",
        "slot_occupancy",
        "channel_coverage",
        "order",
        "tensor",
        "comparison_counts",
        "refusals",
    }
)


def sanitize(report: dict) -> dict:
    """Keep only whitelisted aggregate keys and refuse anything that looks like an identifier."""
    unknown = set(report) - SANITIZED_KEYS
    if unknown:
        raise ValueError(f"Unsanitized report keys: {sorted(unknown)}")

    def check(value: object) -> None:
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item)
        elif isinstance(value, str) and value.count(".") >= 3 and value.replace(".", "").isdigit():
            raise ValueError("DICOM UID-like value in exported aggregates")

    check(report)
    return report
