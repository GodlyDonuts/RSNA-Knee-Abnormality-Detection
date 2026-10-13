"""Checkpoint-bound public DINOv2 input recipe and rendering contract checks.

Fields are copied from retained audits, each with its evidence. `source_verified`
stays False until the exact v15 source review binds the same constant; README prose
is never treated as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

AUDIT = "initiative/audits/"
FINGERPRINT = AUDIT + "public-dinov2-fingerprint-20261007-v1/actual-receipt.json"
COLLECTION = AUDIT + "public-solution-intelligence-20261007-v1/README.md#manifest"


@dataclass(frozen=True)
class Field:
    value: object
    evidence: str
    source_verified: bool = False


RECIPE: dict[str, Field] = {
    "image_side": Field(336, FINGERPRINT + " fingerprint_config.img / actual_input_config.side"),
    "slots": Field(6, FINGERPRINT + " fingerprint_config.n_slot / actual_input_config.slots"),
    "channels_per_image": Field(3, FINGERPRINT + " fingerprint_config.group / channels"),
    "selected_slices": Field(12, COLLECTION + " (downloaded weights manifest field)"),
    "field_of_view_mm": Field(130.0, COLLECTION + " (downloaded weights manifest field)"),
    "slice_band": Field((0.2, 0.8), COLLECTION + " (downloaded weights manifest field)"),
    "slot_semantics": Field(
        "anatomy/contrast/fat-suppression", COLLECTION + " (downloaded weights manifest field)"
    ),
}
UNBOUND = (
    "slot order and assignment rule",
    "missing-slot fill and mask polarity",
    "slice ordering key and laterality rule",
    "intensity normalization",
    "crop/resize interpolation",
    "tensor layout between slices and three-slice channels",
    "inference window aggregation",
)


def validate_rendering(images: Tensor, slot_mask: Tensor) -> dict:
    """Check only checkpoint-bound facts; unknown layout fields are reported, not assumed."""
    side, slots, channels = (RECIPE[k].value for k in ("image_side", "slots", "channels_per_image"))
    if images.ndim < 4 or tuple(images.shape[-3:]) != (channels, side, side):
        raise ValueError(f"Rendering must end in ({channels},{side},{side})")
    if images.shape[0] != slots or tuple(slot_mask.shape) != (slots,):
        raise ValueError(f"Rendering and slot mask need {slots} leading slots")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("Rendering must be finite floating point")
    if slot_mask.dtype != torch.bool:
        raise ValueError("Slot mask must be boolean")
    return {"shape": list(images.shape), "occupied_slots": int(slot_mask.sum())}
