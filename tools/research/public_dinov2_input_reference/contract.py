"""Checkpoint-bound public DINOv2 input recipe and rendering contract checks.

Each field names the retained audit and the reviewed v15 node that binds it. Member
`rules` and slot scheme are per-member manifest fields kept only on the owning node;
they are listed in UNBOUND and must be read from that manifest, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

AUDIT = "initiative/audits/"
FINGERPRINT = AUDIT + "public-dinov2-fingerprint-20261007-v1/actual-receipt.json"
COLLECTION = AUDIT + "public-solution-intelligence-20261007-v1/README.md (manifest fields)"


@dataclass(frozen=True)
class Field:
    value: object
    evidence: str
    source_verified: bool = True


RECIPE: dict[str, Field] = {
    "image_side": Field(336, FINGERPRINT + "; adopt_config_globals cfg['img'] -> IMG/CACHE_IMG"),
    "slots": Field(6, FINGERPRINT + "; SLOTS_RECOVERED/SLOTS_PUBLIC both define 6"),
    "channels_per_image": Field(3, FINGERPRINT + "; cfg['group'] -> GROUP, window_starts"),
    "selected_slices": Field(12, COLLECTION + "; cfg['slices'] -> CACHE_SLICES, read_slot"),
    "field_of_view_mm": Field(130.0, COLLECTION + "; cfg['crop_mm'] -> CROP_MM, read_slot"),
    "slice_band": Field((0.2, 0.8), COLLECTION + "; cfg['band'] -> SLICE_BAND, read_slot"),
    "slot_semantics": Field("plane x fluid x fat-suppression", "SLOTS_RECOVERED, pick_slots"),
    "intensity": Field("per-series 1st-99th percentile clip to [0,1] then uint8", "read_slot"),
    "resize": Field("bilinear, align_corners=False, after 130mm centre crop", "read_slot"),
    "missing_slot": Field("zero cache row, mask 0.0", "build_cache"),
    "laterality": Field("R knees: coronal/axial width flip, sagittal slice reversal", "lat_of"),
    "window_aggregation": Field("overlapping 3-slice windows, mean of sigmoid", "predict_member"),
    "model_input_scaling": Field("/255, ImageNet mean/std", "Model.forward (fingerprint audit)"),
}
UNBOUND = (
    "member pixel_group rules (order/lat/slot_fallback/decode_fill; native if absent)",
    "member slot scheme (recovered vs public slot names)",
)


def validate_rendering(images: Tensor, slot_mask: Tensor) -> dict:
    """Check an original cache row `[slots, slices, side, side]` uint8 plus bool mask."""
    side, slots = RECIPE["image_side"].value, RECIPE["slots"].value
    slices = RECIPE["selected_slices"].value
    if tuple(images.shape) != (slots, slices, side, side):
        raise ValueError(f"Rendering must be ({slots},{slices},{side},{side})")
    if images.dtype != torch.uint8:
        raise ValueError("Original cache rows are uint8")
    if slot_mask.dtype != torch.bool or tuple(slot_mask.shape) != (slots,):
        raise ValueError(f"Slot mask must be boolean ({slots},)")
    if bool(images[~slot_mask].any()):
        raise ValueError("Absent slots must stay zero")
    return {"shape": list(images.shape), "occupied_slots": int(slot_mask.sum())}
