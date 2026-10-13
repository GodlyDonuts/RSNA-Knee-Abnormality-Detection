"""Original Pilkwang v15 rows for the streaming source-bound public triplet cache.

Each row is rendered by the exact hashed ``dino_adapter.render_inputs`` (original
nodes plus explicit original-header geometry). The raw uint8 [6,12,336,336] row,
slot mask, five sequence features and [6,12,2] positions it passes to the
adapter's own ``study_inputs`` are captured unchanged; the float windows are
validated there and discarded. No labels are read: the adapter receives an
all-masked placeholder so evaluation and fitting frames stay the only source.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from rsna.competitive import public_input
from rsna.schema import TARGETS

PACKAGE = __package__ or "tools.research.public_dinov2_input_reference"


def source_row_renderer(
    policy: dict,
    namespace: dict,
    receipt: dict,
    pixel_group: dict,
    split_root: Path,
    plane_map: dict,
) -> Callable[[str], dict[str, np.ndarray]]:
    """Bind the configured original namespace to the policy; return ``uid -> row``."""
    public_input.validate_public_input(policy)
    if (
        receipt.get("source_sha256") != policy["source_sha256"]
        or receipt.get("node_text_sha256") != policy["node_text_sha256"]
    ):
        raise ValueError("Renderer namespace is not the policy's reviewed source")
    if public_input.canonical_sha256(pixel_group) != policy["pixel_group_sha256"]:
        raise ValueError("Pixel group differs from the bound policy")
    if [slot[0] for slot in namespace["SLOTS"]] != list(pixel_group["slots"]):
        raise ValueError("Renderer namespace was not configured with the bound pixel group")
    adapter = public_input.load_adapter(policy)
    adapter.__package__ = PACKAGE
    original = adapter.study_inputs
    captured: list[tuple] = []

    def capture(images, series_mask, features, slice_positions, labels, label_mask, *, study_uid):
        original(
            images, series_mask, features, slice_positions, labels, label_mask, study_uid=study_uid
        )
        captured.append((study_uid, images, series_mask, features, slice_positions))
        return None

    adapter.study_inputs = capture
    split_root = Path(split_root)

    def render(uid: str) -> dict[str, np.ndarray]:
        study = split_root / uid
        if not study.is_dir() or study.is_symlink():
            raise ValueError("Original study directory is unavailable")
        captured.clear()
        placeholder = torch.zeros(len(TARGETS))
        adapter.render_inputs(namespace, study, plane_map, placeholder, placeholder.clone())
        if len(captured) != 1 or captured[0][0] != uid:
            raise ValueError("Original renderer did not produce exactly this study")
        _, images, mask, features, positions = captured.pop()
        return {
            "images": np.ascontiguousarray(images.numpy()),
            "slot_mask": np.ascontiguousarray(mask.numpy()),
            "features": np.ascontiguousarray(features.numpy()),
            "positions": np.ascontiguousarray(positions.numpy()),
        }

    return render
