"""Identical full-map inputs for cached engineering and online frozen encoding.

Private indexes and clinical rows belong on the admitted owning node. The
provider never trains its parent. No image crop, source selection or encoder
precision changes are introduced here.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.compact_bag import candidate_windows, window_tensor
from rsna.competitive.criterion_geometry import within_series_relations
from rsna.competitive.spatial_reader import build_plan
from rsna.data import select_series


def source_graphs(dataset, uid: str, windows: list[tuple[int, int]]) -> list[dict]:
    """Reproduce the accepted original scout's metadata/support rows exactly."""
    rows = []
    for slot, entry in enumerate(select_series(dataset.series[uid], 6, False)):
        sid = entry["SeriesInstanceUID"]
        metadata_sha, meta = dataset._metadata[(uid, sid)]
        selected = (
            np.linspace(0, meta["n_slices"] - 1, min(16, meta["n_slices"])).round().astype(np.int64)
        )
        graph = within_series_relations(meta, selected)
        if [j for s, j in windows if s == slot] != list(range(len(graph.supports))):
            raise ValueError("Online window/support membership differs from original scout")
        rows.append(
            {
                "slot": slot,
                "series_uid": sid,
                "source_metadata_sha256": metadata_sha,
                "selected_source_indices": selected.tolist(),
                "supports": graph.supports.tolist(),
                "centers_mm": graph.centers_mm.tolist(),
                "continuity_edges": graph.continuity_edges.tolist(),
                "continuity_gaps_mm": graph.continuity_gaps_mm.tolist(),
                "repeat_edges": graph.repeat_edges.tolist(),
                "repeat_gaps_mm": graph.repeat_gaps_mm.tolist(),
                "geometry_known": graph.geometry_known,
                "native_original_shape": meta["original_shape"],
                "native_resized_shape": meta["resized_shape"],
                "native_cache_side": meta["side"],
                "pixel_spacing": meta["pixel_spacing"],
                "image_orientation_patient": meta["orientation"],
                "selected_image_positions_patient": [meta["positions"][int(j)] for j in selected],
                "full_frame_letterbox_align_corners_false": True,
                "input_augmentation_or_mirror": False,
            }
        )
    return rows


class PlanCache:
    """Bounded metadata cache; no MRI feature or model cache."""

    def __init__(self, seed: int, maximum=8):
        self.seed, self.maximum = seed, maximum
        self.values: OrderedDict[str, object] = OrderedDict()

    def get(self, uid, graphs, windows, shape):
        key = hashlib.sha256(
            json.dumps([uid, graphs, windows.tolist(), list(shape)], sort_keys=True).encode()
        ).hexdigest()
        if key not in self.values:
            self.values[key] = build_plan(graphs, windows, shape, seed=self.seed)
            if len(self.values) > self.maximum:
                self.values.popitem(last=False)
        self.values.move_to_end(key)
        return self.values[key]


def _supervision(labels, masks):
    labels = torch.as_tensor(labels, dtype=torch.float32)
    masks = torch.as_tensor(masks)
    if masks.shape != (12,) or labels.shape != (12,) or not ((masks == 0) | (masks == 1)).all():
        raise ValueError("Original twelve binary supervision masks required")
    masks = masks.bool()
    if (
        not torch.isfinite(labels[masks]).all()
        or not ((labels[masks] == 0) | (labels[masks] == 1)).all()
    ):
        raise ValueError("Accepted original observed binary labels required")
    return torch.where(masks, labels, torch.zeros_like(labels)), masks


class CachedSpatialData:
    """Already admitted immutable original maps; stat identity follows full proof."""

    def __init__(self, root: Path, records: list[dict], labels, masks, binding: dict, *, seed: int):
        self.root, self.records, self.binding = root, records, binding
        self.labels, self.masks = labels, masks
        self.plans = PlanCache(seed)
        self.verified: set[str] = set()
        if len(records) != len(labels) or len(records) != len(masks) or not records:
            raise ValueError("Complete source-bound supervised roster required")
        self.signatures = {}
        for row in records:
            path = self._path(row)
            stat = path.stat()
            if stat.st_size != row["bytes"] or not row.get("sha256"):
                raise ValueError("Original privately verified feature file differs")
            self.signatures[row["file"]] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _path(self, row):
        path = self.root / row["file"]
        if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Original owned feature member required")
        return path

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        path = self._path(row)
        stat = path.stat()
        if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != self.signatures[row["file"]]:
            raise ValueError("Admitted original feature member changed")
        raw = path.read_bytes()
        if row["file"] not in self.verified:
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Original immutable feature bytes differ")
            self.verified.add(row["file"])
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            maps = torch.from_numpy(arrays["spatial_maps"].copy())
            base = torch.from_numpy(arrays["base_logits"].copy())
            windows = arrays["windows"].copy()
        if maps.dtype != torch.float32 or base.dtype != torch.float32:
            raise ValueError("Original unquantized FP32 representation required")
        plan = self.plans.get(row["study_uid"], row["source_graphs"], windows, maps.shape[1:])
        labels, masks = _supervision(self.labels[index], self.masks[index])
        return {
            "study_uid": row["study_uid"],
            "maps": maps.contiguous(),
            "base_logits": base,
            "plan": plan,
            "labels": labels,
            "label_mask": masks,
        }


class OnlineSpatialData:
    """One unchanged frozen parent forward; learned projection sees FULL maps."""

    def __init__(self, dataset, parent, binding: dict, *, seed: int, chunk_size=12):
        if parent.training or any(p.requires_grad for p in parent.parameters()):
            raise ValueError("Frozen eval parent required before creating online data")
        if (
            dataset.augment
            or dataset.side != 384
            or dataset.slices != 16
            or dataset.max_series != 6
            or dataset.slice_sampling != "uniform"
            or dataset.series_sampling != "selected"
            or dataset.neighbor_channels is not None
            or dataset.cache_contrast is not None
            or dataset.input_channels != 1
        ):
            raise ValueError("Original fixed N3 native input policy required")
        self.dataset, self.parent, self.binding = dataset, parent, binding
        self.chunk_size, self.plans = chunk_size, PlanCache(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.parent.training or any(p.requires_grad for p in self.parent.parameters()):
            raise ValueError("Online provider cannot fit the original parent")
        sample = self.dataset[index]
        windows = candidate_windows(sample["series_mask"], sample["slice_mask"])
        device = next(self.parent.parameters()).device
        captured = []
        handle = self.parent.encoder.head.global_pool.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach())
        )
        try:
            # no_grad rather than inference_mode: these constants must be legal
            # saved inputs to the learned projection's actual backward pass.
            with torch.no_grad():
                base = self.parent(
                    window_tensor(sample["images"], windows).to(device), chunk_size=self.chunk_size
                )
                maps = torch.cat(captured).float().contiguous()
        finally:
            handle.remove()
        graphs = source_graphs(self.dataset, sample["study_uid"], windows)
        plan = self.plans.get(
            sample["study_uid"], graphs, np.asarray(windows, dtype=np.int64), maps.shape[1:]
        )
        labels, masks = _supervision(sample["labels"], sample["label_mask"])
        return {
            "study_uid": sample["study_uid"],
            "maps": maps,
            "base_logits": base,
            "plan": plan,
            "labels": labels,
            "label_mask": masks,
        }
