"""Source-specific MRNet exam tensors and three-task MRI adapter warm-up head.

No MRNet labels map to the twelve RSNA targets. Physical spacing and unobserved
sequence attributes are not invented. Only encoder adapter tensors transfer.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from rsna.competitive.model import _masked_softmax

MRNET_TARGETS = ("acl", "meniscus", "abnormal")
MRNET_PLANES = ("axial", "coronal", "sagittal")


class MRNetStudyDataset(Dataset):
    def __init__(
        self, root: Path, receipt: Path, split: str, *, image_size: int = 256, slices: int = 16
    ):
        if split not in ("train", "valid") or image_size % 16 or not 32 <= image_size <= 512:
            raise ValueError("Explicit source split and DINO patch-compatible size required")
        if not 1 <= slices <= 256:
            raise ValueError("Invalid MRNet slice budget")
        admitted = json.loads((root / "source-admission.json").read_text())
        accepted = json.loads(receipt.read_text())
        journal = root / "verified-files.jsonl"
        if (
            admitted["source"] != "aimi.mrnet_knee_mri_s:4a2c:v1_0"
            or admitted["target_order"] != list(MRNET_TARGETS)
            or admitted["rsna_target_mapping"] is not None
            or accepted["status"] != "verbatim_private_MRNet_images_and_labels_verified"
            or accepted["source"] != admitted["source"]
            or hashlib.sha256(journal.read_bytes()).hexdigest() != accepted["file_journal_sha256"]
        ):
            raise ValueError("Original accepted private MRNet acquisition required")
        manifest = root / "downloads.json"
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != accepted["manifest_sha256"]:
            raise ValueError("Original source table manifest changed")
        original = json.loads(manifest.read_text())
        if len(original) != 8:
            raise ValueError("All eight original source tables required")
        for row in original:
            if row["name"] not in {
                "train",
                "valid",
                *(f"{s}-{t}" for s in ("train", "valid") for t in MRNET_TARGETS),
            }:
                raise ValueError("Unknown source table")
            path = root / (row["name"] + ".csv")
            raw = path.read_bytes()
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Original source table bytes changed")
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            key = (row["split"], row["name"])
            if key in self.rows:
                raise ValueError("Duplicate original source volume")
            self.rows[key] = row
        if len(self.rows) != accepted["files"]:
            raise ValueError("Incomplete original volume journal")
        spec = admitted["splits"][split]
        self.uids = tuple(spec["supervised_cases"])
        self.labels = spec["labels"]
        for target in MRNET_TARGETS:
            with (root / f"{split}-{target}.csv").open(newline="") as stream:
                reader = csv.reader(stream)
                next(reader)
                observed = {uid.zfill(4): int(value) for uid, value in reader}
            if observed != self.labels[target]:
                raise ValueError("Derived labels differ from original source values")
        if set(self.uids) != set(self.labels[MRNET_TARGETS[0]]):
            raise ValueError("Supervised membership differs from original source values")
        self.root, self.split, self.image_size, self.slices = root, split, image_size, slices
        self.training_permitted = spec["training_permitted"]
        if self.training_permitted is not (split == "train"):
            raise ValueError("Source validation must remain out of fitting")
        for uid in self.uids:
            if uid in spec["unlabeled_cases"] or any(
                uid not in self.labels[t] for t in MRNET_TARGETS
            ):
                raise ValueError("Missing labels cannot become supervised examples")
            if any((split, f"{plane}/{uid}.npy") not in self.rows for plane in MRNET_PLANES):
                raise ValueError("Missing source plane")

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        uid = self.uids[index]
        images = torch.zeros(3, self.slices, 1, self.image_size, self.image_size)
        mask = torch.zeros(3, self.slices, dtype=torch.bool)
        ordinal = torch.zeros(3, self.slices)
        for plane_index, plane in enumerate(MRNET_PLANES):
            name = f"{plane}/{uid}.npy"
            row = self.rows[(self.split, name)]
            path = self.root / self.split / name
            raw = path.read_bytes()
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Original source volume changed after acquisition")
            array = np.load(path, allow_pickle=False)
            if array.dtype != np.uint8 or list(array.shape) != row["shape"]:
                raise ValueError("Original uint8 volume shape mismatch")
            depth = array.shape[0]
            count = min(self.slices, depth)
            indices = np.linspace(0, depth - 1, count).round().astype(np.int64)
            selected = torch.from_numpy(array[indices].copy()).float().unsqueeze(1) / 255
            if self.image_size != 256:
                selected = F.interpolate(
                    selected,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).clamp(0, 1)
            images[plane_index, :count] = selected
            mask[plane_index, :count] = True
            # Ordinal array position is observed; it is not a physical distance.
            ordinal[plane_index, :count] = torch.from_numpy(indices.astype(np.float32)) / max(
                depth - 1, 1
            )
        return {
            "source_case_id": uid,
            "source_split": self.split,
            "images": images,
            "slice_mask": mask,
            "ordinal": ordinal,
            "labels": torch.tensor(
                [self.labels[t][uid] for t in MRNET_TARGETS], dtype=torch.float32
            ),
        }


class MRNetAdapterNet(nn.Module):
    """Task attention over all three planes, with no per-slice disease labels."""

    def __init__(
        self, encoder: nn.Module, embedding_dim: int, hidden_dim: int = 128, chunk: int = 32
    ):
        super().__init__()
        if min(embedding_dim, hidden_dim, chunk) < 1:
            raise ValueError("Positive MRNet model dimensions required")
        self.encoder, self.chunk = encoder, chunk
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.plane = nn.Embedding(3, hidden_dim)
        self.ordinal = nn.Linear(1, hidden_dim, bias=False)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 3)
        )
        self.output_weight = nn.Parameter(torch.empty(3, hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(3))
        nn.init.xavier_uniform_(self.output_weight)

    def forward(self, images: Tensor, mask: Tensor, ordinal: Tensor) -> Tensor:
        if images.ndim != 6 or images.shape[1] != 3 or images.shape[3] != 1:
            raise ValueError("MRNet exam images must be [B,3,K,1,H,W]")
        if (
            mask.shape != images.shape[:3]
            or mask.dtype != torch.bool
            or ordinal.shape != mask.shape
        ):
            raise ValueError("Exact source slice masks and ordinal positions required")
        if any(t.device != images.device for t in (mask, ordinal)) or not mask.any(dim=2).all():
            raise ValueError("Every source plane needs observed slices on one device")
        selected = images[mask]
        if not torch.isfinite(selected).all() or (selected < 0).any() or (selected > 1).any():
            raise ValueError("Valid source pixels must be finite gray values in [0,1]")
        if (
            not torch.isfinite(ordinal[mask]).all()
            or (ordinal[mask] < 0).any()
            or (ordinal[mask] > 1).any()
        ):
            raise ValueError("Observed source ordinal positions must be in [0,1]")
        features = torch.cat([self.encoder(part) for part in selected.split(self.chunk)])
        projected = self.projection(features)
        hidden = projected.new_zeros(*mask.shape, projected.shape[-1])
        hidden[mask] = projected
        safe_ordinal = ordinal.masked_fill(~mask, 0)
        hidden = (
            hidden
            + self.plane.weight[None, :, None]
            + self.ordinal(safe_ordinal[..., None].to(hidden.dtype))
        )
        hidden = hidden.masked_fill(~mask[..., None], 0)
        flattened = hidden.flatten(1, 2)
        weights = _masked_softmax(self.attention(flattened), mask.flatten(1, 2)[..., None], dim=1)
        pooled = torch.einsum("bnt,bnh->bth", weights, flattened)
        return (pooled * self.output_weight).sum(-1) + self.output_bias


def transfer_encoder_adapters(
    source: nn.Module,
    destination: nn.Module,
    *,
    source_provenance: dict[str, Any],
    destination_provenance: dict[str, Any],
) -> int:
    """Copy only complete compatible residual tensors; never transfer source heads.

    Require identical original base provenance and complete adapter policy before
    copying anything. Source heads and base weights never transfer here.
    """
    keys = (
        "architecture",
        "config_sha256",
        "weights_sha256",
        "source_id",
        "source_revision",
        "input_contract",
        "adapter",
    )
    if source_provenance.get("architecture") != "dinov3_l16" or any(
        key not in source_provenance
        or key not in destination_provenance
        or source_provenance[key] != destination_provenance[key]
        for key in keys
    ):
        raise ValueError("Identical original base provenance and adapter policy required")
    source_state, destination_state = source.state_dict(), destination.state_dict()
    source_names = {name for name in source_state if name.endswith((".lora_A", ".lora_B"))}
    destination_names = {
        name for name in destination_state if name.endswith((".lora_A", ".lora_B"))
    }
    if not source_names or source_names != destination_names:
        raise ValueError("Exact complete encoder adapter graph required")
    for name in source_names:
        if (
            source_state[name].shape != destination_state[name].shape
            or not torch.isfinite(source_state[name]).all()
        ):
            raise ValueError("Adapter rank or numerical data incompatible")
    with torch.no_grad():
        for name in source_names:
            destination_state[name].copy_(source_state[name])
    return len(source_names)
