"""Trainable DINO slice adapter connected to the patch hierarchy."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from .patch_model import PatchKneeAlibiModel, PatchKneeModelConfig
except ImportError:
    from patch_model import PatchKneeAlibiModel, PatchKneeModelConfig


def _encoder_layers(backbone: nn.Module) -> list[nn.Module]:
    candidates = [
        getattr(getattr(backbone, "encoder", None), "layer", None),
        getattr(getattr(backbone, "encoder", None), "layers", None),
        getattr(backbone, "blocks", None),
        getattr(backbone, "layer", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, (nn.ModuleList, nn.Sequential)):
            return list(candidate)
    return []


def freeze_except_last_blocks(backbone: nn.Module, trainable_blocks: int) -> int:
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    layers = _encoder_layers(backbone)
    if trainable_blocks > len(layers):
        raise ValueError(
            f"requested {trainable_blocks} trainable blocks but backbone exposes {len(layers)}"
        )
    for layer in layers[-trainable_blocks:] if trainable_blocks else []:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    return len(layers)


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(F.gelu(self.down(self.norm(x))))


class DinoSliceAdapter(nn.Module):
    """DINO backbone plus a small trainable residual token adapter."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int,
        patch_grid: int = 4,
        adapter_bottleneck: int = 64,
        trainable_blocks: int = 0,
    ) -> None:
        super().__init__()
        if patch_grid < 1:
            raise ValueError("patch_grid must be positive")
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.patch_grid = patch_grid
        self.adapter = ResidualAdapter(hidden_dim, adapter_bottleneck)
        self.exposed_blocks = freeze_except_last_blocks(backbone, trainable_blocks)
        self.num_register_tokens = int(
            getattr(getattr(backbone, "config", None), "num_register_tokens", 0) or 0
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        patch_grid: int = 4,
        adapter_bottleneck: int = 64,
        trainable_blocks: int = 0,
        local_files_only: bool = False,
    ) -> "DinoSliceAdapter":
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden_dim = int(getattr(backbone.config, "hidden_size"))
        return cls(
            backbone,
            hidden_dim,
            patch_grid,
            adapter_bottleneck,
            trainable_blocks,
        )

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(
            pixel_values=pixel_values, interpolate_pos_encoding=True
        ).last_hidden_state
        hidden = self.adapter(hidden)
        cls = hidden[:, 0]
        patches = hidden[:, 1 + self.num_register_tokens :]
        side = int(round(math.sqrt(patches.shape[1])))
        if side * side != patches.shape[1]:
            raise ValueError(f"DINO emitted {patches.shape[1]} non-square patch tokens")
        spatial = patches.transpose(1, 2).reshape(
            patches.shape[0], patches.shape[2], side, side
        )
        compact = F.adaptive_avg_pool2d(
            spatial.float(), (self.patch_grid, self.patch_grid)
        ).flatten(2).transpose(1, 2)
        summary = torch.cat([cls, patches.mean(dim=1)], dim=-1)
        return summary, compact


class EndToEndPatchKneeModel(nn.Module):
    """Encode valid 2.5-D slices and apply the target-specific hierarchy."""

    def __init__(
        self,
        slice_encoder: DinoSliceAdapter,
        hierarchy_config: PatchKneeModelConfig,
        encoder_batch_size: int = 32,
    ) -> None:
        super().__init__()
        if hierarchy_config.feature_dim != 2 * slice_encoder.hidden_dim:
            raise ValueError("hierarchy feature_dim must equal 2 * DINO hidden size")
        if hierarchy_config.patch_dim != slice_encoder.hidden_dim:
            raise ValueError("hierarchy patch_dim must equal DINO hidden size")
        self.slice_encoder = slice_encoder
        self.hierarchy = PatchKneeAlibiModel(hierarchy_config)
        self.encoder_batch_size = encoder_batch_size

    def forward(
        self,
        pixel_values: Tensor,
        positions_mm: Tensor,
        slice_mask: Tensor,
        series_mask: Tensor,
        plane: Tensor,
        fluid: Tensor,
        fatsat: Tensor,
        return_aux: bool = False,
    ) -> Tensor | Dict[str, Tensor]:
        # pixel_values: [B, R, S, 3, H, W]
        batch, n_series, n_slices = pixel_values.shape[:3]
        valid_mask = slice_mask.bool() & series_mask[..., None].bool()
        flat_pixels = pixel_values.reshape(-1, *pixel_values.shape[3:])
        valid_indices = valid_mask.flatten().nonzero(as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            raise ValueError("an end-to-end batch must contain at least one valid slice")
        summaries = []
        patches = []
        for start in range(0, valid_indices.numel(), self.encoder_batch_size):
            index = valid_indices[start : start + self.encoder_batch_size]
            summary, patch = self.slice_encoder(flat_pixels.index_select(0, index))
            summaries.append(summary)
            patches.append(patch)
        valid_summary = torch.cat(summaries, dim=0)
        valid_patches = torch.cat(patches, dim=0)
        total = flat_pixels.shape[0]
        summary_flat = valid_summary.new_zeros(total, valid_summary.shape[-1]).index_copy(
            0, valid_indices, valid_summary
        )
        patch_flat = valid_patches.new_zeros(
            total, valid_patches.shape[1], valid_patches.shape[2]
        ).index_copy(0, valid_indices, valid_patches)
        features = summary_flat.reshape(batch, n_series, n_slices, -1)
        patch_features = patch_flat.reshape(
            batch, n_series, n_slices, valid_patches.shape[1], valid_patches.shape[2]
        )
        patch_mask = valid_mask[..., None].expand(-1, -1, -1, valid_patches.shape[1])
        return self.hierarchy(
            features=features,
            patch_features=patch_features,
            patch_mask=patch_mask,
            positions_mm=positions_mm,
            slice_mask=slice_mask,
            series_mask=series_mask,
            plane=plane,
            fluid=fluid,
            fatsat=fatsat,
            return_aux=return_aux,
        )

    def trainable_parameter_report(self) -> Dict[str, int]:
        backbone_trainable = sum(
            parameter.numel()
            for parameter in self.slice_encoder.backbone.parameters()
            if parameter.requires_grad
        )
        adapter_trainable = sum(parameter.numel() for parameter in self.slice_encoder.adapter.parameters())
        hierarchy_trainable = sum(parameter.numel() for parameter in self.hierarchy.parameters())
        return {
            "backbone_trainable": backbone_trainable,
            "adapter_trainable": adapter_trainable,
            "hierarchy_trainable": hierarchy_trainable,
            "total_trainable": backbone_trainable + adapter_trainable + hierarchy_trainable,
        }
