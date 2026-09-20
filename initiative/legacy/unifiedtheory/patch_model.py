"""Target-specific patch -> slice -> series -> study MRI hierarchy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import Tensor, nn

try:
    from .model import AlibiBlock
except ImportError:
    from model import AlibiBlock


@dataclass
class PatchKneeModelConfig:
    feature_dim: int
    patch_dim: int
    hidden_dim: int = 256
    n_heads: int = 8
    series_depth: int = 2
    study_depth: int = 2
    dropout: float = 0.1
    aggregator: str = "physical_alibi"
    num_targets: int = 12
    series_dropout: float = 0.15
    token_adapter_bottleneck: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TargetPatchPool(nn.Module):
    """Let every pathology attend to a different spatial patch pattern."""

    def __init__(self, patch_dim: int, hidden_dim: int, num_targets: int) -> None:
        super().__init__()
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
        )
        self.queries = nn.Parameter(torch.empty(num_targets, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.scale = hidden_dim**-0.5

    def forward(self, patches: Tensor, patch_mask: Tensor) -> tuple[Tensor, Tensor]:
        # patches: [B, R, S, P, D], output: [B, R, S, T, H]
        projected = self.patch_projection(patches)
        scores = torch.einsum("brsph,th->brstp", projected, self.queries) * self.scale
        scores = scores.masked_fill(~patch_mask[..., None, :].bool(), -torch.inf)
        empty = ~patch_mask.any(dim=-1)
        scores = torch.where(empty[..., None, None], torch.zeros_like(scores), scores)
        attention = torch.softmax(scores, dim=-1)
        attention = attention * patch_mask[..., None, :].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("brstp,brsph->brsth", attention, projected)
        return pooled, attention


class CachedTokenAdapter(nn.Module):
    """Residual bottleneck applied in the frozen DINO token space.

    This is the cached-feature analogue of lightweight backbone adaptation. It
    shares one adapter across CLS, global-patch, and compact spatial tokens so
    it can be trained without repeatedly decoding hundreds of gigabytes of
    DICOM data.  A zero bottleneck is an exact identity control.
    """

    def __init__(self, token_dim: int, bottleneck: int) -> None:
        super().__init__()
        self.enabled = bottleneck > 0
        if self.enabled:
            self.norm = nn.LayerNorm(token_dim)
            self.down = nn.Linear(token_dim, bottleneck)
            self.up = nn.Linear(bottleneck, token_dim)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, token: Tensor) -> Tensor:
        if not self.enabled:
            return token
        return token + self.up(torch.nn.functional.gelu(self.down(self.norm(token))))


class PatchKneeAlibiModel(nn.Module):
    """Target-conditioned spatial evidence with controlled slice aggregation."""

    def __init__(self, config: PatchKneeModelConfig) -> None:
        super().__init__()
        if config.aggregator not in {"mean", "index_alibi", "physical_alibi"}:
            raise ValueError("unknown patch-hierarchy aggregator")
        self.config = config
        h, t = config.hidden_dim, config.num_targets
        if config.feature_dim != 2 * config.patch_dim:
            raise ValueError("patch model expects CLS || mean(patch) summary features")
        self.token_adapter = CachedTokenAdapter(
            config.patch_dim, config.token_adapter_bottleneck
        )
        self.summary_projection = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, h),
        )
        self.patch_pool = TargetPatchPool(config.patch_dim, h, t)
        self.target_embedding = nn.Parameter(torch.empty(t, h))
        self.series_cls = nn.Parameter(torch.empty(t, h))
        self.study_cls = nn.Parameter(torch.empty(t, h))
        for parameter in (self.target_embedding, self.series_cls, self.study_cls):
            nn.init.trunc_normal_(parameter, std=0.02)

        if config.aggregator == "mean":
            self.series_blocks = nn.ModuleList()
            self.series_mean_post = nn.Sequential(
                nn.LayerNorm(h),
                nn.Linear(h, h),
                nn.GELU(),
            )
        else:
            self.series_blocks = nn.ModuleList(
                [
                    AlibiBlock(
                        h,
                        config.n_heads,
                        mlp_ratio=4.0,
                        dropout=config.dropout,
                        distance_mode=config.aggregator,
                    )
                    for _ in range(config.series_depth)
                ]
            )
            self.series_mean_post = nn.Identity()
        self.series_norm = nn.LayerNorm(h)
        self.plane_embedding = nn.Embedding(4, h)
        self.fluid_embedding = nn.Embedding(3, h)
        self.fatsat_embedding = nn.Embedding(3, h)
        study_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.n_heads,
            dim_feedforward=4 * h,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.study_encoder = nn.TransformerEncoder(
            study_layer,
            num_layers=config.study_depth,
            norm=nn.LayerNorm(h),
        )
        self.target_weights = nn.Parameter(torch.empty(t, h))
        self.target_bias = nn.Parameter(torch.zeros(t))
        nn.init.trunc_normal_(self.target_weights, std=0.02)

    def _drop_series(self, mask: Tensor) -> Tensor:
        if not self.training or self.config.series_dropout <= 0:
            return mask.bool()
        keep = torch.rand(mask.shape, device=mask.device) >= self.config.series_dropout
        result = mask.bool() & keep
        lost = mask.any(dim=1) & ~result.any(dim=1)
        if lost.any():
            first_valid = mask.to(torch.int64).argmax(dim=1)
            result[lost, first_valid[lost]] = True
        return result

    def forward(
        self,
        features: Tensor,
        patch_features: Tensor,
        patch_mask: Tensor,
        positions_mm: Tensor,
        slice_mask: Tensor,
        series_mask: Tensor,
        plane: Tensor,
        fluid: Tensor,
        fatsat: Tensor,
        return_aux: bool = False,
    ) -> Tensor | Dict[str, Tensor]:
        batch, n_series, n_slices, _ = features.shape
        targets = self.config.num_targets
        effective_series = self._drop_series(series_mask)
        effective_slices = slice_mask.bool() & effective_series[..., None]

        patch_features = self.token_adapter(patch_features)
        summary_tokens = features.reshape(
            batch, n_series, n_slices, 2, self.config.patch_dim
        )
        features = self.token_adapter(summary_tokens).flatten(-2)
        spatial, patch_attention = self.patch_pool(patch_features, patch_mask)
        global_slice = self.summary_projection(features)[..., None, :]
        slices = spatial + global_slice + self.target_embedding[None, None, None, :, :]
        slices = slices * effective_slices[..., None, None].to(slices.dtype)

        # [B,R,S,T,H] -> [B*R*T,S,H]
        flat = slices.permute(0, 1, 3, 2, 4).reshape(
            batch * n_series * targets, n_slices, self.config.hidden_dim
        )
        positions = positions_mm[:, :, None, :].expand(-1, -1, targets, -1).reshape(
            batch * n_series * targets, n_slices
        )
        masks = effective_slices[:, :, None, :].expand(-1, -1, targets, -1).reshape(
            batch * n_series * targets, n_slices
        )
        if self.config.aggregator == "mean":
            denominator = masks.sum(dim=1, keepdim=True).clamp_min(1).to(flat.dtype)
            encoded_series = (flat * masks[..., None].to(flat.dtype)).sum(dim=1)
            encoded_series = self.series_mean_post(encoded_series / denominator)
        else:
            cls = self.series_cls[None, None, :, :].expand(
                batch, n_series, -1, -1
            ).reshape(batch * n_series * targets, 1, self.config.hidden_dim)
            encoded = torch.cat([cls, flat], dim=1)
            for block in self.series_blocks:
                encoded = block(encoded, positions, masks)
            encoded_series = encoded[:, 0]
        series_tokens = self.series_norm(encoded_series).reshape(
            batch, n_series, targets, self.config.hidden_dim
        )
        series_tokens = series_tokens * effective_series[..., None, None].to(series_tokens.dtype)

        metadata = (
            self.plane_embedding(plane.clamp(0, 3))
            + self.fluid_embedding(fluid.clamp(0, 2))
            + self.fatsat_embedding(fatsat.clamp(0, 2))
        )
        study_series = series_tokens.permute(0, 2, 1, 3)
        study_series = study_series + metadata[:, None, :, :]
        study_series = study_series + self.target_embedding[None, :, None, :]
        flat_study = study_series.reshape(
            batch * targets, n_series, self.config.hidden_dim
        )
        study_cls = self.study_cls[None, :, :].expand(batch, -1, -1).reshape(
            batch * targets, 1, self.config.hidden_dim
        )
        study_input = torch.cat([study_cls, flat_study], dim=1)
        study_mask = effective_series[:, None, :].expand(-1, targets, -1).reshape(
            batch * targets, n_series
        )
        padding = torch.cat(
            [
                torch.zeros(batch * targets, 1, dtype=torch.bool, device=features.device),
                ~study_mask,
            ],
            dim=1,
        )
        study_encoded = self.study_encoder(study_input, src_key_padding_mask=padding)
        target_study = study_encoded[:, 0].reshape(batch, targets, self.config.hidden_dim)
        logits = (target_study * self.target_weights[None, :, :]).sum(dim=-1)
        logits = logits + self.target_bias
        if not return_aux:
            return logits
        return {
            "logits": logits,
            "target_study_embedding": target_study,
            "patch_attention": patch_attention,
            "effective_series_mask": effective_series,
        }
