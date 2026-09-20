"""Hierarchical DINO slice aggregation with physical-distance ALiBi.

The model consumes cached per-slice image features.  Within each MRI series,
attention is biased by the physical DICOM slice separation.  Series summaries
are then fused at study level with acquisition metadata and target-specific
queries.

Three controlled aggregation modes are supported:

* ``mean``: masked mean pooling (the null model);
* ``index_alibi``: relative bias from ordinal slice distance;
* ``physical_alibi``: relative bias from DICOM position in millimetres.

The implementation deliberately has no dependency on transformers or pydicom,
so cached-feature experiments and smoke tests remain lightweight.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _power_of_two_slopes(n_heads: int) -> Tensor:
    """Return deterministic ALiBi slopes, including non-power-of-two counts."""

    if n_heads < 1:
        raise ValueError("n_heads must be positive")

    def slopes_power_of_two(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        values = slopes_power_of_two(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        values = slopes_power_of_two(closest)
        extra = slopes_power_of_two(2 * closest)[0::2]
        values.extend(extra[: n_heads - closest])
    return torch.tensor(values, dtype=torch.float32)


def _normalise_physical_positions(positions: Tensor, mask: Tensor) -> Tensor:
    """Express millimetre positions in units of the median valid slice gap.

    ``positions`` and ``mask`` have shape ``[batch, slices]``.  The result is
    invariant to translations and robust to a single missing/irregular slice.
    Degenerate single-slice series use unit spacing.
    """

    if positions.ndim != 2 or mask.shape != positions.shape:
        raise ValueError("positions and mask must both have shape [B, S]")
    batch, length = positions.shape
    if length <= 1:
        scale = torch.ones(batch, device=positions.device, dtype=positions.dtype)
    else:
        # Sort only for estimating the physical quantum.  Attention itself
        # remains permutation equivariant when slices and positions are
        # permuted together.
        ordered = torch.where(mask, positions, torch.inf).sort(dim=-1).values
        gaps = (ordered[:, 1:] - ordered[:, :-1]).abs()
        pair_mask = torch.isfinite(ordered[:, 1:]) & (gaps > 1e-7)
        inf = torch.full_like(gaps, torch.inf)
        sorted_gaps = torch.where(pair_mask, gaps, inf).sort(dim=-1).values
        counts = pair_mask.sum(dim=-1)
        middle = ((counts.clamp_min(1) - 1) // 2).unsqueeze(-1)
        scale = sorted_gaps.gather(1, middle).squeeze(1)
        scale = torch.where(counts > 0, scale, torch.ones_like(scale))
        scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
    origin = torch.where(mask, positions, torch.inf).amin(dim=-1)
    origin = torch.where(torch.isfinite(origin), origin, torch.zeros_like(origin))
    return (positions - origin[:, None]) / scale.clamp_min(1e-6)[:, None]


class PhysicalAlibiAttention(nn.Module):
    """Multi-head self-attention with symmetric physical-distance bias."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        distance_mode: str = "physical_alibi",
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if distance_mode not in {"physical_alibi", "index_alibi", "none"}:
            raise ValueError(f"unknown distance mode: {distance_mode}")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dropout = float(dropout)
        self.distance_mode = distance_mode
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer("slopes", _power_of_two_slopes(n_heads), persistent=True)

    def attention_bias(self, positions: Tensor, mask: Tensor, has_cls: bool = True) -> Tensor:
        """Return additive attention bias with shape ``[B, H, L, L]``."""

        if self.distance_mode == "none":
            coordinate = torch.zeros_like(positions)
        elif self.distance_mode == "index_alibi":
            coordinate = torch.arange(
                positions.shape[-1], device=positions.device, dtype=positions.dtype
            )[None, :].expand_as(positions)
        else:
            coordinate = _normalise_physical_positions(positions, mask)

        if has_cls:
            coordinate = torch.cat([torch.zeros_like(coordinate[:, :1]), coordinate], dim=1)
            token_mask = torch.cat(
                [torch.ones_like(mask[:, :1], dtype=torch.bool), mask.bool()], dim=1
            )
        else:
            token_mask = mask.bool()

        distance = (coordinate[:, :, None] - coordinate[:, None, :]).abs()
        if has_cls:
            # The summary token is a global readout, not a physical slice.
            distance[:, 0, :] = 0
            distance[:, :, 0] = 0
        bias = -distance[:, None, :, :] * self.slopes[None, :, None, None].to(distance)
        key_invalid = ~token_mask[:, None, None, :]
        bias = bias.masked_fill(key_invalid, torch.finfo(bias.dtype).min)
        return bias

    def forward(self, x: Tensor, positions: Tensor, mask: Tensor, has_cls: bool = True) -> Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        bias = self.attention_bias(positions, mask, has_cls=has_cls)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.hidden_dim)
        return self.out(attended)


class AlibiBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        distance_mode: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicalAlibiAttention(hidden_dim, n_heads, dropout, distance_mode)
        self.norm2 = nn.LayerNorm(hidden_dim)
        inner = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, positions: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), positions, mask, has_cls=True)
        x = x + self.mlp(self.norm2(x))
        return x


class SeriesEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        n_heads: int,
        depth: int,
        dropout: float,
        aggregator: str,
    ) -> None:
        super().__init__()
        if aggregator not in {"mean", "index_alibi", "physical_alibi"}:
            raise ValueError(f"unknown aggregator: {aggregator}")
        self.aggregator = aggregator
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        if aggregator == "mean":
            self.blocks = nn.ModuleList()
            self.mean_post = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    AlibiBlock(
                        hidden_dim,
                        n_heads,
                        mlp_ratio=4.0,
                        dropout=dropout,
                        distance_mode=aggregator,
                    )
                    for _ in range(depth)
                ]
            )
            self.mean_post = nn.Identity()
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: Tensor, positions: Tensor, slice_mask: Tensor) -> Tensor:
        """Encode ``[B, R, S, F]`` slice features into ``[B, R, H]``."""

        batch, n_series, n_slices, feature_dim = features.shape
        flat_features = features.reshape(batch * n_series, n_slices, feature_dim)
        flat_positions = positions.reshape(batch * n_series, n_slices)
        flat_mask = slice_mask.reshape(batch * n_series, n_slices).bool()
        projected = self.projection(flat_features)

        if self.aggregator == "mean":
            denom = flat_mask.sum(dim=1, keepdim=True).clamp_min(1).to(projected.dtype)
            summary = (projected * flat_mask[..., None]).sum(dim=1) / denom
            summary = self.mean_post(summary)
        else:
            cls = self.cls.expand(batch * n_series, -1, -1)
            x = torch.cat([cls, projected], dim=1)
            for block in self.blocks:
                x = block(x, flat_positions, flat_mask)
            summary = x[:, 0]

        summary = self.final_norm(summary)
        valid_series = flat_mask.any(dim=1)
        summary = summary * valid_series[:, None].to(summary.dtype)
        return summary.reshape(batch, n_series, -1)


@dataclass
class KneeModelConfig:
    feature_dim: int
    hidden_dim: int = 256
    n_heads: int = 8
    series_depth: int = 2
    study_depth: int = 2
    dropout: float = 0.1
    aggregator: str = "physical_alibi"
    num_targets: int = 12
    report_dim: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KneeAlibiModel(nn.Module):
    """DINO slice encoder hierarchy for multilabel knee classification."""

    def __init__(self, config: KneeModelConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.series_encoder = SeriesEncoder(
            feature_dim=config.feature_dim,
            hidden_dim=h,
            n_heads=config.n_heads,
            depth=config.series_depth,
            dropout=config.dropout,
            aggregator=config.aggregator,
        )
        self.plane_embedding = nn.Embedding(4, h)
        self.fluid_embedding = nn.Embedding(3, h)
        self.fatsat_embedding = nn.Embedding(3, h)
        self.study_cls = nn.Parameter(torch.zeros(1, 1, h))
        nn.init.trunc_normal_(self.study_cls, std=0.02)
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
        self.target_queries = nn.Parameter(torch.empty(config.num_targets, h))
        self.target_weights = nn.Parameter(torch.empty(config.num_targets, h))
        self.target_bias = nn.Parameter(torch.zeros(config.num_targets))
        nn.init.trunc_normal_(self.target_queries, std=0.02)
        nn.init.trunc_normal_(self.target_weights, std=0.02)
        self.report_projection: Optional[nn.Linear]
        if config.report_dim > 0:
            self.report_projection = nn.Linear(h, config.report_dim)
        else:
            self.report_projection = None

    def forward(
        self,
        features: Tensor,
        positions_mm: Tensor,
        slice_mask: Tensor,
        series_mask: Tensor,
        plane: Tensor,
        fluid: Tensor,
        fatsat: Tensor,
        return_aux: bool = False,
    ) -> Tensor | Dict[str, Tensor]:
        series = self.series_encoder(features, positions_mm, slice_mask)
        series = (
            series
            + self.plane_embedding(plane.clamp(0, 3))
            + self.fluid_embedding(fluid.clamp(0, 2))
            + self.fatsat_embedding(fatsat.clamp(0, 2))
        )
        series = series * series_mask[..., None].to(series.dtype)
        cls = self.study_cls.expand(features.shape[0], -1, -1)
        tokens = torch.cat([cls, series], dim=1)
        padding = torch.cat(
            [
                torch.zeros(series_mask.shape[0], 1, dtype=torch.bool, device=series_mask.device),
                ~series_mask.bool(),
            ],
            dim=1,
        )
        encoded = self.study_encoder(tokens, src_key_padding_mask=padding)
        study = encoded[:, 0]
        series_out = encoded[:, 1:]

        scores = torch.einsum("bsh,th->bts", series_out, self.target_queries)
        scores = scores / math.sqrt(self.config.hidden_dim)
        scores = scores.masked_fill(~series_mask[:, None, :].bool(), -torch.inf)
        # A study cache always has at least one series.  The fallback keeps an
        # accidentally empty synthetic/cache record finite and auditable.
        empty = ~series_mask.any(dim=1)
        if empty.any():
            scores[empty] = 0
        attention = torch.softmax(scores, dim=-1)
        attention = attention * series_mask[:, None, :].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target_repr = torch.einsum("bts,bsh->bth", attention, series_out)
        target_repr = target_repr + study[:, None, :]
        logits = (target_repr * self.target_weights[None, :, :]).sum(dim=-1)
        logits = logits + self.target_bias

        if not return_aux:
            return logits
        out: Dict[str, Tensor] = {
            "logits": logits,
            "study_embedding": study,
            "target_attention": attention,
        }
        if self.report_projection is not None:
            out["report_embedding"] = self.report_projection(study)
        return out
