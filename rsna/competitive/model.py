"""Masked slice/series pooling of global image embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rsna.schema import TARGETS


def _masked_softmax(scores: Tensor, mask: Tensor, dim: int) -> Tensor:
    # Empty padded series have a zero distribution rather than NaN softmax.
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=dim) * mask.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)


class CompetitiveStudyNet(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        pooling: str = "attention",
        hidden_dim: int = 128,
        encode_chunk_size: int = 64,
    ):
        super().__init__()
        if (
            pooling not in {"attention", "max"}
            or min(embedding_dim, hidden_dim, encode_chunk_size) < 1
        ):
            raise ValueError("Invalid pooling or model dimensions")
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.encode_chunk_size = encode_chunk_size
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.position_embedding = nn.Linear(2, hidden_dim, bias=False)
        self.series_embedding = nn.Linear(5, hidden_dim, bias=False)
        self.slice_tanh = nn.Linear(hidden_dim, hidden_dim)
        self.slice_gate = nn.Linear(hidden_dim, hidden_dim)
        self.slice_score = nn.Linear(hidden_dim, len(TARGETS), bias=False)
        self.series_tanh = nn.Linear(hidden_dim, hidden_dim)
        self.series_gate = nn.Linear(hidden_dim, hidden_dim)
        self.series_score = nn.Parameter(torch.empty(len(TARGETS), hidden_dim))
        self.output_weight = nn.Parameter(torch.empty(len(TARGETS), hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(len(TARGETS)))
        nn.init.xavier_uniform_(self.series_score)
        nn.init.xavier_uniform_(self.output_weight)
        if pooling == "max":
            # Keep comparator state explicit but exclude unused attention parameters
            # from the optimizer/DDP parameter set.
            for module in [
                self.slice_tanh,
                self.slice_gate,
                self.slice_score,
                self.series_tanh,
                self.series_gate,
            ]:
                module.requires_grad_(False)
            self.series_score.requires_grad_(False)

    def train(self, mode: bool = True) -> CompetitiveStudyNet:
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self

    def forward(
        self,
        images: Tensor,
        features: Tensor,
        series_mask: Tensor,
        slice_mask: Tensor,
        positions: Tensor,
    ) -> Tensor:
        if images.ndim != 6 or images.shape[3] != 1 or not images.is_floating_point():
            raise ValueError("images must be floating [B,S,K,1,H,W]")
        b, s, k = images.shape[:3]
        if min(images.shape) < 1 or features.shape != (b, s, 5) or positions.shape != (b, s, k, 2):
            raise ValueError("Invalid feature or position shape")
        if series_mask.shape != (b, s) or slice_mask.shape != (b, s, k):
            raise ValueError("Invalid mask shape")
        if series_mask.dtype != torch.bool or slice_mask.dtype != torch.bool:
            raise ValueError("Masks must be boolean")
        if not torch.equal(slice_mask.any(dim=2), series_mask) or not series_mask.any(dim=1).all():
            raise ValueError(
                "Each study/valid series needs genuine valid slices; padding must be masked"
            )
        if any(t.device != images.device for t in [features, series_mask, slice_mask, positions]):
            raise ValueError("Model inputs must share a device")
        valid_images = images[slice_mask]
        valid_features, valid_positions = features[series_mask], positions[slice_mask]
        if not all(
            torch.isfinite(t).all() for t in [valid_images, valid_features, valid_positions]
        ):
            raise ValueError("Valid input entries must be finite")
        if (valid_images < 0).any() or (valid_images > 1).any():
            raise ValueError("Valid images must be grayscale values in [0,1]")
        known = valid_positions[:, 1]
        if not ((known == 0) | (known == 1)).all() or (valid_positions[known == 0, 0] != 0).any():
            raise ValueError("Position known flags must be binary; unknown distances must be zero")
        # Invalid images (including NaN padding) never reach the encoder.
        chunks = [self.encoder(part) for part in valid_images.split(self.encode_chunk_size)]
        encoded = torch.cat(chunks, dim=0)
        if (
            encoded.shape != (len(valid_images), self.embedding_dim)
            or not torch.isfinite(encoded).all()
        ):
            raise ValueError("Encoder must return finite [valid_slices,embedding_dim]")
        projected = self.projection(encoded)
        full = projected.new_zeros((b, s, k, projected.shape[-1]))
        full[slice_mask] = projected + self.position_embedding(valid_positions.to(projected.dtype))
        series_features = full.new_zeros((b, s, full.shape[-1]))
        series_features[series_mask] = self.series_embedding(valid_features.to(full.dtype))
        if self.pooling == "max":
            pooled = full.masked_fill(~slice_mask[..., None], -torch.inf).amax(dim=2)
            pooled = torch.where(series_mask[..., None], pooled, torch.zeros_like(pooled))
            pooled = pooled + series_features
            study = pooled.masked_fill(~series_mask[..., None], -torch.inf).amax(dim=1)
            return torch.einsum("bh,th->bt", study, self.output_weight) + self.output_bias
        scores = self.slice_score(
            torch.tanh(self.slice_tanh(full)) * torch.sigmoid(self.slice_gate(full))
        )
        weights = _masked_softmax(scores, slice_mask[..., None], dim=2)
        pooled = torch.einsum("bskt,bskh->bsth", weights, full) + series_features[:, :, None, :]
        gated = torch.tanh(self.series_tanh(pooled)) * torch.sigmoid(self.series_gate(pooled))
        series_scores = torch.einsum("bsth,th->bst", gated, self.series_score)
        series_weights = _masked_softmax(series_scores, series_mask[..., None], dim=1)
        study = torch.einsum("bst,bsth->bth", series_weights, pooled)
        return torch.einsum("bth,th->bt", study, self.output_weight) + self.output_bias
