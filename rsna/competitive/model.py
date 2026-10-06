"""Masked slice/series pooling of global or spatial DINOv3 embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rsna.competitive.context import NeighborSliceContext
from rsna.competitive.feature_summary import PatchMeanProjection
from rsna.competitive.plane_memory import PlaneMemoryBranch
from rsna.competitive.relations import SliceDistanceHead, SliceRelations
from rsna.competitive.witness import PatchWitness
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
        slice_context: str = "none",
        relation_hidden_dim: int | None = None,
        relation_head_seed: int = 0,
        witness: dict | None = None,
        plane_memory: dict | None = None,
        input_channels: int = 1,
        patch_summary: dict | None = None,
    ):
        super().__init__()
        if (
            pooling not in {"attention", "max", "patch_attention"}
            or min(embedding_dim, hidden_dim, encode_chunk_size) < 1
        ):
            raise ValueError("Invalid pooling or model dimensions")
        if type(input_channels) is not int or input_channels not in {1, 3}:
            raise ValueError("Bound model input channel count must be one or three")
        self.input_channels = input_channels
        self.encoder = encoder
        if slice_context not in {"none", "neighbor3"} or (
            slice_context != "none" and pooling != "attention"
        ):
            raise ValueError("Neighbor context requires global-feature attention pooling")
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.encode_chunk_size = encode_chunk_size
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        if pooling == "patch_attention":
            if not callable(getattr(encoder, "forward_features", None)):
                raise ValueError("Spatial pooling needs an encoder with patch features")
            self.patch_projection: nn.Module | None = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
            self.patch_position: nn.Module | None = nn.Linear(2, hidden_dim, bias=False)
            self.patch_tanh: nn.Module | None = nn.Linear(hidden_dim, hidden_dim)
            self.patch_gate: nn.Module | None = nn.Linear(hidden_dim, hidden_dim)
            self.patch_score: nn.Module | None = nn.Linear(hidden_dim, len(TARGETS), bias=False)
        else:
            self.patch_projection = None
            self.patch_position = None
            self.patch_tanh = None
            self.patch_gate = None
            self.patch_score = None
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
        # Preserve shared initial weights and the caller's CPU RNG stream in a
        # matched context/no-context comparison. The new residual starts at zero.
        self.slice_context: NeighborSliceContext | None = None
        if slice_context == "neighbor3":
            with torch.random.fork_rng(devices=[]):
                self.slice_context = NeighborSliceContext(hidden_dim)
        self.relation_head = (
            SliceDistanceHead(embedding_dim, relation_hidden_dim, seed=relation_head_seed)
            if relation_hidden_dim is not None
            else None
        )
        self.witness = None
        if witness is not None:
            if (
                pooling != "attention"
                or slice_context != "none"
                or not callable(getattr(encoder, "forward_features", None))
            ):
                raise ValueError("Witness experiment requires spatial encoder and plain attention")
            self.witness = PatchWitness(embedding_dim, hidden_dim, witness)

        self.plane_memory: PlaneMemoryBranch | None = None
        if plane_memory is not None:
            if (
                pooling != "attention"
                or slice_context != "none"
                or witness is not None
                or relation_hidden_dim is not None
                or input_channels != 1
                or not callable(getattr(encoder, "forward_features", None))
            ):
                raise ValueError(
                    "Initial sequence-memory comparison requires one plain spatial encoder"
                )
            self.plane_memory = PlaneMemoryBranch(embedding_dim, plane_memory)

        self.patch_summary: PatchMeanProjection | None = None
        if patch_summary is not None:
            if (
                pooling != "attention"
                or slice_context != "none"
                or witness is not None
                or plane_memory is not None
                or relation_hidden_dim is not None
                or input_channels != 1
                or not callable(getattr(encoder, "forward_features", None))
            ):
                raise ValueError("First CLS/patch-mean comparison requires plain spatial DINO")
            self.patch_summary = PatchMeanProjection(embedding_dim, hidden_dim, patch_summary)

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
        *,
        relations: SliceRelations | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if (
            images.ndim != 6
            or images.shape[3] != self.input_channels
            or not images.is_floating_point()
        ):
            raise ValueError("images must match bound floating [B,S,K,C,H,W] input")
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
            raise ValueError("Valid images must be bounded raw channels in [0,1]")
        known = valid_positions[:, 1]
        if not ((known == 0) | (known == 1)).all() or (valid_positions[known == 0, 0] != 0).any():
            raise ValueError("Position known flags must be binary; unknown distances must be zero")
        # Invalid images (including NaN padding) never reach the encoder.
        if (
            self.pooling == "patch_attention"
            or self.witness is not None
            or self.plane_memory is not None
            or self.patch_summary is not None
        ):
            forward_features = getattr(self.encoder, "forward_features")
            spatial_chunks = [
                forward_features(part) for part in valid_images.split(self.encode_chunk_size)
            ]
            encoded = torch.cat([chunk[0] for chunk in spatial_chunks], dim=0)
            patches = torch.cat([chunk[1] for chunk in spatial_chunks], dim=0)
            if (
                patches.ndim != 4
                or patches.shape[0] != len(valid_images)
                or patches.shape[-1] != self.embedding_dim
                or not torch.isfinite(patches).all()
            ):
                raise ValueError("Encoder must return finite spatial patch embeddings")
        else:
            chunks = [self.encoder(part) for part in valid_images.split(self.encode_chunk_size)]
            encoded = torch.cat(chunks, dim=0)
        if (
            encoded.shape != (len(valid_images), self.embedding_dim)
            or not torch.isfinite(encoded).all()
        ):
            raise ValueError("Encoder must return finite [valid_slices,embedding_dim]")
        if self.patch_summary is None:
            projected = self.projection(encoded)
        else:
            combined = self.projection[0](encoded) + self.patch_summary(patches)
            projected = self.projection[2](self.projection[1](combined))
        slice_position = self.position_embedding(valid_positions.to(projected.dtype))
        if self.pooling == "patch_attention":
            assert (
                self.patch_projection is not None
                and self.patch_position is not None
                and self.patch_tanh is not None
                and self.patch_gate is not None
                and self.patch_score is not None
            )
            height, width = patches.shape[1:3]
            if min(height, width) < 1:
                raise ValueError("Encoder returned an empty patch grid")
            y = torch.linspace(-1, 1, height, device=patches.device, dtype=patches.dtype)
            x = torch.linspace(-1, 1, width, device=patches.device, dtype=patches.dtype)
            coordinates = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
            if height == 1:
                coordinates[..., 0] = 0
            if width == 1:
                coordinates[..., 1] = 0
            patch_features = self.patch_projection(patches) + self.patch_position(coordinates)
            patch_features = patch_features.flatten(1, 2)
            gated_patches = torch.tanh(self.patch_tanh(patch_features)) * torch.sigmoid(
                self.patch_gate(patch_features)
            )
            patch_weights = torch.softmax(self.patch_score(gated_patches), dim=1)
            patch_context = torch.einsum("npt,nph->nth", patch_weights, patch_features)
            target_slices = (projected[:, None, :] + patch_context) * 0.5
            full = projected.new_zeros((b, s, k, len(TARGETS), projected.shape[-1]))
            full[slice_mask] = target_slices + slice_position[:, None, :]
        else:
            full = projected.new_zeros((b, s, k, projected.shape[-1]))
            full[slice_mask] = projected + slice_position
        if self.slice_context is not None:
            full = self.slice_context(full, slice_mask, positions)
        series_features = full.new_zeros((b, s, full.shape[-1]))
        # Autocast linear output can be BF16 while LayerNorm/pooling buffers are
        # FP32. Indexed writes require exact dtype agreement on CUDA.
        series_features[series_mask] = self.series_embedding(valid_features.to(full.dtype)).to(
            series_features.dtype
        )
        if self.pooling == "max":
            pooled = full.masked_fill(~slice_mask[..., None], -torch.inf).amax(dim=2)
            pooled = torch.where(series_mask[..., None], pooled, torch.zeros_like(pooled))
            pooled = pooled + series_features
            study = pooled.masked_fill(~series_mask[..., None], -torch.inf).amax(dim=1)
            logits = torch.einsum("bh,th->bt", study, self.output_weight) + self.output_bias
            return self._relation_output(logits, encoded, slice_mask, relations)
        gated_slices = torch.tanh(self.slice_tanh(full)) * torch.sigmoid(self.slice_gate(full))
        scores = (
            torch.einsum("bskth,th->bskt", gated_slices, self.slice_score.weight)
            if self.pooling == "patch_attention"
            else self.slice_score(gated_slices)
        )
        weights = _masked_softmax(scores, slice_mask[..., None], dim=2)
        pooled = (
            torch.einsum("bskt,bskth->bsth", weights, full)
            if self.pooling == "patch_attention"
            else torch.einsum("bskt,bskh->bsth", weights, full)
        ) + series_features[:, :, None, :]
        gated = torch.tanh(self.series_tanh(pooled)) * torch.sigmoid(self.series_gate(pooled))
        series_scores = torch.einsum("bsth,th->bst", gated, self.series_score)
        series_weights = _masked_softmax(series_scores, series_mask[..., None], dim=1)
        study = torch.einsum("bst,bsth->bth", series_weights, pooled)
        logits = torch.einsum("bth,th->bt", study, self.output_weight) + self.output_bias
        if self.witness is not None:
            logits = logits + self.witness(patches, slice_mask)
        if self.plane_memory is not None:
            logits = logits + self.plane_memory(
                patches, features, series_mask, slice_mask, positions
            )
        return self._relation_output(logits, encoded, slice_mask, relations)

    def _relation_output(self, logits, encoded, slice_mask, relations):
        if relations is None:
            return logits
        if self.relation_head is None:
            raise ValueError("Relation outputs require a configured auxiliary head")
        raw = encoded.new_zeros((*slice_mask.shape, self.embedding_dim))
        raw[slice_mask] = encoded
        if len(relations.indices):
            study, series, first, second = relations.indices.to(slice_mask.device).unbind(1)
            if not (slice_mask[study, series, first] & slice_mask[study, series, second]).all():
                raise ValueError("Relation references a padded image")
        return logits, self.relation_head.study_sum(raw, relations)
