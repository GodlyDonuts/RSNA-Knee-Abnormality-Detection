"""Opt-in geometry supervision; existing production experiments stay unchanged.

Predict unsigned physical slice separation from raw encoder features, before
position/protocol embeddings. Geometry supplies targets only. No disease label
is assigned to an individual slice, and no feature-equality loss is imposed.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SliceRelations:
    indices: Tensor  # N x 4: study, series, first slice, second slice
    targets: Tensor  # log1p(unsigned millimeters) / log1p(100)
    weights: Tensor  # equal contribution per represented study
    shape: tuple[int, int, int]

    @property
    def study_count(self) -> int:
        return len(self.indices[:, 0].unique())


def validate_relation(policy: Mapping) -> dict:
    if not isinstance(policy, Mapping) or set(policy) != {
        "weight",
        "max_pairs",
        "target_mode",
        "hidden_dim",
        "head_seed",
    }:
        raise ValueError("Exact relation weight/pair/target/head policy required")
    if (
        isinstance(policy["weight"], bool)
        or not isinstance(policy["weight"], (int, float))
        or not math.isfinite(policy["weight"])
        or policy["weight"] < 0
    ):
        raise ValueError("Finite nonnegative relation weight required")
    if (
        type(policy["max_pairs"]) is not int
        or not 1 <= policy["max_pairs"] <= 512
        or type(policy["hidden_dim"]) is not int
        or not 1 <= policy["hidden_dim"] <= 1024
        or type(policy["head_seed"]) is not int
        or not 0 <= policy["head_seed"] < 2**63
        or not isinstance(policy["target_mode"], str)
        or policy["target_mode"] not in {"physical", "shuffled"}
    ):
        raise ValueError("Invalid bounded relation policy")
    return dict(policy)


def relation_model_kwargs(policy: Mapping | None) -> dict:
    if policy is None:
        return {}
    validated = validate_relation(policy)
    return {
        "relation_hidden_dim": validated["hidden_dim"],
        "relation_head_seed": validated["head_seed"],
    }


def relations_for_batch(batch: dict, policy: Mapping, *, seed: int, epoch: int) -> SliceRelations:
    policy = validate_relation(policy)
    uids = batch["study_uid"]
    if (
        not isinstance(uids, (list, tuple))
        or len(uids) != len(batch["positions"])
        or any(not isinstance(uid, str) or not uid for uid in uids)
        or type(epoch) is not int
        or epoch < 0
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError("Persisted study identities and epoch/seed required")
    seeds = [
        int.from_bytes(
            hashlib.sha256(f"{seed}:{epoch}:{uid}:relation-v1".encode()).digest()[:8], "big"
        )
        % 2**63
        for uid in uids
    ]
    return slice_relations(
        batch["positions"],
        batch["slice_mask"],
        batch["series_mask"],
        batch["sample_valid"],
        seed=seed,
        max_pairs=policy["max_pairs"],
        shuffle_targets=policy["target_mode"] == "shuffled",
        study_seeds=seeds,
    )


def relation_study_count(batches: Sequence[SliceRelations], device: torch.device) -> Tensor:
    count = torch.tensor(
        sum(batch.study_count for batch in batches), device=device, dtype=torch.float64
    )
    if dist.is_initialized():
        dist.all_reduce(count)
    return count


def effective_relation_loss(study_sum: Tensor, count: Tensor) -> Tensor:
    if count.ndim != 0 or not torch.isfinite(count) or count < 0:
        raise ValueError("Finite scalar effective-batch relation count required")
    if not count:
        return study_sum * 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return study_sum * world / count.to(study_sum.dtype)


def slice_relations(
    positions: Tensor,
    slice_mask: Tensor,
    series_mask: Tensor,
    sample_mask: Tensor,
    *,
    seed: int,
    max_pairs: int = 32,
    shuffle_targets: bool = False,
    study_seeds: Sequence[int] | None = None,
) -> SliceRelations:
    """Sample within-series pairs without consuming the trainer's random stream.

    Positions follow data._positions: centered millimeters / 100 and a binary
    physical-geometry flag. Ordinal fallbacks, padding and distributed tail
    duplicates cannot generate supervision. The caller must admit training
    membership and derive seeds from persisted epoch/study identities.
    """
    if positions.ndim != 4 or positions.shape[-1] != 2 or min(positions.shape) < 1:
        raise ValueError("Expected nonempty B,S,K,2 positions")
    shape = (positions.shape[0], positions.shape[1], positions.shape[2])
    if (
        slice_mask.shape != shape
        or series_mask.shape != shape[:2]
        or sample_mask.shape != shape[:1]
        or any(mask.dtype != torch.bool for mask in (slice_mask, series_mask, sample_mask))
    ):
        raise ValueError("Relation masks must be boolean with matching dimensions")
    if (
        type(seed) is not int
        or not 0 <= seed < 2**63
        or type(max_pairs) is not int
        or max_pairs < 1
    ):
        raise ValueError("Invalid relation sampling budget or seed")
    if not positions.is_floating_point() or type(shuffle_targets) is not bool:
        raise ValueError("Floating geometry and boolean shuffle policy required")
    if study_seeds is not None and (
        len(study_seeds) != shape[0]
        or any(type(value) is not int or not 0 <= value < 2**63 for value in study_seeds)
    ):
        raise ValueError("One valid persisted seed per study required")
    valid = (
        slice_mask.detach().cpu()
        & series_mask.detach().cpu()[..., None]
        & sample_mask.detach().cpu()[:, None, None]
    )
    physical = positions.detach().to(device="cpu", dtype=torch.float64)
    values = physical[valid]
    if not torch.isfinite(values).all() or not ((values[:, 1] == 0) | (values[:, 1] == 1)).all():
        raise ValueError("Valid geometry must be finite with binary flags")
    if (values[values[:, 1] == 0, 0] != 0).any():
        raise ValueError("Unknown physical geometry must have zero coordinates")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    target_generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x52454C4154494F4E)
    rows: list[list[int]] = []
    targets: list[float] = []
    for study in range(shape[0]):
        if study_seeds is not None:
            generator.manual_seed(study_seeds[study])
            target_generator.manual_seed(study_seeds[study] ^ 0x52454C4154494F4E)
        for series in range(shape[1]):
            known = torch.where(valid[study, series] & (physical[study, series, :, 1] == 1))[0]
            if len(known) < 2:
                continue
            coords = physical[study, series, known, 0]
            gaps = coords.diff()
            if not ((gaps > 0).all() or (gaps < 0).all()):
                raise ValueError("Known slice positions must be strictly monotone")
            pairs = torch.combinations(known, r=2)
            if len(pairs) > max_pairs:
                pairs = pairs[torch.randperm(len(pairs), generator=generator)[:max_pairs]]
            separation = (
                physical[study, series, pairs[:, 0], 0] - physical[study, series, pairs[:, 1], 0]
            ).abs() * 100
            labels = torch.log1p(separation) / math.log1p(100)
            if shuffle_targets and len(labels) > 1:
                labels = labels[torch.randperm(len(labels), generator=target_generator)]
            rows.extend([study, series, int(i), int(j)] for i, j in pairs)
            targets.extend(labels.tolist())
    indices = torch.tensor(rows, dtype=torch.long).reshape(-1, 4)
    weights = torch.empty(len(rows), dtype=torch.float32)
    if rows:
        counts = torch.bincount(indices[:, 0], minlength=shape[0])
        studies = (counts > 0).sum()
        weights = 1 / (counts[indices[:, 0]].float() * studies)
    return SliceRelations(indices, torch.tensor(targets, dtype=torch.float32), weights, shape)


class SliceDistanceHead(nn.Module):
    """Symmetric image-feature-only head, dormant during classification inference."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 128, *, seed: int = 0):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (embedding_dim, hidden_dim)):
            raise ValueError("Positive feature/head dimensions required")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Invalid auxiliary initialization seed")
        # Auxiliary initialization must not perturb matched classification heads.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(2 * embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
            )
        self.embedding_dim = embedding_dim

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        if first.shape != second.shape or first.ndim != 2 or first.shape[-1] != self.embedding_dim:
            raise ValueError("Expected matching N,D raw encoder features")
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError("Relation features must be finite")
        return self.network(torch.cat(((first - second).abs(), (first + second) * 0.5), -1))[:, 0]

    def loss(self, features: Tensor, relations: SliceRelations) -> Tensor:
        if features.shape != (*relations.shape, self.embedding_dim):
            raise ValueError("Raw encoder feature layout differs from relation layout")
        if not len(relations.indices):
            # Padding may be NaN; an empty selection remains differentiable zero.
            return sum((p.sum() * 0 for p in self.parameters()), features.reshape(-1)[:0].sum())
        study, series, first, second = relations.indices.to(features.device).unbind(1)
        prediction = self(features[study, series, first], features[study, series, second])
        loss = F.smooth_l1_loss(
            prediction.float(), relations.targets.to(features.device), reduction="none"
        )
        return (loss * relations.weights.to(features.device)).sum()

    def study_sum(self, features: Tensor, relations: SliceRelations) -> Tensor:
        return self.loss(features, relations) * relations.study_count
