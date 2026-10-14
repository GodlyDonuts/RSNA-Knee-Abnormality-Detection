"""Optional whole-window-bag variant of the source-matched slot classifier (issue #49).

The question is whether letting each disease choose evidence across a study's windows
beats fixed probability averaging. Both arms train on the same bag, the four
non-overlapping source groups (window starts 0/3/6/9). Both evaluate on the same ten
overlapping windows. Neither arm assigns the study label to one arbitrary window.

Per window ``w`` the unchanged reviewed ``SourceSlotHead`` gives a target-conditioned
window feature ``c[w,t]`` (its per-target slot-attention context) and window logit
``u[w,t]``. Window weights are ``beta = (1-rho) softmax_w(s) + rho/W``, with disease-query
score ``s = selection * <c[w,t], q[t]> / sqrt(H)`` and ``q`` initialized at zero. The study
probability is the mixture ``p = sum_w beta[w,t] sigmoid(u[w,t])``. It is returned as its
exact logit, computed stably in log space, so the trainer applies one study-level BCE to the
whole pooled bag and prediction applies one sigmoid.

``window_weighting`` is ``plain_mean`` (``selection = 0``: ``beta = 1/W``, which is fixed
probability averaging) or ``disease_query`` (``selection = 1``). The two arms have the same
parameters, initial values, RNG draws, inputs, floor and arithmetic. Only the score's
effect differs. Windows show different anatomy. Their disagreement is never used as
uncertainty or as a weight.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rsna.competitive import source_slot as ss
from rsna.competitive.model import _masked_softmax
from rsna.schema import TARGETS

KIND = "source_window_bag_v1"
TRAINING_WINDOWS = "source_four_group_bag_every_update_v1"
EVALUATION_WINDOWS = "source_overlap10_bag_v1"
BAG_POOL = "probability_mixture_exact_logit_v1"
WEIGHTINGS = {"plain_mean": 0.0, "disease_query": 1.0}
UNIFORM_FLOOR = 0.25
TRAINING_VIEWS, EVALUATION_VIEWS = ss.TRAINING_VIEWS, ss.EVALUATION_VIEWS
STRATEGY_FIXED = {
    **{k: v for k, v in ss.STRATEGY_FIXED.items() if k not in {"training_windows"}},
    "kind": KIND,
    "training_windows": TRAINING_WINDOWS,
    "evaluation_windows": EVALUATION_WINDOWS,
    "bag_pool": BAG_POOL,
    "uniform_floor": UNIFORM_FLOOR,
}
STRATEGY_KEYS = set(STRATEGY_FIXED) | {"augmentation", "window_weighting"}


def validate_strategy(strategy: object) -> dict:
    """Exact versioned policy; only ``augmentation`` and ``window_weighting`` vary."""
    if (
        not isinstance(strategy, Mapping)
        or set(strategy) != STRATEGY_KEYS
        or any(
            type(strategy[key]) is not type(value) or strategy[key] != value
            for key, value in STRATEGY_FIXED.items()
        )
        or not isinstance(strategy["augmentation"], str)
        or strategy["augmentation"] not in ss.AUGMENTATIONS
        or not isinstance(strategy["window_weighting"], str)
        or strategy["window_weighting"] not in WEIGHTINGS
    ):
        raise ValueError("Exact source window-bag classifier strategy required")
    return dict(strategy)


def slot_strategy(strategy: Mapping) -> dict:
    """The unchanged ``source_matched_slot_v1`` policy whose encoder/pool/head are reused."""
    return {**ss.STRATEGY_FIXED, "augmentation": validate_strategy(strategy)["augmentation"]}


def target_window_features(head: ss.SourceSlotHead, x: Tensor, mask: Tensor) -> tuple:
    """``SourceSlotHead.forward`` with its per-target context exposed: (``[N,T,H]``, ``[N,T]``).

    Same operations in the same order, so the logits are bitwise ``head(x, mask)`` and the
    head's forward contract is untouched.
    """
    h = head.proj(x) + head.slot_emb
    att = torch.einsum("bsh,oh->bos", h, head.query) / head.hidden**0.5
    att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
    ctx = head.drop(torch.einsum("bos,bsh->boh", att, h))
    return ctx, (ctx * head.out.weight.unsqueeze(0)).sum(-1) + head.out.bias


def bag_pool(
    window_logits: Tensor,
    context: Tensor,
    query: Tensor,
    window_valid: Tensor,
    *,
    selection: float,
    uniform_floor: float = UNIFORM_FLOOR,
) -> tuple[Tensor, Tensor]:
    """``[B,W,T]`` logits, ``[B,W,T,H]`` features, ``[T,H]`` query, ``[B,W]`` mask -> study
    logit ``[B,T]`` and weights ``[B,W,T]``.

    FP32 with autocast disabled. Invalid windows never contribute (their values are
    ignored). Each valid window weighs between ``rho/W`` and ``1-rho+rho/W``. The mixture
    logit ``log p - log(1-p)`` comes from log-sigmoids, so extreme window logits keep finite
    values and gradients.
    """
    if (
        window_logits.ndim != 3
        or context.ndim != 4
        or context.shape[:3] != window_logits.shape
        or query.shape != (window_logits.shape[2], context.shape[3])
        or window_valid.shape != window_logits.shape[:2]
        or window_valid.dtype != torch.bool
    ):
        raise ValueError("Aligned [B,W,T] window logits, [B,W,T,H] features and [B,W] mask")
    if selection not in WEIGHTINGS.values() or not 0 < uniform_floor <= 1:
        raise ValueError("Window selection must be 0 or 1 and the floor in (0, 1]")
    if not window_valid.any(dim=1).all():
        raise ValueError("Every study needs a genuine window; empty bags are refused")
    with torch.autocast(window_logits.device.type, enabled=False):
        valid = window_valid[..., None].expand_as(window_logits)
        u = torch.where(valid, window_logits.float(), 0.0)
        c = torch.where(valid[..., None], context.float(), 0.0)
        if not (
            torch.isfinite(u).all() and torch.isfinite(c).all() and torch.isfinite(query).all()
        ):
            raise ValueError("Valid window logits and features must be finite")
        scores = selection * torch.einsum("bwth,th->bwt", c, query.float()) / c.shape[-1] ** 0.5
        attention = _masked_softmax(scores, valid, dim=1)
        uniform = valid.float() / valid.sum(dim=1, keepdim=True).float()
        beta = (1 - uniform_floor) * attention + uniform_floor * uniform
        log_beta = torch.where(valid, torch.where(valid, beta, 1.0).log(), -torch.inf)
        positive = torch.logsumexp(log_beta + F.logsigmoid(u), dim=1)
        negative = torch.logsumexp(log_beta + F.logsigmoid(-u), dim=1)
        return positive - negative, beta


class WindowBagSlotStudyNet(ss.SourceSlotStudyNet):
    """Reviewed source-slot encoder/pool/head plus a zero-initialized window query.

    ``forward`` takes a ``[B,6,W,3,H,W]`` bag with ``W`` four (training groups) or ten
    (evaluation windows) and returns study logits. ``window_probabilities`` applies one
    sigmoid to the ten-window bag.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        strategy: Mapping,
        encode_chunk_size: int = 64,
        input_channels: int = 3,
    ):
        bag = validate_strategy(strategy)
        super().__init__(
            encoder, embedding_dim, slot_strategy(bag), encode_chunk_size, input_channels
        )
        self.strategy = bag
        self.selection = WEIGHTINGS[bag["window_weighting"]]
        self.uniform_floor = bag["uniform_floor"]
        self.window_query = nn.Parameter(torch.zeros(len(TARGETS), bag["hidden"]))

    def bag_outputs(self, images, features, series_mask, slice_mask, positions):
        """Study logits ``[B,T]``, window logits ``[B,W,T]`` and weights ``[B,W,T]``."""
        views = images.shape[2] if images.ndim == 6 else -1
        if views not in {len(TRAINING_VIEWS), len(EVALUATION_VIEWS)}:
            raise ValueError("Window bags hold the four source groups or the ten windows")
        self._check(images, features, series_mask, slice_mask, positions, views)
        b, s = series_mask.shape
        windows = images.transpose(1, 2).flatten(0, 1)
        masks = series_mask[:, None].expand(b, views, s).flatten(0, 1)
        context, logits = target_window_features(
            self.head, self.slot_features(windows, masks), masks.float()
        )
        window_logits = logits.unflatten(0, (b, views))
        study, beta = bag_pool(
            window_logits,
            context.unflatten(0, (b, views)),
            self.window_query,
            slice_mask.any(dim=1),
            selection=self.selection,
            uniform_floor=self.uniform_floor,
        )
        return study, window_logits, beta

    def forward(self, images, features, series_mask, slice_mask, positions) -> Tensor:
        return self.bag_outputs(images, features, series_mask, slice_mask, positions)[0]

    def window_probabilities(self, images, features, series_mask, slice_mask, positions):
        if images.ndim != 6 or images.shape[2] != len(EVALUATION_VIEWS):
            raise ValueError("Window-bag evaluation reads the ten original windows")
        return torch.sigmoid(self.forward(images, features, series_mask, slice_mask, positions))


def training_inputs(
    batch: dict, device: torch.device, strategy: Mapping, *, seed: int, epoch: int, update: int
) -> tuple[Tensor, ...]:
    """The four source groups of every study, every update.

    Augmentation reuses ``source_slot``'s per-study generator and ``augment`` on the twelve
    stacked slices. Each slot's single rigid/intensity draw then applies to all of its
    windows (one anatomy, one transform), and group ``g`` equals the one-window recipe's
    augmented group ``g``. ``update`` is accepted for interface parity; no window is drawn.
    """
    strategy = validate_strategy(strategy)
    if any(type(value) is not int or value < 0 for value in (seed, epoch, update)):
        raise ValueError("Window-bag inputs need nonnegative integer counters")
    images = batch["images"]
    if images.ndim != 6 or images.shape[2] != len(EVALUATION_VIEWS):
        raise ValueError("Window-bag training reads the ten-view public row")
    views = list(TRAINING_VIEWS)
    bag = images[:, :, views]
    if strategy["augmentation"] == ss.AUGMENT:
        series_mask = batch["series_mask"]
        rows = []
        for index, uid in enumerate(batch["study_uid"]):
            study = bag[index].cpu()
            augmented = ss.source_augment(
                study.flatten(1, 2), ss.augmentation_generator(seed, epoch, uid)
            ).view_as(study)
            rows.append(
                torch.where(series_mask[index, :, None, None, None, None].cpu(), augmented, 0)
            )
        bag = torch.stack(rows)
    return (
        bag.to(device, non_blocking=True),
        batch["features"].to(device, non_blocking=True),
        batch["series_mask"].to(device, non_blocking=True),
        batch["slice_mask"][:, :, views].to(device, non_blocking=True),
        batch["positions"][:, :, views].to(device, non_blocking=True),
    )


def deployment_probabilities(
    model: WindowBagSlotStudyNet,
    images: Tensor,
    slot_mask: Tensor,
    device: torch.device,
    precision: str = "fp16",
) -> Tensor:
    """MRI-only inference: rendered uint8 ``[6,12,336,336]`` row and slot mask -> ``[12]``.

    The same row/mask/precision contract as ``source_slot.deployment_probabilities``.
    No labels, reports, teacher or committee.
    """
    if not isinstance(model, WindowBagSlotStudyNet):
        raise ValueError("Window-bag deployment requires the window-bag student")
    return ss.deployment_probabilities(model, images, slot_mask, device, precision)


def classifier_module(strategy: object):
    """``source_slot`` or this module, chosen only by an exact versioned ``kind``."""
    if isinstance(strategy, Mapping) and strategy.get("kind") == KIND:
        return sys.modules[__name__]
    return ss
