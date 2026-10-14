"""Identity-preserving disease-specific patch focus for the source-slot classifier (issue #62).

``SourceSlotStudyNet.slot_features`` reduces register-free spatial patch tokens to their mean
before the per-target slot head, so slot attention cannot see spatial co-occurrence that
the mean discards. This variant tests (as a hypothesis, motivated by Query2Label's
label-specific spatial pooling, arXiv:2107.10834) whether per-disease patch queries help.
It is distinct from window pooling (#38/#49/#60) and from the earlier replacement
``patch_attention`` head: the reviewed source-slot path is kept and the focus is a residual.

Per acquired slot, disease ``t`` and patch ``p``, with zero-initialized queries ``Q[T,D]``::

    w[t,p]    = softmax_p(<patch_p, Q_t> / sqrt(D))          u = softmax_p(0) = 1/P
    focus[t]  = mean_p(patch_p) + sum_p (w[t,p] - u[p]) patch_p
    logits    = head(x) + focus_gate * (H(x_Q) - H(x_0))

``x`` is the unchanged ``[CLS, patch mean]`` slot feature, ``x_Q`` replaces the patch mean of
target ``t`` with ``focus[t]`` (``x_0`` is the same construction with a zero query, which is
exactly ``x`` broadcast over targets). ``H`` (``target_conditioned_head``) is the reviewed
``SourceSlotHead`` applied per target: same parameters, LayerNorm/Linear/GELU projection, slot
attention, per-target context dropout ``[B,T,H]`` and output row. Both residual terms use
the same dropout draws, and the global RNG is restored afterwards, so the original head
consumes exactly the draws it would consume alone. At ``Q = 0`` the two terms are computed
by identical operations on identical values and cancel exactly in training and evaluation,
while ``dQ`` is generally nonzero. ``focus_gate`` is 0 (``disabled``, control: same
parameters, compute and draws, residual and query gradient exactly zero) or 1
(``query_residual``, candidate).

Pooling is content-only over the acquired slots' register-free tokens from the one encoder
pass; absent slots are never encoded. No token coordinates are used. Token grid positions
are normalized indices of the resampled image, not physical anatomy.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from rsna.competitive import source_slot as ss
from rsna.competitive import window_bag
from rsna.schema import TARGETS

KIND = "source_patch_focus_v1"
PATCH_POOL = "zero_query_patch_softmax_residual_v1"
FOCUS_GATES = {"disabled": 0.0, "query_residual": 1.0}
STRATEGY_FIXED = {**ss.STRATEGY_FIXED, "kind": KIND, "patch_pool": PATCH_POOL}
STRATEGY_KEYS = set(STRATEGY_FIXED) | {"augmentation", "patch_focus"}


def validate_strategy(strategy: object) -> dict:
    """Exact versioned policy; only ``augmentation`` and ``patch_focus`` vary."""
    if (
        not isinstance(strategy, Mapping)
        or set(strategy) != STRATEGY_KEYS
        or any(
            type(strategy[key]) is not type(value) or strategy[key] != value
            for key, value in STRATEGY_FIXED.items()
        )
        or not isinstance(strategy["augmentation"], str)
        or strategy["augmentation"] not in ss.AUGMENTATIONS
        or not isinstance(strategy["patch_focus"], str)
        or strategy["patch_focus"] not in FOCUS_GATES
    ):
        raise ValueError("Exact source patch-focus classifier strategy required")
    return dict(strategy)


def slot_strategy(strategy: Mapping) -> dict:
    """The unchanged ``source_matched_slot_v1`` policy whose encoder/head/inputs are reused."""
    return {**ss.STRATEGY_FIXED, "augmentation": validate_strategy(strategy)["augmentation"]}


def focus_features(base: Tensor, patches: Tensor, series_mask: Tensor, query: Tensor) -> Tensor:
    """``[B,S,2D]`` features, acquired ``[N,P,D]`` patches, ``[B,S]`` mask, ``[T,D]`` query ->
    per-target slot features ``[B,T,S,2D]``.

    The CLS half is ``base``'s, untouched. The patch half is ``base``'s mean plus the
    query-weighted deviation, computed in FP32 with autocast disabled. A zero query gives
    ``w == u`` bitwise, so the deviation is exactly zero. Absent slots stay zero.
    """
    b, s = series_mask.shape
    t, d = query.shape
    if (
        series_mask.dtype != torch.bool
        or base.shape != (b, s, 2 * d)
        or patches.ndim != 3
        or patches.shape[0] != int(series_mask.sum())
        or patches.shape[2] != d
        or patches.shape[1] < 1
    ):
        raise ValueError("Aligned [B,S,2D] features, [N,P,D] acquired patches and [T,D] query")
    with torch.autocast(base.device.type, enabled=False):
        tokens = patches.float()
        if not (torch.isfinite(tokens).all() and torch.isfinite(query).all()):
            raise ValueError("Patch tokens and queries must be finite")
        scores = torch.einsum("npd,td->ntp", tokens, query.float()) / d**0.5
        weights = scores.softmax(-1) - torch.zeros_like(scores).softmax(-1)
        deviation = torch.einsum("ntp,npd->ntd", weights, tokens)
        full = deviation.new_zeros((b, s, t, d))
        full[series_mask] = deviation
    cls, mean = base[..., :d], base[..., d:]
    focus = mean[:, None] + full.transpose(1, 2).to(base.dtype)
    return torch.cat([cls[:, None].expand(b, t, s, d), focus], dim=-1)


def target_conditioned_head(head: ss.SourceSlotHead, x: Tensor, mask: Tensor) -> Tensor:
    """``SourceSlotHead.forward`` with a separate slot feature per target: ``[B,T,S,2D]`` -> ``[B,T]``.

    Target ``t`` reads only ``x[:, t]``, its own query row and output row. With ``x`` constant
    over targets it equals ``head(x[:, 0], mask)`` up to floating rounding (different GEMM
    shapes), so it is used only inside the cancelling residual, never for the original logits.
    """
    if x.ndim != 4 or x.shape[1] != head.query.shape[0] or mask.shape != (x.shape[0], x.shape[2]):
        raise ValueError("Target-conditioned head reads [B,T,S,2D] features and a [B,S] mask")
    h = head.proj(x) + head.slot_emb
    att = torch.einsum("btsh,th->bts", h, head.query) / head.hidden**0.5
    att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
    ctx = head.drop(torch.einsum("bts,btsh->bth", att, h))
    return (ctx * head.out.weight.unsqueeze(0)).sum(-1) + head.out.bias


def _rng_state(device: torch.device) -> Tensor:
    return torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()


def _set_rng_state(device: torch.device, state: Tensor) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)


class PatchFocusSlotStudyNet(ss.SourceSlotStudyNet):
    """Reviewed source-slot encoder/pool/head plus zero-initialized ``patch_query`` ``[T,D]``.

    Inputs, window selection, ten-window probability averaging and deployment are the
    source-slot ones; only ``window_logits`` adds the gated residual.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        strategy: Mapping,
        encode_chunk_size: int = 64,
        input_channels: int = 3,
    ):
        focus = validate_strategy(strategy)
        super().__init__(
            encoder, embedding_dim, slot_strategy(focus), encode_chunk_size, input_channels
        )
        if type(self.head) is not ss.SourceSlotHead or any(True for _ in self.head.buffers()):
            raise ValueError("Patch focus needs the stateless reviewed source slot head")
        self.strategy = focus
        self.focus_gate = FOCUS_GATES[focus["patch_focus"]]
        # Created after the reviewed head; zeros draw nothing from the factory RNG.
        self.patch_query = nn.Parameter(torch.zeros(len(TARGETS), embedding_dim))

    def slot_tokens(self, window: Tensor, series_mask: Tensor) -> tuple[Tensor, Tensor]:
        """``slot_features`` and the acquired slots' register-free patches ``[N,h*w,D]``.

        Same operations as ``SourceSlotStudyNet.slot_features`` from one encoder pass.
        """
        valid = window[series_mask]
        forward_features = getattr(self.encoder, "forward_features")
        parts = [forward_features(part) for part in valid.split(self.encode_chunk_size)]
        cls = torch.cat([part[0] for part in parts])
        patches = torch.cat([part[1] for part in parts])
        if (
            cls.shape != (len(valid), self.embedding_dim)
            or patches.ndim != 4
            or patches.shape[0] != len(valid)
            or patches.shape[-1] != self.embedding_dim
        ):
            raise ValueError("Encoder must return [N,D] CLS and [N,h,w,D] register-free patches")
        features = torch.cat([cls, patches.flatten(1, 2).mean(1)], dim=1)
        full = features.new_zeros((*series_mask.shape, features.shape[-1]))
        full[series_mask] = features
        return full, patches.flatten(1, 2)

    def slot_features(self, window: Tensor, series_mask: Tensor) -> Tensor:
        return self.slot_tokens(window, series_mask)[0]

    def focused_logits(self, base: Tensor, patches: Tensor, series_mask: Tensor) -> Tensor:
        """Original head logits plus the gated, exactly cancelling-at-zero focus residual."""
        mask = series_mask.float()
        logits = self.head(base, mask)
        device = logits.device
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            state = _rng_state(device)
            focused = target_conditioned_head(
                self.head, focus_features(base, patches, series_mask, self.patch_query), mask
            )
            _set_rng_state(device, state)
            reference = target_conditioned_head(
                self.head,
                focus_features(base, patches, series_mask, torch.zeros_like(self.patch_query)),
                mask,
            )
        return logits + self.focus_gate * (focused - reference)

    def window_logits(self, window: Tensor, series_mask: Tensor) -> Tensor:
        base, patches = self.slot_tokens(window, series_mask)
        return self.focused_logits(base, patches, series_mask)


def training_inputs(
    batch: dict, device: torch.device, strategy: Mapping, *, seed: int, epoch: int, update: int
) -> tuple[Tensor, ...]:
    """The unchanged source-slot update window and augmentation."""
    return ss.training_inputs(
        batch, device, slot_strategy(strategy), seed=seed, epoch=epoch, update=update
    )


def deployment_probabilities(
    model: PatchFocusSlotStudyNet,
    images: Tensor,
    slot_mask: Tensor,
    device: torch.device,
    precision: str = "fp16",
) -> Tensor:
    """MRI-only inference with the ``source_slot.deployment_probabilities`` contract."""
    if not isinstance(model, PatchFocusSlotStudyNet):
        raise ValueError("Patch-focus deployment requires the patch-focus student")
    return ss.deployment_probabilities(model, images, slot_mask, device, precision)


def classifier_module(strategy: object):
    """This module for its exact ``kind``; otherwise the window-bag/source-slot dispatch."""
    if isinstance(strategy, Mapping) and strategy.get("kind") == KIND:
        return sys.modules[__name__]
    return window_bag.classifier_module(strategy)
