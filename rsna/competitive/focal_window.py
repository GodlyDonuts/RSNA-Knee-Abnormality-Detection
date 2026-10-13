"""Disease-query sparse window pooling and window-level frozen-teacher supervision.

Standalone research mechanism (issue #37); not wired into the trainer. A public-triplet
window ``k`` is rendered slices ``k, k+1, k+2`` of every acquired slot, exactly the
original source's ``window_starts(12, 3) == range(10)``. The frozen teacher's ``k``-th
logit (issue #35) is one study-wide forward of that same window across all slots, so the
student's per-window study logit below is its like-for-like counterpart.

Disagreement across windows is anatomical: different windows show different anatomy.
It is never treated as uncertainty or used to down-weight supervision. Prediction
variability under transformations of the *same* window is a different quantity; the
#35 evidence contains one prediction per window, so it is unavailable here.

Teacher evidence is fitting-only soft supervision. It never replaces labels, masks or
folds, never reaches evaluation and is not needed at student inference. Its training
roster is unknown, so no clean held-out teacher/student claim is possible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from rsna.competitive.loss import effective_batch_loss
from rsna.competitive.model import _masked_softmax
from rsna.schema import TARGETS

CACHE_SLICES, GROUP, SLOTS = 12, 3, 6
WINDOW_STARTS = tuple(range(CACHE_SLICES - GROUP + 1))
WINDOWS = len(WINDOW_STARTS)
LOGIT_CAP = 6.0
UNIFORM_FLOOR = 0.25
MAX_WINDOW_TEACHER_WEIGHT = 1.0


def window_slices(window: int) -> tuple[int, ...]:
    """Rendered cache slice indices of one window; its position is the center slice."""
    if type(window) is not int or window not in WINDOW_STARTS:
        raise ValueError("Window index outside the ten original overlapping windows")
    return tuple(range(window, window + GROUP))


class FocalWindowPool(nn.Module):
    """Slots -> per-window study logit -> bounded sparse study logit, per disease query.

    1. Per window ``k`` and target ``t``, gated attention over valid slots (disease query
       ``slot_query[t]``) gives a window feature and a window logit ``u[k,t]``, softly
       capped to ``(-cap, cap)`` so one extreme window cannot be arbitrarily certain.
    2. Gated attention over valid windows (``window_query[t]``) is mixed with a fixed
       uniform floor ``rho``: ``beta = (1-rho)*softmax + rho/K``. The study logit is
       ``sum_k beta[k,t] * u[k,t]``, a convex combination: ``|z| < cap``; any single
       window weighs at most ``1-rho+rho/K`` and at least ``rho/K``.

    Queries start at zero, so initial pooling is the plain mean over slots and windows.
    The parameter count equals the existing slice/series attention head it would replace.
    """

    def __init__(
        self,
        hidden_dim: int,
        targets: int = len(TARGETS),
        *,
        logit_cap: float = LOGIT_CAP,
        uniform_floor: float = UNIFORM_FLOOR,
    ):
        super().__init__()
        if min(hidden_dim, targets) < 1:
            raise ValueError("Invalid focal window dimensions")
        if not (math.isfinite(logit_cap) and logit_cap > 0 and 0 < uniform_floor <= 1):
            raise ValueError("Focal pooling requires a positive cap and a floor in (0, 1]")
        self.logit_cap, self.uniform_floor = float(logit_cap), float(uniform_floor)
        self.slot_tanh = nn.Linear(hidden_dim, hidden_dim)
        self.slot_gate = nn.Linear(hidden_dim, hidden_dim)
        self.slot_query = nn.Parameter(torch.zeros(targets, hidden_dim))
        self.window_tanh = nn.Linear(hidden_dim, hidden_dim)
        self.window_gate = nn.Linear(hidden_dim, hidden_dim)
        self.window_query = nn.Parameter(torch.zeros(targets, hidden_dim))
        self.output_weight = nn.Parameter(torch.empty(targets, hidden_dim))
        self.output_bias = nn.Parameter(torch.zeros(targets))
        nn.init.normal_(self.output_weight, std=hidden_dim**-0.5)

    def forward(
        self, features: Tensor, slice_mask: Tensor, series_features: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """features [B,S,K,H], slice_mask [B,S,K] -> study [B,T], window [B,K,T], beta [B,K,T].

        Padded slots/windows never contribute; their feature values are ignored.
        """
        if features.ndim != 4 or slice_mask.shape != features.shape[:3]:
            raise ValueError("Expected [B,S,K,H] window features and a [B,S,K] mask")
        if slice_mask.dtype != torch.bool:
            raise ValueError("Window masks must be boolean")
        window_valid = slice_mask.any(dim=1)
        if not window_valid.any(dim=1).all():
            raise ValueError("Every study needs a genuine window; empty studies are refused")
        x = features
        if series_features is not None:
            if series_features.shape != (*features.shape[:2], features.shape[3]):
                raise ValueError("Series features must be [B,S,H]")
            x = x + series_features[:, :, None, :]
        x = torch.where(slice_mask[..., None], x, torch.zeros_like(x))
        if not torch.isfinite(x).all():
            raise ValueError("Valid window features must be finite")
        gated = torch.tanh(self.slot_tanh(x)) * torch.sigmoid(self.slot_gate(x))
        alpha = _masked_softmax(
            torch.einsum("bskh,th->bskt", gated, self.slot_query), slice_mask[..., None], dim=1
        )
        window = torch.einsum("bskt,bskh->bkth", alpha, x)
        raw = torch.einsum("bkth,th->bkt", window, self.output_weight) + self.output_bias
        logits = self.logit_cap * torch.tanh(raw / self.logit_cap)
        logits = torch.where(window_valid[..., None], logits, torch.zeros_like(logits))
        gated_window = torch.tanh(self.window_tanh(window)) * torch.sigmoid(
            self.window_gate(window)
        )
        valid = window_valid[..., None].expand_as(logits)
        attention = _masked_softmax(
            torch.einsum("bkth,th->bkt", gated_window, self.window_query), valid, dim=1
        )
        uniform = valid.to(logits.dtype) / valid.sum(dim=1, keepdim=True).to(logits.dtype)
        beta = (1 - self.uniform_floor) * attention + self.uniform_floor * uniform
        return (beta * logits).sum(dim=1), logits, beta


def window_teacher_targets(
    study_uids: Sequence[str], arrays: dict[str, np.ndarray], series_mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Per-window teacher soft targets and weights for a fitting batch.

    ``arrays`` follows the #35 evidence contract (``study_uid``, ``window_logits``
    [N,10,12], ``slot_mask`` [N,6]). Weight is that window's own decisiveness
    ``|2p-1|``; across-window spread never reduces it. Studies without evidence get
    zero weight. A slot mask differing from the student's means the teacher saw other
    inputs, so it is refused rather than silently reinterpreted.
    """
    logits = np.asarray(arrays["window_logits"])
    slot_mask = np.asarray(arrays["slot_mask"])
    uids = [str(u) for u in np.asarray(arrays["study_uid"]).tolist()]
    if (
        logits.dtype != np.float32
        or logits.shape != (len(uids), WINDOWS, len(TARGETS))
        or slot_mask.dtype != np.bool_
        or slot_mask.shape != (len(uids), SLOTS)
        or len(set(uids)) != len(uids)
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Teacher window evidence differs from the #35 contract")
    if series_mask.shape != (len(study_uids), SLOTS) or series_mask.dtype != torch.bool:
        raise ValueError("Student series mask must be boolean [B,6]")
    index = {uid: i for i, uid in enumerate(uids)}
    soft = torch.zeros(len(study_uids), WINDOWS, len(TARGETS))
    weight = torch.zeros_like(soft)
    for row, uid in enumerate(study_uids):
        if uid not in index:
            continue
        i = index[uid]
        if not np.array_equal(slot_mask[i], series_mask[row].cpu().numpy()):
            raise ValueError("Teacher and student slot masks differ for a study")
        p = torch.sigmoid(torch.from_numpy(logits[i]).double())
        soft[row], weight[row] = p.float(), (2 * p - 1).abs().float()
    return soft, weight


def window_variability(window_logits: Tensor, replicate_logits: Tensor | None = None) -> dict:
    """Descriptive only; neither quantity is used as a supervision weight.

    ``anatomical_window_range``: max-min window probability [N,T]; expected to be large
    for focal findings. ``transformation_spread``: max-min probability across ``R>=2``
    predictions of the same window [R,N,K,T] -> [N,K,T]; ``None`` when not collected.
    """
    if window_logits.ndim != 3 or not torch.isfinite(window_logits).all():
        raise ValueError("Finite [N,K,T] window logits required")
    p = torch.sigmoid(window_logits.double())
    spread = None
    if replicate_logits is not None:
        if (
            replicate_logits.ndim != 4
            or replicate_logits.shape[0] < 2
            or replicate_logits.shape[1:] != window_logits.shape
            or not torch.isfinite(replicate_logits).all()
        ):
            raise ValueError("Same-window replicates must be finite [R>=2,N,K,T]")
        r = torch.sigmoid(replicate_logits.double())
        spread = r.amax(0) - r.amin(0)
    return {"anatomical_window_range": p.amax(1) - p.amin(1), "transformation_spread": spread}


def window_teacher_loss(
    window_logits: Tensor, soft: Tensor, weight: Tensor, counts: Tensor
) -> Tensor:
    """Weighted soft BCE per (window, target), normalized like the label loss.

    ``counts`` are per-target weight sums over the whole effective batch, e.g.
    ``observed_counts([w.flatten(0, 1) for w in update_weights])``.
    """
    if window_logits.ndim != 3 or not (window_logits.shape == soft.shape == weight.shape):
        raise ValueError("Aligned [B,K,T] window logits, targets and weights required")
    return effective_batch_loss(
        window_logits.flatten(0, 1), soft.flatten(0, 1), weight.flatten(0, 1), counts
    )


def combined_loss(label_loss: Tensor, auxiliary: Callable[[], Tensor], weight: object) -> Tensor:
    """Weight zero returns the label loss object itself; the teacher term is not evaluated."""
    if (
        type(weight) not in {int, float}
        or not math.isfinite(float(weight))  # type: ignore[arg-type]
        or not 0 <= float(weight) <= MAX_WINDOW_TEACHER_WEIGHT  # type: ignore[arg-type]
    ):
        raise ValueError("Window teacher weight must be finite in [0, 1]")
    if float(weight) == 0:  # type: ignore[arg-type]
        return label_loss
    return label_loss + float(weight) * auxiliary()  # type: ignore[arg-type]
