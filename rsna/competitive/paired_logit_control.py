"""Training-only deterministic parent logit corrections; full inference is unchanged.

This is an estimator candidate, not accepted full-encoder training or disease
accuracy. Cached parent features do not reproduce stochastic training features.
"""

from __future__ import annotations

import hashlib

import torch

from rsna.competitive.licensed_importance import Recipe, sample, validate_proposal


def correction(parent_full, parent_sample):
    if (
        not isinstance(parent_full, torch.Tensor)
        or not isinstance(parent_sample, torch.Tensor)
        or parent_full.shape != parent_sample.shape
        or parent_full.ndim != 2
        or parent_full.shape[1] != 12
        or parent_full.dtype != torch.float32
        or parent_sample.dtype != torch.float32
        or parent_full.requires_grad
        or parent_sample.requires_grad
        or not torch.isfinite(parent_full).all()
        or not torch.isfinite(parent_sample).all()
    ):
        raise ValueError("Matching detached finite FP32 parent view/finding logits required")
    return (parent_full - parent_sample).detach().clone()


@torch.no_grad()
@torch.random.fork_rng(devices=[])
def schedule(readout, features, uid, recipe, proposal):
    """Compute only3×12constants with the exact existing IID16 windows/view.

    readout(features, log_proposal) must be a frozen deterministic CPU parent
    head. Its numerical source/head/feature hashes must be bound separately by
    an actual schedule artifact admission. No encoder is invoked here.
    """
    validate_proposal(proposal)
    if (
        not isinstance(recipe, Recipe)
        or recipe.epochs != 3
        or recipe.role != "influence"
        or features.shape != (2, 94, 1024)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or features.requires_grad
        or not torch.isfinite(features).all()
    ):
        raise ValueError(
            "Original fixed3 influence recipe and full frozen two-view features required"
        )
    before = torch.get_rng_state().clone()
    full = readout(features, None)
    if full.shape != (2, 12) or full.device.type != "cpu":
        raise ValueError("Complete original CPU full-parent logits required")
    positions, mirrors, shifts, log_probabilities = [], [], [], []
    for epoch in range(3):
        indices, mirror, logq = sample(uid, recipe, epoch, proposal)
        view = int(mirror)
        observed = readout(features[view : view + 1, indices], logq)
        shifts.append(correction(full[view : view + 1], observed)[0])
        positions.append(indices)
        mirrors.append(mirror)
        log_probabilities.append(logq.clone())
    if not torch.equal(before, torch.get_rng_state()):
        raise ValueError("Cached correction baseline must not consume stochastic training RNG")
    return {
        "study_uid_sha256": hashlib.sha256(uid.encode()).hexdigest(),
        "log_proposal": torch.stack(log_probabilities),
        "positions": torch.tensor(positions, dtype=torch.uint8),
        "mirror": torch.tensor(mirrors, dtype=torch.bool),
        "shift": torch.stack(shifts),
    }


def lookup(entry, uid, positions, mirror, log_proposal):
    """Bind to actual images/view, avoiding stale mutable epoch state on a model."""
    if (
        not isinstance(entry, dict)
        or set(entry) != {"study_uid_sha256", "positions", "mirror", "shift", "log_proposal"}
        or not isinstance(uid, str)
        or not uid
        or entry["study_uid_sha256"] != hashlib.sha256(uid.encode()).hexdigest()
        or any(
            not isinstance(entry[n], torch.Tensor)
            for n in ("positions", "mirror", "shift", "log_proposal")
        )
        or entry["positions"].shape != (3, 16)
        or entry["positions"].dtype != torch.uint8
        or entry["mirror"].shape != (3,)
        or entry["mirror"].dtype != torch.bool
        or entry["shift"].shape != (3, 12)
        or entry["shift"].dtype != torch.float32
        or entry["log_proposal"].shape != (3, 16)
        or entry["log_proposal"].dtype != torch.float32
        or any(
            entry[n].device.type != "cpu" or entry[n].requires_grad
            for n in ("positions", "mirror", "shift", "log_proposal")
        )
        or not isinstance(log_proposal, torch.Tensor)
        or log_proposal.shape != (16,)
        or log_proposal.dtype != torch.float32
        or log_proposal.device.type != "cpu"
        or log_proposal.requires_grad
        or not torch.isfinite(log_proposal).all()
        or not torch.isfinite(entry["shift"]).all()
        or type(mirror) is not bool
        or not isinstance(positions, list)
        or len(positions) != 16
        or any(type(i) is not int or not 0 <= i < 94 for i in positions)
    ):
        raise ValueError(
            "Bound original3epoch sampling schedule and exact current image draw required"
        )
    matches = (entry["positions"] == torch.tensor(positions, dtype=torch.uint8)).all(1)
    matches &= entry["mirror"] == mirror
    matches &= (entry["log_proposal"] == log_proposal).all(1)
    values = entry["shift"][matches]
    if not len(values) or not torch.equal(values, values[:1].expand_as(values)):
        raise ValueError("Current image/view has no unique matching immutable correction")
    return values[0].detach().clone()


class PairedLogitReadout:
    """Compose before the unchanged ImportanceReadout and original licensed class.

    The training caller must provide each exact case's detached shift explicitly;
    no cached batch state or silent fallback. With no training arguments, the
    original full94 inference graph and complete parameter/state graph remain.
    """

    def forward(self, images, log_proposal=None, *, baseline_shift=None):
        if log_proposal is None:
            if baseline_shift is not None:
                raise ValueError("A parent correction must never enter ordinary full inference")
            return getattr(super(), "forward")(images)
        if (
            not isinstance(baseline_shift, torch.Tensor)
            or baseline_shift.shape != (images.shape[0], 12)
            or baseline_shift.dtype != torch.float32
            or baseline_shift.device != images.device
            or baseline_shift.requires_grad
            or not torch.isfinite(baseline_shift).all()
        ):
            raise ValueError("Explicit detached complete case-matched parent correction required")
        logits = getattr(super(), "forward")(images, log_proposal)
        if logits.shape != baseline_shift.shape or not torch.isfinite(logits).all():
            raise ValueError("Original complete finite finding logits required")
        return logits.float() + baseline_shift
