"""Exact deterministic fitting bags and private frozen-parent sampled logits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import torch
from torch.nn import functional as F

from rsna.competitive.licensed_importance import Recipe, sample, validate_proposal


def plan(uid, recipe, q):
    if not isinstance(recipe, Recipe) or recipe.role != "influence" or not 1 <= recipe.epochs <= 3:
        raise ValueError("Exact original influence recipe, at most three epochs required")
    proposal = torch.from_numpy(q.copy())
    validate_proposal(proposal)
    rows = [sample(uid, recipe, epoch, proposal) for epoch in range(recipe.epochs)]
    return np.array([r[0] for r in rows], dtype=np.uint8), np.array(
        [r[1] for r in rows], dtype=bool
    )


@torch.no_grad()
def teacher_logits(features, state, positions, mirrors, q):
    if (
        not isinstance(features, torch.Tensor)
        or features.shape != (2, 94, 1024)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not torch.isfinite(features).all()
        or positions.dtype != np.uint8
        or positions.ndim != 2
        or positions.shape[1] != 16
        or (positions >= 94).any()
        or mirrors.dtype != np.bool_
        or mirrors.shape != (len(positions),)
    ):
        raise ValueError("Exact CPU FP32 cached views and scheduled sixteen-window bags required")
    proposal = torch.from_numpy(q.copy())
    validate_proposal(proposal)
    rows = []
    for indices, mirror in zip(positions, mirrors, strict=True):
        ix = indices.astype(np.int64)
        h = F.layer_norm(
            features[int(mirror), ix], (1024,), state["norm.weight"], state["norm.bias"], eps=1e-5
        )
        scores = F.linear(
            F.linear(h, state["att.0.weight"], state["att.0.bias"]).tanh(),
            state["att.3.weight"],
            state["att.3.bias"],
        )
        logq = proposal[ix].log().float()
        logq = logq - logq.mean()
        weights = (scores.float() - logq[:, None]).softmax(0)
        pooled = torch.einsum("kt,kd->td", weights, h)
        logits = (pooled * state["clsW"]).sum(-1) + state["clsb"]
        if logits.shape != (12,) or not torch.isfinite(logits).all():
            raise ValueError("Complete finite deterministic parent sampled readout required")
        rows.append(logits.numpy().copy())
    return np.stack(rows).astype(np.float32)


class Schedule:
    """No mutable last-case state or RNG consumption during table lookup."""

    def __init__(self, uids, positions, mirrors, logits, proposals, recipe):
        if not isinstance(recipe, Recipe):
            raise ValueError("Exact original recipe required")
        n = len(uids)
        e = recipe.epochs
        if (
            not n
            or len(set(uids)) != n
            or any(not isinstance(u, str) or not u for u in uids)
            or positions.shape != (e, n, 16)
            or positions.dtype != np.uint8
            or mirrors.shape != (e, n)
            or mirrors.dtype != np.bool_
            or logits.shape != (e, n, 12)
            or logits.dtype != np.float32
            or not np.isfinite(logits).all()
            or proposals.shape != (n, 94)
            or proposals.dtype != np.float64
        ):
            raise ValueError("Exact aligned immutable fitting schedule required")
        for i, uid in enumerate(uids):
            p, m = plan(uid, recipe, proposals[i])
            if not np.array_equal(p, positions[:, i]) or not np.array_equal(m, mirrors[:, i]):
                raise ValueError("Schedule differs from original UID/epoch/mirror/proposal recipe")
            for a in range(e):
                for b in range(a):
                    if (
                        np.array_equal(p[a], p[b])
                        and m[a] == m[b]
                        and not np.array_equal(logits[a, i], logits[b, i])
                    ):
                        raise ValueError(
                            "Identical bags cannot have different frozen-parent logits"
                        )
        self.uids = tuple(uids)
        self.recipe = recipe
        self.positions = positions.copy()
        self.mirrors = mirrors.copy()
        self.logits = logits.copy()
        self.proposals = proposals.copy()
        for value in (self.positions, self.mirrors, self.logits, self.proposals):
            value.flags.writeable = False
        self.binding = {
            "version": "exact_frozen_parent_same_IID16_bag_logits_v1",
            "recipe": asdict(recipe),
            "population_sha256": hashlib.sha256(
                json.dumps(uids, separators=(",", ":")).encode()
            ).hexdigest(),
            "arrays_sha256": {
                k: hashlib.sha256(v.tobytes()).hexdigest()
                for k, v in [
                    ("positions", self.positions),
                    ("mirrors", self.mirrors),
                    ("logits", self.logits),
                    ("proposals", self.proposals),
                ]
            },
            "deterministic_FP32_readout_not_actual_stochastic_FP16_acceptance": True,
        }

    def lookup(self, index, positions, mirror):
        if (
            type(index) is not int
            or not 0 <= index < len(self.uids)
            or type(mirror) is not bool
            or not isinstance(positions, list)
            or len(positions) != 16
            or any(type(p) is not int or not 0 <= p < 94 for p in positions)
        ):
            raise ValueError("Exact admitted fitting bag identity required")
        for epoch in range(self.recipe.epochs):
            if self.mirrors[epoch, index] == mirror and np.array_equal(
                self.positions[epoch, index], positions
            ):
                return torch.from_numpy(self.logits[epoch, index].copy())
        raise ValueError("Unknown bag cannot borrow another epoch teacher correction")
