"""One isolated FP32 logit correction in the unchanged optimizer/recovery engine."""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import torch
from schedule import Schedule

from rsna.competitive.licensed_importance import ProposalCases
from rsna.competitive.licensed_importance_train import fit as original_fit

ENGINE_SHA = "e024a8ee91d5d64967e6128bfeac1ab2774468613ccaa43d13fc8ea569a5de7d"


class CorrectedCases(ProposalCases):
    def __init__(self, data, schedule, *, mode):
        if (
            not isinstance(data, ProposalCases)
            or not isinstance(schedule, Schedule)
            or mode not in ("zero", "stable_logit_delta")
            or data.uids != list(schedule.uids)
            or not torch.equal(
                torch.stack([data.proposal(i) for i in range(len(data))]),
                torch.from_numpy(schedule.proposals.copy()),
            )
        ):
            raise ValueError(
                "Exact original fitting identities/proposals and correction schedule required"
            )
        super().__init__(data.original, torch.from_numpy(schedule.proposals.copy()))
        self.schedule = schedule
        self.mode = mode

    def case(self, index, positions, mirror):
        sampled = self.schedule.lookup(index, positions, mirror)
        case = super().case(index, positions, mirror)
        parent = case["parent_logits"]
        if (
            not isinstance(parent, torch.Tensor)
            or parent.dtype != torch.float32
            or parent.shape != (12,)
            or parent.device.type != "cpu"
            or parent.requires_grad
            or not torch.isfinite(parent).all()
        ):
            raise ValueError("Exact immutable original FP32 view-specific teacher logits required")
        delta = torch.zeros_like(parent) if self.mode == "zero" else parent - sampled
        return {**case, "parent_pooling_delta": delta}


def _engine():
    if hashlib.sha256(Path(inspect.getfile(original_fit)).read_bytes()).hexdigest() != ENGINE_SHA:
        raise ValueError("Exact unchanged original full optimizer/recovery engine required")
    tree = ast.parse(inspect.getsource(original_fit))
    count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "anchored_loss"
        ):
            if (
                len(node.args) != 4
                or not isinstance(node.args[0], ast.Name)
                or node.args[0].id != "logits"
            ):
                raise ValueError("Unique original anchored loss signature required")
            node.args[0] = ast.parse(
                "logits + case['parent_pooling_delta'].to(device).reshape(1,12).float()",
                mode="eval",
            ).body
            count += 1
    if count != 1:
        raise ValueError("Exactly one training-only loss-input correction required")
    namespace = {**original_fit.__globals__}
    exec(
        compile(ast.fix_missing_locations(tree), "isolated_parent_delta_optimizer_engine", "exec"),
        namespace,
    )
    return namespace["fit"]


def fit(model, data, output, binding, recipe, **kwargs):
    if (
        not isinstance(data, CorrectedCases)
        or data.schedule.recipe != recipe
        or not isinstance(binding, dict)
        or "parent_delta" in binding
    ):
        raise ValueError(
            "Explicit original recipe, bound schedule and isolated correction mode required"
        )
    proof = {
        "version": "frozen_parent_same_bag_logit_residual_training_only_v1",
        "mode": data.mode,
        "schedule": data.schedule.binding,
        "original_engine_sha256": ENGINE_SHA,
        "full94_inference_unchanged": True,
        "new_learned_parameters": 0,
        "cached_deterministic_FP32_does_not_cancel_all_FP16_or_dropout_noise": True,
    }
    return _engine()(model, data, output, {**binding, "parent_delta": proof}, recipe, **kwargs)
