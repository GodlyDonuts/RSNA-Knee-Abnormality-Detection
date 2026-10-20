"""CPU checks of the untrained T4 template's nondegenerate precision protocol."""

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from rsna.competitive.predictive.cases import RawGroup, assemble
from rsna.competitive.predictive.infer import predict
from rsna.competitive.predictive.model import Predictive

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "initiative/research/predictive-T4-throughput-20261020-v1/benchmark.py.txt"
)


def checker(predict_call):
    tree = ast.parse(TEMPLATE.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "nonzero_shift_precision_check"
    ]
    assert len(selected) == 1
    namespace = {"torch": torch, "predict": predict_call}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(TEMPLATE), "exec"), namespace)
    return namespace["nonzero_shift_precision_check"]


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(33)
        raw = RawGroup(
            torch.rand(2, 64, 64),
            torch.arange(2),
            torch.zeros(2, dtype=torch.int64),
            torch.arange(2),
        )
        contexts = assemble([raw], side=16, halo=2, context_cap=2, classes=2)
        encoder = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        model = Predictive(encoder, 4, hidden=8, classes=2).eval()
    return model, contexts


def test_stress_exercises_nonzero_evidence_and_restores_original_model():
    model, contexts = fixture()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    probability_before = predict(model, contexts)
    requests = []

    def cpu_protocol_check(model, contexts, *, precision):
        # CPU exercises the actual evidence path/protocol, not CUDA FP16
        # numerics. The actual T4 execution remains a separate required result.
        requests.append(precision)
        return predict(model, contexts, precision="fp32")

    result = checker(cpu_protocol_check)(model, contexts)
    assert requests == ["fp16", "fp32"]
    assert result["maximum_absolute_FP32_log_bayes_factor"] > 1e-6
    assert result["maximum_FP16_FP32_log_bayes_factor_gap"] == 0
    assert not result["trained_head_precision_acceptance"]
    assert not model._forward_hooks
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert torch.equal(probability_before, predict(model, contexts))


def test_failed_precision_reference_still_restores_model_and_removes_hook():
    model, contexts = fixture()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    def unavailable_reference(model, contexts, *, precision):
        if precision == "fp32":
            raise RuntimeError("synthetic reference failure")
        return predict(model, contexts)

    with pytest.raises(RuntimeError, match="synthetic reference failure"):
        checker(unavailable_reference)(model, contexts)
    assert not model._forward_hooks
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_stress_refuses_to_overwrite_a_learned_shift():
    model, contexts = fixture()
    with torch.no_grad():
        model.shift.bias[0] = 1
    with pytest.raises(AssertionError):
        checker(predict)(model, contexts)
    assert model.shift.bias[0].item() == 1
    assert not model._forward_hooks
