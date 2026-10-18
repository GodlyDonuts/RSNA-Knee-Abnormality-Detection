"""New arithmetic tests only, with synthetic nonhomogeneous directed metadata."""

import importlib
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "connected-witness-paths-20261017-v1")]
v = importlib.import_module("path_vectorized")
r = importlib.import_module("variable_paths")


@pytest.mark.parametrize("length,tokens", [(1, 3), (2, 4), (7, 5), (13, 8)])
@pytest.mark.parametrize("arm", ["physical", "independent"])
def test_new_score_and_gradient_match_directed_reference(length, tokens, arm):
    g = torch.Generator().manual_seed(41 + length)
    unary = (torch.randn(length, tokens, 12, generator=g, dtype=torch.float64) * 3).requires_grad_()
    kappa = torch.linspace(-3, 3, 12, dtype=torch.float64, requires_grad=True)
    joints = torch.rand(length - 1, tokens, tokens, generator=g, dtype=torch.float64)
    if length > 1:
        joints[joints < 0.4] = 0
        joints += torch.eye(tokens, dtype=torch.float64)[None] * 0.1
    starts = torch.rand(length, tokens, generator=g, dtype=torch.float64)
    starts /= starts.sum(1, keepdim=True)
    gaps = torch.linspace(0.01, 100, max(0, length - 1), dtype=torch.float64)
    expected = r.witness_score(unary, joints, starts, gaps, kappa, arm=arm)
    actual = v.witness_score(unary, joints, starts, gaps, kappa, arm=arm, chunk_bytes=8192)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    ea = torch.autograd.grad(expected.sum(), (unary, kappa), retain_graph=True)
    aa = torch.autograd.grad(actual.sum(), (unary, kappa))
    for a, b in zip(aa, ea, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


def test_float32_zero_and_extreme_sparse_support():
    g = torch.Generator().manual_seed(50)
    unary = torch.randn(8, 9, 12, generator=g) * 40
    joints = torch.eye(9).repeat(7, 1, 1)
    starts = torch.ones(8, 9) / 9
    gaps = torch.linspace(0.001, 1000, 7)
    kappa = torch.linspace(-20, 20, 12)
    for arm in ("physical", "independent"):
        a = v.witness_score(unary, joints, starts, gaps, kappa, arm=arm)
        b = r.witness_score(unary, joints, starts, gaps, kappa, arm=arm)
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-6)
        z = v.witness_score(torch.zeros_like(unary), joints, starts, gaps, kappa, arm=arm)
        assert torch.equal(z, torch.zeros_like(z))


def test_candidate_workspace_limit_prevents_large_materialization():
    with pytest.raises(MemoryError):
        v.witness_score(
            torch.zeros(3, 4, 12),
            torch.eye(4).repeat(2, 1, 1),
            torch.ones(3, 4) / 4,
            torch.ones(2),
            torch.zeros(12),
            arm="independent",
            max_work_bytes=1,
        )


@pytest.mark.parametrize("arm", ["aligned", "null"])
def test_new_complete_head_keeps_all_summary_features(arm):
    from types import SimpleNamespace

    import numpy as np
    from path_reader import PathPlan, PathReader

    torch.manual_seed(19)
    reference = PathReader()
    fast = v.FastPathReader()
    with torch.no_grad():
        reference.output.copy_(torch.randn_like(reference.output) * 0.1)
        reference.continuation.copy_(torch.randn_like(reference.continuation))
    fast.load_state_dict(reference.state_dict(), strict=True)
    source = SimpleNamespace(
        windows=5, channels=1024, height=2, width=2, valid_tokens=[np.arange(4) for _ in range(5)]
    )
    chain = SimpleNamespace(
        indices=tuple(range(5)),
        tokens=np.arange(4),
        joints=torch.eye(4).repeat(4, 1, 1),
        starts=torch.ones(5, 4) / 4,
        gaps_mm=torch.tensor([1.0, 3.0, 2.0, 4.0]),
    )
    plan = PathPlan(source, (chain,), {"unsupported_windows": 0, "qualified_multistep_chains": 1})
    maps = torch.randn(5, 1024, 2, 2)
    base = torch.randn(12)
    expected = reference(maps, base, plan, arm=arm)
    actual = fast(maps, base, plan, arm=arm)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        fast.statistics(maps, plan, arm=arm),
        reference.statistics(maps, plan, arm=arm),
        atol=2e-6,
        rtol=2e-6,
    )


def test_new_optional_inference_policy_preserves_checkpoint_parameters(tmp_path):
    sys.path.insert(0, str(HERE.parent / "path-serving-20261018-v1"))
    serving = importlib.import_module("serving")
    model = v.FastPathReader()
    policy = {
        "schema": serving.SCHEMA,
        "reader": importlib.import_module("path_reader").VERSION,
        "arm": "null",
        "epochs": 3,
        "updates": 843,
        "parent_checkpoint_sha256": serving.PARENT,
        "encoder_precision": "float32",
        "sampling": "dense",
        "targets": list(serving.TARGETS),
        "channels": 1024,
        "width": 64,
        "chunk_size": 12,
    }
    head = tmp_path / "synthetic-head.pt"
    torch.save({"policy": policy, "model": model.state_dict()}, head)
    a = serving.PathInference(
        head, serving.digest(head), parent_sha256=serving.PARENT, device="cpu"
    )
    b = serving.PathInference(
        head,
        serving.digest(head),
        parent_sha256=serving.PARENT,
        device="cpu",
        vectorized_scoring=True,
    )
    assert not a.vectorized_scoring and b.vectorized_scoring
    assert isinstance(b.reader, v.FastPathReader)
    for n, t in a.reader.state_dict().items():
        assert torch.equal(t, b.reader.state_dict()[n])
    with pytest.raises(ValueError):
        serving.PathInference(
            head,
            serving.digest(head),
            parent_sha256=serving.PARENT,
            device="cpu",
            vectorized_scoring=1,
        )
