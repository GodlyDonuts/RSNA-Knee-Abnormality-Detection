import pytest
import torch
from torch.nn import functional as F

from rsna.competitive.cached_attention_control import moments, readout


def fixture():
    w = torch.tensor([[[0.65, 0.1], [0.25, 0.6], [0.1, 0.3]]], dtype=torch.float64)
    v = torch.tensor([[[2.0, -1.0], [-2.0, 3.0], [0.5, -0.5]]], dtype=torch.float64)
    q = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    b = torch.tensor([0.1, -0.2], dtype=torch.float64)
    return w, v, q, b


def test_exact_reference_logits_for_any_draw_duplicates_and_no_rng_change():
    w, v, q, b = fixture()
    before = torch.get_rng_state()
    for i in [torch.tensor([[0]]), torch.tensor([[1, 1, 1]]), torch.tensor([[2, 0, 2, 1]])]:
        delta = torch.zeros((*i.shape, 2), dtype=torch.float64)
        z = readout(w, v, i, q[i].log(), delta, delta, b)
        assert torch.equal(z, (w * v).sum(1) + b)
    assert torch.equal(before, torch.get_rng_state())


def test_dense_enumeration_recovers_full_current_attention_after_drift():
    w, v, q, b = fixture()
    ds = torch.tensor([[[0.1, -0.3], [-0.2, 0.4], [0.6, 0.1]]], dtype=w.dtype)
    dv = torch.tensor([[[0.2, 0.6], [0.1, -0.1], [-0.4, 0.3]]], dtype=w.dtype)
    i = torch.arange(3).reshape(1, 3)
    logq = torch.full((1, 3), -torch.log(torch.tensor(3.0)).item(), dtype=w.dtype)
    # Use the exact double-precision full-population probability.
    logq = torch.full_like(logq, 1 / 3).log()
    z = readout(w, v, i, logq, ds, dv, b)
    weights = w * ds.exp()
    expected = (weights * (v + dv)).sum(1) / weights.sum(1) + b
    assert torch.allclose(z, expected, atol=1e-14, rtol=0)


def test_expected_full_loss_gradient_exact_at_reference_by_exhaustive_sampling():
    w, v, q, b = fixture()
    ds = torch.zeros_like(w, requires_grad=True)
    dv = torch.zeros_like(v, requires_grad=True)
    bias = b.clone().requires_grad_(True)
    target = torch.tensor([[0.15, 0.8]], dtype=w.dtype)
    a = w * ds.exp()
    exact = (a * (v + dv)).sum(1) / a.sum(1) + bias
    full_grad = torch.autograd.grad(
        F.binary_cross_entropy_with_logits(exact, target), (ds, dv, bias)
    )
    expected = [torch.zeros_like(x) for x in full_grad]
    for j in range(3):
        i = torch.tensor([[j]])
        z = readout(w, v, i, q[i].log(), ds[:, j : j + 1], dv[:, j : j + 1], bias)
        grads = torch.autograd.grad(F.binary_cross_entropy_with_logits(z, target), (ds, dv, bias))
        expected = [x + q[j] * g for x, g in zip(expected, grads)]
    assert all(torch.allclose(a, b, atol=1e-14, rtol=0) for a, b in zip(expected, full_grad))
    assert full_grad[0].abs().sum() > 0  # An ordinary one-instance softmax has zero score gradient.


def test_unbiased_moments_do_not_make_ratio_unbiased_after_drift():
    w, v, q, b = fixture()
    ds = torch.tensor([[[0.2, -0.1], [0.05, 0.3], [-0.2, 0.1]]], dtype=w.dtype)
    dv = ds * 2
    means = [torch.zeros((1, 2), dtype=w.dtype) for _ in range(3)]
    for j in range(3):
        i = torch.tensor([[j]])
        n, d = moments(w, v, i, q[i].log(), ds[:, j : j + 1], dv[:, j : j + 1])
        z = readout(w, v, i, q[i].log(), ds[:, j : j + 1], dv[:, j : j + 1], b)
        means = [x + q[j] * y for x, y in zip(means, (n, d, z))]
    a = w * ds.exp()
    numerator = (a * (v + dv)).sum(1)
    denominator = a.sum(1)
    assert torch.allclose(means[0], numerator, atol=1e-14, rtol=0)
    assert torch.allclose(means[1], denominator, atol=1e-14, rtol=0)
    assert not torch.allclose(means[2], numerator / denominator + b, atol=1e-8, rtol=0)


def test_negative_denominator_is_exposed_and_never_silently_clamped():
    w = torch.tensor([[[0.9], [0.1]]], dtype=torch.float64)
    v = torch.ones_like(w)
    i = torch.tensor([[0]])
    logq = torch.tensor([[0.5]], dtype=w.dtype).log()
    ds = torch.tensor([[[-5.0]]], dtype=w.dtype)
    dv = torch.zeros_like(ds)
    _, d = moments(w, v, i, logq, ds, dv)
    assert d.item() < 0
    with pytest.raises(ValueError, match="no fallback"):
        readout(w, v, i, logq, ds, dv, torch.zeros(1, dtype=w.dtype))


@pytest.mark.parametrize("change", ["reference_grad", "q_grad", "shape", "q", "index", "nan"])
def test_invalid_or_trainable_reference_refused(change):
    w, v, q, b = fixture()
    i = torch.tensor([[0, 1]])
    logq = q[i].log()
    ds = torch.zeros((1, 2, 2), dtype=w.dtype)
    if change == "reference_grad":
        w.requires_grad_(True)
    elif change == "q_grad":
        logq.requires_grad_(True)
    elif change == "shape":
        ds = ds[:, :1]
    elif change == "q":
        logq[0, 0] = 1
    elif change == "index":
        i[0, 0] = 9
    else:
        ds[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        readout(w, v, i, logq, ds, ds, b)


def test_relocated_CPU_screen_keeps_all_original_numerical_loops_and_stress_levels():
    import ast
    from pathlib import Path

    d = Path("initiative/research/cached-attention-control-20261020-v1")
    original = ast.parse((d / "screen.py.txt").read_text())
    relocated = ast.parse((d / "screen_relocated_v2.py.txt").read_text())

    # Host controls/funding differ; every actual readout/draw/statistic loop is identical.
    def loops(tree):
        return [ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.For)]

    assert loops(original) == loops(relocated)
    for tree in (original, relocated):
        values = {
            n.targets[0].id: n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        }
        assert ast.literal_eval(values["levels"]) == (0.0, 0.05, 0.2, 0.5)
        assert ast.literal_eval(values["draws"]) == 128
        assert ast.unparse(values["selected"]) == "np.linspace(0, 3886, 32, dtype=np.int64)"
    source = (d / "screen_relocated_v2.py.txt").read_text()
    assert source.index("paid_controls") < source.index("D.mkdir")
    assert "256 * 2**10 + 2 * 2**20 < 8 * 2**20" in source
