"""Engineering invariants only; this prototype has no clinical result."""

import torch

from scripts.research.plane_memory_prototype import PlaneMemory


def data():
    torch.manual_seed(17)
    x = torch.randn(2, 3, 4, 5, 8)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[0, 2] = False
    mask[1, 1, 2:] = False
    plane = torch.tensor([[0, 1, 3], [2, 0, 1]])
    pos = torch.randn(2, 3, 4, 2)
    return x, mask, plane, pos


def test_zero_start_preserves_shared_rng_and_activates_finite_head_gradient():
    state = torch.get_rng_state().clone()
    m = PlaneMemory(8, 16, 3)
    assert torch.equal(state, torch.get_rng_state())
    out = m(*data())
    assert torch.equal(out, torch.zeros(2, 3))
    out.sum().backward()
    assert torch.isfinite(m.score.grad).all() and m.score.grad.abs().sum() > 0


def test_genuine_metadata_and_padding_only_are_used():
    x, mask, plane, pos = data()
    m = PlaneMemory(8, 16, 3).eval()
    reference = m.evidence(x, mask, plane, pos)
    x[~mask] = float("nan")
    pos[~mask] = float("nan")
    plane[~mask.any(dim=2)] = -500
    assert torch.equal(reference, m.evidence(x, mask, plane, pos))


def test_sequence_order_permutation_preserves_output():
    x, mask, plane, pos = data()
    m = PlaneMemory(8, 16, 3).eval()
    order = [2, 0, 1]
    torch.testing.assert_close(
        m.evidence(x, mask, plane, pos),
        m.evidence(x[:, order], mask[:, order], plane[:, order], pos[:, order]),
    )


def test_missing_sequences_produce_finite_output_without_imputing_images():
    x, mask, plane, pos = data()
    mask[:, 1:] = False
    m = PlaneMemory(8, 16, 3)
    assert torch.isfinite(m.evidence(x, mask, plane, pos)).all()
