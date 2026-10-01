import pytest
import torch
from torch import nn

from rsna.competitive.adapters import ResidualLinear
from rsna.competitive.relations import SliceDistanceHead, slice_relations


def geometry():
    positions = torch.zeros(2, 2, 5, 2)
    positions[..., 0] = torch.tensor([-0.12, -0.06, 0, 0.07, 0.15])
    positions[..., 1] = 1
    return positions, torch.ones(2, 2, 5, dtype=torch.bool)


def pairs(positions, mask, **kwargs):
    return slice_relations(
        positions,
        mask,
        torch.ones(2, 2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        seed=32,
        **kwargs,
    )


def test_pair_rng_gauge_invariance_and_shuffled_marginals():
    positions, mask = geometry()
    rng = torch.get_rng_state().clone()
    reference = pairs(positions, mask, max_pairs=6)
    assert torch.equal(torch.get_rng_state(), rng)
    changed = positions.clone()
    changed[..., 0] = -changed[..., 0] + 2
    shifted = pairs(changed, mask, max_pairs=6)
    assert torch.equal(reference.indices, shifted.indices)
    torch.testing.assert_close(reference.targets, shifted.targets, atol=1e-6, rtol=1e-6)
    shuffled_capped = pairs(positions, mask, max_pairs=6, shuffle_targets=True)
    assert torch.equal(reference.indices, shuffled_capped.indices)
    # Use all pairs: negative control preserves each series' target distribution.
    reference = pairs(positions, mask)
    shuffled = pairs(positions, mask, shuffle_targets=True)
    assert torch.equal(reference.indices, shuffled.indices)
    assert not torch.equal(reference.targets, shuffled.targets)
    for study in range(2):
        for series in range(2):
            selected = (reference.indices[:, 0] == study) & (reference.indices[:, 1] == series)
            torch.testing.assert_close(
                reference.targets[selected].sort().values, shuffled.targets[selected].sort().values
            )


def test_padding_unknown_geometry_tail_duplicates_and_equal_study_weight():
    positions, mask = geometry()
    mask[0, 0, 3:] = False
    positions[~mask] = torch.nan
    positions[0, 1] = 0  # Unknown geometry is not fabricated from ordinal index.
    relations = pairs(positions, mask)
    assert len(relations.indices) == 23
    for study in range(2):
        torch.testing.assert_close(
            relations.weights[relations.indices[:, 0] == study].sum(), torch.tensor(0.5)
        )
    excluded = slice_relations(
        positions, mask, torch.ones(2, 2, dtype=torch.bool), torch.tensor([True, False]), seed=1
    )
    assert len(excluded.indices) == 3 and not excluded.indices[:, :2].any()


@pytest.mark.parametrize("mutation", ["flags", "nonfinite", "nonmonotone", "unknown"])
def test_invalid_actual_geometry_is_rejected(mutation):
    positions, mask = geometry()
    if mutation == "flags":
        positions[0, 0, 0, 1] = 0.5
    elif mutation == "nonfinite":
        positions[0, 0, 0, 0] = torch.nan
    elif mutation == "nonmonotone":
        positions[0, 0, 2, 0] = -0.1
    else:
        positions[0, 0, 0, 1] = 0
    with pytest.raises(ValueError):
        pairs(positions, mask)


def test_auxiliary_initialization_symmetry_and_strict_reload():
    rng = torch.get_rng_state().clone()
    head = SliceDistanceHead(8, seed=44)
    assert torch.equal(torch.get_rng_state(), rng)
    first, second = torch.randn(4, 8), torch.randn(4, 8)
    torch.testing.assert_close(head(first, second), head(second, first), rtol=0, atol=0)
    restored = SliceDistanceHead(8, seed=91)
    restored.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(head(first, second), restored(first, second), rtol=0, atol=0)


def test_real_adapter_update_keeps_base_frozen_and_excluded_inputs_without_gradient():
    positions, mask = geometry()
    mask[0, 0, 4] = False
    positions[~mask] = torch.nan
    relations = slice_relations(
        positions, mask, torch.ones(2, 2, dtype=torch.bool), torch.tensor([True, False]), seed=4
    )
    encoder = ResidualLinear(nn.Linear(3, 8), rank=2, alpha=2, dropout=0)
    original = encoder.weight.detach().clone()
    residual = encoder.lora_B.detach().clone()
    images = torch.randn(2, 2, 5, 3, requires_grad=True)
    head = SliceDistanceHead(8)
    optimizer = torch.optim.SGD([p for p in encoder.parameters() if p.requires_grad], lr=0.2)
    loss = head.loss(encoder(images), relations)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert encoder.weight.grad is None
    assert not images.grad[1].any() and not images.grad[~mask].any()
    assert images.grad[0, 0, :4].abs().sum() > 0
    optimizer.step()
    assert not torch.equal(encoder.lora_B, residual)
    assert torch.equal(encoder.weight, original)


def test_empty_geometry_is_finite_differentiable_zero_even_with_nan_padding():
    positions, mask = geometry()
    mask.zero_()
    positions[:] = torch.nan
    features = torch.full((2, 2, 5, 8), torch.nan, requires_grad=True)
    loss = SliceDistanceHead(8).loss(features, pairs(positions, mask))
    assert loss.item() == 0
    loss.backward()
    assert features.grad is not None and not features.grad.any()


def test_image_only_distance_signal_is_learnable_on_new_synthetic_pairs():
    # Engineering check only: feature already encodes synthetic position.
    # Disjoint test coordinates prevent memorization of this tiny pair roster.
    head = SliceDistanceHead(2, hidden_dim=16, seed=19)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.02)
    train = torch.linspace(-0.2, 0.2, 9)
    grid = torch.combinations(train, r=2)
    first = torch.stack((grid[:, 0], grid[:, 0].square()), -1)
    second = torch.stack((grid[:, 1], grid[:, 1].square()), -1)
    targets = torch.log1p((grid[:, 0] - grid[:, 1]).abs() * 100) / torch.log1p(torch.tensor(100.0))
    for _ in range(180):
        optimizer.zero_grad()
        (head(first, second) - targets).square().mean().backward()
        optimizer.step()
    test = torch.combinations(torch.linspace(-0.19, 0.19, 8), r=2)
    expected = torch.log1p((test[:, 0] - test[:, 1]).abs() * 100) / torch.log1p(torch.tensor(100.0))
    actual = head(
        torch.stack((test[:, 0], test[:, 0].square()), -1),
        torch.stack((test[:, 1], test[:, 1].square()), -1),
    )
    assert (actual - expected).square().mean() < (targets.mean() - expected).square().mean() * 0.25
