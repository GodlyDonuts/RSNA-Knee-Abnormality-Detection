import pytest
import torch
from torch import nn

from rsna.competitive.context import NeighborSliceContext
from rsna.competitive.model import CompetitiveStudyNet


class RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 8)

    def forward(self, images):
        return self.linear(images.mean((2, 3)))


def inputs():
    torch.manual_seed(12)
    images = torch.rand(2, 2, 3, 1, 16, 16)
    series_mask = torch.tensor([[True, True], [True, False]])
    slice_mask = torch.tensor(
        [
            [[True, True, False], [True, False, False]],
            [[True, True, True], [False, False, False]],
        ]
    )
    return images, torch.rand(2, 2, 5), series_mask, slice_mask, torch.zeros(2, 2, 3, 2)


def context_inputs():
    torch.manual_seed(72)
    x = torch.randn(2, 3, 5, 8)
    mask = torch.tensor(
        [
            [[True] * 5, [True, True, False, False, False], [False] * 5],
            [[True, False, False, False, False], [True] * 5, [True] * 5],
        ]
    )
    positions = torch.zeros(2, 3, 5, 2)
    positions[..., 0] = torch.arange(5) * 0.02
    positions[..., 1] = 1
    return x, mask, positions


def test_zero_context_exactly_reproduces_valid_features_and_original_model():
    x, mask, positions = context_inputs()
    module = NeighborSliceContext(8)
    torch.testing.assert_close(module(x, mask, positions)[mask], x[mask], rtol=0, atol=0)
    reference = CompetitiveStudyNet(RecordingEncoder(), 8, hidden_dim=12).eval()
    candidate = CompetitiveStudyNet(
        RecordingEncoder(), 8, hidden_dim=12, slice_context="neighbor3"
    ).eval()
    result = candidate.load_state_dict(reference.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys and all(k.startswith("slice_context.") for k in result.missing_keys)
    torch.testing.assert_close(candidate(*inputs()), reference(*inputs()), rtol=0, atol=0)


def test_context_preserves_shared_initial_state_and_random_stream_for_matched_control():
    torch.manual_seed(384)
    reference = CompetitiveStudyNet(RecordingEncoder(), 8, hidden_dim=12).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(384)
    candidate = CompetitiveStudyNet(
        RecordingEncoder(), 8, hidden_dim=12, slice_context="neighbor3"
    ).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(candidate(*inputs()), reference(*inputs()), rtol=0, atol=0)


@pytest.mark.parametrize("autocast", [False, True])
def test_learned_context_ignores_padding_and_does_not_cross_series(autocast):
    x, mask, positions = context_inputs()
    module = NeighborSliceContext(8).eval()
    nn.init.eye_(module.output.weight)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        original = module(x, mask, positions)
        changed = x.clone()
        changed[0, 0, 0, 2] += 3
        changed[~mask] = torch.nan
        padded_positions = positions.clone()
        padded_positions[~mask] = torch.nan
        result = module(changed, mask, padded_positions)
    assert torch.isfinite(result).all()
    assert not torch.equal(result[0, 0, 1], original[0, 0, 1])
    torch.testing.assert_close(result[0, 0, 2:], original[0, 0, 2:], rtol=0, atol=0)
    torch.testing.assert_close(result[0, 1:], original[0, 1:], rtol=0, atol=0)
    torch.testing.assert_close(result[1], original[1], rtol=0, atol=0)
    assert not result[~mask].any()


def test_context_is_order_sensitive_and_strictly_reloadable():
    x, mask, positions = context_inputs()
    module = NeighborSliceContext(8).eval()
    nn.init.eye_(module.output.weight)
    # Unknown physical geometry still retains source ordinal order.
    positions.zero_()
    original = module(x, mask, positions)
    permuted = x.clone()
    permuted[0, 0] = x[0, 0].flip(0)
    result = module(permuted, mask, positions)
    assert not torch.allclose(result[0, 0].flip(0), original[0, 0], atol=1e-6)
    restored = NeighborSliceContext(8).eval()
    restored.load_state_dict(module.state_dict(), strict=True)
    torch.testing.assert_close(restored(x, mask, positions), original, rtol=0, atol=0)


def test_context_gradient_reaches_neighbors_with_no_padding_gradient():
    x, mask, positions = context_inputs()
    x.requires_grad_()
    module = NeighborSliceContext(8)
    nn.init.eye_(module.output.weight)
    module(x, mask, positions)[0, 0, 1].square().sum().backward()
    assert torch.count_nonzero(x.grad[0, 0, 0]) > 0
    assert torch.count_nonzero(x.grad[0, 0, 2]) > 0
    assert not x.grad[~mask].any() and not x.grad[0, 1:].any()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("change", ["reverse_geometry", "unknown_distance", "flag", "nan_valid"])
def test_bad_context_input_is_rejected(change):
    x, mask, positions = context_inputs()
    if change == "reverse_geometry":
        positions[0, 0, :, 0] = positions[0, 0, :, 0].flip(0)
    elif change == "unknown_distance":
        positions[0, 0, 1, 1] = 0
    elif change == "flag":
        positions[0, 0, 1, 1] = 0.5
    else:
        x[0, 0, 1, 0] = torch.nan
    with pytest.raises(ValueError):
        NeighborSliceContext(8)(x, mask, positions)
