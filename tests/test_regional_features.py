
import pytest
import torch

from rsna.competitive.regional_features import RegionalCapture, regional_storage, regional_summary


def test_same_global_average_does_not_erase_focal_spatial_difference():
    uniform = torch.ones(1, 1024, 10, 10)
    focal = torch.zeros_like(uniform)
    focal[:, :, 2, 2] = 100
    assert torch.equal(uniform.mean((2, 3)), focal.mean((2, 3)))
    assert not regional_summary(uniform).count_nonzero()
    regional = regional_summary(focal)
    assert (
        regional.shape == (1, 7, 1024) and regional[:, 5].min() > 0 and regional[:, 6].min() == 99
    )
    assert regional[:, 0].min() > 0 and regional[:, 3].max() < 0


def test_summary_is_invariant_to_channelwise_constant_offset():
    torch.manual_seed(717)
    x = torch.randn(2, 1024, 10, 10)
    offset = torch.randn(1, 1024, 1, 1)
    torch.testing.assert_close(
        regional_summary(x + offset), regional_summary(x), atol=1e-6, rtol=1e-5
    )


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(1, 1024, 8, 8),
        torch.zeros(1, 512, 10, 10),
        torch.zeros(0, 1024, 10, 10),
        torch.zeros(1, 1024, 10, 10, dtype=torch.float64),
        torch.full((1, 1024, 10, 10), float("nan")),
    ],
)
def test_wrong_feature_graph_or_nonfinite_values_are_rejected(bad):
    with pytest.raises(ValueError):
        regional_summary(bad)


def test_summary_does_not_modify_input_and_storage_is_explicitly_quantized():
    torch.manual_seed(871)
    x = torch.randn(2, 1024, 10, 10, requires_grad=True)
    before = x.detach().clone()
    summary = regional_summary(x)
    stored, error = regional_storage(summary)
    assert not summary.requires_grad and summary.dtype == torch.float32
    assert stored.dtype == torch.float16 and error <= 0.001
    assert torch.equal(x, before)
    with pytest.raises(ValueError):
        regional_storage(torch.full_like(summary, 1e6))


def test_owned_hook_preserves_original_output_and_forward_membership():
    class SelectAdaptivePool2d(torch.nn.Module):
        def forward(self, x):
            return x.mean((2, 3))

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Module()
            self.head.global_pool = SelectAdaptivePool2d()
            self.num_features = 1024

        def forward(self, x):
            return self.head.global_pool(x)

    base = Backbone().eval()
    capture = RegionalCapture(base)
    x = torch.randn(3, 1024, 10, 10)
    expected = base(x).clone()
    capture.begin()
    assert torch.equal(base(x), expected)
    with pytest.raises(ValueError):
        capture.begin()
    result = capture.end(3)
    assert result.shape == (3, 7, 1024) and capture.qualified
    assert capture.maximum_CPU_GPU_scaled_error == 0
    with pytest.raises(ValueError):
        capture.end(3)
    assert torch.equal(base(x), expected) and not capture.values
    capture.handle.remove()
    with pytest.raises(ValueError):
        RegionalCapture(base.train())
