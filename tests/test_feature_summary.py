"""A conventional feature readout must preserve initial state and missing-image masks."""

import pytest
import torch
from torch.nn import functional as F

from rsna.competitive.backbones import create_encoder
from rsna.competitive.feature_summary import PatchMeanProjection, validate_patch_summary
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig


def model(weight=None):
    torch.manual_seed(42)
    encoder, dim, _ = create_encoder({"architecture": "diagnostic_tiny", "embedding_dim": 8})
    return CompetitiveStudyNet(
        encoder,
        dim,
        hidden_dim=16,
        encode_chunk_size=8,
        patch_summary=None if weight is None else {"weight": weight},
    )


def arguments():
    torch.manual_seed(43)
    images = torch.rand(1, 3, 2, 1, 16, 16)
    features = torch.zeros(1, 3, 5)
    features[0, :, :3] = torch.eye(3)
    return (
        images,
        features,
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 3, 2, dtype=torch.bool),
        torch.zeros(1, 3, 2, 2),
    )


def test_common_initialization_matches_legacy_values_rng_and_outputs():
    baseline = model()
    rng = torch.get_rng_state()
    candidate = model(1)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name])
    control = model(0)
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, control.state_dict()[name])
    inputs = arguments()
    assert torch.equal(baseline(*inputs), control(*inputs))
    assert torch.equal(baseline(*inputs), candidate(*inputs))
    candidate(*inputs).square().sum().backward()
    assert candidate.patch_summary.weight.grad.abs().sum() > 0
    control(*inputs).square().sum().backward()
    assert not control.patch_summary.weight.grad.any()


def test_split_projection_matches_concatenated_feature_algebra():
    torch.manual_seed(9)
    cls, patches = torch.randn(4, 8), torch.randn(4, 2, 3, 8)
    first, second = torch.randn(16, 8), PatchMeanProjection(8, 16, {"weight": 1})
    with torch.no_grad():
        second.weight.copy_(torch.randn_like(second.weight))
    actual = F.linear(cls, first) + second(patches)
    expected = F.linear(
        torch.cat([cls, patches.mean((1, 2))], -1), torch.cat([first, second.weight], -1)
    )
    torch.testing.assert_close(actual, expected)


def test_nan_padding_stays_outside_summary_and_classification():
    m = model(1)
    with torch.no_grad():
        m.patch_summary.weight.copy_(torch.randn_like(m.patch_summary.weight))
    inputs = list(arguments())
    inputs[3][0, 0, 1] = False
    expected = m(*inputs)
    inputs[0][0, 0, 1] = float("nan")
    assert torch.equal(expected, m(*inputs))


@pytest.mark.parametrize(
    "policy",
    [{"weight": True}, {"weight": 0.5}, {"weight": float("nan")}, {"weight": 1, "extra": 2}],
)
def test_fixed_readout_policy_rejects_drift(policy):
    with pytest.raises(ValueError):
        validate_patch_summary(policy)


def test_baseline_readout_does_not_mix_unproved_mechanisms():
    TrainConfig(diagnostic_study_limit=16, patch_summary={"weight": 1}).validate()
    with pytest.raises(ValueError):
        TrainConfig(
            diagnostic_study_limit=16, patch_summary={"weight": 1}, pooling="patch_attention"
        ).validate()
    with pytest.raises(ValueError):
        TrainConfig(
            diagnostic_study_limit=16,
            patch_summary={"weight": 1},
            plane_memory={"weight": 1, "seed": 1, "hidden_dim": 16},
        ).validate()
