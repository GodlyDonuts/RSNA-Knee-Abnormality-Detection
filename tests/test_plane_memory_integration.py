"""Exact zero addition, metadata boundaries and shared initialization are essential."""

import pytest
import torch

from rsna.competitive.backbones import create_encoder
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.plane_memory import planes_from_features, validate_plane_memory
from rsna.competitive.train import TrainConfig


def policy(weight=1):
    return {"seed": 20261006, "weight": weight, "hidden_dim": 16}


def model(weight=None):
    torch.manual_seed(10)
    encoder, dim, _ = create_encoder({"architecture": "diagnostic_tiny", "embedding_dim": 8})
    return CompetitiveStudyNet(
        encoder,
        dim,
        hidden_dim=16,
        encode_chunk_size=8,
        plane_memory=None if weight is None else policy(weight),
    )


def test_zero_start_matches_unchanged_model_and_shared_factory_rng():
    base = model()
    state = torch.get_rng_state()
    memory = model(1)
    assert torch.equal(state, torch.get_rng_state())
    for name, value in base.state_dict().items():
        assert torch.equal(value, memory.state_dict()[name])
    torch.manual_seed(12)
    images = torch.rand(1, 3, 2, 1, 16, 16)
    features = torch.zeros(1, 3, 5)
    features[0, :, :3] = torch.eye(3)
    series = torch.ones(1, 3, dtype=torch.bool)
    mask = torch.ones(1, 3, 2, dtype=torch.bool)
    positions = torch.zeros(1, 3, 2, 2)
    arguments = (images, features, series, mask, positions)
    assert torch.equal(base(*arguments), memory(*arguments))
    zero = model(0)
    for name, p in memory.named_parameters():
        assert torch.equal(p, dict(zero.named_parameters())[name])
    output = memory(*arguments)
    output.sum().backward()
    assert memory.plane_memory.memory.score.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in memory.parameters() if p.grad is not None)


def test_original_plane_codes_and_unknown_are_not_guessed():
    features = torch.zeros(1, 4, 5)
    features[0, :3, :3] = torch.eye(3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert planes_from_features(features, mask).tolist() == [[0, 1, 2, 3]]
    features[0, 0, 1] = 1
    with pytest.raises(ValueError):
        planes_from_features(features, mask)


@pytest.mark.parametrize(
    "change", [{"seed": -1}, {"hidden_dim": 32}, {"weight": 0.5}, {"extra": True}]
)
def test_exact_policy_rejects_unsupported_drift(change):
    with pytest.raises(ValueError):
        validate_plane_memory(policy() | change)


def test_initial_memory_and_local_or_relation_mechanisms_do_not_mix():
    args = {"diagnostic_study_limit": 16, "plane_memory": policy()}
    TrainConfig(**args).validate()
    with pytest.raises(ValueError):
        TrainConfig(**args, witness={"weight": 1, "patches": 4, "slices": 2, "seed": 3}).validate()
    with pytest.raises(ValueError):
        TrainConfig(**args, slice_context="neighbor3").validate()
