import copy

import pytest
import torch

from rsna.competitive.backbones import create_encoder
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig
from rsna.competitive.witness import pool_witness_evidence, validate_witness


def batch():
    images = torch.rand(2, 2, 3, 1, 32, 32)
    mask = torch.tensor([[[1, 1, 0], [1, 0, 0]], [[1, 1, 1], [0, 0, 0]]]).bool()
    return images, torch.rand(2, 2, 5), mask.any(2), mask, torch.zeros(2, 2, 3, 2)


def net(weight, pooling=None):
    torch.manual_seed(7)
    encoder, dim, _ = create_encoder({"architecture": "diagnostic_tiny", "embedding_dim": 16})
    return CompetitiveStudyNet(
        encoder,
        dim,
        hidden_dim=12,
        encode_chunk_size=2,
        witness=dict(
            patches=2,
            slices=2,
            seed=11,
            weight=weight,
            **({} if pooling is None else {"pooling": pooling, "beta": 8}),
        ),
    )


@pytest.mark.parametrize("pooling", [None, "normalized_logmeanexp"])
def test_shared_initial_state_rng_and_logits_equal_across_witness_arms(pooling):
    control = net(0, pooling)
    control_rng = torch.get_rng_state().clone()
    candidate = net(1, pooling)
    assert torch.equal(control_rng, torch.get_rng_state())
    assert control.state_dict().keys() == candidate.state_dict().keys()
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    inputs = batch()
    torch.testing.assert_close(control(*inputs), candidate(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize("pooling", [None, "normalized_logmeanexp"])
def test_padding_cannot_create_witness_and_all_parameters_receive_finite_gradients(pooling):
    model = net(1, pooling)
    inputs = list(batch())
    inputs[0].requires_grad_()
    # A learned head is needed to exercise projection and nonselected-image gradients.
    with torch.no_grad():
        model.witness.score.weight.normal_()
    expected = model(*inputs)
    changed = [x.clone() for x in inputs]
    changed[0][~inputs[3]] = torch.nan
    changed[1][~inputs[2]] = torch.nan
    changed[4][~inputs[3]] = torch.nan
    torch.testing.assert_close(model(*changed), expected, rtol=0, atol=0)
    expected.square().mean().backward()
    assert not inputs[0].grad[~inputs[3]].any()
    for name, value in model.named_parameters():
        if value.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all(), name
    restored = net(1, pooling)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(*inputs), expected, rtol=0, atol=0)


def test_focal_evidence_is_not_diluted_by_unselected_background_or_padding():
    scores = torch.full((1, 1, 1, 4, 2), -2.0)
    scores[..., :2, 0] = 3
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    expected = pool_witness_evidence(scores, mask, 2, 2)
    torch.testing.assert_close(expected, torch.tensor([[3.0, -2.0]]))
    enlarged = torch.full((1, 3, 6, 12, 2), torch.nan)
    enlarged[0, 0, 0] = -2
    enlarged[0, 0, 0, :4] = scores[0, 0, 0]
    enlarged_mask = torch.zeros(1, 3, 6, dtype=torch.bool)
    enlarged_mask[0, 0, 0] = True
    torch.testing.assert_close(pool_witness_evidence(enlarged, enlarged_mask, 2, 2), expected)


@pytest.mark.parametrize("pooling", [None, "normalized_logmeanexp"])
def test_bf16_backward_with_variable_real_slice_counts(pooling):
    model = net(1, pooling)
    inputs = batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(*inputs)
    assert torch.isfinite(logits).all()
    logits.float().square().sum().backward()
    for value in model.parameters():
        if value.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all()


@pytest.mark.parametrize(
    "field,value",
    [
        ("patches", 0),
        ("slices", True),
        ("seed", -1),
        ("weight", float("nan")),
        ("weight", True),
        ("weight", 0.5),
        ("extra", 1),
    ],
)
def test_reject_unaccepted_witness_policy(field, value):
    policy = {"patches": 4, "slices": 2, "seed": 20261003, "weight": 1}
    policy[field] = value
    with pytest.raises(ValueError):
        validate_witness(policy)


def test_config_rejects_confounded_architecture_and_relation_combination():
    policy = {"patches": 4, "slices": 2, "seed": 20261003, "weight": 1}
    TrainConfig(witness=policy, diagnostic_study_limit=8).validate()
    for changes in ({"pooling": "patch_attention"}, {"slice_context": "neighbor3"}):
        config = TrainConfig(witness=copy.deepcopy(policy), diagnostic_study_limit=8, **changes)
        with pytest.raises(ValueError):
            config.validate()
