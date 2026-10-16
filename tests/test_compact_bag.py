import copy

import pytest
import torch
from torch import nn

from rsna.competitive.compact_bag import (
    CompactBagClassifier,
    bag_seed,
    candidate_windows,
    disable_encoder_dropout,
    sample_windows,
    window_tensor,
)


def masks(counts):
    return torch.tensor([n > 0 for n in counts]), torch.arange(16)[None] < torch.tensor(counts)[
        :, None
    ]


def test_complete_native_study_and_exact_channels():
    sm, km = masks([16] * 6)
    candidates = candidate_windows(sm, km)
    assert len(candidates) == 84
    assert candidates[13:16] == [(0, 13), (1, 0), (1, 1)]
    images = torch.arange(6 * 16).reshape(6, 16, 1, 1, 1).float() / 95
    out = window_tensor(images, [(0, 13), (1, 0)])
    assert out.shape == (2, 3, 1, 1)
    assert torch.equal(out.flatten(), torch.tensor([13, 14, 15, 16, 17, 18]) / 95 * 2 - 1)


def test_stratification_is_deterministic_independent_and_without_replacement():
    sm, km = masks([16] * 6)
    state = torch.get_rng_state().clone()
    first = sample_windows(sm, km, run_seed=20260922, epoch=1, study_uid="private-fixture")
    assert torch.equal(state, torch.get_rng_state())
    assert first == sample_windows(sm, km, run_seed=20260922, epoch=1, study_uid="private-fixture")
    assert len(set(first)) == 12
    assert [sum(s == i for s, _ in first) for i in range(6)] == [2] * 6
    assert first != sample_windows(sm, km, run_seed=20260922, epoch=2, study_uid="private-fixture")
    assert first != sample_windows(sm, km, run_seed=20260922, epoch=1, study_uid="other")


@pytest.mark.parametrize("counts", [[3, 2, 0], [4, 5, 0], [16, 3, 2, 0], [16, 4, 4, 4, 4, 4]])
def test_missing_short_series_and_fill(counts):
    sm, km = masks(counts)
    all_windows = candidate_windows(sm, km)
    picked = sample_windows(sm, km, run_seed=7, epoch=0, study_uid="x")
    assert len(picked) == min(12, len(all_windows))
    assert len(set(picked)) == len(picked)
    assert set(picked) <= set(all_windows)
    for s, count in enumerate(counts):
        assert sum(i == s for i, _ in picked) >= min(2, max(0, count - 2))
    if len(all_windows) <= 12:
        assert picked == all_windows


def test_padding_holes_and_empty_bags_are_rejected():
    sm, km = masks([2, 1])
    with pytest.raises(ValueError, match="no valid"):
        candidate_windows(sm, km)
    sm, km = masks([6, 0])
    km[0, 2] = False
    with pytest.raises(ValueError, match="precede padding"):
        candidate_windows(sm, km)
    sm, km = masks([6, 0])
    sm[0] = False
    with pytest.raises(ValueError, match="Inactive"):
        candidate_windows(sm, km)


@pytest.mark.parametrize("value", [float("nan"), -0.01, 1.01])
def test_native_normalization_refuses_corruption(value):
    with pytest.raises(ValueError):
        window_tensor(torch.full((1, 3, 1, 2, 2), value), [(0, 0)])


def test_seed_convention_and_window_bounds():
    assert bag_seed(1, 0, "x") == bag_seed(1, 0, "x")
    assert 0 <= bag_seed(-1, 0, "x") < 2**63
    with pytest.raises(ValueError):
        bag_seed(1, -1, "x")
    for windows in [[(0, 1)], [(1, 0)], [(0, 0), (0, 0)]]:
        with pytest.raises(ValueError):
            window_tensor(torch.zeros(1, 3, 1, 2, 2), windows)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.hidden_dropout = nn.Dropout(0.125)

    def forward(self, x):
        return self.hidden_dropout(self.bn(self.projection(x))).mean((2, 3))


def test_uniform_attention_pools_all_features_not_a_scalar():
    model = CompactBagClassifier(TinyEncoder(), dimension=4).eval()
    for p in model.attention.parameters():
        p.data.zero_()
    features = torch.randn(7, 4)
    pooled = model.norm(features).mean(0)
    expected = torch.cat([head(pooled) for head in model.classifiers])
    assert torch.allclose(model.pool(features), expected, atol=1e-7)


def test_trunk_and_head_train_bn_once_and_eval_chunks_preserve_saved_state():
    model = CompactBagClassifier(TinyEncoder(), dimension=4).train()
    assert model.encoder.hidden_dropout.p == 0
    assert model.dropout.p == 0.2
    assert model.encoder.bn.training
    windows = torch.randn(12, 3, 4, 4)
    before = model.encoder.projection.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    nn.functional.binary_cross_entropy_with_logits(model(windows), torch.rand(12)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    assert not torch.equal(before, model.encoder.projection.weight)
    assert model.encoder.bn.num_batches_tracked == 1
    with pytest.raises(ValueError, match="evaluation-only"):
        model(windows, chunk_size=3)
    model.eval()
    bn_before = model.encoder.bn.running_mean.clone()
    with torch.no_grad():
        assert torch.allclose(model(windows), model(windows, chunk_size=3), atol=1e-6)
        assert torch.isfinite(model(windows[:1])).all()
    assert torch.equal(bn_before, model.encoder.bn.running_mean)


def test_model_and_bn_optimizer_rng_resume_is_exact_for_real_updates():
    torch.manual_seed(8)
    model = CompactBagClassifier(TinyEncoder(), dimension=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    windows = torch.linspace(-1, 1, 12 * 3 * 4 * 4).reshape(12, 3, 4, 4)

    def update(m, optimizer):
        optimizer.zero_grad(set_to_none=True)
        nn.functional.binary_cross_entropy_with_logits(
            m(windows), torch.full((12,), 0.5)
        ).backward()
        optimizer.step()

    update(model, opt)
    saved = copy.deepcopy(
        {"model": model.state_dict(), "opt": opt.state_dict(), "rng": torch.get_rng_state()}
    )
    update(model, opt)
    resumed = CompactBagClassifier(TinyEncoder(), dimension=4)
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed.load_state_dict(saved["model"])
    resumed_opt.load_state_dict(saved["opt"])
    torch.set_rng_state(saved["rng"])
    update(resumed, resumed_opt)
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in model.state_dict().items())
    assert all(
        torch.equal(v["exp_avg"], resumed_opt.state_dict()["state"][k]["exp_avg"])
        for k, v in opt.state_dict()["state"].items()
    )
    model.eval()
    resumed.eval()
    assert torch.equal(model(windows), resumed(windows))


def test_dropout_policy_does_not_hide_stochastic_depth():
    encoder = TinyEncoder()
    assert disable_encoder_dropout(encoder) == ["hidden_dropout"]
    encoder.drop_prob = 0.1
    with pytest.raises(ValueError, match="stochastic depth"):
        disable_encoder_dropout(encoder)
