import pytest
import torch
from torch import nn

from rsna.competitive.compact_bag import CompactBagClassifier, candidate_windows, sample_windows
from rsna.competitive.compact_mc import BAGS, RUN_SEED, VIEW_OFFSET, predict_both


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x.mean(dim=(2, 3))


def inputs(slices=16):
    model = CompactBagClassifier(Encoder(), dimension=3).eval()
    series = torch.tensor([True, True])
    mask = torch.ones((2, slices), dtype=torch.bool)
    count = len(candidate_windows(series, mask))
    windows = torch.randn(count, 3, 2, 2)
    return model, windows, series, mask


def test_mc_matches_explicit_bags_with_only_one_encoder_pass():
    model, windows, series, mask = inputs()
    state = torch.random.get_rng_state().clone()
    result = predict_both(model, windows, series, mask, study_uid="test", chunk_size=7)
    assert model.encoder.calls == 4
    assert torch.equal(state, torch.random.get_rng_state())
    candidates = candidate_windows(series, mask)
    lookup = {w: i for i, w in enumerate(candidates)}
    with torch.inference_mode():
        features = windows.mean(dim=(2, 3))
        explicit = torch.stack(
            [
                model.pool(
                    features[
                        [
                            lookup[w]
                            for w in sample_windows(
                                series,
                                mask,
                                run_seed=RUN_SEED,
                                epoch=VIEW_OFFSET + view,
                                study_uid="test",
                            )
                        ]
                    ]
                ).sigmoid()
                for view in range(BAGS)
            ]
        ).mean(0)
    torch.testing.assert_close(result["mc128"], explicit, atol=1e-6, rtol=1e-6)
    with torch.inference_mode():
        torch.testing.assert_close(result["full"], model.pool(features).sigmoid())
    repeat = predict_both(model, windows, series, mask, study_uid="test", chunk_size=7)
    assert torch.equal(result["mc128"], repeat["mc128"])


def test_small_bags_identical():
    model, windows, series, mask = inputs(slices=7)
    result = predict_both(model, windows, series, mask, study_uid="test")
    assert torch.equal(result["full"], result["mc128"])


def test_train_mode_and_incomplete_windows_refused():
    model, windows, series, mask = inputs()
    model.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        predict_both(model, windows, series, mask, study_uid="test")
    model.eval()
    with pytest.raises(ValueError, match="canonical"):
        predict_both(model, windows[:-1], series, mask, study_uid="test")
