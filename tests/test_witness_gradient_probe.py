"""Initial ties, invalid padding and bag duplication are mechanism checks."""

import pytest
import torch

from scripts.research.witness_gradient_probe import normalized_smooth_evidence, probe


def test_exact_zero_start_covers_all_genuine_score_cells_without_changing_logits():
    result = probe()["cases"]
    hard = result["original_hard_top4_patch_top2_slice"]
    soft = result["standalone_normalized_smooth_beta8"]
    assert hard["output_exactly_zero"] and soft["output_exactly_zero"]
    assert hard["nonzero_gradient_cells"] == 576
    assert soft["nonzero_gradient_cells"] == 663552
    assert soft["finite_gradients"]


def test_invalid_padding_is_not_evidence_and_cannot_poison_gradients():
    scores = torch.tensor(
        [[[[[1.0, -1.0], [0.0, 0.0]], [[float("nan")] * 2] * 2]]], requires_grad=True
    )
    mask = torch.tensor([[[True, False]]])
    out = normalized_smooth_evidence(scores, mask)
    ref = normalized_smooth_evidence(scores[:, :, :1], mask[:, :, :1])
    torch.testing.assert_close(out, ref)
    out.sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert not scores.grad[:, :, 1].any()


def test_repeating_identical_series_does_not_manufacture_more_evidence():
    torch.manual_seed(3)
    scores = torch.randn(1, 1, 3, 7, 12)
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    torch.testing.assert_close(
        normalized_smooth_evidence(scores, mask),
        normalized_smooth_evidence(scores.repeat(1, 4, 1, 1, 1), mask.repeat(1, 4, 1)),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pooling_preserves_branch_precision_and_finite_zero_start(dtype):
    scores = torch.zeros(1, 2, 3, 7, 12, dtype=dtype, requires_grad=True)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    output = normalized_smooth_evidence(scores, mask)
    assert output.dtype == dtype and output.eq(0).all()
    output.sum().backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.ne(0).all()


def test_signed_scores_and_strong_local_evidence_survive_stable_pooling():
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    scores = torch.full((1, 1, 1, 576, 1), -2.0)
    torch.testing.assert_close(normalized_smooth_evidence(scores, mask), torch.tensor([[-2.0]]))
    scores[0, 0, 0, 0, 0] = 8
    out = normalized_smooth_evidence(scores, mask)
    assert 7 < out.item() < 8
    out = normalized_smooth_evidence(scores * 1000, mask)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("beta", [0, -1, float("nan"), float("inf"), True])
def test_invalid_sharpness_is_rejected(beta):
    with pytest.raises(ValueError):
        normalized_smooth_evidence(
            torch.zeros(1, 1, 1, 1, 1), torch.ones(1, 1, 1, dtype=torch.bool), beta
        )
