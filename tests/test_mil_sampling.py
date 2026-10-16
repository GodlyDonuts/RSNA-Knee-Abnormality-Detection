import numpy as np
import pytest

from rsna.competitive.mil_sampling import benchmark, probability, proposal, sampled_probabilities


def moments():
    w = np.full((2, 100, 3), 0.7 / 99)
    w[:, 0] = 0.3
    v = np.zeros_like(w)
    v[:, 0] = [5, -5, 2]
    return w, v, np.zeros(3)


def test_all_windows_uniform_enumeration_reconstructs_exact_teacher_moments():
    w, v, b = moments()
    p = sampled_probabilities(w, v, b, np.arange(100, dtype=np.int64)[None], np.full(100, 0.01))
    assert np.allclose(p[0], probability((w * v).sum(1) + b).mean(0), atol=1e-14, rtol=0)


def test_iid_importance_correction_equals_attention_logit_offset_including_duplicates():
    w, v, b = moments()
    q = proposal(w, v, b)
    indices = np.array([[0, 0, 0, 1, 2], [3, 5, 0, 8, 8]], dtype=np.int64)
    result = sampled_probabilities(w, v, b, indices, q)
    logits = np.log(w[:, indices]) - np.log(q[indices])[None, :, :, None]
    a = np.exp(logits - logits.max(2, keepdims=True))
    a /= a.sum(2, keepdims=True)
    expected = probability((a * v[:, indices]).sum(2) + b).mean(0)
    assert np.allclose(result, expected, atol=1e-14, rtol=0)


def test_constant_teacher_has_uniform_fallback_and_no_invented_noise():
    w = np.full((2, 94, 12), 1 / 94)
    v = np.ones_like(w)
    q = proposal(w, v, np.zeros(12))
    assert np.allclose(q, np.full(94, 1 / 94))
    result = benchmark(w, v, np.zeros(12))
    assert result["uniform_per_target_MSE"].max() < 1e-25
    assert result["influence_per_target_MSE"].max() < 1e-25


def test_toy_sparse_model_influence_reduces_proxy_error_with_explicit_uniform_floor():
    w, v, b = moments()
    q = proposal(w, v, b)
    assert q.min() >= 0.25 / 100 and np.isclose(q.sum(), 1)
    r = benchmark(w, v, b, draws=4096)
    assert r["influence_per_target_MSE"].mean() < 0.2 * r["uniform_per_target_MSE"].mean()
    assert r["clinical_accuracy_evaluated"] is False


def test_proposal_and_draws_are_deterministic_without_consuming_global_rng():
    w, v, b = moments()
    state = np.random.get_state()
    a = benchmark(w, v, b)
    c = benchmark(w, v, b)
    assert np.array_equal(a["influence_per_target_MSE"], c["influence_per_target_MSE"])
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]


@pytest.mark.parametrize("change", ["weights", "vote", "floor", "index", "q"])
def test_invalid_moments_and_missing_support_are_rejected(change):
    w, v, b = moments()
    if change == "weights":
        w[0, 0, 0] = 0
    elif change == "vote":
        v[0, 0, 0] = np.nan
    if change == "floor":
        with pytest.raises(ValueError):
            proposal(w, v, b, uniform_fraction=0)
    elif change in ("index", "q"):
        q = np.full(100, 0.01)
        if change == "q":
            q[0] = 0
        with pytest.raises(ValueError):
            sampled_probabilities(
                w, v, b, np.array([[100 if change == "index" else 1]], dtype=np.int64), q
            )
    else:
        with pytest.raises(ValueError):
            proposal(w, v, b)
