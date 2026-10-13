import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsna.competitive import focal_window as fw  # noqa: E402
from rsna.competitive.loss import effective_batch_loss, observed_counts  # noqa: E402
from rsna.schema import TARGETS  # noqa: E402

CAP, RHO, K = fw.LOGIT_CAP, fw.UNIFORM_FLOOR, fw.WINDOWS


def aggregate_confidence(probabilities, window_logits):
    """Issue #35 aggregate weight (commit c3431c00 teacher_evidence.confidence)."""
    windows = torch.sigmoid(window_logits.float())
    spread = windows.amax(1) - windows.amin(1)
    return ((2 * probabilities.float() - 1).abs() - spread).clamp(0, 1)


def keyed_pool(floor=RHO):
    """Hidden 4, one target: dim0 drives window attention, dim1 is the evidence logit."""
    pool = fw.FocalWindowPool(4, 1, uniform_floor=floor)
    with torch.no_grad():
        for p in pool.parameters():
            p.zero_()
        pool.window_tanh.weight[0, 0] = 5.0
        pool.window_gate.bias.fill_(20.0)
        pool.window_query[0, 0] = 10.0
        pool.output_weight[0, 1] = 1.0
    return pool


def study(evidence, selector, slots=1):
    """One study, `slots` valid identical slots, per-window (selector, evidence)."""
    x = torch.zeros(1, fw.SLOTS, K, 4)
    x[0, :slots, :, 0] = torch.as_tensor(selector, dtype=torch.float32)
    x[0, :slots, :, 1] = torch.as_tensor(evidence, dtype=torch.float32)
    mask = torch.zeros(1, fw.SLOTS, K, dtype=torch.bool)
    mask[0, :slots] = True
    return x, mask


def focal(value=4.0, background=-2.0, at=3):
    evidence = [background] * K
    evidence[at] = value
    selector = [-1.0] * K
    selector[at] = 1.0
    return evidence, selector


def capped(v):
    return CAP * np.tanh(v / CAP)


def test_window_identity_matches_exact_source_and_student_adapter():
    from tools.research.public_dinov2_input_reference import dino_adapter, reference

    namespace, _ = reference.load()
    reference.configure(
        namespace,
        {
            **reference.MANIFEST_BOUND_PIXEL_GROUP,
            "slots": [
                "SAG_FLUID",
                "COR_FLUID",
                "AX_FLUID",
                "SAG_STRUCT",
                "COR_STRUCT",
                "AX_STRUCT",
            ],
        },
    )
    assert list(namespace["window_starts"](fw.CACHE_SLICES, fw.GROUP)) == list(fw.WINDOW_STARTS)
    assert namespace["GROUP"] == fw.GROUP and namespace["CACHE_SLICES"] == fw.CACHE_SLICES
    generator = torch.Generator().manual_seed(37)
    images = torch.randint(0, 256, (6, 12, 336, 336), dtype=torch.uint8, generator=generator)
    slot_mask = torch.tensor([True, True, False, True, False, True])
    images[~slot_mask] = 0
    positions = torch.zeros(6, 12, 2)
    positions[slot_mask, :, 0] = torch.arange(12, dtype=torch.float32) / 100
    positions[slot_mask, :, 1] = 1
    features = torch.zeros(6, 5)
    features[slot_mask, 0] = 1
    student = dino_adapter.study_inputs(
        images, slot_mask, features, positions, torch.zeros(12), torch.ones(12), study_uid="s"
    )
    source = reference.windows(namespace, images)
    assert source.shape[0] == K == student["images"].shape[1]
    for k in fw.WINDOW_STARTS:
        assert torch.equal(source[k], images[:, list(fw.window_slices(k))])
        assert torch.equal(student["images"][:, k].mul(255).round().to(torch.uint8), source[k])
        center = fw.window_slices(k)[1]
        assert torch.equal(student["positions"][:, k], positions[:, center])
    with pytest.raises(ValueError):
        fw.window_slices(10)


def test_parameters_match_replaced_attention_head_and_init_is_plain_mean():
    pool = fw.FocalWindowPool(128)
    assert sum(p.numel() for p in pool.parameters()) == 70_668
    torch.manual_seed(0)
    x = torch.randn(2, 6, K, 128)
    mask = torch.ones(2, 6, K, dtype=torch.bool)
    mask[1, 2:] = False
    series = torch.randn(2, 6, 128)
    z, u, beta = pool(x, mask, series)
    torch.testing.assert_close(beta, torch.full_like(beta, 1 / K))
    torch.testing.assert_close(z, u.mean(1))
    e = (x + series[:, :, None]) * mask[..., None]
    mean = e.sum(1) / mask.sum(1)[..., None]
    raw = torch.einsum("bkh,th->bkt", mean, pool.output_weight) + pool.output_bias
    torch.testing.assert_close(u, CAP * torch.tanh(raw / CAP))
    z.sum().backward()
    assert pool.output_weight.grad.abs().sum() > 0
    assert list(inspect.signature(pool.forward).parameters) == [
        "features",
        "slice_mask",
        "series_features",
    ]


def test_queries_learn_from_zero_initialization():
    pool = fw.FocalWindowPool(16, 2)
    torch.manual_seed(1)
    x = torch.randn(1, 6, K, 16)
    z, _, _ = pool(x, torch.ones(1, 6, K, dtype=torch.bool))
    z.sum().backward()
    assert pool.window_query.grad.abs().sum() > 0 and pool.slot_query.grad.abs().sum() > 0


def test_one_focal_positive_survives_where_mean_pooling_dilutes_it():
    evidence, selector = focal()
    z, u, beta = keyed_pool()(*study(evidence, selector))
    z_mean, _, _ = keyed_pool(floor=1.0)(*study(evidence, selector))
    expected_mean = (capped(4.0) + 9 * capped(-2.0)) / K
    assert z_mean.item() == pytest.approx(expected_mean, abs=1e-5) and z_mean.item() < 0
    assert beta[0, 3, 0].item() == pytest.approx(1 - RHO + RHO / K, abs=1e-4)
    assert z.item() > 2.0 and z.item() < CAP
    assert u[0, 3, 0].item() == pytest.approx(capped(4.0), abs=1e-5)


def test_focal_teacher_window_is_zeroed_by_aggregate_weight_but_kept_per_window():
    logits = torch.full((1, K, len(TARGETS)), -4.0)
    logits[0, 3, TARGETS.index("ACL")] = 4.0
    probability = torch.sigmoid(logits).mean(1)
    assert aggregate_confidence(probability, logits)[0, TARGETS.index("ACL")] == 0
    arrays = {
        "study_uid": np.array(["a"]),
        "window_logits": logits.numpy().astype(np.float32),
        "slot_mask": np.ones((1, 6), dtype=np.bool_),
    }
    soft, weight = fw.window_teacher_targets(["a"], arrays, torch.ones(1, 6, dtype=torch.bool))
    acl = TARGETS.index("ACL")
    assert soft[0, 3, acl] > 0.98 and soft[0, 2, acl] < 0.02
    torch.testing.assert_close(
        weight[0, :, acl], torch.full((K,), 2 * torch.sigmoid(torch.tensor(4.0)).item() - 1)
    )


def test_diffuse_weak_positives_are_not_amplified_into_certainty():
    z, u, _ = keyed_pool()(*study([0.5] * K, [float(i % 2) for i in range(K)]))
    assert z.item() == pytest.approx(capped(0.5), abs=1e-5)
    noisy_or = 1 - (1 - torch.sigmoid(torch.tensor(0.5))) ** K
    assert torch.sigmoid(z).item() < 0.63 < 0.99 < noisy_or.item()
    _, weight = fw.window_teacher_targets(
        ["a"],
        {
            "study_uid": np.array(["a"]),
            "window_logits": np.full((1, K, 12), 0.5, np.float32),
            "slot_mask": np.ones((1, 6), np.bool_),
        },
        torch.ones(1, 6, dtype=torch.bool),
    )
    assert weight.max().item() < 0.25


def test_one_extreme_outlier_window_is_bounded_and_gradients_stay_finite():
    evidence, selector = focal(value=1e6)
    x, mask = study(evidence, selector)
    x.requires_grad_(True)
    pool = keyed_pool()
    z, u, _ = pool(x, mask)
    bound = (1 - RHO + RHO / K) * CAP + (RHO * (K - 1) / K) * capped(-2.0)
    assert u.max().item() <= CAP and z.item() <= bound + 1e-5 and torch.sigmoid(z) < 0.99
    z.sum().backward()
    assert torch.isfinite(x.grad).all()
    unselected, _ = focal(value=1e6)
    z_flat, _, _ = keyed_pool()(*study(unselected, [0.0] * K))
    assert z_flat.item() == pytest.approx((CAP + 9 * capped(-2.0)) / K, abs=1e-4)
    assert z_flat.item() < 0


def test_empty_slots_and_windows_never_contribute():
    pool = fw.FocalWindowPool(8, 3)
    torch.manual_seed(2)
    with torch.no_grad():
        pool.window_query.normal_()
        pool.slot_query.normal_()
    x = torch.randn(1, 6, K, 8)
    mask = torch.zeros(1, 6, K, dtype=torch.bool)
    mask[0, [0, 2]] = True
    mask[0, :, 7] = False
    z, u, beta = pool(x, mask)
    poisoned = x.clone()
    poisoned[~mask] = float("nan")
    series = torch.zeros(1, 6, 8)
    series[0, [1, 3, 4, 5]] = float("nan")
    z2, u2, beta2 = pool(poisoned, mask)
    assert torch.equal(z, z2) and torch.equal(u, u2) and torch.equal(beta, beta2)
    assert (beta[0, 7] == 0).all() and (u[0, 7] == 0).all()
    torch.testing.assert_close(beta.sum(1), torch.ones(1, 3))
    z3, _, _ = pool(x, mask, series)
    torch.testing.assert_close(z3, pool(x, mask, torch.zeros(1, 6, 8))[0])
    with pytest.raises(ValueError, match="empty"):
        pool(x, torch.zeros(1, 6, K, dtype=torch.bool))
    with pytest.raises(ValueError):
        pool(x, mask.float())


def test_teacher_targets_refuse_mismatched_slots_and_zero_absent_studies():
    arrays = {
        "study_uid": np.array(["a", "b"]),
        "window_logits": np.zeros((2, K, 12), np.float32),
        "slot_mask": np.array([[1, 1, 0, 1, 0, 1], [1] * 6], dtype=np.bool_),
    }
    student = torch.tensor([[1, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    soft, weight = fw.window_teacher_targets(["a", "z"], arrays, student)
    assert (weight[1] == 0).all() and (soft[1] == 0).all()
    with pytest.raises(ValueError, match="slot masks"):
        fw.window_teacher_targets(["a", "b"], arrays, student)
    for name, value in [
        ("window_logits", np.zeros((2, K, 12), np.float64)),
        ("window_logits", np.zeros((2, 9, 12), np.float32)),
        ("window_logits", np.full((2, K, 12), np.nan, np.float32)),
        ("slot_mask", np.ones((2, 6), np.int64)),
        ("study_uid", np.array(["a", "a"])),
    ]:
        with pytest.raises(ValueError):
            fw.window_teacher_targets(["a"], {**arrays, name: value}, student[:1])


def test_anatomical_variability_is_distinct_from_same_anatomy_transformation_spread():
    logits = torch.full((1, K, 1), -4.0)
    logits[0, 3] = 4.0
    out = fw.window_variability(logits)
    assert out["anatomical_window_range"].item() > 0.9 and out["transformation_spread"] is None
    same = fw.window_variability(logits, torch.stack([logits, logits]))
    assert (same["transformation_spread"] == 0).all()
    flat = torch.zeros(1, K, 1)
    noisy = fw.window_variability(flat, torch.stack([flat - 2, flat + 2]))
    assert noisy["anatomical_window_range"].item() == 0
    assert (noisy["transformation_spread"] > 0.7).all()
    with pytest.raises(ValueError):
        fw.window_variability(flat, flat[None])
    arrays = {
        "study_uid": np.array(["a"]),
        "window_logits": logits.expand(1, K, 12).numpy().astype(np.float32).copy(),
        "slot_mask": np.ones((1, 6), np.bool_),
    }
    _, focal_weight = fw.window_teacher_targets(["a"], arrays, torch.ones(1, 6, dtype=torch.bool))
    arrays["window_logits"][0, [0, 1, 2, 4, 5, 6, 7, 8, 9]] = 4.0
    _, flat_weight = fw.window_teacher_targets(["a"], arrays, torch.ones(1, 6, dtype=torch.bool))
    assert torch.equal(focal_weight[0, 3], flat_weight[0, 3])


def test_window_loss_normalizes_like_label_loss_and_zero_weight_is_identity():
    torch.manual_seed(3)
    logits = torch.randn(2, K, 12, requires_grad=True)
    soft = torch.rand(2, K, 12)
    weight = torch.rand(2, K, 12)
    weight[1] = 0
    counts = observed_counts([weight.flatten(0, 1)])
    loss = fw.window_teacher_loss(logits, soft, weight, counts)
    expected = effective_batch_loss(logits[0], soft[0], weight[0], counts)
    torch.testing.assert_close(loss, expected)
    label = torch.tensor(0.7, requires_grad=True)

    def refuse():
        raise AssertionError("zero-weight arm must not evaluate the teacher term")

    assert fw.combined_loss(label, refuse, 0) is label
    assert fw.combined_loss(label, refuse, 0.0) is label
    torch.testing.assert_close(fw.combined_loss(label, lambda: loss, 0.5), label + 0.5 * loss)
    for bad in (True, -0.1, 1.5, float("nan"), "0.5"):
        with pytest.raises(ValueError):
            fw.combined_loss(label, refuse, bad)
