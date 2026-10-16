import copy

import pytest
import torch

from rsna.competitive.released_context import (
    ContextResidual,
    anchored_loss,
    tta_probabilities,
    window_metadata,
)


def inputs():
    g = torch.Generator().manual_seed(41)
    f = torch.randn(2, 6, 8, generator=g)
    teacher = torch.randn(2, 3, generator=g)
    slots, depth = window_metadata(torch.tensor([1, 25, 26, 47, 48, 94]))
    present = torch.ones(2, 6, dtype=torch.bool)
    keep = present.clone()
    return f, teacher, slots.expand(2, -1), depth.expand(2, -1), present, keep


def model():
    return ContextResidual(8, 8, 3, layers=2)


def test_zero_start_is_exact_parent_and_does_not_consume_global_initialization_rng():
    rng = torch.get_rng_state().clone()
    m = model()
    assert torch.equal(rng, torch.get_rng_state())
    x = inputs()
    out, delta = m(*x)
    assert torch.equal(out, x[1]) and torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(tta_probabilities(out[None]), tta_probabilities(x[1][None]))


def test_correction_is_bounded_and_only_readout_moves_on_first_update():
    m = model()
    x = inputs()
    teacher = x[1].clone()
    opt = torch.optim.AdamW(m.parameters(), lr=0.1)
    labels = torch.ones_like(teacher)
    labels[0, 0] = float("nan")
    mask = torch.isfinite(labels)
    for i in range(2):
        out, delta = m(*x)
        loss = anchored_loss(out, x[1], labels, mask)
        loss.backward()
        assert torch.isfinite(loss)
        if i == 0:
            assert m.output.grad.abs().sum() > 0
            assert m.project[1].weight.grad.abs().sum() == 0
        else:
            assert m.project[1].weight.grad.abs().sum() > 0
        opt.step()
        opt.zero_grad()
        assert (delta.abs() <= 0.5).all()
    with torch.no_grad():
        m.output.fill_(1e5)
    out, delta = m(*x)
    assert (delta.abs() <= 0.5).all() and torch.isfinite(out).all()
    assert torch.equal(teacher, x[1])


def test_both_transformer_layers_have_independent_initial_weights():
    m = model()
    assert not torch.equal(m.layers[0].linear1.weight, m.layers[1].linear1.weight)


def test_mirror_metadata_reflects_only_sagittal_depth():
    c = torch.tensor([1, 25, 26, 47, 48, 65, 66, 77, 78, 94])
    slot, plain = window_metadata(c)
    other, mirror = window_metadata(c, mirrored=True)
    assert slot.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert torch.equal(slot, other)
    assert torch.equal(mirror[slot >= 2], plain[slot >= 2])
    assert torch.equal(mirror[slot < 2], 1 - plain[slot < 2])


def test_joint_token_and_metadata_permutation_preserves_context_but_false_positions_do_not():
    m = model().eval()
    with torch.no_grad():
        m.output.normal_(std=0.1)
    x = inputs()
    perm = torch.tensor([5, 4, 3, 2, 1, 0])
    other = [v[:, perm] if i != 1 else v for i, v in enumerate(x)]
    left = m(*x)[0]
    assert torch.allclose(left, m(*other)[0], atol=1e-6, rtol=1e-6)
    other[2:] = list(x[2:])
    assert not torch.allclose(left, m(*other)[0], atol=1e-6, rtol=1e-6)


def test_masked_padding_can_contain_nan_without_changing_real_context():
    m = model().eval()
    with torch.no_grad():
        m.output.normal_(std=0.1)
    x = list(inputs())
    baseline = m(*x)[0]
    for i in [0, 2, 3, 4, 5]:
        v = x[i]
        shape = list(v.shape)
        shape[1] = 2
        pad = torch.zeros(shape, dtype=v.dtype)
        if i in (0, 3):
            pad.fill_(float("nan"))
        elif i == 2:
            pad.fill_(99)
        x[i] = torch.cat([v, pad], dim=1)
    assert torch.allclose(baseline, m(*x)[0], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("change", ["empty", "slots", "depth", "feature", "parent", "dtype"])
def test_invalid_genuine_values_and_empty_bags_are_rejected(change):
    x = [v.clone() for v in inputs()]
    if change == "empty":
        x[-1][0] = False
    elif change == "slots":
        x[2][0, 0] = 5
    elif change == "depth":
        x[3][0, 0] = 1.1
    elif change == "feature":
        x[0][0, 0] = float("nan")
    elif change == "parent":
        x[1][0, 0] = float("inf")
    else:
        x[0] = x[0].half()
    with pytest.raises(ValueError):
        model()(*x)


def test_loss_has_no_teacher_grad_and_missing_source_is_teacher_only():
    logits = torch.randn(2, 3, requires_grad=True)
    teacher = torch.randn(2, 3, requires_grad=True)
    labels = torch.full((2, 3), float("nan"))
    mask = torch.zeros_like(labels, dtype=torch.bool)
    actual = anchored_loss(logits, teacher, labels, mask)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, teacher.sigmoid())
    assert torch.equal(actual, expected)
    actual.backward()
    assert teacher.grad is None and logits.grad is not None


def test_mean_probability_tta_differs_from_mean_logit():
    logits = torch.tensor([[[4.0, 0], [-1.0, 1]]])
    assert torch.equal(tta_probabilities(logits), logits.sigmoid().mean(1))
    assert not torch.equal(tta_probabilities(logits), logits.mean(1).sigmoid())


def test_fresh_full_stop_resume_preserves_all_head_and_adamw_state_and_predictions():
    x = inputs()
    labels = torch.full_like(x[1], 0.8)
    mask = torch.ones_like(labels, dtype=torch.bool)

    def create():
        m = model()
        o = torch.optim.AdamW(m.parameters(), lr=1e-3)
        return m, o

    def update(m, opt):
        opt.zero_grad()
        out, _ = m(*x)
        anchored_loss(out, x[1], labels, mask).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
        opt.step()

    full, fo = create()
    for _ in range(4):
        update(full, fo)
    stopped, so = create()
    update(stopped, so)
    state = copy.deepcopy((stopped.state_dict(), so.state_dict(), torch.get_rng_state()))
    resumed, ro = create()
    resumed.load_state_dict(state[0])
    ro.load_state_dict(state[1])
    torch.set_rng_state(state[2])
    for _ in range(3):
        update(resumed, ro)
    for name, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
    for key, value in fo.state_dict()["state"].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, ro.state_dict()["state"][key][name])
    assert fo.state_dict()["param_groups"] == ro.state_dict()["param_groups"]
    assert torch.equal(full(*x)[0], resumed(*x)[0])
