"""Synthetic table tests; no clinical or actual GPU fitting acceptance."""

import copy

import numpy as np
import pytest
import torch
from schedule import Schedule, plan, teacher_logits

from rsna.competitive.licensed_importance import Recipe, sample
from rsna.competitive.mil_sampling import readout_moments


def fixture(n=3):
    uids = [f"synthetic-{i}" for i in range(n)]
    recipe = Recipe(role="influence")
    q = np.full((n, 94), 1 / 94, dtype=np.float64)
    rows = [plan(u, recipe, q[i]) for i, u in enumerate(uids)]
    positions = np.stack([p for p, m in rows], axis=1)
    mirrors = np.stack([m for p, m in rows], axis=1)
    logits = np.arange(3 * n * 12, dtype=np.float32).reshape(3, n, 12) / 100
    return uids, positions, mirrors, logits, q, recipe


def test_original_sampler_exact_and_lookup_does_not_advance_RNG():
    args = fixture()
    s = Schedule(*args)
    state = torch.get_rng_state().clone()
    for i, uid in enumerate(args[0]):
        for epoch in range(3):
            positions, mirror, _ = sample(uid, args[-1], epoch, torch.from_numpy(args[-2][i]))
            torch.testing.assert_close(
                s.lookup(i, positions, mirror), torch.from_numpy(args[3][epoch, i]), rtol=0, atol=0
            )
    assert torch.equal(state, torch.get_rng_state())
    args[3].fill(0)
    assert s.logits.sum() > 0 and not s.logits.flags.writeable


@pytest.mark.parametrize("damage", ["order", "positions", "mirror", "nan", "q", "dtype", "shape"])
def test_changed_or_unaligned_schedule_refused(damage):
    a = list(copy.deepcopy(fixture()))
    if damage == "order":
        a[0].reverse()
    elif damage == "positions":
        a[1][0, 0, 0] = (int(a[1][0, 0, 0]) + 1) % 94
    elif damage == "mirror":
        a[2][0, 0] = ~a[2][0, 0]
    elif damage == "nan":
        a[3][0, 0, 0] = np.nan
    elif damage == "q":
        a[4][0, 0] = 0
    elif damage == "dtype":
        a[3] = a[3].astype(np.float64)
    else:
        a[1] = a[1][:2]
    with pytest.raises(ValueError):
        Schedule(*a)


def test_wrong_case_or_unscheduled_bag_refused():
    s = Schedule(*fixture())
    p = s.positions[0, 0].astype(int).tolist()
    m = bool(s.mirrors[0, 0])
    for i, positions, mirror in [(4, p, m), (0, p, not m), (0, [-1] * 16, m), (0, p[:15], m)]:
        with pytest.raises(ValueError):
            s.lookup(i, positions, mirror)


def test_FP32_teacher_matches_independent_FP64_importance_readout():
    g = torch.Generator().manual_seed(2)

    def r(*shape):
        return torch.randn(*shape, generator=g) * 0.05

    state = {
        "norm.weight": torch.ones(1024),
        "norm.bias": torch.zeros(1024),
        "att.0.weight": r(256, 1024),
        "att.0.bias": r(256),
        "att.3.weight": r(12, 256),
        "att.3.bias": r(12),
        "clsW": r(12, 1024),
        "clsb": r(12),
    }
    from rsna.competitive.mil_sampling import PARENT_SHA, PARENT_SOURCE

    head = {
        "format": "licensed_parent_MIL_readout_v1",
        "parent_checkpoint_sha256": PARENT_SHA,
        "source_notebook_sha256": PARENT_SOURCE,
        "model": state,
    }
    f = r(2, 94, 1024)
    w, v, b, _ = readout_moments(head, f)
    q = np.full(94, 1 / 94, dtype=np.float64)
    p, m = plan("synthetic", Recipe(role="influence"), q)
    actual = teacher_logits(f, state, p, m, q)
    for epoch in range(3):
        ix = p[epoch].astype(np.int64)
        a = w[int(m[epoch]), ix] / q[ix, None]
        expected = 1 / (1 + np.exp(-((a * v[int(m[epoch]), ix]).sum(0) / a.sum(0) + b)))
        np.testing.assert_allclose(
            torch.from_numpy(actual[epoch]).sigmoid().numpy(), expected, atol=2e-6, rtol=0
        )
