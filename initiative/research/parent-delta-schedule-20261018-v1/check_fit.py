"""Actual isolated CPU optimizer loop on synthetic images; not real-MRI/GPU evidence."""

import numpy as np
import pytest
import torch
from delta_fit import CorrectedCases, fit
from schedule import Schedule, plan
from test_licensed_parent_train import Cases, TinyMRI, binding, equal

from rsna.competitive.licensed_importance import ProposalCases, Recipe
from rsna.competitive.licensed_importance_train import fit as original_fit


class TinyImportance(TinyMRI):
    def forward(self, x, log_proposal=None):
        return super().forward(x)


def data(mode):
    original = Cases(n=5)
    recipe = Recipe(epochs=2, effective_batch=4, role="influence")
    q = np.full((5, 94), 1 / 94, dtype=np.float64)
    rows = [plan(u, recipe, q[i]) for i, u in enumerate(original.uids)]
    p = np.stack([x[0] for x in rows], axis=1)
    m = np.stack([x[1] for x in rows], axis=1)
    teacher = np.stack(
        [original.teacher[np.arange(5), m[e].astype(int)].numpy() - 0.1 for e in range(2)]
    )
    schedule = Schedule(original.uids, p, m, teacher, q, recipe)
    base = ProposalCases(original, torch.from_numpy(q))
    return base, CorrectedCases(base, schedule, mode=mode), recipe


def test_case_preserves_original_images_labels_masks_and_teacher():
    base, wrapped, recipe = data("stable_logit_delta")
    for epoch in range(2):
        p = wrapped.schedule.positions[epoch, 0].astype(int).tolist()
        m = bool(wrapped.schedule.mirrors[epoch, 0])
        a = base.case(0, p, m)
        b = wrapped.case(0, p, m)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        torch.testing.assert_close(
            b["parent_pooling_delta"], torch.full((12,), 0.1), rtol=0, atol=1e-7
        )


def test_zero_mode_all_actual_numerical_optimizer_and_RNG_state_matches_original(tmp_path):
    torch.set_num_threads(1)
    base, wrapped, recipe = data("zero")
    device = torch.device("cpu")
    torch.manual_seed(713)
    original_fit(TinyImportance(), base, tmp_path / "original", binding(), recipe, device=device)
    torch.manual_seed(713)
    fit(TinyImportance(), wrapped, tmp_path / "zero", binding(), recipe, device=device)
    a = torch.load(tmp_path / "original/checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "zero/checkpoint.pt", weights_only=True)
    aa = a.pop("binding")
    bb = b.pop("binding")
    assert "parent_delta" not in aa and bb["parent_delta"]["mode"] == "zero"
    assert {k: v for k, v in bb.items() if k != "parent_delta"} == aa
    equal(a, b)


@pytest.mark.parametrize("mode", ["zero", "stable_logit_delta"])
def test_complete_actual_CPU_engine_recovery_including_partial_final_batch(tmp_path, mode):
    torch.set_num_threads(1)
    _, wrapped, recipe = data(mode)
    device = torch.device("cpu")
    torch.manual_seed(713)
    fit(TinyImportance(), wrapped, tmp_path / "full", binding(), recipe, device=device)
    torch.manual_seed(713)
    fit(
        TinyImportance(),
        wrapped,
        tmp_path / "resume",
        binding(),
        recipe,
        device=device,
        stop_after=1,
    )
    torch.manual_seed(1)
    fit(
        TinyImportance(),
        wrapped,
        tmp_path / "resume",
        binding(),
        recipe,
        device=device,
        resume=True,
    )
    a = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/checkpoint.pt", weights_only=True)
    equal(a, b)
    assert [r["studies"] for r in a["history"]] == [4, 1, 4, 1]
    for epoch in (1, 2):
        equal(
            torch.load(
                tmp_path / f"full/epoch-checkpoints/epoch-{epoch:03d}.pt", weights_only=True
            ),
            torch.load(
                tmp_path / f"resume/epoch-checkpoints/epoch-{epoch:03d}.pt", weights_only=True
            ),
        )
    with pytest.raises(ValueError):
        fit(
            TinyImportance(),
            wrapped,
            tmp_path / "resume",
            binding(),
            recipe,
            device=device,
            resume=True,
            stop_after=1,
        )


def test_nonzero_correction_changes_learned_state_without_adding_parameters(tmp_path):
    torch.set_num_threads(1)
    _, zero, recipe = data("zero")
    _, delta, _ = data("stable_logit_delta")
    for mode, d in [("zero", zero), ("delta", delta)]:
        torch.manual_seed(713)
        fit(TinyImportance(), d, tmp_path / mode, binding(), recipe, device=torch.device("cpu"))
    a = torch.load(tmp_path / "zero/checkpoint.pt", weights_only=True)["model"]
    b = torch.load(tmp_path / "delta/checkpoint.pt", weights_only=True)["model"]
    assert a.keys() == b.keys() and any(not torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("damage", ["recipe", "binding", "mode", "uids"])
def test_invalid_admission_refused_before_optimizer(tmp_path, damage):
    base, wrapped, recipe = data("zero")
    if damage == "mode":
        with pytest.raises(ValueError):
            CorrectedCases(base, wrapped.schedule, mode="unknown")
    elif damage == "uids":
        base.uids.reverse()
        with pytest.raises(ValueError):
            CorrectedCases(base, wrapped.schedule, mode="zero")
    else:
        from dataclasses import replace

        r = replace(recipe, seed=1) if damage == "recipe" else recipe
        b = {**binding(), "parent_delta": {}} if damage == "binding" else binding()
        with pytest.raises(ValueError):
            fit(TinyImportance(), wrapped, tmp_path / "bad", b, r, device=torch.device("cpu"))
        assert not (tmp_path / "bad").exists()
