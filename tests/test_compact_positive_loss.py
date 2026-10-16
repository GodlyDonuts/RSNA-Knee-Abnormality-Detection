import copy
import subprocess
import sys
import types

import pandas as pd
import pytest
import torch
from test_compact_training import FittingFixture, equal, fresh

from rsna.competitive.compact_loss import ENGINE, derive_policy, loss_weights, validate_policy
from rsna.competitive.compact_train import Settings, fit, masked_cell_loss, predict
from rsna.schema import TARGETS


class Fitting(FittingFixture):
    def __init__(self):
        super().__init__()
        self.studies = pd.DataFrame(
            [
                {
                    "study_uid": r["study_uid"],
                    **dict(zip(TARGETS, r["labels"].tolist(), strict=True)),
                    **dict(
                        zip([t + "__mask" for t in TARGETS], r["label_mask"].tolist(), strict=True)
                    ),
                }
                for r in self.samples
            ]
        )


def test_formula_is_fitting_only_masked_probability_mass_and_row_order_independent():
    frame = Fitting().studies
    frame[TARGETS[0]] = 0.0
    frame[TARGETS[1]] = 1.0
    policy = derive_policy(frame)
    validate_policy(policy)
    assert policy["positive_weights"][:2] == [10.0, 1.0]
    assert policy["observed_cells"][:2] == [2, 1]
    assert derive_policy(frame.iloc[::-1]) == policy
    for target in TARGETS:
        frame.loc[frame[target + "__mask"] == 0, target] = float("nan")
    assert derive_policy(frame) == policy  # Unknown cells contribute neither mass nor labels.
    frame.loc[frame[TARGETS[2] + "__mask"] == 1, TARGETS[2]] = float("inf")
    with pytest.raises(ValueError, match="Observed"):
        derive_policy(frame)


def test_missing_targets_cannot_be_invented_and_policy_formula_cannot_be_tampered():
    frame = Fitting().studies
    policy = derive_policy(frame)
    wrong = copy.deepcopy(policy)
    wrong["positive_weights"][0] += 0.1
    with pytest.raises(ValueError, match="formula"):
        validate_policy(wrong)
    frame[TARGETS[0] + "__mask"] = 0
    with pytest.raises(ValueError, match="observed fitting"):
        derive_policy(frame)
    data = Fitting()
    policy = derive_policy(data.studies)
    data.studies.loc[0, TARGETS[-1]] = 0.999
    with pytest.raises(ValueError, match="Actual fitting"):
        loss_weights(policy, data, {"diagnostic_only": True}, torch.device("cpu"))


def test_positive_term_gradient_matches_analytic_soft_target_formula_with_masked_nan():
    logits = torch.linspace(-2, 2, 12, requires_grad=True)
    labels = torch.linspace(0.1, 0.9, 12)
    mask = torch.tensor([1.0] * 5 + [0.0] * 7)
    weights = torch.linspace(1, 10, 12)
    labels[5:] = float("nan")
    denominator = 19.0
    loss = masked_cell_loss(logits, labels, mask, denominator, pos_weight=weights)
    loss.backward()
    p = logits.detach().sigmoid()
    safe = torch.where(mask.bool(), labels, 0)
    expected = ((1 - safe) * p - weights * safe * (1 - p)) * mask / denominator
    torch.testing.assert_close(logits.grad, expected, rtol=1e-6, atol=1e-8)
    assert logits.grad[5:].eq(0).all()
    for bad in [weights[:-1], weights.double(), weights * 0, weights * float("nan")]:
        with pytest.raises(ValueError, match="positive weights"):
            masked_cell_loss(logits, labels, mask, denominator, pos_weight=bad)


def test_weighted_fit_full_stop_resume_complete_state_and_csv_equality(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = Fitting()
        policy = derive_policy(data.studies)
        settings = Settings(seed=7, epochs=2, effective_batch=2)
        binding = {"diagnostic_only": True, "loss_policy": policy}
        model = fresh()
        fit(model, data, tmp_path / "full", binding, settings, device=torch.device("cpu"))
        expected = predict(model, data, torch.device("cpu")).to_csv(index=False)
        fit(
            fresh(),
            data,
            tmp_path / "resume",
            binding,
            settings,
            device=torch.device("cpu"),
            stop_after_updates=1,
        )
        model = fresh()
        fit(
            model,
            data,
            tmp_path / "resume",
            binding,
            settings,
            device=torch.device("cpu"),
            resume=True,
        )
        left = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
        right = torch.load(tmp_path / "resume/checkpoint.pt", weights_only=True)
        equal(left, right)
        assert left["binding"]["engine"] == ENGINE and left["binding"]["loss_policy"] == policy
        assert left["optimizer_steps"] == 4 and left["scheduler"]["last_epoch"] == 4
        assert expected == predict(model, data, torch.device("cpu")).to_csv(index=False)
        with pytest.raises(ValueError, match="binding"):
            fit(
                fresh(),
                data,
                tmp_path / "resume",
                {"diagnostic_only": True},
                settings,
                device=torch.device("cpu"),
                resume=True,
            )
        fit(
            fresh(),
            data,
            tmp_path / "unweighted",
            {"diagnostic_only": True},
            settings,
            device=torch.device("cpu"),
        )
        old = torch.load(tmp_path / "unweighted/checkpoint.pt", weights_only=True)
        assert any(not torch.equal(old["model"][k], left["model"][k]) for k in old["model"])
    finally:
        torch.set_num_threads(old_threads)


def test_default_numerics_equal_original_frozen_trainer_complete_state(tmp_path):
    source = subprocess.check_output(
        [
            "git",
            "show",
            "3803b717b69db6c5e89e11724c4daee15490cbe2:rsna/competitive/compact_train.py",
        ],
        text=True,
    )
    legacy = types.ModuleType("_frozen_n3_trainer")
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, "frozen_n3_trainer", "exec"), legacy.__dict__)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = Fitting()
        settings = Settings(seed=7, epochs=2, effective_batch=2)
        binding = {"diagnostic_only": True}
        a = fresh()
        legacy.fit(a, data, tmp_path / "old", binding, settings, device=torch.device("cpu"))
        b = fresh()
        fit(b, data, tmp_path / "new", binding, settings, device=torch.device("cpu"))
        equal(
            torch.load(tmp_path / "old/checkpoint.pt", weights_only=True),
            torch.load(tmp_path / "new/checkpoint.pt", weights_only=True),
        )
        assert legacy.predict(a, data, torch.device("cpu")).to_csv(index=False) == predict(
            b, data, torch.device("cpu")
        ).to_csv(index=False)
    finally:
        torch.set_num_threads(old_threads)
        del sys.modules[legacy.__name__]
