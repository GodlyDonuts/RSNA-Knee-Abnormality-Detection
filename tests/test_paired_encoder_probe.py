"""New actual-autograd wiring on a synthetic small encoder; not a GPU acceptance."""

import copy

import pytest
import torch
from test_licensed_importance import Model, OriginalCases, Recipe, equal, proposal

from rsna.competitive.checkpoint import capture_rng
from rsna.competitive.paired_encoder_probe import measure_case
from rsna.competitive.paired_logit_control import PairedLogitReadout


class Corrected(PairedLogitReadout, Model):
    pass


def test_zero_control_matches_full_encoder_gradients_under_matched_dropout_and_preserves_state():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = Corrected()
    original = OriginalCases(1)
    views = [original.case(0, list(range(94)), mirror) for mirror in (False, True)]
    features = torch.zeros(2, 94, 4)

    def zero_teacher(features, logq):
        return torch.zeros(features.shape[0], 12)

    before = copy.deepcopy(model.state_dict())
    modes = [x.training for x in model.modules()]
    rng = capture_rng(torch.device("cpu"))
    events = []
    out = measure_case(
        model,
        views,
        features,
        zero_teacher,
        proposal(),
        "new-engineering-only",
        Recipe(role="influence"),
        device=torch.device("cpu"),
        expected_parameters=sum(p.numel() for p in model.parameters()),
        reference_draws=2,
        paired_draws=2,
        progress=events.append,
    )
    equal(before, model.state_dict())
    equal(rng, capture_rng(torch.device("cpu")))
    assert [x.training for x in model.modules()] == modes
    assert len(events) == 8 and out["optimizer_or_parameter_updates"] == 0
    for v in out["views"]:
        for group in ("all", "encoder", "head"):
            assert v["groups"][group]["MSE_ratio"] == 1
            assert v["groups"][group]["empirical_mean_error_ratio"] == 1
        assert v["loss_uses_one_view"] and v["dropout_arms_seed_matched"]
        assert v["actual_optimizer_step_timing_measured"] is False


def test_stochastic_cached_parent_is_refused_and_original_rng_is_restored():
    model = Corrected()
    original = OriginalCases(1)
    views = [original.case(0, list(range(94)), m) for m in (False, True)]
    state = capture_rng(torch.device("cpu"))

    def random_parent(features, logq):
        return torch.rand(features.shape[0], 12)

    with pytest.raises(ValueError, match="Deterministic"):
        measure_case(
            model,
            views,
            torch.zeros(2, 94, 4),
            random_parent,
            proposal(),
            "new-engineering-only",
            Recipe(role="influence"),
            device=torch.device("cpu"),
            expected_parameters=sum(p.numel() for p in model.parameters()),
            reference_draws=2,
            paired_draws=2,
        )
    equal(state, capture_rng(torch.device("cpu")))


def test_partial_encoder_parameter_count_is_refused_before_teacher_or_forward():
    model = Corrected()
    original = OriginalCases(1)
    views = [original.case(0, list(range(94)), m) for m in (False, True)]

    def forbidden(features, logq):
        raise AssertionError("Reader must not run after an incomplete-model contract")

    with pytest.raises(ValueError, match="Complete actual encoder"):
        measure_case(
            model,
            views,
            torch.zeros(2, 94, 4),
            forbidden,
            proposal(),
            "new-engineering-only",
            Recipe(role="influence"),
            device=torch.device("cpu"),
            expected_parameters=1,
            reference_draws=2,
            paired_draws=2,
        )
