"""Full readout contrast preserves fixed cohort/budget and canonical binding fields."""

import copy
import json
from pathlib import Path

import pytest

from scripts.newton.run_plane_memory_comparison import expected_binding_config, verify_recipe


def policy(role="candidate"):
    values = json.loads(
        Path("initiative/audits/smooth-witness-full-fit-v2/base-config.json").read_text()
    )
    return dict(
        values,
        witness=None,
        plane_memory={"seed": 20261006, "weight": int(role == "candidate"), "hidden_dim": 128},
    )


def test_fixed_roles_share_every_config_field_except_one_branch_weight():
    control, candidate = policy("control"), policy()
    assert verify_recipe(control, "control", copy.deepcopy(control)) == control
    assert verify_recipe(candidate, "candidate", copy.deepcopy(candidate)) == candidate
    candidate["plane_memory"]["weight"] = 0
    assert control == candidate


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 10),
        ("diagnostic_study_limit", 16),
        ("warmup_updates", 2),
        ("precision", "bf32"),
        ("fold", 1),
        ("seed", 20261004),
        ("side", 224),
    ],
)
def test_full_budget_or_cohort_drift_is_rejected_even_if_supplied_as_expected(field, value):
    values = policy()
    values[field] = value
    with pytest.raises(ValueError):
        verify_recipe(values, "candidate", copy.deepcopy(values))


def test_unsupported_readout_or_added_mechanism_is_rejected():
    values = policy()
    values["plane_memory"]["hidden_dim"] = 16
    with pytest.raises(ValueError):
        verify_recipe(values, "candidate", values)
    values = policy()
    values["witness"] = {"weight": 1, "patches": 4, "slices": 2, "seed": 20261003}
    with pytest.raises(ValueError):
        verify_recipe(values, "candidate", values)


def test_bound_full_config_follows_canonical_optional_serialization():
    values = policy()
    provenance = {"pinned_test_provenance": True}
    result = expected_binding_config(values, provenance)
    assert result["encoder"] == provenance and result["plane_memory"] == values["plane_memory"]
    assert result["amp_init_scale"] == 128
    assert not set(result) & {"witness", "relation", "fitting_supervision", "neighbor_channels"}
