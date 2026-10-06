"""Full label-source contrast preserves fixed cohort/budget and canonical binding fields."""

import copy
import json
from pathlib import Path

import pytest

from scripts.newton.run_public_label_comparison import expected_binding_config, verify_recipe


def policy(role="public"):
    return json.loads(
        Path(
            f"initiative/experiments/competitive-v1/public-label-source-v1/{role}-config.json"
        ).read_text()
    )


def test_fixed_roles_share_every_config_field_except_one_branch_weight():
    control, candidate = policy("control"), policy()
    assert verify_recipe(control, "control", copy.deepcopy(control)) == control
    assert verify_recipe(candidate, "public", copy.deepcopy(candidate)) == candidate
    candidate["fitting_supervision"]["enabled"] = False
    assert control == candidate


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 3),
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
        verify_recipe(values, "public", copy.deepcopy(values))


def test_unsupported_readout_or_added_mechanism_is_rejected():
    values = policy()
    values["fitting_supervision"]["enabled"] = False
    with pytest.raises(ValueError):
        verify_recipe(values, "public", values)
    values = policy()
    values["witness"] = {"weight": 1, "patches": 4, "slices": 2, "seed": 20261003}
    with pytest.raises(ValueError):
        verify_recipe(values, "public", values)


def test_bound_full_config_follows_canonical_optional_serialization():
    values = policy()
    provenance = {"pinned_test_provenance": True}
    result = expected_binding_config(values, provenance)
    assert (
        result["encoder"] == provenance
        and result["fitting_supervision"] == values["fitting_supervision"]
    )
    assert result["amp_init_scale"] == 128
    assert not set(result) & {
        "witness",
        "relation",
        "patch_summary",
        "neighbor_channels",
        "plane_memory",
    }
