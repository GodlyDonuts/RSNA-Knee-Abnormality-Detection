"""GPU wrapper accepts one final fitting recipe; synthetic tests cannot admit CUDA."""

import copy
import json
from pathlib import Path

import pytest

from scripts.newton.probe_final_public_source_recovery import (
    engineering_config,
    validate_final_checkpoint,
)


def recipe():
    values = json.loads(
        Path(
            "initiative/experiments/competitive-v1/public-label-source-v1/public-config.json"
        ).read_text()
    )
    values.update(fit_scope="all_development", warmup_updates=426)
    values["fitting_supervision"] = dict(
        kind="label_source_final_v1", enabled=True, sha256="a" * 64
    )
    return values


def test_bounded_wrapper_changes_only_recovery_budget():
    values = recipe()
    result = engineering_config(values)
    changed = {k for k in values if values[k] != result[k]}
    assert changed == {
        "epochs",
        "diagnostic_study_limit",
        "warmup_updates",
        "submission_every_epochs",
    }
    assert result["fit_scope"] == "all_development" and result["precision"] == "fp16"
    assert (result["epochs"], result["diagnostic_study_limit"]) == (2, 16)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fit_scope", "outer_fold"),
        ("warmup_updates", 281),
        ("epochs", 3),
        ("side", 224),
        ("precision", "bf16"),
        ("amp_init_scale", 65536),
        ("diagnostic_study_limit", 100),
        ("augment", True),
        ("accumulation_steps", 4),
    ],
)
def test_wrong_recipe_is_refused(field, value):
    values = recipe()
    values[field] = value
    with pytest.raises(ValueError):
        engineering_config(values)


@pytest.mark.parametrize("change", ["base", "rank", "scope", "enabled"])
def test_wrong_model_or_source_is_refused(change):
    values = recipe()
    if change == "base":
        values["encoder"]["weights_sha256"] = "b" * 64
    elif change == "rank":
        values["encoder"]["adapter"]["rank"] = 64
    elif change == "scope":
        values["fitting_supervision"]["kind"] = "label_source_v1"
    else:
        values["fitting_supervision"]["enabled"] = False
    with pytest.raises(ValueError):
        engineering_config(values)


@pytest.mark.parametrize("change", ["scope", "source", "scaler"])
def test_final_state_cannot_fabricate_other_scope_or_scaler(change):
    state = {"binding": {"config": recipe()}, "amp_scaler": {"_growth_tracker": 4}}
    resumed = copy.deepcopy(state)
    if change == "scope":
        state["binding"]["config"]["fit_scope"] = "outer_fold"
    elif change == "source":
        state["binding"]["config"]["fitting_supervision"]["kind"] = "label_source_v1"
    else:
        state["amp_scaler"]["_growth_tracker"] = 3
    with pytest.raises(ValueError, match="scope/source/scaler"):
        validate_final_checkpoint(state, resumed, {}, {}, {}, {})


def test_recovery_placement_preserves_original_and_requires_explicit_relocation():
    from scripts.newton.probe_final_public_source_recovery import validate_placement

    validate_placement({}, "evc31")
    validate_placement({"owning_node": "evc26"}, "evc26")
    for manifest, node in [
        ({}, "evc26"),
        ({"owning_node": "evc26"}, "evc31"),
        ({"owning_node": "evc38"}, "evc38"),
    ]:
        with pytest.raises(ValueError):
            validate_placement(manifest, node)
