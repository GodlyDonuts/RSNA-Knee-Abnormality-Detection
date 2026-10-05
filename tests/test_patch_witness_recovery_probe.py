import copy
import json
from pathlib import Path

import pytest

from scripts.newton.probe_patch_witness_trainer_recovery import validate_placement, validate_recipe

ROOT = Path("initiative/audits/patch-witness-gpu-recovery-v1")


@pytest.mark.parametrize("mode", ["control", "candidate"])
def test_exact_original_engineering_arms_are_admitted(mode):
    config = json.loads((ROOT / (mode + "-engineering-config.json")).read_text())
    validate_recipe(config, mode, copy.deepcopy(config))


@pytest.mark.parametrize("mode", ["control", "candidate"])
@pytest.mark.parametrize("mutation", ["weight", "budget", "rank", "labels", "neighbors", "seed"])
def test_mutual_admission_cannot_accept_a_changed_recipe(mode, mutation):
    config = json.loads((ROOT / (mode + "-engineering-config.json")).read_text())
    if mutation == "weight":
        config["witness"]["weight"] = 1 - config["witness"]["weight"]
    elif mutation == "budget":
        config["epochs"] = 3
    elif mutation == "rank":
        config["encoder"]["adapter"]["rank"] = 64
    elif mutation == "labels":
        config["fitting_supervision"] = {"unadmitted": True}
    elif mutation == "neighbors":
        config["neighbor_channels"] = {"mode": "gray", "strength": 0}
    else:
        config["seed"] += 1
    with pytest.raises(ValueError):
        validate_recipe(config, mode, copy.deepcopy(config))


def test_unreviewed_config_and_role_are_rejected_before_execution():
    config = json.loads((ROOT / "control-engineering-config.json").read_text())
    expected = copy.deepcopy(config)
    config["head_lr"] *= 2
    with pytest.raises(ValueError):
        validate_recipe(config, "control", expected)
    with pytest.raises(ValueError):
        validate_recipe(expected, "unreviewed", expected)


def test_legacy_original_and_distinct_funded_node_only():
    assert validate_placement({"node_future_bytes": 8 * 2**30}) == "evc31"
    assert validate_placement({"owning_node": "evc48", "node_future_bytes": 16 * 2**30}) == "evc48"


@pytest.mark.parametrize("node,budget", [("evc34", 16), ("evc48", 8), ("evc31", 16)])
def test_unaccepted_node_or_reused_node_budget_is_rejected(node, budget):
    with pytest.raises(ValueError):
        validate_placement({"owning_node": node, "node_future_bytes": budget * 2**30})
