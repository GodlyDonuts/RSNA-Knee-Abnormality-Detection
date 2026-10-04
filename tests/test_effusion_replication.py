"""Keep the second-seed scientific contrast and original assets isolated."""

import copy
import json

import pytest

from scripts.newton.dispatch_effusion_replication import (
    CONTROL,
    PREFIX,
    ROOT,
    SEED,
    STAGE,
    launch_files,
    validate_configs,
)


def configs():
    return {
        role: json.loads(
            (
                ROOT
                / "initiative/audits/effusion-supervision-screen-final-v1/second-seed-replication"
                / (role + "-screen.json")
            ).read_text()
        )
        for role in ("control", "candidate")
    }


def test_predeclared_second_seed_preserves_original_contrast():
    pair = configs()
    validate_configs(pair)
    a, b = (copy.deepcopy(pair[r]) for r in ("control", "candidate"))
    assert a["seed"] == b["seed"] == SEED
    assert not a["fitting_supervision"]["enabled"] and b["fitting_supervision"]["enabled"]
    b["fitting_supervision"] = a["fitting_supervision"]
    assert a == b


@pytest.mark.parametrize(
    "field,value",
    [("seed", 20260922), ("epochs", 10), ("fold", 1), ("side", 224), ("encoder_lr", 0.0002)],
)
def test_reject_even_symmetric_scientific_confounds(field, value):
    pair = configs()
    for c in pair.values():
        c[field] = value
    with pytest.raises(ValueError):
        validate_configs(pair)


@pytest.mark.parametrize("role", ["control", "candidate"])
def test_reject_wrong_fitting_policy_or_weight_source(role):
    pair = configs()
    pair[role]["fitting_supervision"]["enabled"] = not pair[role]["fitting_supervision"]["enabled"]
    with pytest.raises(ValueError):
        validate_configs(pair)
    pair = configs()
    pair[role]["encoder"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_configs(pair)


def test_reused_wrapper_has_no_extraction_and_frozen_trainer_bytes_stay_original():
    files = launch_files()
    run = files["scripts/newton/run_effusion_screen.py"].decode()
    assert "def prepare():" not in run and "runtime_environment.py" not in run
    assert 'choices=("arm",)' in run
    assert "Only exact retained original node-local assets may be reused" in run
    assert CONTROL in run
    pub = files["scripts/hf/publish_effusion_screen_epoch.py"].decode()
    assert PREFIX in pub and 'expected["seed"] = 20261004' in pub
    assert 'meta.get("seed") != 20261004' in pub
    for role in ("control", "candidate"):
        assert (CONTROL + "/" + role + ".out") in files[role + ".sbatch"].decode()
        assert b"--gres=gpu:nvidia_h100_pcie:1" in files[role + ".sbatch"]
    compile(STAGE, "original_second_seed_staging", "exec")
