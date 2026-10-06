"""Reject drift that would break the fixed same-input/same-budget clinical contrast."""

import copy
import json
from pathlib import Path

import pytest

from scripts.newton.run_smooth_witness_comparison import verify_config

ROOT = Path("initiative/audits/smooth-witness-full-fit-v1")


@pytest.mark.parametrize("role", ["control", "candidate"])
def test_exact_full_config(role):
    base = json.loads((ROOT / "base-config.json").read_text())
    config = json.loads((ROOT / (role + ".json")).read_text())
    assert verify_config(config, role, base) is config
    assert config["diagnostic_study_limit"] is None and config["submission_every_epochs"] == 1
    assert config["encoder"]["adapter"]["rank"] == 32 and config["warmup_updates"] == 141


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", 4),
        ("diagnostic_study_limit", 100),
        ("precision", "bf16"),
        ("accumulation_steps", 4),
        ("seed", 42),
        ("side", 224),
        ("series_sampling", "all"),
        ("encoder_lr", 0.00001),
        ("warmup_updates", 2),
        ("amp_init_scale", 256),
        ("encode_chunk_size", 16),
        ("fitting_supervision", {"unadmitted": True}),
    ],
)
def test_drift_rejected(key, value):
    base = json.loads((ROOT / "base-config.json").read_text())
    config = json.loads((ROOT / "candidate.json").read_text())
    config[key] = value
    with pytest.raises(ValueError):
        verify_config(config, "candidate", base)


@pytest.mark.parametrize("role", ["control", "candidate"])
def test_no_cross_role_configuration(role):
    base = json.loads((ROOT / "base-config.json").read_text())
    config = json.loads(
        (ROOT / ("candidate.json" if role == "control" else "control.json")).read_text()
    )
    with pytest.raises(ValueError):
        verify_config(config, role, base)


def test_beta_and_encoder_graph_are_fixed():
    base = json.loads((ROOT / "base-config.json").read_text())
    config = json.loads((ROOT / "candidate.json").read_text())
    altered = copy.deepcopy(config)
    altered["witness"]["beta"] = 16
    with pytest.raises(ValueError):
        verify_config(altered, "candidate", base)
    altered = copy.deepcopy(config)
    altered["encoder"]["adapter"]["rank"] = 64
    with pytest.raises(ValueError):
        verify_config(altered, "candidate", base)
