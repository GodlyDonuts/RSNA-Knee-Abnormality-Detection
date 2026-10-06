"""Mounted paths may differ; the admitted numerical recipe may not drift."""

import copy
import json
from pathlib import Path

import pytest

from scripts.kaggle.plane_memory_recovery import engineering_config


def base():
    return json.loads(
        Path("initiative/audits/smooth-witness-full-fit-v2/base-config.json").read_text()
    )


def test_recovery_arms_keep_original_encoder_and_isolate_one_weight():
    original = base()
    spec = dict(original["encoder"], weights_path="/mounted/weights", config_path="/mounted/config")
    a = engineering_config(original, "control", spec)
    b = engineering_config(original, "candidate", spec)
    assert a["plane_memory"]["weight"] == 0 and b["plane_memory"]["weight"] == 1
    b["plane_memory"]["weight"] = 0
    assert a == b
    assert a["encoder"] == spec and a["diagnostic_study_limit"] == 16
    assert original == base()


@pytest.mark.parametrize(
    "field,value",
    [("precision", "bf16"), ("side", 224), ("amp_init_scale", 65536), ("accumulation_steps", 4)],
)
def test_unaccepted_numeric_recipe_is_rejected(field, value):
    original = base()
    original[field] = value
    with pytest.raises(ValueError):
        engineering_config(original, "candidate", original["encoder"])


def test_changed_base_or_adapter_is_rejected():
    original = base()
    changed = copy.deepcopy(original["encoder"])
    changed["adapter"]["rank"] = 64
    with pytest.raises(ValueError, match="encoder policy"):
        engineering_config(original, "candidate", changed)
