"""Real CPU tensor receipt must bind the isolated GPU label-source recipe."""

import copy
import json
from pathlib import Path

import pytest

from scripts.kaggle.public_label_source_recovery import engineering_config, validate_cpu_receipt


def receipt():
    return json.loads(
        Path("initiative/audits/public-label-source-inputs-v1/actual-CPU-accepted.json").read_text()
    )


def test_actual_CPU_contract_and_only_label_policy_differs():
    cpu = receipt()
    proof = {
        "input_tensor_sha256": {
            "training": cpu["image_tensor_sha256"]["control"],
            "outer_engineering": cpu["image_tensor_sha256"]["outer_engineering"],
        },
        "nonimage_tensor_sha256": {
            "training": cpu["other_tensor_sha256"]["control"],
            "outer_engineering": cpu["other_tensor_sha256"]["outer_engineering"],
        },
    }
    validate_cpu_receipt(cpu, proof)
    base = cpu["configs"]["control"]
    spec = dict(base["encoder"], weights_path="/mounted/base", config_path="/mounted/config")
    control = engineering_config(base, "control", spec)
    public = engineering_config(base, "public", spec)
    assert public["fitting_supervision"]["enabled"] is True
    other = copy.deepcopy(public)
    other["fitting_supervision"]["enabled"] = False
    assert other == control and control["epochs"] == 2 and control["diagnostic_study_limit"] == 16
    assert control["encoder"]["adapter"]["rank"] == 32


@pytest.mark.parametrize(
    "field,value", [("precision", "bf16"), ("amp_init_scale", 65536), ("side", 224)]
)
def test_GPU_recipe_drift_is_rejected(field, value):
    base = receipt()["configs"]["control"]
    base[field] = value
    with pytest.raises(ValueError):
        engineering_config(base, "public", base["encoder"])
