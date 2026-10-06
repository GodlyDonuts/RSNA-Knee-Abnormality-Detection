"""The real CPU receipt must bind the new GPU diagnostic without relaxing input equality."""

import copy
import json
from pathlib import Path

import pytest

from scripts.kaggle.cls_patch_summary_recovery import engineering_config, validate_cpu_receipt


def receipt():
    return json.loads(
        Path("initiative/audits/cls-patch-summary-CPU-v2/actual-CPU-accepted.json").read_text()
    )


def test_actual_cpu_contract_and_isolated_recovery_recipe():
    cpu = receipt()
    proof = {
        "input_tensor_sha256": cpu["original_input_tensor_hashes"],
        "nonimage_tensor_sha256": cpu["original_other_tensor_hashes"],
    }
    validate_cpu_receipt(cpu, proof)
    base = cpu["configs"]["control"]
    spec = dict(base["encoder"], weights_path="/mounted/weights", config_path="/mounted/config")
    a = engineering_config(base, "control", spec)
    b = engineering_config(base, "candidate", spec)
    assert a["patch_summary"] == {"weight": 0}
    assert b["patch_summary"] == {"weight": 1}
    b["patch_summary"] = {"weight": 0}
    assert a == b and a["epochs"] == 2 and a["diagnostic_study_limit"] == 16
    assert base == receipt()["configs"]["control"]


@pytest.mark.parametrize("field", ["original_input_tensor_hashes", "original_other_tensor_hashes"])
def test_changed_original_tensor_rejected(field):
    cpu = receipt()
    proof = {
        "input_tensor_sha256": copy.deepcopy(cpu["original_input_tensor_hashes"]),
        "nonimage_tensor_sha256": copy.deepcopy(cpu["original_other_tensor_hashes"]),
    }
    cpu[field] = {"changed": "not-the-original"}
    with pytest.raises(ValueError, match="Original MRI"):
        validate_cpu_receipt(cpu, proof)


@pytest.mark.parametrize(
    "field,value", [("precision", "bf16"), ("side", 224), ("accumulation_steps", 4)]
)
def test_changed_gpu_numeric_recipe_rejected(field, value):
    base = receipt()["configs"]["control"]
    base[field] = value
    with pytest.raises(ValueError):
        engineering_config(base, "candidate", base["encoder"])
