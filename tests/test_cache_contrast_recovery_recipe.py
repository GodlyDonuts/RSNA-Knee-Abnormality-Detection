import copy
import json
from pathlib import Path

import pytest

from scripts.newton.cache_contrast_recovery import engineering_config, validate_cpu_receipt


def test_only_input_policy_changes_in_fixed_engineering_recipe():
    base = json.loads(
        Path("initiative/audits/public-label-source-inputs-v1/actual-CPU-accepted.json").read_text()
    )["configs"]["control"]
    spec = dict(base["encoder"], weights_path="/mounted/base", config_path="/mounted/config")
    a = engineering_config(base, "control", spec)
    b = engineering_config(base, "candidate", spec)
    assert a["cache_contrast"] is None and b["cache_contrast"] == {"mode": "p01_p99"}
    b = copy.deepcopy(b)
    b["cache_contrast"] = None
    assert a == b
    assert (
        a["fitting_supervision"] is None and a["epochs"] == 2 and a["diagnostic_study_limit"] == 16
    )


@pytest.mark.parametrize(
    "field,value",
    [("precision", "bf16"), ("side", 224), ("amp_init_scale", 65536), ("encode_chunk_size", 16)],
)
def test_unadmitted_recipe_drift_refused(field, value):
    base = json.loads(
        Path("initiative/audits/public-label-source-inputs-v1/actual-CPU-accepted.json").read_text()
    )["configs"]["control"]
    base[field] = value
    with pytest.raises(ValueError):
        engineering_config(base, "candidate", base["encoder"])


def test_current16_study_gate_cannot_reserve_a_trainer_GPU():
    actual = json.loads(
        Path(
            "initiative/audits/cache-contrast-canonical-inputs-267fcb65-v1/actual-receipt.json"
        ).read_text()
    )
    with pytest.raises(ValueError):
        validate_cpu_receipt(actual, {})
