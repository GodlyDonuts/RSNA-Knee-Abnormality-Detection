"""The accelerated route must preserve the full MRI/adapter scientific scope."""

import copy
import json
from pathlib import Path

import pytest

from scripts.kaggle.witness_amp_recovery import validate_recipe


def source_config(role):
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "initiative/audits/patch-witness-gpu-recovery-v1"
            / (role + "-engineering-config.json")
        ).read_text()
    )


def recipe(role):
    old = source_config(role)
    v = dict(old, precision="fp16", encode_chunk_size=8, min_free_gib=3, amp_init_scale=128)
    v["witness"] = dict(v["witness"], pooling="normalized_logmeanexp", beta=8)
    return old, v


@pytest.mark.parametrize("role", ["control", "candidate"])
def test_exact_T4_recipe_preserves_resolution_views_adapter_and_labels(role):
    old, v = recipe(role)
    validate_recipe(v, role, old)
    assert v["side"] == 384 and v["slices"] == 16 and v["max_series"] == 6
    assert v["encoder"]["adapter"]["rank"] == 32 and v["freeze_encoder"]


@pytest.mark.parametrize(
    "change",
    [
        ("side", 224),
        ("slices", 8),
        ("accumulation_steps", 4),
        ("precision", "fp32"),
        ("amp_init_scale", 64),
        ("encode_chunk_size", 1),
    ],
)
def test_resource_route_does_not_admit_an_easier_or_different_experiment(change):
    old, v = recipe("candidate")
    v = copy.deepcopy(v)
    v[change[0]] = change[1]
    with pytest.raises(ValueError):
        validate_recipe(v, "candidate", old)
