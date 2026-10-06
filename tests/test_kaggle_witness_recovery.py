"""T4 recovery admits an explicit policy without broadening original diagnosis."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from rsna.schema import TARGETS
from scripts.kaggle.witness_recovery import digest, validate_predictions, validate_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_research_predictions_retain_fold_seed_and_exact_roster():
    identifiers = [f"engineering-{i:02d}" for i in range(16)]
    frame = pd.DataFrame(
        {
            "study_uid": identifiers,
            "fold": 0,
            "seed": 20261003,
            **{target: [0.5] * 16 for target in TARGETS},
        }
    )
    roster = digest(sorted(identifiers))
    validate_predictions(frame, roster, 0, 20261003)
    with pytest.raises(ValueError):
        validate_predictions(frame.drop(columns=["fold", "seed"]), roster, 0, 20261003)
    for column, value in [
        ("fold", 1),
        ("seed", 42),
        (TARGETS[0], float("nan")),
        (TARGETS[0], 1.1),
        ("study_uid", identifiers[1]),
    ]:
        altered = frame.copy()
        altered.loc[0, column] = value
        with pytest.raises(ValueError):
            validate_predictions(altered, roster, 0, 20261003)


def original(role):
    return json.loads(
        (
            ROOT
            / "initiative/audits/patch-witness-gpu-recovery-v1"
            / (role + "-engineering-config.json")
        ).read_text()
    )


@pytest.mark.parametrize("role", ["control", "candidate"])
def test_explicit_FP32_chunk1_recipe_preserves_original_policy(role):
    prior = original(role)
    values = copy.deepcopy(prior)
    values.update(precision="fp32", encode_chunk_size=1, min_free_gib=3)
    values["encoder"]["weights_path"] = "/kaggle/input/pinned-base/weights.safetensors"
    values["encoder"]["config_path"] = "/kaggle/input/pinned-base/config.json"
    validate_recipe(values, role, prior)
    assert prior["precision"] == "bf16" and prior["encode_chunk_size"] == 16


@pytest.mark.parametrize(
    "change", ["weights_sha", "rank", "epochs", "side", "role_weight", "precision"]
)
def test_recovery_cannot_change_base_adapter_or_numerical_scope(change):
    prior = original("candidate")
    values = copy.deepcopy(prior)
    values.update(precision="fp32", encode_chunk_size=1, min_free_gib=3)
    if change == "weights_sha":
        values["encoder"]["weights_sha256"] = "0" * 64
    elif change == "rank":
        values["encoder"]["adapter"]["rank"] = 64
    elif change == "epochs":
        values["epochs"] = 10
    elif change == "side":
        values["side"] = 224
    elif change == "role_weight":
        values["witness"]["weight"] = 0
    elif change == "precision":
        values["precision"] = "bf16"
    with pytest.raises(ValueError):
        validate_recipe(values, "candidate", prior)
