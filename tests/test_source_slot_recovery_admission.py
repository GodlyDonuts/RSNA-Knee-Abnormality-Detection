"""Reject confounded recipes before funding or reserving a trainer GPU."""

import json
from pathlib import Path

import pytest

from scripts.newton.probe_source_slot_recovery import RECIPE, recipe

ROOT = Path(__file__).resolve().parents[1]


def original():
    return {
        "configs": {
            "public": json.loads(
                (
                    ROOT
                    / "initiative/experiments/competitive-v1/public-input-recipe-v1/public-triplet-config.json"
                ).read_text()
            )
        }
    }


def test_only_classifier_changes():
    value = recipe(ROOT, original())
    assert value["side"] == 336
    assert value["batch_size"] * value["accumulation_steps"] == 8
    assert value["encoder"]["adapter"]["rank"] == 32


@pytest.mark.parametrize("change", ["learning_rate", "labels", "source", "sampling"])
def test_rejects_undeclared_recipe_changes(tmp_path, change):
    values = json.loads((ROOT / RECIPE).read_text())
    if change == "learning_rate":
        values["encoder_lr"] *= 2
    elif change == "labels":
        values["fitting_supervision"]["sha256"] = "0" * 64
    elif change == "source":
        values["classifier_strategy"]["source_sha256"] = "0" * 64
    else:
        values["slice_sampling"] = "stratified"
    path = tmp_path / RECIPE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        recipe(tmp_path, original())
