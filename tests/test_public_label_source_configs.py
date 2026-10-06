"""The supervision comparison keeps the strong adapter model and full budget."""

import copy
import hashlib
import json
from pathlib import Path

from rsna.competitive.train import TrainConfig


def test_fixed_full_recipe_changes_only_label_source_enabled():
    root = Path("initiative/experiments/competitive-v1/public-label-source-v1")
    control = json.loads((root / "control-config.json").read_text())
    public = json.loads((root / "public-config.json").read_text())
    for values in (control, public):
        TrainConfig(**values).validate()
        assert values["encoder"]["adapter"]["rank"] == 32
        assert values["epochs"] == 10 and values["warmup_updates"] == 281
        assert values["batch_size"] * values["accumulation_steps"] == 8
        assert values["side"] == 384 and values["precision"] == "fp16"
        assert (
            values["fitting_supervision"]["sha256"]
            == hashlib.sha256((root / "fold0-overlay.json").read_bytes()).hexdigest()
        )
    assert public["fitting_supervision"]["enabled"] is True
    other = copy.deepcopy(public)
    other["fitting_supervision"]["enabled"] = False
    assert control == other
