"""Saved provenance must be checked independently of physical input paths."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rsna.competitive.train import TrainConfig
from scripts.kaggle.witness_observe import digest, update_log, validate_bound_config


def test_saved_encoder_provenance_and_numerical_recipe_are_both_bound():
    original = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "initiative/audits/patch-witness-gpu-recovery-v1/candidate-engineering-config.json"
        ).read_text()
    )
    bound = asdict(
        TrainConfig(**dict(original, precision="fp32", encode_chunk_size=1, min_free_gib=3))
    )
    bound.pop("amp_init_scale", None)
    for name in ("relation", "fitting_supervision", "neighbor_channels", "witness"):
        if bound[name] is None:
            bound.pop(name)
    # A stored derived record intentionally has a different graph from the
    # physical encoder input spec; its whole original hash admits that graph.
    bound["encoder"] = {
        "fixture": "derived saved provenance",
        "weights_sha256": original["encoder"]["weights_sha256"],
    }
    sha = digest(bound["encoder"])
    validate_bound_config(bound, original, sha)
    for key, value in [("precision", "bf16"), ("side", 224), ("seed", 42)]:
        changed = copy.deepcopy(bound)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_bound_config(changed, original, sha)
    changed = copy.deepcopy(bound)
    changed["encoder"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_bound_config(changed, original, sha)


def test_actual_optimizer_log_discards_only_nonnumerical_timing(tmp_path):
    p = tmp_path / "stop.stdout"
    p.write_text(
        json.dumps({"status": "runtime"})
        + "\n"
        + json.dumps(
            {
                "epoch": 0,
                "update": 1,
                "optimizer_steps": 1,
                "loss": 0.7,
                "gradient_norm": 2.0,
                "elapsed_seconds": 123.0,
                "peak_gpu_allocated_bytes": 100,
            }
        )
        + "\n"
    )
    assert update_log(p) == [
        {"epoch": 0, "update": 1, "optimizer_steps": 1, "loss": 0.7, "gradient_norm": 2.0}
    ]
    p.write_text(json.dumps({"optimizer_steps": 1, "loss": float("nan")}) + "\n")
    with pytest.raises(ValueError):
        update_log(p)
