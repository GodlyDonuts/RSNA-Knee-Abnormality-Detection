"""Sampling comparisons must isolate data selection from initial model capacity."""

import json
from pathlib import Path

import pytest
import torch

from scripts.newton.verify_sampling_initialization import matched_configs, tensor_hash

FOLDER = Path("initiative/experiments/competitive-v1/stratified-slices-v1")


@pytest.mark.parametrize("kind", ["config", "pilot-config"])
def test_actual_sampling_comparison_configs_match(kind):
    control = json.loads((FOLDER / f"control-{kind}.json").read_text())
    candidate = json.loads((FOLDER / f"candidate-{kind}.json").read_text())
    first, second = matched_configs(control, candidate)
    assert first.slice_sampling == "uniform" and second.slice_sampling == "stratified"


@pytest.mark.parametrize(
    "change", [{"seed": 7}, {"side": 384}, {"augment": True}, {"head_lr": 0.01}]
)
def test_comparison_rejects_an_extra_intervention(change):
    control = json.loads((FOLDER / "control-pilot-config.json").read_text())
    candidate = {**control, "slice_sampling": "stratified", **change}
    with pytest.raises(ValueError, match="only configuration difference"):
        matched_configs(control, candidate)


def test_tensor_hash_binds_values_and_accepts_noncontiguous_layout():
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    assert tensor_hash(tensor) == tensor_hash(tensor.contiguous())
    assert tensor_hash(tensor) != tensor_hash(tensor + 1)
