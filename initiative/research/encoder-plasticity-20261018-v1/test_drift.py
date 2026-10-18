import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("new_drift", Path(__file__).with_name("drift.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_actual_movement_and_frozen_buffers_are_separate():
    a = {
        "backbone.stages.0.weight": torch.tensor([3.0, 4.0]),
        "head.weight": torch.tensor([1.0]),
        "backbone.bn.running_mean": torch.zeros(2),
        "backbone.bn.num_batches_tracked": torch.tensor(0),
    }
    b = {k: v.clone() for k, v in a.items()}
    b["backbone.stages.0.weight"] *= 2
    r = m.summarize(a, b)
    assert r["encoder"]["relative_L2_movement"] == 1 and r["encoder"]["cosine"] == 1
    assert r["encoder"]["changed_elements"] == 2 and r["head"]["relative_L2_movement"] == 0
    assert r["BN_buffers"]["changed_elements"] == 0
    assert torch.equal(a["backbone.stages.0.weight"], torch.tensor([3.0, 4.0]))


@pytest.mark.parametrize("case", ["graph", "shape", "nonfinite", "integer"])
def test_mismatched_states_are_not_interpreted_as_plasticity(case):
    a = {"backbone.weight": torch.ones(2), "counter": torch.tensor(0)}
    b = {k: v.clone() for k, v in a.items()}
    if case == "graph":
        b["new"] = torch.zeros(1)
    if case == "shape":
        b["backbone.weight"] = torch.ones(3)
    if case == "nonfinite":
        b["backbone.weight"][0] = float("nan")
    if case == "integer":
        b["counter"] += 1
    with pytest.raises(ValueError):
        m.summarize(a, b)
