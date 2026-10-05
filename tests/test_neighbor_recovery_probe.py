"""A successful-looking exit cannot substitute for complete recovery evidence."""

import copy

import pytest
import torch

from scripts.newton.probe_neighbor_trainer_recovery import compare_recovery


def records():
    full = dict(
        schema_version=2,
        optimizer_boundary=True,
        epoch=2,
        next_update=0,
        step=4,
        binding={"config": {"neighbor_channels": {"mode": "physical", "strength": 0.25}}},
        model={"adapter": torch.tensor([0.25])},
        model_storage={"format": "frozen_base_delta_v1", "base_sha256": "a" * 64},
        optimizer={"state": {0: {"exp_avg": torch.tensor([0.5])}}},
        rng_by_rank=[{"torch": torch.tensor([1, 2], dtype=torch.uint8)}],
    )
    partial = copy.deepcopy(full)
    partial.update(epoch=0, next_update=1, step=1)
    rows = [
        dict(
            epoch=e,
            update=u,
            optimizer_steps=s,
            loss=0.5,
            gradient_norm=1.0,
            observed_cells=12,
            elapsed_seconds=float(s),
        )
        for e, u, s in [(0, 1, 1), (0, 2, 2), (1, 1, 3), (1, 2, 4)]
    ]
    return (
        full,
        copy.deepcopy(full),
        partial,
        {"updates": rows},
        {"updates": rows[:1]},
        {"updates": rows[1:]},
        b"predictions",
        b"predictions",
    )


def test_complete_recovery_accepts_equal_state_despite_different_wall_clock():
    args = list(records())
    args[4] = copy.deepcopy(args[4])
    args[4]["updates"][0]["elapsed_seconds"] = 999
    assert len(compare_recovery(*args)) == 64


@pytest.mark.parametrize(
    "mutation", ["AdamW", "RNG", "base", "CSV", "update", "model_missing", "AdamW_missing"]
)
def test_recovery_rejects_any_missing_or_changed_semantic_obligation(mutation):
    args = list(records())
    resumed = args[1]
    if mutation == "AdamW":
        resumed["optimizer"]["state"][0]["exp_avg"][0] = 0.4
    elif mutation == "RNG":
        resumed["rng_by_rank"][0]["torch"][0] = 3
    elif mutation == "base":
        resumed["model_storage"]["base_sha256"] = "b" * 64
    elif mutation == "CSV":
        args[7] = b"different"
    elif mutation == "update":
        args[4] = copy.deepcopy(args[4])
        args[4]["updates"][0]["loss"] = 0.1
    elif mutation == "model_missing":
        resumed["model"] = {}
    else:
        resumed["optimizer"]["state"] = {}
    with pytest.raises(ValueError):
        compare_recovery(*args)
