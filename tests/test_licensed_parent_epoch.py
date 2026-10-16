import copy
import random

import pytest
import torch

from rsna.competitive.licensed_parent_epoch import verify


@pytest.fixture
def completed():
    binding = {
        "source_git": "synthetic",
        "population": 3887,
        "world_size": 2,
        "total_steps": 1458,
        "recipe": {"epochs": 3, "windows": 94},
    }
    model = {"backbone.weight": torch.ones(2), "head.weight": torch.ones(1)}
    history = [
        {
            "update": i + 1,
            "epoch": 1,
            "batch": i + 1,
            "studies": 7 if i == 485 else 8,
            "loss": 0.6,
            "gradient_norm": 0.2,
            "encoder_lr": 1e-6,
            "head_lr": 1e-5,
            "scale": 1024,
        }
        for i in range(486)
    ]
    optimizer = {
        "param_groups": [{"params": [0], "lr": 1e-6}, {"params": [1], "lr": 1e-5}],
        "state": {
            i: {
                "step": torch.tensor(486.0),
                "exp_avg": torch.zeros(n),
                "exp_avg_sq": torch.zeros(n),
            }
            for i, n in ((0, 2), (1, 1))
        },
    }
    state = {
        "binding": binding,
        "completed": 486,
        "epoch": 1,
        "cursor": 0,
        "optimizer_boundary": True,
        "history": history,
        "schedule": {"completed": 486, "total": 1458, "kind": "warmup_cosine"},
        "rng_by_rank": [
            {
                "torch": torch.get_rng_state(),
                "cuda": torch.ones(16, dtype=torch.uint8),
                "python": random.getstate(),
                "numpy": ["MT19937", [0] * 624, 0, 0, 0],
            }
            for _ in range(2)
        ],
        "scaler_by_rank": [
            {
                "scale": 1024.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "_growth_tracker": 486,
            }
            for _ in range(2)
        ],
        "model": model,
        "optimizer": optimizer,
    }
    snapshot = {"binding": binding, "completed": 486, "epoch": 1, "model": copy.deepcopy(model)}
    receipt = {
        "status": "actual_whole94_full_parent_epoch_session_complete",
        "segment": 1,
        "completed": 486,
        "binding": {"source_git": "synthetic"},
    }
    return state, snapshot, receipt


def test_actual_epoch_requires_complete_encoder_head_and_AdamW_graph(completed):
    result = verify(*completed, expected_parameters=3)
    assert result["actual_AdamW_parameters"] == 3 and result["finite_contiguous_updates"] == 486


@pytest.mark.parametrize(
    "change",
    [
        "weights",
        "optimizer",
        "step",
        "source",
        "partial",
        "history",
        "nonfinite",
        "padding",
        "RNG",
        "LR",
    ],
)
def test_partial_or_changed_epoch_cannot_be_published_as_complete(completed, change):
    state, snapshot, receipt = completed
    if change == "weights":
        snapshot["model"]["backbone.weight"][0] = 2
    elif change == "optimizer":
        state["optimizer"]["state"].pop(0)
    elif change == "step":
        state["optimizer"]["state"][0]["step"] = torch.tensor(485.0)
    elif change == "source":
        receipt["binding"]["source_git"] = "changed"
    elif change == "partial":
        state["cursor"] = 1
    elif change == "history":
        state["history"].pop()
    elif change == "nonfinite":
        state["history"][0]["gradient_norm"] = float("nan")
    elif change == "padding":
        state["history"][-1]["studies"] = 8
    elif change == "RNG":
        state["rng_by_rank"].pop()
    else:
        state["optimizer"]["param_groups"][0]["lr"] = 0.1
    with pytest.raises(ValueError):
        verify(state, snapshot, receipt, expected_parameters=3)
