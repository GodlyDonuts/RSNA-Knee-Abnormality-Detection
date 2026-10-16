import copy

import pytest
import torch
from torch import nn

from rsna.competitive.compact_average import (
    BASE_ENGINE,
    ENGINE,
    average_parameters,
    derive_checkpoint,
    validate_sources,
)


def sources():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    params = set(dict(model.named_parameters()))
    binding = {
        "engine": BASE_ENGINE,
        "diagnostic_only": False,
        "fitting_count": 2247,
        "total_steps": 4496,
        "settings": {"epochs": 16, "effective_batch": 8},
        "fitting_uids_sha256": "fixed-original-fit",
        "source_sha256": {"frozen_trainer": "original-source"},
    }
    values = []
    for epoch, scale in zip([12, 14, 16], [1, 3, 8], strict=True):
        state = {k: v.clone() for k, v in model.state_dict().items()}
        for name, value in state.items():
            value.fill_(scale if name in params else 10 * scale)
        values.append(
            {
                "schema_version": 1,
                "binding": copy.deepcopy(binding),
                "epoch": epoch,
                "optimizer_steps": epoch * 281,
                "model": state,
            }
        )
    return params, values


def test_average_learned_parameters_but_preserve_actual_final_BN_state_and_inputs():
    params, values = sources()
    old = copy.deepcopy(values)
    result = derive_checkpoint(values, params, ["a" * 64, "b" * 64, "c" * 64])
    assert result["binding"]["engine"] == ENGINE != BASE_ENGINE
    for key, value in result["model"].items():
        assert torch.equal(value, torch.full_like(value, 4 if key in params else 80))
        assert value.data_ptr() != values[-1]["model"][key].data_ptr()
    for actual, original in zip(values, old, strict=True):
        for key in actual["model"]:
            assert torch.equal(actual["model"][key], original["model"][key])


@pytest.mark.parametrize("change", ["order", "cohort", "source", "diagnostic", "weighted", "step"])
def test_sources_from_other_fits_or_objectives_cannot_be_averaged(change):
    _, values = sources()
    if change == "order":
        values.reverse()
    elif change == "cohort":
        values[0]["binding"]["fitting_uids_sha256"] = "another-patient-set"
    elif change == "source":
        values[0]["binding"]["source_sha256"] = {}
    elif change == "diagnostic":
        for v in values:
            v["binding"]["diagnostic_only"] = True
    elif change == "weighted":
        for v in values:
            v["binding"]["loss_policy"] = {"different": True}
    else:
        values[1]["optimizer_steps"] -= 1
    with pytest.raises(ValueError):
        validate_sources(values)


@pytest.mark.parametrize("bad", ["nan_buffer", "shape", "half", "missing", "unknown_parameter"])
def test_corrupt_or_incompatible_graph_cannot_be_silently_averaged(bad):
    params, values = sources()
    states = [v["model"] for v in values]
    if bad == "nan_buffer":
        states[0]["1.running_var"][0] = float("nan")
    elif bad == "shape":
        states[1]["0.weight"] = torch.ones(3, 2)
    elif bad == "half":
        for s in states:
            s["0.weight"] = s["0.weight"].half()
    elif bad == "missing":
        del states[0]["1.running_mean"]
    else:
        params.add("nonexistent")
    with pytest.raises(ValueError):
        average_parameters(states, params)


def test_entire_final_buffer_policy_includes_non_BN_buffers():
    states = [
        {"weight": torch.tensor([float(i)]), "anything": torch.tensor([i])} for i in [1, 3, 8]
    ]
    out = average_parameters(states, {"weight"})
    assert out["weight"].item() == 4
    assert out["anything"].item() == 8
