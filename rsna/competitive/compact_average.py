"""Predeclared late-parameter average; final-epoch buffers are never averaged.

Derived weights have their own engine identity. The original scheduled-epoch
inference loader must continue rejecting them until distinct serving acceptance.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence

import torch

ENGINE = "native_coatnet_bag12_N3_late_parameter_average_v1"
EPOCHS = (12, 14, 16)
POLICY = {
    "version": "N3_equal_parameters_E12_E14_E16_final_buffers_v1",
    "epochs": list(EPOCHS),
    "parameter_weights": [1 / 3, 1 / 3, 1 / 3],
    "accumulation": "CPU_float64_then_cast_once_to_original_float32",
    "all_buffers": "retain_exact_epoch16_including_BN_running_statistics_and_counters",
    "selection": "fixed_before_observing_derived_model_or_final_epoch_scores",
}
BASE_ENGINE = "native_coatnet_bag12_N3_full_v1"


def average_parameters(
    states: Sequence[Mapping[str, torch.Tensor]], parameter_names: set[str]
) -> dict[str, torch.Tensor]:
    """Average only learned FP32 parameters; retain ALL final-state buffers.

    Input state dictionaries are immutable and must describe the exact same graph.
    CPU float64 accumulation limits order-dependent rounding without assuming that
    averaging BN statistics or integer counters is a sensible operation.
    """
    if len(states) != 3 or not parameter_names:
        raise ValueError("Exactly three full states and explicit learned parameter names required")
    keys = set(states[0])
    if not parameter_names <= keys or any(set(state) != keys for state in states):
        raise ValueError("Complete identical model graph and actual parameter names required")
    result = {}
    for name in sorted(keys):
        values = [state[name] for state in states]
        if any(
            not isinstance(v, torch.Tensor)
            or v.layout != torch.strided
            or v.device.type != "cpu"
            or v.shape != values[0].shape
            or v.dtype != values[0].dtype
            for v in values
        ):
            raise ValueError("Equal dense CPU tensor shapes and original dtypes required")
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in values):
            raise ValueError("Nonfinite source model tensor")
        if name in parameter_names:
            if values[0].dtype != torch.float32:
                raise ValueError("Actual learned parameters must retain full original FP32 values")
            total = values[0].double().clone()
            total.add_(values[1].double()).add_(values[2].double()).div_(3)
            result[name] = total.float()
        else:
            result[name] = values[-1].clone()
    return result


def validate_sources(checkpoints: Sequence[dict]) -> dict:
    """Only fixed original N3 epochs12/14/16 from one fit may enter this average."""
    if len(checkpoints) != 3:
        raise ValueError("Exactly the three preregistered late epochs required")
    binding = checkpoints[-1].get("binding")
    if not isinstance(binding, dict) or (
        binding.get("engine") != BASE_ENGINE
        or binding.get("diagnostic_only") is not False
        or binding.get("fitting_count") != 2247
        or binding.get("total_steps") != 4496
        or binding.get("settings", {}).get("epochs") != 16
        or binding.get("settings", {}).get("effective_batch") != 8
        or binding.get("input_policy") is not None
        or binding.get("loss_policy") is not None
        or binding.get("weight_average") is not None
    ):
        raise ValueError("One original full16 unweighted selected-slot N3 fit required")
    for epoch, value in zip(EPOCHS, checkpoints, strict=True):
        if (
            set(value) != {"schema_version", "binding", "epoch", "optimizer_steps", "model"}
            or value["schema_version"] != 1
            or type(value["epoch"]) is not int
            or value["epoch"] != epoch
            or type(value["optimizer_steps"]) is not int
            or value["optimizer_steps"] != epoch * 281
            or value["binding"] != binding
        ):
            raise ValueError(
                "Exact same source/cohort/runtime/schedule and fixed epoch order required"
            )
    return copy.deepcopy(binding)


def derive_checkpoint(
    checkpoints: Sequence[dict], parameter_names: set[str], source_shas: list[str]
):
    binding = validate_sources(checkpoints)
    if (
        len(source_shas) != 3
        or len(set(source_shas)) != 3
        or any(len(s) != 64 or any(c not in "0123456789abcdef" for c in s) for s in source_shas)
    ):
        raise ValueError("Three distinct whole-byte source checkpoint SHA256 identities required")
    learned = hashlib.sha256(json.dumps(sorted(parameter_names)).encode()).hexdigest()
    binding["engine"] = ENGINE
    binding["weight_average"] = {
        "policy": copy.deepcopy(POLICY),
        "source_checkpoint_sha256": list(source_shas),
        "learned_parameter_names_sha256": learned,
        "source_engine": BASE_ENGINE,
        "derived_not_original_scheduled_epoch": True,
    }
    return {
        "schema_version": 1,
        "binding": binding,
        "epoch": 16,
        "optimizer_steps": 4496,
        "model": average_parameters([c["model"] for c in checkpoints], parameter_names),
    }
