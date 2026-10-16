"""Separate evaluation loader for the fixed N3 E12/E14/E16 parameter average.

The scheduled-epoch loader remains strict and rejects this derived engine. Saved
BN statistics are restored; derived graph caches are rebuilt by the same blueprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_average import BASE_ENGINE, ENGINE, POLICY
from rsna.competitive.compact_bag import MODEL_NAME, SAMPLER_VERSION, TIMM_VERSION, WEIGHT_SHA256
from rsna.competitive.compact_infer import DeploymentBagClassifier, _blueprint
from rsna.schema import TARGETS


def validate_derived(value: dict, source_shas: list[str]) -> dict:
    if (
        set(value) != {"schema_version", "binding", "epoch", "optimizer_steps", "model"}
        or value["schema_version"] != 1
        or value["epoch"] != 16
        or value["optimizer_steps"] != 4496
        or len(source_shas) != 3
        or len(set(source_shas)) != 3
        or any(len(x) != 64 or any(c not in "0123456789abcdef" for c in x) for x in source_shas)
    ):
        raise ValueError("Exact derived full16 state and three original source identities required")
    b = value["binding"]
    a = b.get("weight_average", {})
    if (
        b.get("engine") != ENGINE
        or b.get("diagnostic_only") is not False
        or b.get("architecture") != MODEL_NAME
        or b.get("timm_version") != TIMM_VERSION
        or b.get("initialization_sha256") != WEIGHT_SHA256
        or b.get("sampler") != SAMPLER_VERSION
        or b.get("targets") != list(TARGETS)
        or b.get("fitting_count") != 2247
        or b.get("total_steps") != 4496
        or b.get("settings", {}).get("epochs") != 16
        or b.get("settings", {}).get("effective_batch") != 8
        or b.get("input_policy") is not None
        or b.get("loss_policy") is not None
        or b.get("native_input")
        != {"side": 384, "slices": 16, "max_series": 6, "mean": [0.5] * 3, "std": [0.5] * 3}
        or a.get("policy") != POLICY
        or a.get("source_engine") != BASE_ENGINE
        or a.get("source_checkpoint_sha256") != source_shas
        or a.get("derived_not_original_scheduled_epoch") is not True
    ):
        raise ValueError(
            "Derived average policy, graph, fitted population or original scope differs"
        )
    if any(
        not isinstance(t, torch.Tensor)
        or (t.is_floating_point() and (t.dtype != torch.float32 or not torch.isfinite(t).all()))
        for t in value["model"].values()
    ):
        raise ValueError("Complete finite original FP32 learned/saved state required")
    return b


def load_model(
    checkpoint: Path,
    expected_sha256: str,
    source_shas: list[str],
    *,
    precision: str = "float32",
    device=torch.device("cpu"),
):
    if file_sha256(checkpoint) != expected_sha256 or checkpoint.stat().st_size > 512 * 2**20:
        raise ValueError("Exact derived full-model checkpoint identity required")
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    b = validate_derived(value, source_shas)
    model = DeploymentBagClassifier(_blueprint(), precision=precision).float()
    names = sorted(dict(model.named_parameters()))
    if (
        hashlib.sha256(json.dumps(names).encode()).hexdigest()
        != b["weight_average"]["learned_parameter_names_sha256"]
    ):
        raise ValueError("Derived learned-parameter inventory differs")
    model.train(True)
    model.load_state_dict(value["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, {
        "checkpoint_sha256": expected_sha256,
        "derived_engine": ENGINE,
        "source_checkpoint_sha256": list(source_shas),
        "diagnostic_only": False,
        "competition_eligible": True,
        "actual_target_runtime_qualified": False,
        "precision": precision,
        "policy": POLICY,
    }
