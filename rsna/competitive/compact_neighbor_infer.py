"""Explicit deployment of the matched-center source-neighbor experiment.

Original N3 deployment continues to reject these weights. This separate loader
requires the new engine AND exact input policy, and consumes original native
volume donors through the same frozen training dataset view.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_bag import (
    MODEL_NAME,
    SAMPLER_VERSION,
    TIMM_VERSION,
    WEIGHT_SHA256,
    candidate_windows,
)
from rsna.competitive.compact_infer import DeploymentBagClassifier, _blueprint
from rsna.competitive.compact_source_neighbors import (
    ENGINE,
    POLICY,
    MatchedSourceNeighborDataset,
    select_window_tensor,
)
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.schema import TARGETS


def load_model(
    checkpoint: Path,
    expected_sha256: str,
    *,
    precision: str = "float32",
    device: torch.device = torch.device("cpu"),
    allow_diagnostic: bool = False,
) -> tuple[DeploymentBagClassifier, dict]:
    """Restore one complete scheduled epoch; reject diagnostic weights by default."""
    if type(allow_diagnostic) is not bool or file_sha256(checkpoint) != expected_sha256:
        raise ValueError("Exact checkpoint identity and explicit diagnostic policy required")
    if checkpoint.stat().st_size > 512 * 2**20:
        raise ValueError("Deploy the immutable full-model epoch, not an optimizer archive")
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        set(value) != {"schema_version", "binding", "epoch", "optimizer_steps", "model"}
        or value["schema_version"] != 1
    ):
        raise ValueError("Expected an immutable scheduled compact epoch")
    binding = value["binding"]
    diagnostic = binding.get("diagnostic_only")
    if type(diagnostic) is not bool or (diagnostic and not allow_diagnostic):
        raise ValueError("Engineering checkpoint is not competition eligible")
    if (
        binding.get("architecture") != MODEL_NAME
        or binding.get("timm_version") != TIMM_VERSION
        or binding.get("initialization_sha256") != WEIGHT_SHA256
        or binding.get("sampler") != SAMPLER_VERSION
        or binding.get("targets") != list(TARGETS)
        or binding.get("engine") != ENGINE
        or binding.get("input_policy") != POLICY
        or binding.get("native_input")
        != {"side": 384, "slices": 16, "max_series": 6, "mean": [0.5] * 3, "std": [0.5] * 3}
    ):
        raise ValueError("Checkpoint architecture/input/target provenance differs")
    epochs, count, steps_per_epoch = (2, 16, 2) if diagnostic else (16, 2247, 281)
    epoch, steps = value["epoch"], value["optimizer_steps"]
    if (
        type(epoch) is not int
        or not 1 <= epoch <= epochs
        or type(steps) is not int
        or steps != epoch * steps_per_epoch
        or binding.get("fitting_count") != count
        or binding.get("settings", {}).get("epochs") != epochs
        or binding.get("settings", {}).get("effective_batch") != 8
        or binding.get("total_steps") != epochs * steps_per_epoch
    ):
        raise ValueError("Checkpoint schedule or fitted scope differs")
    if any(v.is_floating_point() and v.dtype != torch.float32 for v in value["model"].values()):
        raise ValueError("Original full-state FP32 tensors required before execution qualification")
    model = DeploymentBagClassifier(_blueprint(), precision=precision).float()
    model.train(True)
    model.load_state_dict(value["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, {
        "checkpoint_sha256": expected_sha256,
        "epoch": epoch,
        "optimizer_steps": steps,
        "diagnostic_only": diagnostic,
        "competition_eligible": not diagnostic,
        "precision": precision,
        "actual_target_runtime_qualified": False,
        "input_policy": dict(POLICY),
        "engine": ENGINE,
    }


def inference_dataset(
    identifiers: list[str], series: pd.DataFrame, cache: Path, *, raw_split: str = "test_series"
):
    """Construct an image-only test view; target placeholders are unobserved zeros."""
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or any(not isinstance(x, str) or not x for x in identifiers)
    ):
        raise ValueError("Distinct original test identities required")
    rows = pd.DataFrame(
        {
            "study_uid": identifiers,
            **{target: 0.0 for target in TARGETS},
            **{target + "__mask": 0.0 for target in TARGETS},
        }
    )
    base = CompetitiveStudyDataset(
        rows,
        series,
        cache=cache,
        slices=16,
        side=384,
        max_series=6,
        augment=False,
        raw_split=raw_split,
        slice_sampling="uniform",
        series_sampling="selected",
    )
    return MatchedSourceNeighborDataset(base)


def prediction_rows(model, data, expected_ids: list[str], *, device):
    """Exact complete roster, with explicit unusable-image fallback only."""
    if getattr(data, "input_policy", None) != POLICY:
        raise ValueError("Explicit matched source-neighbor input required")
    if (
        not expected_ids
        or len(set(expected_ids)) != len(expected_ids)
        or any(not isinstance(x, str) or not x for x in expected_ids)
    ):
        raise ValueError("Distinct complete test identities required")
    allowed = set(expected_ids)
    rows = {}
    encoded = 0
    insufficient = 0
    edge_fallbacks = 0
    model.to(device).eval()
    with torch.inference_mode():
        for sample in data:
            uid = sample["study_uid"]
            if uid not in allowed or uid in rows:
                raise ValueError("Unexpected or duplicate test study")
            series_mask, slice_mask = sample["series_mask"], sample["slice_mask"]
            if not any(
                bool(active) and int(mask.sum()) >= 3
                for active, mask in zip(series_mask, slice_mask)
            ):
                rows[uid] = [0.5] * 12
                insufficient += 1
                continue
            candidates = candidate_windows(series_mask, slice_mask)
            windows = select_window_tensor(sample, candidates).to(device)
            probabilities = model(windows, chunk_size=12).sigmoid().float().cpu()
            if probabilities.shape != (12,) or not torch.isfinite(probabilities).all():
                raise ValueError("Finite twelve-finding predictions required")
            rows[uid] = probabilities.tolist()
            encoded += len(candidates)
            edge_fallbacks += sample["legacy_edge_fallback_windows"]
    if set(rows) != allowed:
        raise ValueError("Incomplete test roster")
    return {
        "rows": rows,
        "encoded_windows": encoded,
        "insufficient_window_studies": insufficient,
        "legacy_edge_fallback_windows": edge_fallbacks,
    }
