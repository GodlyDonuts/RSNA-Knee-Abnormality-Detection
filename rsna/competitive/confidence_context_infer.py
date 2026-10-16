"""Strict final source-confidence component export and unchanged FP32 serving.

The licensed MRI parent stays unchanged. This component contains only the new
head and whitelisted provenance, never clinical rows, features or predictions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from rsna.competitive.released_context import ContextResidual, tta_probabilities, window_metadata
from rsna.competitive.released_context_train import PARENT_SHA, Recipe, learning_rate
from rsna.schema import TARGETS

FORMAT = "licensed_parent_fixed10_source_confidence_component_v1"
VERSION = "licensed_source_confidence_context_trainer_v1"
ORIGINAL_SOURCE_GIT = "2ee0cffeae9dd023fef7ff98403e0508d4d22554"
FIXED_RECIPE = {**asdict(Recipe()), "objective": "source_confidence"}
SOURCE_GIT = "75bac344741e243264cbba645138a285d2630eda"
PARENT_SOURCE = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"
HELPER_SHA = "4ed7cd1978a4acac6c233c734914dbe713aed818f229eabd7f23fa10580fbee9"
POPULATION_SHA = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
LABEL_SOURCE_SHA = "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4"
FEATURE_SOURCE_SHA = "6c6ed23f5e7d32d824d22bf2bf282a319c4526efe7794589ce73127700c9e575"
PROVENANCE_KEYS = (
    "source_git",
    "parent_weights_sha256",
    "parent_source_sha256",
    "feature_pipeline_helper_sha256",
    "safe_population_uid_sha256",
    "label_source_sha256",
    "feature_index_sha256",
    "new_feature_archive_sha256",
    "eligible_release_sha256",
    "actual_engine_recovery_sha256",
    "original_feature_recovery_source_sha256",
    "confidence_source_git",
    "original_full_input_manifest_sha256",
    "confidence_engine_recovery_sha256",
    "uniform_full_parity_sha256",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def is_sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_final(state, terminal, recovery, parity):
    recipe = Recipe()
    fixed = {
        "source_git": ORIGINAL_SOURCE_GIT,
        "confidence_source_git": SOURCE_GIT,
        "confidence_engine": VERSION,
        "original_full_input_manifest_sha256": "a4c5b4910dc818293cdfb6ed59ba92bfa16faa5e0cb2c39c7a5e5b41df877ab0",
        "parent_weights_sha256": PARENT_SHA,
        "parent_source_sha256": PARENT_SOURCE,
        "feature_pipeline_helper_sha256": HELPER_SHA,
        "safe_population_uid_sha256": POPULATION_SHA,
        "label_source_sha256": LABEL_SOURCE_SHA,
        "original_feature_recovery_source_sha256": FEATURE_SOURCE_SHA,
    }
    b = state["binding"]
    if (
        state["version"] != VERSION
        or state.get("original_trainer_sha256")
        != "5addea8212501ba31021c739543a7fa627cf70066ad439fa0889ba5c5b3fa59f"
        or state["recipe"] != FIXED_RECIPE
        or b.get("objective") != "source_confidence"
        or any(b.get(k) != v for k, v in fixed.items())
        or state["population"] != 3887
        or state["completed"] != 2430
        or state["epoch"] != 10
        or state["cursor"] != 0
        or state["order"] is not None
        or state["scheduler"]
        != {"kind": "linear_warmup_cosine_v1", "batches_per_epoch": 243, "completed": 2430}
        or terminal.get("status") != "fixed_final_complete"
        or terminal.get("completed_updates") != 2430
        or terminal.get("epochs") != 10
        or terminal.get("studies") != 3887
        or terminal.get("trainable_parameters") != 402956
        or terminal.get("binding") != b
        or terminal.get("clinical_accuracy_evaluated") is not False
        or terminal.get("independent_internal_validation") is not False
        or terminal.get("CNN_parameters_fitted") != 0
        or terminal.get("engine") != VERSION
        or terminal.get("objective") != "source_confidence"
        or terminal.get("recipe") != FIXED_RECIPE
        or recovery.get("status")
        != "actual32_source_confidence_engine_recovery_and_uniform_parity_accepted"
        or recovery.get("fresh_processes") != 7
        or recovery.get("uniform_complete_numerical_state_and_predictions_equal_frozen_original")
        is not True
        or set(recovery.get("roles", {})) != {"uniform", "source_confidence"}
        or any(
            r.get("complete_model_AdamW_LR_order_counters_allRNG_history_equal") is not True
            or r.get("all32x12_prediction_bytes_equal") is not True
            or r.get("updates") != 4
            for r in recovery.get("roles", {}).values()
        )
        or parity.get("status")
        != "all3887_full_uniform_numerical_state_history_and_prediction_bytes_equal_original"
        or parity.get("updates") != 2430
        or parity.get("complete_model_AdamW_LR_order_counters_all_RNG_history") is not True
    ):
        raise ValueError(
            "Exact final2430-step admitted population, parent, source and recovery required"
        )
    history = state["history"]
    if len(history) != 2430:
        raise ValueError("Complete original optimizer history required")
    for i, record in enumerate(history):
        epoch, batch = divmod(i, 243)
        if (
            record["update"] != i + 1
            or record["epoch"] != epoch + 1
            or record["batch"] != batch + 1
            or record["studies"] != (15 if batch == 242 else 16)
            or record["lr"] != learning_rate(recipe, i, 243)
            or any(
                not math.isfinite(record[k])
                for k in ("loss", "gradient_norm", "max_abs_correction")
            )
            or not 0 <= record["max_abs_correction"] <= 0.5
        ):
            raise ValueError("Incomplete or altered fixed optimizer update record")
    return b


def export_component(checkpoint, terminal_path, recovery_path, parity_path, destination):
    checkpoint, terminal_path, recovery_path, parity_path, destination = map(
        Path, (checkpoint, terminal_path, recovery_path, parity_path, destination)
    )
    if destination.exists() or checkpoint.stat().st_size > 16 * 2**20:
        raise ValueError("Fresh small fixed-final component destination required")
    terminal, recovery = (json.loads(p.read_text()) for p in (terminal_path, recovery_path))
    original_sha = sha256(checkpoint)
    if terminal["checkpoint_sha256"] != original_sha:
        raise ValueError("Actual final checkpoint differs from its terminal")
    # Original trusted own-engine checkpoint only; exported serving file below
    # contains tensors/plain metadata and supports weights_only=True loading.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    b = validate_final(state, terminal, recovery, json.loads(parity_path.read_text()))
    provenance = {
        k: b[k]
        for k in PROVENANCE_KEYS
        if k not in ("confidence_engine_recovery_sha256", "uniform_full_parity_sha256")
    }
    provenance.update(
        confidence_engine_recovery_sha256=sha256(recovery_path),
        uniform_full_parity_sha256=sha256(parity_path),
    )
    if any(
        not is_sha(provenance[k])
        for k in provenance
        if k not in ("source_git", "confidence_source_git")
    ):
        raise ValueError("Complete original source/input SHA provenance required")
    value = {
        "format": FORMAT,
        "recipe": FIXED_RECIPE,
        "model": state["model"],
        "provenance": provenance,
        "targets": list(TARGETS),
        "population": 3887,
        "epoch": 10,
        "completed_updates": 2430,
        "original_checkpoint_sha256": original_sha,
        "terminal_sha256": sha256(terminal_path),
        "attributed_external_parent": True,
        "independent_internal_validation": False,
        "clinical_accuracy_claim": False,
    }
    model_from_component(value)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        __import__("os").fchmod(stream.fileno(), 0o600)
        torch.save(value, stream)
    return {
        "format": FORMAT,
        "component_sha256": sha256(destination),
        "component_bytes": destination.stat().st_size,
        "epoch": 10,
        "completed_updates": 2430,
        "population": 3887,
        "original_checkpoint_sha256": original_sha,
        "provenance": provenance,
        "attributed_external_parent": True,
        "independent_internal_validation": False,
        "clinical_accuracy_claim": False,
    }


def model_from_component(value):
    expected = {
        "format",
        "recipe",
        "model",
        "provenance",
        "targets",
        "population",
        "epoch",
        "completed_updates",
        "original_checkpoint_sha256",
        "terminal_sha256",
        "attributed_external_parent",
        "independent_internal_validation",
        "clinical_accuracy_claim",
    }
    if set(value) != expected:
        raise ValueError("Whitelisted fixed-final context component schema required")
    p = value["provenance"]
    if (
        value["format"] != FORMAT
        or set(p) != set(PROVENANCE_KEYS)
        or any(not is_sha(p[k]) for k in p if k not in ("source_git", "confidence_source_git"))
        or value["recipe"] != FIXED_RECIPE
        or value["targets"] != list(TARGETS)
        or (value["population"], value["epoch"], value["completed_updates"]) != (3887, 10, 2430)
        or value["attributed_external_parent"] is not True
        or value["independent_internal_validation"] is not False
        or value["clinical_accuracy_claim"] is not False
        or p.get("source_git") != ORIGINAL_SOURCE_GIT
        or p.get("confidence_source_git") != SOURCE_GIT
        or p.get("original_full_input_manifest_sha256")
        != "a4c5b4910dc818293cdfb6ed59ba92bfa16faa5e0cb2c39c7a5e5b41df877ab0"
        or p.get("parent_weights_sha256") != PARENT_SHA
        or p.get("parent_source_sha256") != PARENT_SOURCE
        or p.get("feature_pipeline_helper_sha256") != HELPER_SHA
        or p.get("safe_population_uid_sha256") != POPULATION_SHA
        or p.get("label_source_sha256") != LABEL_SOURCE_SHA
        or p.get("original_feature_recovery_source_sha256") != FEATURE_SOURCE_SHA
        or any(not is_sha(value[k]) for k in ("original_checkpoint_sha256", "terminal_sha256"))
    ):
        raise ValueError("Only exact admitted full3887/final10 context adaptation can serve")
    if any(
        not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or not torch.isfinite(x).all()
        for x in value["model"].values()
    ):
        raise ValueError("Complete finite FP32 context state required")
    model = ContextResidual().float().cpu()
    model.load_state_dict(value["model"], strict=True)
    if sum(x.numel() for x in model.parameters()) != 402956 or not model.output.count_nonzero():
        raise ValueError("Actual learned primary context graph required")
    return model.eval()


def load_component(path, expected_sha, *, device="cpu"):
    path = Path(path)
    if not is_sha(expected_sha) or path.stat().st_size > 4 * 2**20 or sha256(path) != expected_sha:
        raise ValueError("Exact small serving component hash required")
    value = torch.load(path, map_location="cpu", weights_only=True)
    return model_from_component(value).to(device), value["provenance"]


@torch.no_grad()
def predict_context_pair(model, features, parent_logits, centers, present):
    """Use both actual views with their registered depth convention, in FP32."""
    if (
        features.shape != (2, 94, 1024)
        or features.dtype != torch.float32
        or parent_logits.shape != (2, 12)
        or parent_logits.dtype != torch.float32
        or centers.shape != (94,)
        or centers.dtype != torch.long
        or present.shape != (94,)
        or present.dtype != torch.bool
        or any(m.training for m in model.modules())
    ):
        raise ValueError("Exact eval-only two-view parent features and center metadata required")
    slot, plain = window_metadata(centers)
    _, mirror = window_metadata(centers, mirrored=True)
    device = features.device
    inputs = (
        features,
        parent_logits,
        slot[None].expand(2, -1).to(device),
        torch.stack([plain, mirror]).to(device),
        present[None].expand(2, -1).to(device),
        torch.ones((2, 94), dtype=torch.bool, device=device),
    )
    with torch.autocast(device.type, enabled=False):
        logits, correction = model(*inputs)
    return tta_probabilities(logits[None])[0], correction
