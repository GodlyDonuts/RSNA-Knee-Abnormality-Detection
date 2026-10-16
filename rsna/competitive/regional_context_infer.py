"""Typed fixed-final spatial-context export; no change to the frozen trainer.

The global-repeat control and spatial candidate have distinct serving identities.
Only small learned tensors and whitelisted provenance leave the fitting node.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch

from rsna.competitive.regional_context_train import (
    ENGINE,
    GENERIC_TRAINER_SHA,
    PARENT_SOURCE,
    POPULATIONS,
    REGIONAL_HELPER_SHA,
    REPRESENTATION,
    ROLES,
    feature_vector,
)
from rsna.competitive.released_context import ContextResidual, tta_probabilities, window_metadata
from rsna.competitive.released_context_infer import is_sha, sha256
from rsna.competitive.released_context_train import PARENT_SHA, VERSION, Recipe, learning_rate
from rsna.schema import TARGETS

FORMAT = "licensed_coherent_regional_fixed10_component_v1"
SOURCE_GIT = "46f5014ddb58da5bff2901995ac56025bf8dbbf7"
RECOVERY_SHA = "3b986f6328fcb236877e88bed9f09900131febaf93d1098726795bc6622c6ec6"
FIXED_PROVENANCE = {
    "source_git": SOURCE_GIT,
    "training_source_git": SOURCE_GIT,
    "representation": REPRESENTATION,
    "parent_weights_sha256": PARENT_SHA,
    "parent_source_sha256": PARENT_SOURCE,
    "regional_helper_sha256": REGIONAL_HELPER_SHA,
    "coherent_feature_pipeline_source_sha256": "16527682e6f216ca7189096c4d0d914c8811929869288e9d20910a9f232d29c4",
    "coherent_feature_index_sha256": "136d673f0351528e5e68f6c7cbbe72af08d0a633a9156be2ca49b4ee23e1dc43",
    "coherent_feature_archive_sha256": "68caf6a12b94780971e66747dde5721b60ad52b57b8e9357296f326ddfc50bc1",
    "population_uid_sha256": POPULATIONS["full3887"][1],
    "safe_fitting_population_sha256": POPULATIONS["full3887"][1],
    "label_source_sha256": "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4",
    "eligible_release_sha256": "2bf49fd2a8ae0b7c5af7e8c105fc3f2337845a92654d136fe67bc41a1e52fae7",
    "numerical_trainer_sha256": GENERIC_TRAINER_SHA,
    "actual_engine_recovery_sha256": RECOVERY_SHA,
    "permission_text_sha256": "34c44b20e4d7ace4eb523b0423b36d36dec826fe6ae54c66346796bba0190e79",
}


def validate_final(state, terminal, recovery):
    """Check the actual registered complete endpoint, never an engineering model."""
    recipe = Recipe(embedding_dim=8192)
    binding = state["binding"]
    role = binding.get("regional_role")
    if (
        role not in ROLES
        or any(binding.get(k) != v for k, v in FIXED_PROVENANCE.items())
        or binding.get("regional_engine") != ENGINE
        or binding.get("window_feature_dimension") != 8192
        or binding.get("diagnostic_only") is not False
        or state["version"] != VERSION
        or state["recipe"] != asdict(recipe)
        or (state["population"], state["epoch"], state["completed"], state["cursor"])
        != (3887, 10, 2430, 0)
        or state["order"] is not None
        or state["scheduler"]
        != {"kind": "linear_warmup_cosine_v1", "batches_per_epoch": 243, "completed": 2430}
        or terminal.get("engine") != ENGINE
        or terminal.get("role") != role
        or terminal.get("status") != "fixed_final_complete"
        or terminal.get("diagnostic_only") is not False
        or terminal.get("immutable_production_epoch_snapshots") is not True
        or terminal.get("completed_updates") != 2430
        or terminal.get("epochs") != 10
        or terminal.get("studies") != 3887
        or terminal.get("trainable_parameters") != 1334796
        or terminal.get("binding") != binding
        or terminal.get("clinical_accuracy_evaluated") is not False
        or terminal.get("independent_internal_validation") is not False
        or terminal.get("CNN_parameters_fitted") != 0
        or recovery.get("status") != "actual32_both_regional_roles_complete_CPU_recovery_accepted"
        or recovery.get("engine") != ENGINE
        or recovery.get("fresh_training_processes") != 6
        or set(recovery.get("roles", {})) != set(ROLES)
        or any(
            recovery["roles"][r].get(k) is not True
            for r in ROLES
            for k in (
                "complete_decoded_state_exact",
                "all32x12_prediction_bytes_exact",
                "actual_nonzero_learning",
            )
        )
    ):
        raise ValueError("Exact complete2430-step matched role, population and recovery required")
    if len(state["history"]) != 2430:
        raise ValueError("Complete original fixed-endpoint update history required")
    for i, row in enumerate(state["history"]):
        epoch, batch = divmod(i, 243)
        if (
            row["update"] != i + 1
            or row["epoch"] != epoch + 1
            or row["batch"] != batch + 1
            or row["studies"] != (15 if batch == 242 else 16)
            or row["lr"] != learning_rate(recipe, i, 243)
            or any(
                not math.isfinite(row[k]) for k in ("loss", "gradient_norm", "max_abs_correction")
            )
            or not 0 <= row["max_abs_correction"] <= 0.5
        ):
            raise ValueError("Original finite fixed-recipe optimizer records required")
    return role


def model_from_component(value, role):
    expected = {
        "format",
        "role",
        "engine",
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
    if (
        set(value) != expected
        or role not in ROLES
        or value["role"] != role
        or value["format"] != FORMAT
        or value["engine"] != ENGINE
        or value["recipe"] != asdict(Recipe(embedding_dim=8192))
        or value["provenance"] != FIXED_PROVENANCE
        or value["targets"] != list(TARGETS)
        or (value["population"], value["epoch"], value["completed_updates"]) != (3887, 10, 2430)
        or value["attributed_external_parent"] is not True
        or value["independent_internal_validation"] is not False
        or value["clinical_accuracy_claim"] is not False
        or any(not is_sha(value[k]) for k in ("original_checkpoint_sha256", "terminal_sha256"))
    ):
        raise ValueError(
            "Whitelisted exact registered role and final spatial-component schema required"
        )
    if any(
        not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or not torch.isfinite(x).all()
        for x in value["model"].values()
    ):
        raise ValueError("Complete finite FP32 spatial readout state required")
    model = ContextResidual(8192).float().cpu()
    model.load_state_dict(value["model"], strict=True)
    if sum(p.numel() for p in model.parameters()) != 1334796 or not model.output.count_nonzero():
        raise ValueError("Actual nonzero learned matched spatial graph required")
    model._regional_role = role
    return model.eval()


def export_component(checkpoint, terminal_path, recovery_path, destination, role):
    checkpoint, terminal_path, recovery_path, destination = map(
        Path, (checkpoint, terminal_path, recovery_path, destination)
    )
    if destination.exists() or checkpoint.stat().st_size > 17 * 2**20:
        raise ValueError("Fresh bounded original fixed-final export required")
    terminal = json.loads(terminal_path.read_text())
    if terminal["checkpoint_sha256"] != sha256(checkpoint) or sha256(recovery_path) != RECOVERY_SHA:
        raise ValueError("Original final checkpoint and actual recovery bytes required")
    # Own trusted trainer checkpoint; the exported file uses weights_only=True.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if validate_final(state, terminal, json.loads(recovery_path.read_text())) != role:
        raise ValueError("Export role differs from the actual original fit")
    value = {
        "format": FORMAT,
        "role": role,
        "engine": ENGINE,
        "recipe": asdict(Recipe(embedding_dim=8192)),
        "model": state["model"],
        "provenance": {k: state["binding"][k] for k in FIXED_PROVENANCE},
        "targets": list(TARGETS),
        "population": 3887,
        "epoch": 10,
        "completed_updates": 2430,
        "original_checkpoint_sha256": sha256(checkpoint),
        "terminal_sha256": sha256(terminal_path),
        "attributed_external_parent": True,
        "independent_internal_validation": False,
        "clinical_accuracy_claim": False,
    }
    model_from_component(value, role)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        torch.save(value, stream)
    return {
        "format": FORMAT,
        "role": role,
        "component_sha256": sha256(destination),
        "component_bytes": destination.stat().st_size,
        "original_checkpoint_sha256": value["original_checkpoint_sha256"],
        "epoch": 10,
        "completed_updates": 2430,
        "population": 3887,
        "clinical_accuracy_claim": False,
    }


def load_component(path, expected_sha, role, *, device="cpu"):
    path = Path(path)
    if not is_sha(expected_sha) or path.stat().st_size > 6 * 2**20 or sha256(path) != expected_sha:
        raise ValueError("Exact small spatial-component bytes required")
    value = torch.load(path, map_location="cpu", weights_only=True)
    return model_from_component(value, role).to(device), value["provenance"]


@torch.no_grad()
def predict_pair(model, role, globals_, regions, parent_logits, centers, present):
    """Same two views and deliberate FP16 regional quantization used in fitting."""
    if (
        role not in ROLES
        or globals_.shape != (2, 94, 1024)
        or regions.shape != (2, 94, 7, 1024)
        or parent_logits.shape != (2, 12)
        or parent_logits.dtype != torch.float32
        or not torch.isfinite(parent_logits).all()
        or centers.shape != (94,)
        or centers.dtype != torch.long
        or present.shape != (94,)
        or present.dtype != torch.bool
        or not present.any()
        or model.embedding_dim != 8192
        or getattr(model, "_regional_role", None) != role
        or any(m.training for m in model.modules())
        or any(t.device != globals_.device for t in (regions, parent_logits, centers, present))
    ):
        raise ValueError("Exact eval-only coherent two-view spatial inputs required")
    features = feature_vector(globals_, regions, role)
    slots, plain = window_metadata(centers)
    _, mirror = window_metadata(centers, mirrored=True)
    with torch.autocast(features.device.type, enabled=False):
        logits, correction = model(
            features,
            parent_logits,
            slots[None].expand(2, -1),
            torch.stack([plain, mirror]),
            present[None].expand(2, -1),
            torch.ones((2, 94), dtype=torch.bool, device=features.device),
        )
    return tta_probabilities(logits[None])[0], correction
