"""Serve the four preserved fixed-final readouts on the unchanged MRI encoder.

Source labels and training coverage are the only factorial choices. This does
not turn released-encoder fitting results into independent OOF evidence.
"""

from __future__ import annotations

import hashlib
import json
from types import MethodType

import torch

from rsna.schema import TARGETS

SOURCE = "eed22c54d56b0416d666c9a7f2fb53e4a370d957"
PARENT = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
PARENT_SOURCE = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"
READOUT = "8ce6546b4ec7a50fe6a3e9c0a0dd51e813d78a418ad61094427282bbbd72b8c5"
POPULATION = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
ENGINE = "cfd8e8a96cebf32137533f4d842993c7f46f013d04b2452245fb2410ce8c9001"
PUBLICATION = "c4cba4c210542307436a3bdc89034f5f99633ce4"
FINAL_SHAS = {
    ("original", 16): "20689ece07a123e9e2d2c475450f1405cb7c2d3729d0b71b3a0683a609a77f4c",
    ("original", 94): "006cfdfc9b9ecfe87544844e41ffd18c2859bd3b68314917aa289ffdd04f3cb6",
    ("released_S6", 16): "366f3ac7cef10cf1f1f42b7e136591e2f4f87ae51936c05751578728f56808d4",
    ("released_S6", 94): "a1852109f7781ea8c953d02fbe95fab57c941331ea04f55c33f3525b5fa58b12",
}
SHAPES = {
    "clsW": (12, 1024),
    "clsb": (12,),
    "norm.weight": (1024,),
    "norm.bias": (1024,),
    "att.0.weight": (256, 1024),
    "att.0.bias": (256,),
    "att.3.weight": (12, 256),
    "att.3.bias": (12,),
}


def fixed_recipe(role, windows):
    if (role, windows) not in FINAL_SHAS or type(windows) is not int:
        raise ValueError("One of four original fixed-final roles required")
    return {
        "label_role": role,
        "windows": windows,
        "epochs": 3,
        "effective_batch": 8,
        "seed": 20261016,
        "head_lr": 3e-5,
        "weight_decay": 0.02,
        "warmup_fraction": 0.1,
        "final_lr_fraction": 0.01,
        "source_weight": 0.25,
        "clip": 1.0,
    }


def _exact(a, b):
    if isinstance(b, dict):
        return (
            isinstance(a, dict)
            and a.keys() == b.keys()
            and all(_exact(a[k], v) for k, v in b.items())
        )
    if isinstance(b, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_exact(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def provenance(full, publication, recovery, role, windows, model_sha):
    fixed_recipe(role, windows)
    key = role + "-" + str(windows)
    row = full.get("roles", {}).get(key, {})
    roles = {r + "-" + str(w) for r, w in FINAL_SHAS}
    if (
        full.get("status") != "actual_four_role_cached_MIL_fixed3_full3887_completed"
        or full.get("source_git") != SOURCE
        or full.get("CPU_job") != "948122"
        or set(full.get("roles", {})) != roles
        or row.get("actual_updates") != 1458
        or row.get("fixed_epochs") != 3
        or row.get("population") != 3887
        or row.get("encoder_trained") is not False
        or row.get("trainable_head_parameters") != 279832
        or row.get("source_label_role") != role
        or row.get("training_windows") != windows
        or row.get("inference_windows") != 94
        or publication.get("status")
        != "all12_cached_MIL_factorial_epochs_private_full_download_size_SHA_verified"
        or publication.get("source_git") != SOURCE
        or publication.get("actual_full_fit_CPU_job") != "948122"
        or publication.get("CPU_job") != "948191"
        or publication.get("revision") != PUBLICATION
        or publication.get("objects") != 60
        or publication.get("encoder_weight_copies") != 0
        or recovery.get("status") != "actual_four_role_CPU_head_engine_recovery_accepted"
        or recovery.get("CPU_job") != "948113"
        or recovery.get("source_git") != SOURCE
        or set(recovery.get("roles", {})) != roles
        or recovery.get("old_new_source_values_actually_change_learned_tensors") is not True
        or full.get("clinical_or_Kaggle_accuracy_claim") is not False
        or model_sha != FINAL_SHAS[role, windows]
    ):
        raise ValueError("Actual fixed-final factorial training/private/recovery proofs required")
    for value in recovery["roles"].values():
        if (
            value.get("finite_updates") != 4
            or value.get("train_studies") != 16
            or value.get("disjoint_engineering_prediction_studies") != 16
            or value.get("prediction_shape") != [16, 12]
            or value.get("prediction_bytes_equal") is not True
            or value.get(
                "complete_decoded_model_AdamW_LR_cursors_allRNG_history_and_epoch_snapshots_equal"
            )
            is not True
            or value.get("actual_learned_tensor_change") is not True
        ):
            raise ValueError("All four actual complete engine recoveries required")
    entries = publication.get("epochs", [])
    identities = {(e.get("role"), e.get("training_windows"), e.get("epoch")) for e in entries}
    if len(entries) != 12 or identities != {(r, w, e) for r, w in FINAL_SHAS for e in (1, 2, 3)}:
        raise ValueError("All twelve distinct privately verified immutable epochs required")
    selected = next(
        e for e in entries if (e["role"], e["training_windows"], e["epoch"]) == (role, windows, 3)
    )
    model = {"bytes": 1126029, "sha256": model_sha}
    if (
        selected.get("head_model") != model
        or selected.get("completed_updates") != 1458
        or selected.get("actual_complete_optimizer_state_verified") is not True
        or row.get("immutable_epochs", [{}])[-1].get("model") != model
    ):
        raise ValueError("Exact private fixed-final artifact and complete optimizer proof required")
    return {"role": role, "training_windows": windows, "serving_windows": 94, "model": model}


def validate(snapshot, full, publication, recovery, role, windows, model_sha):
    accepted = provenance(full, publication, recovery, role, windows, model_sha)
    b = snapshot.get("binding", {})
    fixed = {
        "version": "licensed_cached_MIL_label_coverage_factorial_v1",
        "source_git": SOURCE,
        "parent_checkpoint_sha256": PARENT,
        "original_readout_sha256": READOUT,
        "population": 3887,
        "population_sha256": POPULATION,
        "total_steps": 1458,
        "target_order": list(TARGETS),
        "diagnostic_only": False,
        "source_label_role": role,
        "protected_gold_confirmation_used": False,
        "source_encoder_exposed_cohort_no_independent_OOF": True,
        "initial_fixed_parent_CNN_unchanged": True,
        "head_dropout_fixed_zero_all_four_roles": True,
        "accepted_masks_unchanged": True,
        "parent_full_teacher_unchanged": True,
        "precision": "CPU_FP32",
        "head_dropout": 0.0,
        "recipe": fixed_recipe(role, windows),
    }
    original_recovery = {k: v for k, v in recovery.items() if k != "elapsed_seconds"}
    expected_recovery_sha = hashlib.sha256(
        json.dumps(original_recovery, indent=2, sort_keys=True).encode()
    ).hexdigest()
    if (
        set(snapshot) != {"binding", "epoch", "completed", "model"}
        or type(snapshot.get("epoch")) is not int
        or snapshot["epoch"] != 3
        or type(snapshot.get("completed")) is not int
        or snapshot["completed"] != 1458
        or any(not _exact(b.get(k), v) for k, v in fixed.items())
        or b.get("actual_recovery_sha256") != expected_recovery_sha
        or b.get("source_files_sha256") != full.get("source_files_sha256")
        or b.get("source_files_sha256", {}).get("rsna/competitive/licensed_cached_mil.py") != ENGINE
    ):
        raise ValueError(
            "Exact final source, recipe, full cohort and original recovery binding required"
        )
    values = snapshot["model"]
    if set(values) != set(SHAPES) or any(
        not isinstance(t, torch.Tensor)
        or t.dtype != torch.float32
        or tuple(t.shape) != SHAPES[n]
        or not torch.isfinite(t).all()
        for n, t in values.items()
    ):
        raise ValueError("All eight finite FP32 learned readout tensors required")
    return {**accepted, "epoch": 3, "completed": 1458, "encoder_changed": False}


def fp32_head(self, features):
    """Keep the accepted FP32 head outside the encoder's fp16 autocast region."""
    if (
        features.ndim != 3
        or features.shape[1:] != (94, 1024)
        or features.device.type not in ("cpu", "cuda")
        or not features.is_floating_point()
        or not torch.isfinite(features).all()
        or self.training
        or self.norm.training
        or self.att.training
    ):
        raise ValueError("Complete full94 finite encoder features and eval-only head required")
    with torch.autocast(device_type=features.device.type, enabled=False):
        h = self.norm(features.float())
        attention = self.att(h).float().softmax(1)
        return (torch.einsum("bkn,bkf->bnf", attention, h) * self.clsW).sum(-1) + self.clsb


def install(model, snapshot):
    """Replace only the validated head tensors on a freshly loaded parent model.

    Caller must validate the complete snapshot and original parent file hashes.
    This function never loads, replaces or converts the encoder's parameters.
    """
    values = snapshot["model"]
    parameters = dict(model.named_parameters())
    if (
        set(values) != set(SHAPES)
        or model.training
        or any(
            n not in parameters
            or tuple(parameters[n].shape) != shape
            or parameters[n].dtype != torch.float32
            or values[n].dtype != torch.float32
            or tuple(values[n].shape) != shape
            or not torch.isfinite(values[n]).all()
            for n, shape in SHAPES.items()
        )
    ):
        raise ValueError("Complete original FP32 readout graph and learned state required")
    with torch.no_grad():
        for n, t in values.items():
            parameters[n].copy_(t.to(parameters[n].device))
    model.head = MethodType(fp32_head, model)
    return model
