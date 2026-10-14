"""Explicit deployment admission for the fixed reviewed-Effusion comparison.

The trained rsna package and checkpoint bytes remain unchanged. Only the prior
experimental-authorization guard is replaced, after exact admission validation;
model loading, source checks, MRI transforms and sigmoid predictions use the
original infer.run. This does not admit external-source or diagnostic weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rsna.competitive import infer

SOURCE = "665c0d515410df24f1f37ed5d9f8cb7a0f0fc480"
OVERLAY = "936e3e2af058726fe3572b7e4ad1680c44f2fd82ad6e9840e2607a1b090ff018"
FIT = "e9d38ee21a114973177b89f4602129c7dc047927b8499991d3be30fd89737f3a"
OUTER = "2f365bb0ce8e052f22dcab78bb2bbb7e03c0919ab28ba94ab9d3d8d6e08f0fdb"

# Owner daily-queue admission: exact all-development RSNA-only epochs only.
# Values are root's sanitized sidecar bindings for original job 908454; see
# initiative/audits/daily-kaggle-comparison-queue-20261014-v1/queue.json.
QUEUE_ID = "daily-kaggle-comparison-queue-20261014-v1"
QUEUE_STATUS = "owner_authorized_daily_kaggle_comparison_queue"
QUEUE_ROLE = "all_development_public_source_epoch"
ALL_DEVELOPMENT_SOURCE = "1bcd6c0e2e2a47d6894d5c545d6950e0c6f0d5b9"
ALL_DEVELOPMENT_BINDING_SHA = "5c8e14732143a479073b43a24ff7851775a1dd550dc8deb0a9291b8f8091597e"
ALL_DEVELOPMENT_CODE_SHA = "86db55a2769a676b0015d24e464de4be025c2c4abc2845363cb2313fc4ea2b7e"
ALL_DEVELOPMENT_EPOCHS = {
    1: ("74a24c6468a3e2e1ff643bb77eb316207c11feab9e7af0514a407c63a0cf66d5", 426),
    2: ("9422444c2144e925da5c6feb845a7890529fbe82a2decd2dafdac6fa9e46f410", 852),
    3: ("59adcdc9ed72dac94e5c8213a8d1fc61c917b5ccb800250be0cc457ccaad7d90", 1278),
    4: ("794de246d2703e87c2f2235bf9fcac5d39673bca1a11a7fbbe2684c58636de8d", 1704),
    5: ("8a81970c35e54f7c0417690c7344546542f6cf34acc74cbf347b294092537140", 2130),
}
ALL_DEVELOPMENT_SUPERVISION = {
    "kind": "label_source_final_v1",
    "overlay_sha256": "54b5a652949120764a596b4d495f11950ff103a909b7efc5fe59da9cd6f2fe85",
    "release_sha256": "2bf49fd2a8ae0b7c5af7e8c105fc3f2337845a92654d136fe67bc41a1e52fae7",
    "folds_sha256": "fe590e5072a0d82abeb377598aeef0021bbe77fe4529a19b9826a7d40d322050",
    "fitting_uid_sha256": "6d00d9aa90799aca0f32b5e0918f9716f00f7a3e516db3a4625aaed46b373844",
    "outer_uid_sha256": hashlib.sha256(b"[]").hexdigest(),
    "applied_cells": 40816,
    "source": {
        "name": "Shio-Koube/knee3",
        "revision": "373f4e1daf36984fa48bf3fc8fd9883fe17cb584",
        "csv_sha256": "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4",
    },
}
OFFICIAL_BASE = {
    "architecture": "dinov3_l16",
    "source_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "source_revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    "weights_sha256": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    "config_sha256": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
}
FORBIDDEN_BINDING = (
    "external_initialization",
    "private_research_only",
    "external_weight_publication",
    "public_input_cache",
)
QUEUE_ADMISSION_KEYS = {
    "schema_version",
    "status",
    "role",
    "queue_id",
    "scientific_source_git",
    "checkpoint_sha256",
    "training_binding_sha256",
    "training_code_sha256",
    "seed",
    "epoch",
    "optimizer_steps",
    "fit_scope",
    "training_studies",
    "outer_studies",
    "supervision",
    "base_encoder",
    "diagnostic_run",
    "protected_confirmation_gold_used",
    "external_initialization",
    "private_research_only",
    "MRI_only_inference",
    "original_checkpoint_and_numerical_source_unchanged",
}


def object_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_admission(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024:
        raise ValueError("Reviewed fitting admission must be a small regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Reviewed fitting admission must be an object")
    return value


def validate_reviewed(checkpoint: dict, admission: dict, checkpoint_sha: str) -> dict:
    if admission.get("role") == "public_source_final10":
        return validate_public_source_final(checkpoint, admission, checkpoint_sha)
    if admission.get("role") == QUEUE_ROLE:
        return validate_all_development_epoch(checkpoint, admission, checkpoint_sha)
    binding = checkpoint["binding"]
    config = binding["config"]
    role = admission.get("role")
    enabled = role == "candidate"
    receipt = binding.get("experimental_fitting_supervision", {})
    if (
        admission.get("schema_version") != 1
        or admission.get("status") != "owner_authorized_fixed_final_kaggle_comparison"
        or role not in {"control", "candidate"}
        or admission.get("scientific_source_git") != SOURCE
        or admission.get("checkpoint_sha256") != checkpoint_sha
        or admission.get("training_binding_sha256") != object_sha(binding)
        or admission.get("training_code_sha256") != object_sha(binding.get("code_sha256"))
        or admission.get("seed") not in {20260922, 20261004}
        or config.get("seed") != admission.get("seed")
        or config.get("epochs") != 3
        or config.get("diagnostic_study_limit") is not None
        or config.get("side") != 384
        or config.get("encoder", {}).get("weights_sha256")
        != "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        or config.get("fitting_supervision")
        != {"kind": "effusion_review_v1", "sha256": OVERLAY, "enabled": enabled}
        or receipt.get("overlay_sha256") != OVERLAY
        or receipt.get("enabled") is not enabled
        or receipt.get("applied_cells") != (31 if enabled else 0)
        or receipt.get("evaluation_masks_changed") is not False
        or receipt.get("experimental_only") is not True
        or receipt.get("fitting_uid_sha256") != FIT
        or receipt.get("outer_uid_sha256") != OUTER
        or binding.get("training_uids_sha256") != FIT
        or binding.get("outer_uids_sha256") != OUTER
        or checkpoint.get("epoch") != 3
        or checkpoint.get("next_update") != 0
        or checkpoint.get("step") != 843
        or admission.get("optimizer_steps") != 843
        or admission.get("training_studies") != 2247
        or admission.get("outer_studies") != 1156
        or admission.get("MRI_only_inference") is not True
        or admission.get("original_checkpoint_and_numerical_source_unchanged") is not True
        or binding.get("external_initialization") is not None
        or binding.get("private_research_only")
    ):
        raise ValueError("Reviewed fitting checkpoint differs from its exact final admission")
    # The original validator must still check every non-authorization invariant.
    # This metadata-only view is never saved or used for model reconstruction.
    view = dict(checkpoint)
    view["binding"] = dict(binding)
    view["binding"].pop("experimental_fitting_supervision")
    view["binding"]["config"] = dict(config)
    view["binding"]["config"].pop("fitting_supervision")
    infer.validate_inference_checkpoint(view, allow_scheduled_epoch=True)
    return binding


def validate_public_source_final(checkpoint: dict, admission: dict, checkpoint_sha: str) -> dict:
    """Authorize only the owner's exact completed public-source final10 comparison."""
    expected_sha = "ebeb3c39428612dc85e60fb6e5d3c0e4fde0acc6987cbe9a1b3f088aa273cfca"
    binding = checkpoint["binding"]
    config = binding["config"]
    if (
        checkpoint_sha != expected_sha
        or admission.get("checkpoint_sha256") != expected_sha
        or object_sha(binding) != "252129384647bacc1838eb0ac643578c92d8cf93de27eb74c3ccceee4495d2aa"
        or admission.get("training_binding_sha256") != object_sha(binding)
        or admission.get("training_code_sha256") != object_sha(binding["code_sha256"])
        or admission.get("schema_version") != 1
        or admission.get("status") != "owner_authorized_fixed_final_kaggle_comparison"
        or admission.get("role") != "public_source_final10"
        or admission.get("scientific_source_git") != "a9902f3db0cf56995c074cddef1f0c2370e3ce47"
        or admission.get("optimizer_steps") != 2810
        or admission.get("seed") != 20260922
        or admission.get("training_studies") != 2247
        or admission.get("outer_studies") != 1156
        or admission.get("MRI_only_inference") is not True
        or admission.get("original_checkpoint_and_numerical_source_unchanged") is not True
        or checkpoint.get("epoch") != 10
        or checkpoint.get("step") != 2810
        or checkpoint.get("next_update") != 0
        or config.get("epochs") != 10
        or binding.get("external_initialization") is not None
        or binding.get("private_research_only")
    ):
        raise ValueError("Public-source checkpoint differs from exact final admission")
    view = dict(checkpoint)
    view["binding"] = dict(binding)
    view["binding"].pop("experimental_fitting_supervision")
    view["binding"]["config"] = dict(config)
    view["binding"]["config"].pop("fitting_supervision")
    infer.validate_inference_checkpoint(view, allow_scheduled_epoch=True)
    return binding


def queue_admission(epoch: int) -> dict:
    """The only admission shape accepted for one exact daily-queue epoch."""
    sha, steps = ALL_DEVELOPMENT_EPOCHS[epoch]
    return {
        "schema_version": 1,
        "status": QUEUE_STATUS,
        "role": QUEUE_ROLE,
        "queue_id": QUEUE_ID,
        "scientific_source_git": ALL_DEVELOPMENT_SOURCE,
        "checkpoint_sha256": sha,
        "training_binding_sha256": ALL_DEVELOPMENT_BINDING_SHA,
        "training_code_sha256": ALL_DEVELOPMENT_CODE_SHA,
        "seed": 20260922,
        "epoch": epoch,
        "optimizer_steps": steps,
        "fit_scope": "all_development",
        "training_studies": 3403,
        "outer_studies": 0,
        "supervision": ALL_DEVELOPMENT_SUPERVISION,
        "base_encoder": OFFICIAL_BASE,
        "diagnostic_run": False,
        "protected_confirmation_gold_used": False,
        "external_initialization": None,
        "private_research_only": False,
        "MRI_only_inference": True,
        "original_checkpoint_and_numerical_source_unchanged": True,
    }


def validate_all_development_epoch(checkpoint: dict, admission: dict, checkpoint_sha: str) -> dict:
    """Authorize only an exact queued all-development RSNA-only epoch, never a guess."""
    binding = checkpoint["binding"]
    config = binding["config"]
    encoder = config.get("encoder", {})
    adapter = encoder.get("adapter", {})
    receipt = binding.get("experimental_fitting_supervision") or {}
    supervision = ALL_DEVELOPMENT_SUPERVISION
    epoch = admission.get("epoch")
    exact = ALL_DEVELOPMENT_EPOCHS.get(epoch) if type(epoch) is int else None
    if (
        exact is None
        or set(admission) != QUEUE_ADMISSION_KEYS
        or admission != queue_admission(epoch)
        or checkpoint_sha != exact[0]
        or object_sha(binding) != ALL_DEVELOPMENT_BINDING_SHA
        or object_sha(binding.get("code_sha256")) != ALL_DEVELOPMENT_CODE_SHA
        or checkpoint.get("epoch") != epoch
        or checkpoint.get("step") != exact[1]
        or checkpoint.get("next_update") != 0
        or checkpoint.get("scheduled_epoch") is not True
        or config.get("fit_scope") != "all_development"
        or config.get("epochs") != 10
        or config.get("seed") != 20260922
        or config.get("fold") != 0
        or config.get("side") != 384
        or config.get("submission_every_epochs") != 1
        or config.get("checkpoint_format") != "frozen_base_delta"
        or config.get("diagnostic_study_limit") is not None
        or config.get("public_input") is not None
        or (adapter.get("method"), adapter.get("rank")) != ("rs_lora_v1", 32)
        or adapter.get("base_frozen") is not True
        or any(encoder.get(k) != v for k, v in OFFICIAL_BASE.items())
        or encoder.get("pretrained") is not True
        or config.get("fitting_supervision")
        != {"kind": supervision["kind"], "sha256": supervision["overlay_sha256"], "enabled": True}
        or receipt.get("overlay_sha256") != supervision["overlay_sha256"]
        or receipt.get("release_sha256") != supervision["release_sha256"]
        or receipt.get("folds_sha256") != supervision["folds_sha256"]
        or receipt.get("fitting_uid_sha256") != supervision["fitting_uid_sha256"]
        or receipt.get("outer_uid_sha256") != supervision["outer_uid_sha256"]
        or receipt.get("applied_cells") != supervision["applied_cells"]
        or receipt.get("swapped_cells") != supervision["applied_cells"]
        or receipt.get("source") != supervision["source"]
        or receipt.get("fitting_studies") != 3403
        or receipt.get("fit_scope") != "all_development"
        or receipt.get("enabled") is not True
        or receipt.get("experimental_only") is not True
        or receipt.get("final_fit_only") is not True
        or receipt.get("internal_validation_available") is not False
        or receipt.get("evaluation_masks_changed") is not False
        or binding.get("training_uids_sha256") != supervision["fitting_uid_sha256"]
        or binding.get("outer_uids_sha256") != supervision["outer_uid_sha256"]
        or binding.get("release", {}).get("manifest_sha256") != supervision["release_sha256"]
        or any(k in binding for k in FORBIDDEN_BINDING)
        or binding.get("diagnostic_run") not in (None, False)
        or binding.get("protected_confirmation_gold_used") not in (None, False)
    ):
        raise ValueError("Queued all-development checkpoint differs from its exact final admission")
    view = dict(checkpoint)
    view["binding"] = dict(binding)
    view["binding"].pop("experimental_fitting_supervision")
    view["binding"]["config"] = dict(config)
    view["binding"]["config"].pop("fitting_supervision")
    infer.validate_inference_checkpoint(view, allow_scheduled_epoch=True)
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fitting-admission", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    admission = read_admission(args.fitting_admission)
    checkpoint_sha = infer.file_sha256(args.checkpoint)
    original = infer.validate_inference_checkpoint

    def explicitly_authorized(checkpoint: dict, **kwargs: object) -> dict:
        if kwargs.get("allow_diagnostic"):
            raise ValueError("Reviewed fitting admission cannot authorize diagnostic weights")
        # validate_reviewed calls the original validator, avoiding recursion.
        infer.validate_inference_checkpoint = original
        try:
            return validate_reviewed(checkpoint, admission, checkpoint_sha)
        finally:
            infer.validate_inference_checkpoint = explicitly_authorized

    try:
        infer.validate_inference_checkpoint = explicitly_authorized
        sys.argv = [sys.argv[0], "--checkpoint", str(args.checkpoint), *remaining]
        infer.main()
    finally:
        infer.validate_inference_checkpoint = original


if __name__ == "__main__":
    main()
