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
    if admission.get("role") == ALL_DEVELOPMENT_ROLE:
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


ALL_DEVELOPMENT_ROLE = "public_source_all_development_epoch"
ALL_DEVELOPMENT_STATUS = "owner_authorized_daily_kaggle_comparison_queue"
ALL_DEVELOPMENT_SOURCE = "1bcd6c0e2e2a47d6894d5c545d6950e0c6f0d5b9"
# Original job 908454 run binding (all epochs) and its frozen rsna module hashes.
ALL_DEVELOPMENT_BINDING = "5c8e14732143a479073b43a24ff7851775a1dd550dc8deb0a9291b8f8091597e"
ALL_DEVELOPMENT_CODE = "86db55a2769a676b0015d24e464de4be025c2c4abc2845363cb2313fc4ea2b7e"
ALL_DEVELOPMENT_OVERLAY = "54b5a652949120764a596b4d495f11950ff103a909b7efc5fe59da9cd6f2fe85"
ALL_DEVELOPMENT_RELEASE = "2bf49fd2a8ae0b7c5af7e8c105fc3f2337845a92654d136fe67bc41a1e52fae7"
ALL_DEVELOPMENT_FOLDS = "fe590e5072a0d82abeb377598aeef0021bbe77fe4529a19b9826a7d40d322050"
ALL_DEVELOPMENT_SOURCE_CSV = "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4"
ALL_DEVELOPMENT_FIT = "6d00d9aa90799aca0f32b5e0918f9716f00f7a3e516db3a4625aaed46b373844"
EMPTY_ROSTER = hashlib.sha256(b"[]").hexdigest()
ALL_DEVELOPMENT_UPDATES = 426
ALL_DEVELOPMENT_STUDIES = 3403
ALL_DEVELOPMENT_PREFIX = (
    "experiments/dinov3-v1/training-epochs/"
    "public-label-source-final-all-development-1bcd6c0-fixed10-seed20260922"
)
OFFICIAL_DINOV3_L = {
    "architecture": "dinov3_l16",
    "source_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "source_revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    "weights_sha256": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    "config_sha256": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
}
# Exact owner queue: epoch -> (checkpoint SHA-256, private HF revision). Nothing else.
ALL_DEVELOPMENT_EPOCHS = {
    1: (
        "74a24c6468a3e2e1ff643bb77eb316207c11feab9e7af0514a407c63a0cf66d5",
        "424827b0bf049e4270f61e48ab004b8ce3070b05",
    ),
    2: (
        "9422444c2144e925da5c6feb845a7890529fbe82a2decd2dafdac6fa9e46f410",
        "9d19a06c3d85fed3746983f274ea87bf2f0d8ad9",
    ),
    3: (
        "59adcdc9ed72dac94e5c8213a8d1fc61c917b5ccb800250be0cc457ccaad7d90",
        "47ef80ce35f481da8bd465438e10648a3166dd6d",
    ),
    4: (
        "794de246d2703e87c2f2235bf9fcac5d39673bca1a11a7fbbe2684c58636de8d",
        "93950d4b0beb50329f5b2211030f610858ee4ac5",
    ),
    5: (
        "8a81970c35e54f7c0417690c7344546542f6cf34acc74cbf347b294092537140",
        "dc44dc8ca541be375070d8d5f3161534dae6f689",
    ),
}
FORBIDDEN_BINDING = (
    "external_initialization",
    "private_research_only",
    "external_weight_publication",
    "public_input_cache",
)


def all_development_admission(epoch: int) -> dict:
    """The only admission accepted for one exact queued all-development epoch."""
    if type(epoch) is not int or epoch not in ALL_DEVELOPMENT_EPOCHS:
        raise ValueError("Epoch is not an exact owner-queued all-development checkpoint")
    sha, revision = ALL_DEVELOPMENT_EPOCHS[epoch]
    return {
        "schema_version": 1,
        "status": ALL_DEVELOPMENT_STATUS,
        "role": ALL_DEVELOPMENT_ROLE,
        "scientific_source_git": ALL_DEVELOPMENT_SOURCE,
        "checkpoint_sha256": sha,
        "training_binding_sha256": ALL_DEVELOPMENT_BINDING,
        "training_code_sha256": ALL_DEVELOPMENT_CODE,
        "private_HF_revision": revision,
        "private_HF_prefix": f"{ALL_DEVELOPMENT_PREFIX}/epoch-{epoch:03d}",
        "seed": 20260922,
        "epoch": epoch,
        "optimizer_steps": ALL_DEVELOPMENT_UPDATES * epoch,
        "fit_scope": "all_development",
        "fitting_supervision": "label_source_final_v1",
        "training_studies": ALL_DEVELOPMENT_STUDIES,
        "outer_studies": 0,
        "official_base": OFFICIAL_DINOV3_L,
        "MRI_only_inference": True,
        "original_checkpoint_and_numerical_source_unchanged": True,
    }


def validate_all_development_epoch(checkpoint: dict, admission: dict, checkpoint_sha: str) -> dict:
    """Authorize one exact queued epoch of original all-development job 908454."""
    epoch = admission.get("epoch")
    try:
        expected = all_development_admission(epoch)  # type: ignore[arg-type]
    except ValueError:
        raise ValueError("All-development checkpoint differs from exact queue admission") from None
    binding = checkpoint["binding"]
    config = binding["config"]
    encoder = config.get("encoder", {})
    receipt = binding.get("experimental_fitting_supervision", {})
    if (
        admission != expected
        or checkpoint_sha != expected["checkpoint_sha256"]
        or object_sha(binding) != ALL_DEVELOPMENT_BINDING
        or object_sha(binding.get("code_sha256")) != ALL_DEVELOPMENT_CODE
        or checkpoint.get("schema_version") != 2
        or checkpoint.get("optimizer_boundary") is not True
        or checkpoint.get("scheduled_epoch") is not True
        or checkpoint.get("epoch") != epoch
        or checkpoint.get("step") != ALL_DEVELOPMENT_UPDATES * epoch
        or checkpoint.get("next_update") != 0
        or config.get("epochs") != 10
        or config.get("warmup_updates") != ALL_DEVELOPMENT_UPDATES
        or config.get("seed") != 20260922
        or config.get("fold") != 0
        or config.get("submission_every_epochs") != 1
        or config.get("fit_scope") != "all_development"
        or config.get("diagnostic_study_limit") is not None
        or config.get("side") != 384
        or config.get("checkpoint_format") != "frozen_base_delta"
        or config.get("public_input") is not None
        or config.get("fitting_supervision")
        != {"kind": "label_source_final_v1", "sha256": ALL_DEVELOPMENT_OVERLAY, "enabled": True}
        or any(encoder.get(k) != v for k, v in OFFICIAL_DINOV3_L.items())
        or encoder.get("adapter", {}).get("rank") != 32
        or receipt.get("overlay_sha256") != ALL_DEVELOPMENT_OVERLAY
        or receipt.get("release_sha256") != ALL_DEVELOPMENT_RELEASE
        or receipt.get("folds_sha256") != ALL_DEVELOPMENT_FOLDS
        or receipt.get("fitting_uid_sha256") != ALL_DEVELOPMENT_FIT
        or receipt.get("outer_uid_sha256") != EMPTY_ROSTER
        or receipt.get("fitting_studies") != ALL_DEVELOPMENT_STUDIES
        or receipt.get("fit_scope") != "all_development"
        or receipt.get("final_fit_only") is not True
        or receipt.get("internal_validation_available") is not False
        or receipt.get("evaluation_masks_changed") is not False
        or receipt.get("experimental_only") is not True
        or receipt.get("enabled") is not True
        or receipt.get("source", {}).get("csv_sha256") != ALL_DEVELOPMENT_SOURCE_CSV
        or binding.get("training_uids_sha256") != ALL_DEVELOPMENT_FIT
        or binding.get("outer_uids_sha256") != EMPTY_ROSTER
        or binding.get("release", {}).get("manifest_sha256") != ALL_DEVELOPMENT_RELEASE
        or any(binding.get(k) for k in FORBIDDEN_BINDING)
    ):
        raise ValueError("All-development checkpoint differs from exact queue admission")
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
