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
