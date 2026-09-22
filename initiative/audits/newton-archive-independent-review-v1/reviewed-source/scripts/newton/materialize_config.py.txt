"""Create a pilot/production config only from fully verified staged encoder bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stage_weights import stage


def materialize(
    recipe: Path,
    manifest: Path,
    repo: Path,
    output: Path,
    mode: str,
    fold: int = 0,
    seed: int = 20260922,
) -> dict:
    if mode not in {"pilot", "production"} or type(fold) is not int or fold not in {0, 1, 2}:
        raise ValueError("explicit_mode_and_valid_fold_required")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("valid_seed_required")
    verified = stage(manifest, repo, download=False)
    plan, pinned = json.loads(recipe.read_text()), json.loads(manifest.read_text())
    encoder = pinned["encoder"].copy()
    for key in ("architecture", "source_id", "source_revision", "weights_sha256"):
        if plan.get(key) != encoder.get(key):
            raise ValueError("staged_encoder_does_not_match_campaign")
    if plan.get("weights_bytes") != verified["artifacts"]["weights"]["bytes"]:
        raise ValueError("staged_weight_size_does_not_match_campaign")
    if plan.get("configuration_sha256") not in {None, encoder["config_sha256"]}:
        raise ValueError("staged_config_does_not_match_campaign")
    for key in ("config_path", "weights_path"):
        encoder[key] = str((repo / encoder[key]).absolute())
    config = plan["train_config_without_encoder"].copy()
    config.update(encoder=encoder, fold=fold, seed=seed)
    if mode == "pilot":
        pilot = plan["pilot"]
        if pilot.get("submission_eligible") is not False:
            raise ValueError("pilot_must_be_ineligible_for_submission")
        config.update(
            diagnostic_study_limit=pilot["diagnostic_study_limit"],
            epochs=pilot["epochs"],
            precision=pilot["initial_precision"],
            submission_every_epochs=0,
        )
    else:
        if config.get("epochs") != 10:
            raise ValueError("production_campaign_requires_ten_epochs")
        config["diagnostic_study_limit"] = None
    raw = (json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file() or output.read_bytes() != raw:
            raise ValueError("existing_configuration_differs")
        state = "verified_existing_identical"
    else:
        with output.open("xb") as handle:
            handle.write(raw)
        state = "created"
    return {
        "status": state,
        "mode": mode,
        "fold": fold,
        "seed": seed,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "encoder_manifest_sha256": verified["manifest_sha256"],
        "diagnostic_only": mode == "pilot",
        "download_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "manifest", "repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "production"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(**vars(args)), indent=2))
    except (OSError, ValueError, KeyError, TypeError):
        print(
            json.dumps({"status": "blocked", "reason": "verified_campaign_configuration_required"})
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
