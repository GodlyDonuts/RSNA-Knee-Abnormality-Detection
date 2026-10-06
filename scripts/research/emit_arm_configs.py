"""Emit verified TrainConfig JSONs for every packaged competitive-v1 arm.

Each emitted file is the production recipe (`dinov3-campaign.json`
``train_config_without_encoder`` + the pinned encoder block) plus exactly one
arm field, deserialized through ``TrainConfig`` so a config that fails
validation never lands on disk. Artifact paths still travel via the matching
CLI flag (`--fitting-supervision`, `--canonical-sides`); the config binds the
sha256 so a mismatched artifact is rejected at load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rsna.competitive.train import TrainConfig

ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _overlay_sha(path: Path) -> str:
    return _sha(path)


def base_config() -> dict:
    campaign = json.loads((ROOT / "configs/competitive/dinov3-campaign.json").read_text())
    recipe = dict(campaign["train_config_without_encoder"])
    encoder = json.loads((ROOT / "configs/competitive/weights/dinov3-large.json").read_text())[
        "encoder"
    ]
    return {**recipe, "encoder": encoder}


def emit(config: dict, name: str, out: Path) -> Path:
    TrainConfig(**json.loads(json.dumps(config))).validate()
    target = out / f"{name}.train.json"
    target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "initiative/experiments/competitive-v1/run-configs",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = base_config()
    written = [emit(base, "base-v2-control", args.output)]

    overlay_root = ROOT / "initiative/experiments/competitive-v1"
    supervision_arms = {
        "label-source-public": overlay_root
        / "label-source-ablation-v1/llm_labels_v4_blend.fold{fold}.overlay.json",
        "label-source-blended": overlay_root
        / "label-source-ablation-v1/blended_v2plus_public.fold{fold}.overlay.json",
        "graded-u05": overlay_root / "graded-targets-v1/graded-states-u05.fold{fold}.overlay.json",
    }
    for fold in (0, 1, 2):
        for name, pattern in supervision_arms.items():
            artifact = Path(str(pattern).format(fold=fold))
            config = {
                **base,
                "fold": fold,
                "fitting_supervision": {
                    "kind": "label_source_v1",
                    "sha256": _overlay_sha(artifact),
                    "enabled": True,
                },
            }
            written.append(emit(config, f"{name}.fold{fold}", args.output))
        sides = overlay_root / f"laterality-normalization-v1/canonical-sides.fold{fold}.json"
        config = {
            **base,
            "fold": fold,
            "canonical_sides": {
                "kind": "canonical_side_v1",
                "sha256": _sha(sides),
                "enabled": True,
            },
        }
        written.append(emit(config, f"canonical-sides.fold{fold}", args.output))

    written.append(
        emit(
            {
                **base,
                "masked_cell": {
                    "enabled": True,
                    "kind": "entropy_max",
                    "weight": 0.05,
                },
            },
            "masked-cell-em005",
            args.output,
        )
    )
    print(json.dumps([str(p.relative_to(ROOT)) for p in written], indent=1))


if __name__ == "__main__":
    main()
