"""Every emitted run config must deserialize and validate through TrainConfig."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rsna.competitive.train import TrainConfig

RUN_CONFIGS = (
    Path(__file__).resolve().parents[1] / "initiative/experiments/competitive-v1/run-configs"
)


def _configs() -> list[Path]:
    return sorted(RUN_CONFIGS.glob("*.train.json"))


def test_run_configs_exist_for_every_packaged_arm():
    names = {p.name.removesuffix(".train.json") for p in _configs()}
    expected = {
        "base-v2-control",
        "masked-cell-em005",
        "masked-cell-consistency",
        *{f"label-source-public.fold{f}" for f in (0, 1, 2)},
        *{f"label-source-blended.fold{f}" for f in (0, 1, 2)},
        *{f"label-source-v21.fold{f}" for f in (0, 1, 2)},
        *{f"label-source-v21pn.fold{f}" for f in (0, 1, 2)},
        *{f"label-source-v21hc.fold{f}" for f in (0, 1, 2)},
        *{f"graded-u05.fold{f}" for f in (0, 1, 2)},
        *{f"canonical-sides.fold{f}" for f in (0, 1, 2)},
    }
    assert names == expected


@pytest.mark.parametrize(
    "path",
    _configs(),
    ids=lambda p: p.name.removesuffix(".train.json"),
)
def test_run_config_deserializes_and_validates(path: Path):
    payload = json.loads(path.read_text())
    TrainConfig(**payload).validate()


def test_artifact_arms_bind_real_overlay_shas():
    overlay_root = RUN_CONFIGS.parent
    pairs = [
        (
            "label-source-ablation-v1/llm_labels_v4_blend",
            "label-source-public",
        ),
        (
            "label-source-ablation-v1/blended_v2plus_public",
            "label-source-blended",
        ),
        (
            "label-source-ablation-v1/v21_severity_rereview",
            "label-source-v21",
        ),
        ("graded-targets-v1/graded-states-u05", "graded-u05"),
    ]

    for artifact_stem, config_stem in pairs:
        for fold in (0, 1, 2):
            artifact = overlay_root / f"{artifact_stem}.fold{fold}.overlay.json"
            config = json.loads((RUN_CONFIGS / f"{config_stem}.fold{fold}.train.json").read_text())
            policy = config["fitting_supervision"]
            assert policy["kind"] == "label_source_v1"
            assert policy["enabled"] is True
            assert policy["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
            assert config["fold"] == fold


def test_canonical_side_config_binds_table_shas():
    for fold in (0, 1, 2):
        artifact = (
            RUN_CONFIGS.parent / f"laterality-normalization-v1/canonical-sides.fold{fold}.json"
        )
        config = json.loads((RUN_CONFIGS / f"canonical-sides.fold{fold}.train.json").read_text())
        policy = config["canonical_sides"]
        assert policy["kind"] == "canonical_side_v1"
        assert policy["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
