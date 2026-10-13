"""Portable recovery admission; these tests are not actual pretrained-GPU evidence."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.kaggle import public_label_final_recovery as probe

ROOT = Path(__file__).resolve().parents[1]


def final_config():
    values = json.loads(
        (
            ROOT / "initiative/experiments/competitive-v1/public-label-source-v1/public-config.json"
        ).read_text()
    )
    values.update(
        fit_scope="all_development",
        warmup_updates=426,
        fitting_supervision=dict(kind="label_source_final_v1", enabled=True, sha256="a" * 64),
    )
    return values


def test_mounted_config_preserves_final_recipe_and_changes_only_bounded_budget_and_paths():
    base = final_config()
    spec = copy.deepcopy(base["encoder"])
    spec.update(weights_path="/mounted/base.safetensors", config_path="/mounted/config.json")
    values = probe.mounted_config(base, spec)
    assert values["encoder"] == spec
    assert values["fit_scope"] == "all_development"
    assert values["fitting_supervision"] == base["fitting_supervision"]
    assert (values["epochs"], values["warmup_updates"], values["diagnostic_study_limit"]) == (
        2,
        2,
        16,
    )
    assert (values["precision"], values["amp_init_scale"], values["encode_chunk_size"]) == (
        "fp16",
        128,
        8,
    )
    assert values["submission_every_epochs"] == 0
    assert base == final_config()


@pytest.mark.parametrize(
    "key,value",
    [
        ("weights_sha256", "b" * 64),
        ("source_revision", "other"),
        ("architecture", "diagnostic_tiny"),
    ],
)
def test_mounted_official_model_substitution_refused(key, value):
    base = final_config()
    spec = copy.deepcopy(base["encoder"])
    spec[key] = value
    with pytest.raises(ValueError, match="Mounted official"):
        probe.mounted_config(base, spec)


@pytest.mark.parametrize(
    "key,value",
    [("fit_scope", "outer_fold"), ("warmup_updates", 281), ("epochs", 9), ("amp_init_scale", 64)],
)
def test_other_recipe_refused(key, value):
    base = final_config()
    base[key] = value
    with pytest.raises(ValueError):
        probe.mounted_config(base, base["encoder"])


def test_source_identity_binds_original_numerical_code_not_private_source_csv_or_CPU_worker(
    tmp_path,
):
    path = tmp_path / "rsna/train.py"
    path.parent.mkdir()
    path.write_text("original trainer")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cpu = {
        "source_modules": {
            "code/rsna/train.py": digest,
            "source.csv": "a" * 64,
            "worker.py": "b" * 64,
        }
    }
    assert probe.source_identity(tmp_path, {"rsna/train.py": digest}, cpu) == {
        "rsna/train.py": digest
    }
    path.write_text("changed trainer")
    with pytest.raises(ValueError, match="bytes differ"):
        probe.source_identity(tmp_path, {"rsna/train.py": digest}, cpu)


def test_source_identity_rejects_missing_original_core(tmp_path):
    with pytest.raises(ValueError, match="scientific source"):
        probe.source_identity(tmp_path, {}, {"source_modules": {"code/rsna/train.py": "a" * 64}})


def test_source_identity_refuses_symlink_even_with_matching_bytes(tmp_path):
    outside = tmp_path / "original.py"
    outside.write_text("original")
    (tmp_path / "rsna").mkdir()
    (tmp_path / "rsna/train.py").symlink_to(outside)
    h = probe.sha(outside)
    with pytest.raises(ValueError, match="path or bytes"):
        probe.source_identity(
            tmp_path, {"rsna/train.py": h}, {"source_modules": {"code/rsna/train.py": h}}
        )


def test_sorted_first16_union_is_covered_by_each_fold_first16_fixture():
    first = [f"{i:04}" for i in range(0, 3403, 2)]
    second = [f"{i:04}" for i in range(1, 3403, 2)]
    assert set(sorted(first + second)[:16]) <= set(first[:16] + second[:16])
