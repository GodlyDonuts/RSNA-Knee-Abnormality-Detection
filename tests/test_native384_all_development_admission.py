"""The new final fit must preserve every eligible mask and reserved boundary."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from rsna.schema import TARGETS
from scripts.newton.admit_native384_all_development import (
    require_scientific_import,
    validate_config,
    verify_population,
)
from scripts.newton.verify_native384_all_recovery import adapter_parameter_count, numerical_updates

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def population():
    manifest = pd.read_parquet(ROOT / "initiative/releases/report-supervision-v1/studies.parquet")
    selected = manifest[manifest.split.eq("train")].copy()
    selected = selected[selected[[f"{t}__mask" for t in TARGETS]].sum(axis=1) > 0]
    train = selected.sort_values("study_uid").reset_index(drop=True)
    ids = train.study_uid.tolist()
    registry = {"training_uids": ids[:2247], "outer_uids": ids[2247:]}
    return manifest, train, train.iloc[:0], registry


def test_all_eligible_rows_and_original_masks_are_accepted(population):
    receipt = verify_population(*population, TARGETS)
    assert receipt["training_studies"] == 3403
    assert receipt["fixed_final_optimizer_steps"] == 4260
    assert receipt["evaluation_or_selection_permitted"] is False


@pytest.mark.parametrize("fault", ["gold", "missing", "duplicate", "mask", "label", "outer"])
def test_forbidden_or_changed_training_population_is_rejected(population, fault):
    manifest, train, outer, registry = population
    if fault == "gold":
        train = pd.concat(
            [train.iloc[:-1], manifest[manifest.split.eq("gold_evaluation")].iloc[:1]]
        )
    elif fault == "missing":
        registry["training_uids"].pop()
    elif fault == "duplicate":
        train.loc[0, "study_uid"] = train.loc[1, "study_uid"]
    elif fault == "mask":
        train.loc[0, "ACL__mask"] = 1 - train.loc[0, "ACL__mask"]
    elif fault == "label":
        train.loc[0, "ACL"] = 0.375
    elif fault == "outer":
        outer = train.iloc[:1]
    with pytest.raises(ValueError):
        verify_population(manifest, train, outer, registry, TARGETS)


def test_final_recipe_changes_only_population_and_corresponding_warmup():
    original = json.loads(
        (ROOT / "initiative/audits/native384-fixed-final-v1/control/config.json").read_text()
    )
    candidate = {**original, "fit_scope": "all_development", "warmup_updates": 213}
    assert set(validate_config(original, candidate)) == {"fit_scope", "warmup_updates"}
    bad = copy.deepcopy(candidate)
    bad["encoder"]["adapter"]["rank"] = 64
    with pytest.raises(ValueError):
        validate_config(original, bad)
    with pytest.raises(ValueError):
        validate_config(original, {**candidate, "diagnostic_study_limit": 16})


def test_original_checkout_alias_is_resolved_without_accepting_another_source(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    imported = real / "train.py"
    imported.write_text("original")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    require_scientific_import(imported, alias)
    with pytest.raises(ValueError):
        require_scientific_import(tmp_path / "other.py", alias)


def test_numerical_recovery_evidence_keeps_gradient_and_loss_excluding_only_timing():
    rows = [
        {
            "optimizer_steps": n,
            "loss": 0.5,
            "gradient_norm": 1.2,
            "encoder_gradient_norm": 0.02,
            "observed_cells": 12,
            "elapsed_seconds": float(n),
            "peak_gpu_allocated_bytes": 100,
        }
        for n in range(1, 7)
    ]
    actual = numerical_updates(rows)
    assert actual[0]["encoder_gradient_norm"] == 0.02
    assert "elapsed_seconds" not in actual[0]
    with pytest.raises(ValueError):
        numerical_updates(rows[:-1])
    rows[0]["loss"] = float("nan")
    with pytest.raises(ValueError):
        numerical_updates(rows)


def test_recovery_verifier_checks_actual_adapter_graph_without_newer_source_helper():
    import torch

    class Encoder:
        def named_parameters(self):
            return iter(self.parameters)

    encoder = Encoder()
    encoder.parameters = [
        ("projection.lora_A", torch.nn.Parameter(torch.zeros(2, 4))),
        ("projection.lora_B", torch.nn.Parameter(torch.zeros(4, 2))),
        ("projection.base", torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)),
    ]
    provenance = {"adapter": {"trainable_parameters": 16, "target_modules": ["projection"]}}
    assert adapter_parameter_count(encoder, provenance) == 16
    with pytest.raises(ValueError, match="actual_adapter_graph"):
        adapter_parameter_count(
            encoder, {"adapter": {"trainable_parameters": 15, "target_modules": ["projection"]}}
        )
    encoder.parameters.append(("unexpected", torch.nn.Parameter(torch.ones(1))))
    with pytest.raises(ValueError, match="only_declared_encoder"):
        adapter_parameter_count(encoder, provenance)
