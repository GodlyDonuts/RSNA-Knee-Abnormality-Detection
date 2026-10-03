"""Synthetic receipts exercise preparation boundaries, never MRI acceptance."""

import copy
import json
from pathlib import Path

import pytest

from scripts.newton.prepare_effusion_screen import ENCODER_CONFIG_SHA, WEIGHTS_SHA, accept

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    # Reuse the actual CPU admission; GPU receipt below is an explicit synthetic
    # negative-test fixture, not an actual job or archived acceptance record.
    base = ROOT / "initiative/audits/effusion-supervision-engineering-v1"
    cpu = json.loads((base / "inputs-accepted.json").read_text())
    cpu["source_manifest_sha256"] = "a" * 64
    source = json.loads((base / "source-manifest.json").read_text())
    screen = ROOT / "initiative/experiments/competitive-v1/effusion-review-contrast-v1"
    configs = {
        n: json.loads((screen / (n + "-screen.json")).read_text()) for n in ("control", "candidate")
    }
    results = {
        n: dict(
            decoded_checkpoint_sha256="c" * 64,
            checkpoint_sha256="d" * 64,
            predictions_sha256="e" * 64,
            exact_complete_state_and_prediction_recovery=True,
            updates=[
                dict(
                    epoch=e,
                    update=u,
                    optimizer_steps=i,
                    loss=0.1,
                    gradient_norm=0.2,
                    observed_cells=5,
                )
                for i, (e, u) in enumerate([(0, 1), (0, 2), (1, 1), (1, 2)], 1)
            ],
        )
        for n in configs
    }
    gpu = dict(
        status="actual_native384_effusion_trainer_recovery_passed",
        job_id="869077",
        node="evc31",
        input_acceptance_sha256="b" * 64,
        diagnostic_only=True,
        clinical_accuracy_evidence=False,
        matched_initial_model_and_RNG=dict(
            complete_model_sha256="f" * 64,
            all_RNG_sha256="0" * 64,
            encoder_provenance=dict(
                weights_sha256=WEIGHTS_SHA,
                config_sha256=ENCODER_CONFIG_SHA,
                architecture="dinov3_l16",
                pretrained=True,
            ),
        ),
        results=results,
    )
    return cpu, gpu, source, configs


def test_fixed_screen_is_only_preparation():
    cpu, gpu, source, configs = fixture()
    result = accept(cpu, gpu, source, "a" * 64, "b" * 64, configs)
    assert result["optimizer_updates_per_arm"] == 843
    assert result["primary_outcome"] == "Effusion_outer_AUC"
    assert result["production_funding_or_dispatch"] is False
    assert result["clinical_accuracy_evidence"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "pending",
        "job",
        "node",
        "base",
        "source",
        "cpu_sha",
        "source_sha",
        "actual_cell",
        "roster",
        "recipe",
        "same_extra_recipe",
        "arm",
        "outer_masks",
        "policy",
        "recovery",
        "counter",
        "nan",
        "initial",
        "clinical",
    ],
)
def test_reject_unaccepted_screen(mutation):
    cpu, gpu, source, configs = fixture()
    if mutation == "pending":
        gpu["status"] = "PENDING"
    elif mutation == "job":
        gpu["job_id"] = "999999"
    elif mutation == "node":
        gpu["node"] = "evc32"
    elif mutation == "base":
        gpu["matched_initial_model_and_RNG"]["encoder_provenance"]["weights_sha256"] = "1" * 64
    elif mutation == "source":
        source["source_git"] = "another"
    elif mutation == "cpu_sha":
        gpu["input_acceptance_sha256"] = "0" * 64
    elif mutation == "source_sha":
        cpu["source_manifest_sha256"] = "0" * 64
    elif mutation == "actual_cell":
        cpu["changed_engineering_cells"] = 0
    elif mutation == "roster":
        cpu["full_outer_studies"] = 1155
    elif mutation == "recipe":
        configs["candidate"]["side"] = 224
    elif mutation == "same_extra_recipe":
        for c in configs.values():
            c["encoder_lr"] = 0.0002
    elif mutation == "arm":
        configs["candidate"]["fitting_supervision"]["enabled"] = False
    elif mutation == "outer_masks":
        cpu["supervision_receipts"]["candidate"]["evaluation_masks_changed"] = True
    elif mutation == "policy":
        configs["candidate"]["fitting_supervision"]["sha256"] = "0" * 64
    elif mutation == "recovery":
        gpu["results"]["control"]["exact_complete_state_and_prediction_recovery"] = False
    elif mutation == "counter":
        gpu["results"]["candidate"]["updates"][3]["optimizer_steps"] = 3
    elif mutation == "nan":
        gpu["results"]["candidate"]["updates"][0]["loss"] = float("nan")
    elif mutation == "initial":
        gpu["matched_initial_model_and_RNG"]["all_RNG_sha256"] = None
    else:
        gpu["clinical_accuracy_evidence"] = True
    with pytest.raises(ValueError):
        accept(cpu, gpu, source, "a" * 64, "b" * 64, copy.deepcopy(configs))
