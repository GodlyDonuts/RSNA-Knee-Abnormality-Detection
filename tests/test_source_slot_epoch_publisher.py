"""Refuse diagnostic or mismatched epochs before a private publication intent."""

import copy

import pytest
import torch

from rsna.schema import TARGETS
from scripts.newton.probe_public_input_recipe_inputs import canonical
from scripts.newton.source_slot_epoch_publisher import validate_epoch


def envelope():
    provenance = {"architecture": "dinov3_l16"}
    original = {
        "encoder_provenance": {"public": provenance},
        "supervision": {"fitting_uid_sha256": "fitting", "outer_uid_sha256": "outer"},
        "public_cohort": {"index_sha256": "public-cache"},
    }
    own = {
        "code_sha256": {"train.py": "source"},
        "initialization": {"learned_parameters": 2, "frozen_base_sha256": canonical({})},
    }
    values = {
        "epochs": 10,
        "classifier_strategy": {"kind": "source_matched_slot_v1"},
        "encoder": {"weights_path": "original"},
    }
    binding = {
        "config": dict(values, encoder=provenance),
        "code_sha256": own["code_sha256"],
        "training_uids_sha256": "fitting",
        "outer_uids_sha256": "outer",
        "experimental_fitting_supervision": original["supervision"],
        "public_input_cache": original["public_cohort"],
        "private_research_only": True,
        "targets": list(TARGETS),
        "device_type": "cuda",
        "world_size": 1,
    }
    record = {
        "binding": binding,
        "diagnostic_run": False,
        "training_studies": 2247,
        "outer_studies": 1156,
    }
    state = {
        "binding": copy.deepcopy(binding),
        "epoch": 1,
        "next_update": 0,
        "step": 281,
        "scheduled_epoch": True,
        "optimizer_boundary": True,
        "schema_version": 2,
        "model_storage": {
            "format": "frozen_base_delta_v1",
            "encoder_provenance": provenance,
            "frozen_tensors": {},
            "learned_tensors": {"head": {"shape": [2], "dtype": "torch.float32"}},
        },
        "model": {"head": torch.ones(2)},
        "optimizer": {"state": {0: {}}},
        "rng_by_rank": [{"torch": 1, "cuda": 1, "python": 1, "numpy": 1}],
        "amp_scaler": {"scale": 128},
    }
    return state, record, original, own, values


def test_completed_epoch_binding():
    state, record, original, own, values = envelope()
    assert validate_epoch(state, record, 1, original, own, values) == record["binding"]


@pytest.mark.parametrize(
    "corruption", ["diagnostic", "counter", "code", "external", "base", "nan", "shape", "rng"]
)
def test_invalid_epoch_refused(corruption):
    state, record, original, own, values = envelope()
    if corruption == "diagnostic":
        record["diagnostic_run"] = True
    elif corruption == "counter":
        state["step"] = 280
    elif corruption == "code":
        record["binding"]["code_sha256"] = {}
    elif corruption == "external":
        record["binding"]["external_initialization"] = "other-data"
    elif corruption == "base":
        state["model_storage"]["frozen_tensors"] = {"changed": True}
    elif corruption == "nan":
        state["model"]["head"][0] = float("nan")
    elif corruption == "shape":
        state["model_storage"]["learned_tensors"]["head"]["shape"] = [1, 2]
    else:
        state["rng_by_rank"][0]["cuda"] = None
    with pytest.raises(ValueError):
        validate_epoch(state, record, 1, original, own, values)
