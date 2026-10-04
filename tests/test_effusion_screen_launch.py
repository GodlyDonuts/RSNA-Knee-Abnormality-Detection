"""Reject wrong scientific inputs before costly screen execution or preservation."""

import copy

import pytest

from scripts.hf.publish_effusion_screen_epoch import validate_binding
from scripts.newton.run_effusion_screen import (
    OVERLAY_SHA,
    REGISTRY_SHA,
    accept_copy,
    fresh_step_environment,
)


def test_staging_CPU_mask_cannot_override_new_steps_or_change_GPU_visibility():
    inherited = {
        "SLURM_CPU_BIND": "mask_cpu:0x1",
        "SLURM_CPU_BIND_LIST": "0x1",
        "SLURM_CPU_BIND_TYPE": "mask_cpu",
        "SLURM_CPU_BIND_VERBOSE": "verbose",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "SLURM_JOB_ID": "123",
    }
    assert fresh_step_environment(inherited) == {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "SLURM_JOB_ID": "123",
    }
    assert inherited["SLURM_CPU_BIND_LIST"] == "0x1"


def receipt():
    return {
        "status": "private_original_native384_copy_fully_byte_verified",
        "job_id": "869745",
        "source_node": "evc31",
        "destination_node": "evc32",
        "members": 37812,
        "bytes": 140626483351,
        "registry_sha256": REGISTRY_SHA,
        "original_cache_retained": True,
        "GPU_reserved": False,
        "target_values_read": False,
        "clinical_accuracy_evidence": False,
        "cache_root": "/tmp/rsna-effusion-native384-evc32-route-869745-example/restored/cache",
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "partial"),
        ("job_id", "869663"),
        ("destination_node", "evc31"),
        ("members", 37811),
        ("bytes", 140626483350),
        ("GPU_reserved", 1),
        ("original_cache_retained", False),
        ("registry_sha256", "0" * 64),
        ("cache_root", "/tmp/rsna-effusion-native384-route-869665-x/../restored/cache"),
        ("cache_root", "/home/sa305415/cache"),
    ],
)
def test_no_partial_or_wrong_placement(key, value):
    record = receipt()
    record[key] = value
    with pytest.raises(ValueError):
        accept_copy(record)


def test_complete_original_placement():
    assert str(accept_copy(receipt())).endswith("/restored/cache")


def binding(role):
    enabled = role == "candidate"
    source = {"files": {"rsna/competitive/train.py": {"sha256": "a" * 64}}}
    expected = {"epochs": 3, "side": 384, "seed": 20260922}
    actual = {
        "config": {
            **expected,
            "fitting_supervision": {
                "kind": "effusion_review_v1",
                "sha256": OVERLAY_SHA,
                "enabled": enabled,
            },
        },
        "code_sha256": {"competitive/train.py": "a" * 64},
        "experimental_fitting_supervision": {
            "overlay_sha256": OVERLAY_SHA,
            "enabled": enabled,
            "applied_cells": 31 if enabled else 0,
            "evaluation_masks_changed": False,
            "experimental_only": True,
        },
    }
    return actual, {"training_studies": 2247, "outer_studies": 1156}, expected, source


@pytest.mark.parametrize("role", ["control", "candidate"])
def test_exact_arm(role):
    actual, record, expected, source = binding(role)
    validate_binding(actual, record, expected, role, source)


@pytest.mark.parametrize("change", ["source", "side", "cells", "outer", "evaluation", "mechanism"])
def test_reject_confounded_or_wrong_epoch_provenance(change):
    actual, record, expected, source = binding("candidate")
    actual = copy.deepcopy(actual)
    if change == "source":
        actual["code_sha256"]["competitive/train.py"] = "b" * 64
    elif change == "side":
        actual["config"]["side"] = 224
    elif change == "cells":
        actual["experimental_fitting_supervision"]["applied_cells"] = 30
    elif change == "outer":
        record["outer_studies"] = 1155
    elif change == "evaluation":
        actual["experimental_fitting_supervision"]["evaluation_masks_changed"] = True
    else:
        actual["config"]["witness"] = {"enabled": True}
    with pytest.raises(ValueError):
        validate_binding(actual, record, expected, "candidate", source)
