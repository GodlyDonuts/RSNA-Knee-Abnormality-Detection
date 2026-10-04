"""Exact authorization must not broaden diagnostics, source scope or inference math."""

import copy
import json
from pathlib import Path

import pytest

from scripts.kaggle import reviewed_infer as reviewed


def state(role="candidate", seed=20260922):
    path = Path("initiative/audits/effusion-supervision-screen-final-v1") / role / "run.json"
    binding = json.loads(path.read_text())["binding"]
    binding["config"]["seed"] = seed
    checkpoint = {
        "binding": binding,
        "schema_version": 2,
        "optimizer_boundary": True,
        "epoch": 3,
        "next_update": 0,
        "step": 843,
        "scheduled_epoch": True,
    }
    sha = "a" * 64
    admission = {
        "schema_version": 1,
        "status": "owner_authorized_fixed_final_kaggle_comparison",
        "role": role,
        "seed": seed,
        "scientific_source_git": reviewed.SOURCE,
        "checkpoint_sha256": sha,
        "training_binding_sha256": reviewed.object_sha(binding),
        "training_code_sha256": reviewed.object_sha(binding["code_sha256"]),
        "optimizer_steps": 843,
        "training_studies": 2247,
        "outer_studies": 1156,
        "MRI_only_inference": True,
        "original_checkpoint_and_numerical_source_unchanged": True,
    }
    return checkpoint, admission, sha


@pytest.mark.parametrize("role", ["control", "candidate"])
@pytest.mark.parametrize("seed", [20260922, 20261004])
def test_explicit_admission_returns_original_binding_without_mutation(role, seed):
    checkpoint, admission, sha = state(role, seed)
    original = copy.deepcopy(checkpoint)
    with pytest.raises(ValueError, match="not admitted"):
        reviewed.infer.validate_inference_checkpoint(checkpoint)
    assert reviewed.validate_reviewed(checkpoint, admission, sha) is checkpoint["binding"]
    assert checkpoint == original


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "prepared"),
        ("role", "other"),
        ("scientific_source_git", "0" * 40),
        ("checkpoint_sha256", "0" * 64),
        ("training_binding_sha256", "0" * 64),
        ("training_code_sha256", "0" * 64),
        ("seed", 9),
        ("optimizer_steps", 842),
        ("training_studies", 16),
        ("outer_studies", 16),
        ("MRI_only_inference", False),
        ("original_checkpoint_and_numerical_source_unchanged", False),
    ],
)
def test_changed_admission_rejected(key, value):
    checkpoint, admission, sha = state()
    admission[key] = value
    with pytest.raises(ValueError, match="exact final admission"):
        reviewed.validate_reviewed(checkpoint, admission, sha)


@pytest.mark.parametrize(
    "change", ["diagnostic", "external", "private", "cells", "evaluation", "progress", "base"]
)
def test_rebound_admission_cannot_broaden_scientific_scope(change):
    checkpoint, admission, sha = state()
    binding = checkpoint["binding"]
    if change == "diagnostic":
        binding["config"]["diagnostic_study_limit"] = 16
    elif change == "external":
        binding["external_initialization"] = {"source": "MRNet"}
    elif change == "private":
        binding["private_research_only"] = True
    elif change == "cells":
        binding["experimental_fitting_supervision"]["applied_cells"] = 30
    elif change == "evaluation":
        binding["experimental_fitting_supervision"]["evaluation_masks_changed"] = True
    elif change == "progress":
        checkpoint["step"] = 842
    else:
        binding["config"]["encoder"]["weights_sha256"] = "0" * 64
    admission["training_binding_sha256"] = reviewed.object_sha(binding)
    with pytest.raises(ValueError, match="exact final admission"):
        reviewed.validate_reviewed(checkpoint, admission, sha)


def test_original_non_authorization_validator_still_runs():
    checkpoint, admission, sha = state()
    checkpoint["optimizer_boundary"] = False
    with pytest.raises(ValueError, match="Unsupported competitive checkpoint"):
        reviewed.validate_reviewed(checkpoint, admission, sha)


def test_adapter_restores_original_validator_and_keeps_original_binding(tmp_path, monkeypatch):
    checkpoint, admission, _ = state()
    payload = tmp_path / "checkpoint.pt"
    payload.write_bytes(b"synthetic-checkpoint-no-MRI-or-weights")
    admission["checkpoint_sha256"] = reviewed.infer.file_sha256(payload)
    proof = tmp_path / "admission.json"
    proof.write_text(json.dumps(admission))
    original = reviewed.infer.validate_inference_checkpoint
    saved = copy.deepcopy(checkpoint)

    def exact_trained_inference_entry():
        assert (
            reviewed.infer.validate_inference_checkpoint(checkpoint, allow_scheduled_epoch=True)
            is checkpoint["binding"]
        )
        assert checkpoint == saved
        with pytest.raises(ValueError, match="diagnostic"):
            reviewed.infer.validate_inference_checkpoint(checkpoint, allow_diagnostic=True)

    monkeypatch.setattr(reviewed.infer, "main", exact_trained_inference_entry)
    monkeypatch.setattr(
        reviewed.sys,
        "argv",
        ["reviewed_infer.py", "--checkpoint", str(payload), "--fitting-admission", str(proof)],
    )
    reviewed.main()
    assert reviewed.infer.validate_inference_checkpoint is original


def test_admission_reader_rejects_large_or_linked_file(tmp_path):
    path = tmp_path / "admission.json"
    path.write_bytes(b" " * (32 * 1024 + 1))
    with pytest.raises(ValueError, match="small regular"):
        reviewed.read_admission(path)
    path.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="small regular"):
        reviewed.read_admission(link)
