"""Numerical and adverse export checks, never an actual MRI/GPU acceptance."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from rsna.competitive import licensed_cached_serving as serving
from rsna.competitive.licensed_cached_mil import Readout, Recipe
from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "initiative/audits/cached-full-study-factorial-20261017-v1"


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 1024)
        self.norm = nn.LayerNorm(1024)
        self.att = nn.Sequential(
            nn.Linear(1024, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 12)
        )
        self.clsW = nn.Parameter(torch.randn(12, 1024))
        self.clsb = nn.Parameter(torch.randn(12))

    def head(self, features):
        raise RuntimeError("Original head must be replaced on this instance")


@pytest.fixture
def evidence():
    return tuple(
        json.loads((AUDIT / name).read_text())
        for name in (
            "actual-four-full-complete.json",
            "actual-all12-private-verified.json",
            "actual-recovery-accepted.json",
        )
    )


def state(evidence, role, windows):
    full, publication, recovery = evidence
    original = {k: v for k, v in recovery.items() if k != "elapsed_seconds"}
    binding = {
        "version": "licensed_cached_MIL_label_coverage_factorial_v1",
        "source_git": serving.SOURCE,
        "parent_checkpoint_sha256": serving.PARENT,
        "original_readout_sha256": serving.READOUT,
        "population": 3887,
        "population_sha256": serving.POPULATION,
        "total_steps": 1458,
        "target_order": list(TARGETS),
        "diagnostic_only": False,
        "source_label_role": role,
        "protected_gold_confirmation_used": False,
        "source_encoder_exposed_cohort_no_independent_OOF": True,
        "initial_fixed_parent_CNN_unchanged": True,
        "head_dropout_fixed_zero_all_four_roles": True,
        "accepted_masks_unchanged": True,
        "parent_full_teacher_unchanged": True,
        "precision": "CPU_FP32",
        "head_dropout": 0.0,
        "recipe": serving.fixed_recipe(role, windows),
        "actual_recovery_sha256": hashlib.sha256(
            json.dumps(original, indent=2, sort_keys=True).encode()
        ).hexdigest(),
        "source_files_sha256": full["source_files_sha256"],
    }
    # Synthetic learned tensors deliberately do not establish actual file SHA.
    torch.manual_seed(983)
    values = {n: torch.randn(shape) / 20 for n, shape in serving.SHAPES.items()}
    values["norm.weight"].add_(1)
    return {"binding": binding, "epoch": 3, "completed": 1458, "model": values}


@pytest.mark.parametrize("role,windows", serving.FINAL_SHAS)
def test_all_four_fixed_final_scopes_and_actual_training_recipe_are_exact(evidence, role, windows):
    from dataclasses import asdict

    assert serving.fixed_recipe(role, windows) == asdict(Recipe(role, windows))
    result = serving.validate(
        state(evidence, role, windows), *evidence, role, windows, serving.FINAL_SHAS[role, windows]
    )
    assert result["epoch"] == 3 and result["completed"] == 1458
    assert result["encoder_changed"] is False and result["serving_windows"] == 94


@pytest.mark.parametrize("role,windows", serving.FINAL_SHAS)
def test_eval_head_is_bit_equal_to_training_graph_and_never_changes_encoder(
    evidence, role, windows
):
    torch.set_num_threads(1)
    snapshot = state(evidence, role, windows)
    model = Parent().eval()
    before = {n: t.clone() for n, t in model.backbone.state_dict().items()}
    reference = Readout(
        {
            "format": "licensed_parent_MIL_readout_v1",
            "parent_checkpoint_sha256": serving.PARENT,
            "source_notebook_sha256": serving.PARENT_SOURCE,
            "model": snapshot["model"],
        }
    ).eval()
    other = Parent().eval()
    serving.install(model, snapshot)
    for n, t in before.items():
        assert torch.equal(t, model.backbone.state_dict()[n])
    assert other.head.__func__ is Parent.head
    features = torch.randn(2, 94, 1024).half()
    expected = reference(features.float())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model.head(features)
    assert actual.dtype == torch.float32 and torch.equal(actual, expected)
    assert all(
        torch.equal(dict(model.named_parameters())[n], t) for n, t in snapshot["model"].items()
    )


@pytest.mark.parametrize(
    "change",
    [
        "partial",
        "diagnostic",
        "source",
        "parent",
        "population",
        "windows",
        "precision",
        "gold",
        "teacher",
        "sourcefiles",
        "mask",
        "recovery",
        "unpublished",
        "duplicate_epoch",
        "head_sha",
        "wrong_role",
        "nonfinite",
        "missing",
        "shape",
        "dtype",
        "bool_epoch",
    ],
)
def test_invalid_or_unpreserved_model_cannot_become_a_candidate(evidence, change):
    evidence = copy.deepcopy(evidence)
    snapshot = state(evidence, "original", 16)
    b = snapshot["binding"]
    role, windows, sha = "original", 16, serving.FINAL_SHAS["original", 16]
    if change == "partial":
        snapshot["completed"] = 972
    elif change == "diagnostic":
        b["diagnostic_only"] = True
    elif change == "source":
        b["source_git"] = "0" * 40
    elif change == "parent":
        b["parent_checkpoint_sha256"] = "0" * 64
    elif change == "population":
        b["population"] = 16
    elif change == "windows":
        b["recipe"]["windows"] = 94
    elif change == "precision":
        b["precision"] = "FP16"
    elif change == "gold":
        b["protected_gold_confirmation_used"] = True
    elif change == "teacher":
        b["parent_full_teacher_unchanged"] = False
    elif change == "sourcefiles":
        b["source_files_sha256"] = {}
    elif change == "mask":
        b["accepted_masks_unchanged"] = False
    elif change == "recovery":
        evidence[2]["roles"]["original-94"]["prediction_bytes_equal"] = False
    elif change == "unpublished":
        evidence[1]["status"] = "uncertain"
    elif change == "duplicate_epoch":
        evidence[1]["epochs"][-1] = evidence[1]["epochs"][0]
    elif change == "head_sha":
        sha = "0" * 64
    elif change == "wrong_role":
        role = "released_S6"
    elif change == "nonfinite":
        snapshot["model"]["clsb"][0] = float("nan")
    elif change == "missing":
        del snapshot["model"]["att.0.weight"]
    elif change == "shape":
        snapshot["model"]["clsW"] = torch.zeros(12, 768)
    elif change == "dtype":
        snapshot["model"]["clsW"] = snapshot["model"]["clsW"].half()
    elif change == "bool_epoch":
        snapshot["epoch"] = True
    with pytest.raises(ValueError):
        serving.validate(snapshot, *evidence, role, windows, sha)


def test_partial_input_and_training_mode_are_rejected(evidence):
    snapshot = state(evidence, "original", 16)
    model = serving.install(Parent().eval(), snapshot)
    with pytest.raises(ValueError):
        model.head(torch.randn(1, 16, 1024))
    model.train()
    with pytest.raises(ValueError):
        model.head(torch.randn(1, 94, 1024))
