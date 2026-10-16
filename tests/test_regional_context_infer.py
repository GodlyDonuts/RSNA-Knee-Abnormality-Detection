"""Synthetic serving tests; fixtures never stand for a trained accepted endpoint."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from rsna.competitive.regional_context_infer import (
    ENGINE,
    FIXED_PROVENANCE,
    FORMAT,
    RECOVERY_SHA,
    export_component,
    load_component,
    model_from_component,
    predict_pair,
    sha256,
    validate_final,
)
from rsna.competitive.regional_context_train import ROLES, feature_vector
from rsna.competitive.released_context import ContextResidual, tta_probabilities, window_metadata
from rsna.competitive.released_context_infer import model_from_component as old_model_from_component
from rsna.competitive.released_context_train import VERSION, Recipe, learning_rate
from rsna.schema import TARGETS


def component(role="regional"):
    model = ContextResidual(8192)
    with torch.no_grad():
        model.output.fill_(0.003)
        model.bias.copy_(torch.linspace(-0.02, 0.03, 12))
    return {
        "format": FORMAT,
        "engine": ENGINE,
        "role": role,
        "recipe": asdict(Recipe(embedding_dim=8192)),
        "model": model.state_dict(),
        "provenance": dict(FIXED_PROVENANCE),
        "targets": list(TARGETS),
        "population": 3887,
        "epoch": 10,
        "completed_updates": 2430,
        "original_checkpoint_sha256": "a" * 64,
        "terminal_sha256": "b" * 64,
        "attributed_external_parent": True,
        "independent_internal_validation": False,
        "clinical_accuracy_claim": False,
    }


def inputs():
    g = torch.Generator().manual_seed(27)
    return (
        torch.randn(2, 94, 1024, generator=g),
        torch.randn(2, 94, 7, 1024, generator=g).half(),
        torch.randn(2, 12, generator=g),
        torch.arange(1, 95),
        torch.ones(94, dtype=torch.bool),
    )


@pytest.mark.parametrize("role", ROLES)
def test_typed_readout_matches_training_inputs_and_both_view_depth_conventions(role):
    model = model_from_component(component(role), role)
    f, r, p, c, present = inputs()
    probability, correction = predict_pair(model, role, f, r, p, c, present)
    slots, depth = window_metadata(c)
    _, mirror = window_metadata(c, mirrored=True)
    with torch.no_grad():
        logits, expected = model(
            feature_vector(f, r, role),
            p,
            slots[None].expand(2, -1),
            torch.stack([depth, mirror]),
            present[None].expand(2, -1),
            torch.ones(2, 94, dtype=torch.bool),
        )
    assert torch.equal(correction, expected)
    assert torch.equal(probability, tta_probabilities(logits[None])[0])
    assert (correction.abs() <= 0.5).all() and torch.isfinite(probability).all()
    other = ROLES[1 - ROLES.index(role)]
    with pytest.raises(ValueError):
        predict_pair(model, other, f, r, p, c, present)
    with pytest.raises(ValueError):
        predict_pair(model.train(), role, f, r, p, c, present)


@pytest.mark.parametrize(
    "change",
    [
        "role",
        "steps",
        "population",
        "source",
        "cache",
        "shape",
        "dtype",
        "zero",
        "extra",
        "clinical",
    ],
)
def test_old_reader_and_changed_endpoint_or_representation_are_rejected(change):
    value = component()
    with pytest.raises(ValueError):
        old_model_from_component(value)
    if change == "role":
        value["role"] = "global_repeat"
    elif change == "steps":
        value["completed_updates"] = 4
    elif change == "population":
        value["population"] = 32
    elif change == "source":
        value["provenance"]["source_git"] = "a" * 40
    elif change == "cache":
        value["provenance"]["coherent_feature_index_sha256"] = "a" * 64
    elif change == "shape":
        value["model"]["project.1.weight"] = value["model"]["project.1.weight"][:, :1024]
    elif change == "dtype":
        value["model"]["output"] = value["model"]["output"].half()
    elif change == "zero":
        value["model"]["output"].zero_()
    elif change == "extra":
        value["private_rows"] = ["FAKE_PRIVATE_UID"]
    else:
        value["clinical_accuracy_claim"] = True
    with pytest.raises((ValueError, RuntimeError)):
        model_from_component(value, "regional")


def fake_final(tmp_path, role):
    value = component(role)
    # This sanitized actual engineering receipt has no clinical rows. It does not
    # make the synthetic final-state header below an actual fitting result.
    audit = Path(__file__).resolve().parents[1] / "initiative/audits"
    recovery = json.loads(
        (audit / "regional-context-engine-20261016-v2/actual-recovery.json").read_text()
    )["actual_recovery"]
    r = tmp_path / "recovery.json"
    r.write_text(json.dumps(recovery, sort_keys=True, indent=2) + "\n")
    assert sha256(r) == RECOVERY_SHA
    binding = dict(
        FIXED_PROVENANCE,
        regional_engine=ENGINE,
        regional_role=role,
        window_feature_dimension=8192,
        diagnostic_only=False,
        private_rows=["FAKE_PRIVATE_UID"],
    )
    recipe = Recipe(embedding_dim=8192)
    state = {
        "version": VERSION,
        "recipe": asdict(recipe),
        "binding": binding,
        "model": value["model"],
        "population": 3887,
        "epoch": 10,
        "completed": 2430,
        "cursor": 0,
        "order": None,
        "scheduler": {
            "kind": "linear_warmup_cosine_v1",
            "batches_per_epoch": 243,
            "completed": 2430,
        },
        "history": [
            {
                "update": i + 1,
                "epoch": i // 243 + 1,
                "batch": i % 243 + 1,
                "studies": 15 if i % 243 == 242 else 16,
                "lr": learning_rate(recipe, i, 243),
                "loss": 0.3,
                "gradient_norm": 0.1,
                "max_abs_correction": 0.05,
            }
            for i in range(2430)
        ],
    }
    c = tmp_path / "original.pt"
    torch.save(state, c)
    terminal = {
        "engine": ENGINE,
        "role": role,
        "status": "fixed_final_complete",
        "diagnostic_only": False,
        "immutable_production_epoch_snapshots": True,
        "completed_updates": 2430,
        "epochs": 10,
        "studies": 3887,
        "trainable_parameters": 1334796,
        "binding": binding,
        "clinical_accuracy_evaluated": False,
        "independent_internal_validation": False,
        "CNN_parameters_fitted": 0,
        "checkpoint_sha256": sha256(c),
    }
    t = tmp_path / "terminal.json"
    t.write_text(json.dumps(terminal))
    return c, t, r, state, terminal, recovery


@pytest.mark.parametrize("role", ROLES)
def test_export_and_weights_only_roundtrip_strip_all_nonwhitelisted_payload(tmp_path, role):
    c, t, r, _, _, _ = fake_final(tmp_path, role)
    out = tmp_path / "head.pt"
    receipt = export_component(c, t, r, out, role)
    model, provenance = load_component(out, receipt["component_sha256"], role)
    assert provenance == FIXED_PROVENANCE
    value = torch.load(out, weights_only=True)
    assert "FAKE_PRIVATE_UID" not in json.dumps({k: v for k, v in value.items() if k != "model"})
    assert not {"history", "optimizer", "order"} & set(value)
    assert out.stat().st_size < 6 * 2**20 and out.stat().st_mode & 0o077 == 0
    assert torch.equal(
        predict_pair(model, role, *inputs())[0],
        predict_pair(model_from_component(value, role), role, *inputs())[0],
    )
    with pytest.raises(ValueError):
        load_component(out, "a" * 64, role)
    with pytest.raises(ValueError):
        export_component(c, t, r, out, role)


def test_changed_history_recovery_and_export_role_are_refused(tmp_path):
    c, t, r, state, terminal, recovery = fake_final(tmp_path, "regional")
    bad = copy.deepcopy(state)
    bad["history"][-1]["lr"] += 0.001
    with pytest.raises(ValueError, match="records"):
        validate_final(bad, terminal, recovery)
    recovery["roles"]["global_repeat"]["complete_decoded_state_exact"] = False
    with pytest.raises(ValueError, match="recovery"):
        validate_final(state, terminal, recovery)
    with pytest.raises(ValueError, match="role"):
        export_component(c, t, r, tmp_path / "wrong.pt", "global_repeat")
    r.write_text(r.read_text() + "\n")
    with pytest.raises(ValueError, match="bytes"):
        export_component(c, t, r, tmp_path / "changed-recovery.pt", "regional")
