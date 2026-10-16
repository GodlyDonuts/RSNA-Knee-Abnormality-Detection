import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from rsna.competitive.released_context import ContextResidual, window_metadata
from rsna.competitive.released_context_infer import (
    FEATURE_SOURCE_SHA,
    FORMAT,
    HELPER_SHA,
    LABEL_SOURCE_SHA,
    PARENT_SOURCE,
    POPULATION_SHA,
    PROVENANCE_KEYS,
    SOURCE_GIT,
    export_component,
    load_component,
    model_from_component,
    predict_context_pair,
    sha256,
    validate_final,
)
from rsna.competitive.released_context_train import PARENT_SHA, VERSION, Recipe, learning_rate
from rsna.schema import TARGETS


def component():
    m = ContextResidual()
    with torch.no_grad():
        m.output.copy_(
            torch.randn(m.output.shape, generator=torch.Generator().manual_seed(7)) * 0.02
        )
        m.bias.copy_(torch.linspace(-0.02, 0.03, 12))
    p = dict.fromkeys(PROVENANCE_KEYS, "a" * 64)
    p.update(
        source_git=SOURCE_GIT,
        parent_weights_sha256=PARENT_SHA,
        parent_source_sha256=PARENT_SOURCE,
        feature_pipeline_helper_sha256=HELPER_SHA,
        safe_population_uid_sha256=POPULATION_SHA,
        label_source_sha256=LABEL_SOURCE_SHA,
        original_feature_recovery_source_sha256=FEATURE_SOURCE_SHA,
    )
    return {
        "format": FORMAT,
        "recipe": asdict(Recipe()),
        "model": m.state_dict(),
        "provenance": p,
        "targets": list(TARGETS),
        "population": 3887,
        "epoch": 10,
        "completed_updates": 2430,
        "original_checkpoint_sha256": "b" * 64,
        "terminal_sha256": "c" * 64,
        "attributed_external_parent": True,
        "independent_internal_validation": False,
        "clinical_accuracy_claim": False,
    }


def inputs():
    g = torch.Generator().manual_seed(5)
    return (
        torch.randn(2, 94, 1024, generator=g),
        torch.randn(2, 12, generator=g),
        torch.arange(1, 95),
        torch.ones(94, dtype=torch.bool),
    )


def test_exact_two_view_readout_uses_registered_metadata_and_mean_probabilities():
    m = model_from_component(component())
    f, p, c, present = inputs()
    probability, correction = predict_context_pair(m, f, p, c, present)
    slots, plain = window_metadata(c)
    _, mirror = window_metadata(c, mirrored=True)
    with torch.no_grad():
        value, expected = m(
            f,
            p,
            slots[None].expand(2, -1),
            torch.stack([plain, mirror]),
            present[None].expand(2, -1),
            torch.ones(2, 94, dtype=torch.bool),
        )
    assert torch.equal(correction, expected) and (correction.abs() <= 0.5).all()
    assert torch.equal(probability, value.sigmoid().mean(0))
    assert torch.isfinite(probability).all()
    with pytest.raises(ValueError):
        predict_context_pair(m.train(), f, p, c, present)


def test_small_weights_only_roundtrip_and_hash_guard(tmp_path):
    value = component()
    path = tmp_path / "head.pt"
    torch.save(value, path)
    m, p = load_component(path, sha256(path))
    assert p == value["provenance"]
    assert np.array_equal(
        predict_context_pair(m, *inputs())[0].numpy(),
        predict_context_pair(model_from_component(value), *inputs())[0].numpy(),
    )
    with pytest.raises(ValueError, match="hash"):
        load_component(path, "0" * 64)


@pytest.mark.parametrize(
    "change",
    [
        "steps",
        "epoch",
        "population",
        "parent",
        "source",
        "feature_source",
        "private_key",
        "dtype",
        "zero",
        "clinical",
        "recipe",
    ],
)
def test_diagnostics_altered_scope_and_extra_payload_are_rejected(change):
    v = component()
    if change == "steps":
        v["completed_updates"] = 4
    elif change == "epoch":
        v["epoch"] = 2
    elif change == "population":
        v["population"] = 32
    elif change == "parent":
        v["provenance"]["parent_weights_sha256"] = "d" * 64
    elif change == "source":
        v["provenance"]["source_git"] = "d" * 40
    elif change == "feature_source":
        v["provenance"]["original_feature_recovery_source_sha256"] = "d" * 64
    elif change == "private_key":
        v["private_rows"] = ["never export"]
    elif change == "dtype":
        v["model"]["output"] = v["model"]["output"].half()
    elif change == "zero":
        v["model"]["output"].zero_()
    elif change == "clinical":
        v["clinical_accuracy_claim"] = True
    else:
        v["recipe"]["source_weight"] = 1
    with pytest.raises((ValueError, RuntimeError)):
        model_from_component(v)


def fake_final(tmp_path):
    # Synthetic header/graph tests only. Never a real training admission/export.
    v = component()
    recovery = {
        "status": "actual32_MRI_full_context_engine_complete_recovery_accepted",
        "source_git": SOURCE_GIT,
        "all_head_AdamW_LR_counters_order_all_RNG_history_exact": True,
        "all32x12_prediction_bytes_exact": True,
    }
    r = tmp_path / "recovery.json"
    r.write_text(json.dumps(recovery))
    binding = copy.deepcopy(v["provenance"])
    binding["actual_engine_recovery_sha256"] = sha256(r)
    binding["private_rows"] = ["FAKE_PRIVATE_UID_NEVER_EXPORT"]
    history = [
        {
            "update": i + 1,
            "epoch": i // 243 + 1,
            "batch": i % 243 + 1,
            "studies": 15 if i % 243 == 242 else 16,
            "lr": learning_rate(Recipe(), i, 243),
            "loss": 0.4,
            "gradient_norm": 0.1,
            "max_abs_correction": 0.02,
        }
        for i in range(2430)
    ]
    state = {
        "version": VERSION,
        "recipe": asdict(Recipe()),
        "model": v["model"],
        "binding": binding,
        "population": 3887,
        "completed": 2430,
        "epoch": 10,
        "cursor": 0,
        "order": None,
        "history": history,
        "scheduler": {
            "kind": "linear_warmup_cosine_v1",
            "batches_per_epoch": 243,
            "completed": 2430,
        },
    }
    terminal = {
        "status": "fixed_final_complete",
        "completed_updates": 2430,
        "epochs": 10,
        "studies": 3887,
        "trainable_parameters": 402956,
        "binding": binding,
        "clinical_accuracy_evaluated": False,
        "independent_internal_validation": False,
        "CNN_parameters_fitted": 0,
    }
    c = tmp_path / "original.pt"
    torch.save(state, c)
    terminal["checkpoint_sha256"] = sha256(c)
    t = tmp_path / "terminal.json"
    t.write_text(json.dumps(terminal))
    return c, t, r, state, terminal, recovery


def test_export_whitelists_provenance_and_never_copies_private_rows_or_training_payload(tmp_path):
    c, t, r, state, _, _ = fake_final(tmp_path)
    out = tmp_path / "head.pt"
    receipt = export_component(c, t, r, out)
    m, _ = load_component(out, receipt["component_sha256"])
    assert not m.training
    v = torch.load(out, weights_only=True)
    assert "history" not in v and "optimizer" not in v
    assert "private_rows" not in v["provenance"]
    assert "FAKE_PRIVATE_UID" not in json.dumps({k: x for k, x in v.items() if k != "model"})
    assert v["original_checkpoint_sha256"] == hashlib.sha256(c.read_bytes()).hexdigest()
    assert state["binding"]["private_rows"]
    with pytest.raises(ValueError, match="Fresh"):
        export_component(c, t, r, out)


@pytest.mark.parametrize("bad", ["missing_update", "lr", "last_batch", "recovery"])
def test_export_requires_complete_actual_counter_and_fixed_schedule_scope(tmp_path, bad):
    _, _, _, s, t, r = fake_final(tmp_path)
    if bad == "missing_update":
        s["history"].pop(10)
    elif bad == "lr":
        s["history"][10]["lr"] *= 2
    elif bad == "last_batch":
        s["history"][-1]["studies"] = 16
    else:
        r["all32x12_prediction_bytes_exact"] = False
    with pytest.raises(ValueError):
        validate_final(s, t, r)
