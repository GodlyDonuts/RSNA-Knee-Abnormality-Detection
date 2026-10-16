import copy
import json

import pytest
import torch
from test_released_context_infer import fake_final, inputs

from rsna.competitive import ranking_context_infer as c
from rsna.competitive import released_context_infer as original


def synthetic(tmp_path):
    checkpoint, terminal, recovery, state, t, _ = fake_final(tmp_path)
    state["version"] = c.VERSION
    state["recipe"] = dict(c.FIXED_RECIPE)
    state["original_trainer_sha256"] = (
        "5addea8212501ba31021c739543a7fa627cf70066ad439fa0889ba5c5b3fa59f"
    )
    state["binding"].update(
        ranking_source_git=c.SOURCE_GIT,
        ranking_engine=c.VERSION,
        objective="source_rank",
        original_full_input_manifest_sha256="a4c5b4910dc818293cdfb6ed59ba92bfa16faa5e0cb2c39c7a5e5b41df877ab0",
    )
    t.update(engine=c.VERSION, objective="source_rank", recipe=dict(c.FIXED_RECIPE))
    r = {
        "status": "actual32_source_rank_engine_recovery_and_uniform_parity_accepted",
        "fresh_processes": 7,
        "uniform_complete_numerical_state_and_predictions_equal_frozen_original": True,
        "roles": {
            role: {
                "complete_model_AdamW_LR_order_counters_allRNG_history_equal": True,
                "all32x12_prediction_bytes_equal": True,
                "updates": 4,
            }
            for role in ("uniform", "source_rank")
        },
    }
    parity = tmp_path / "parity.json"
    p = {
        "status": "fixed_final_complete",
        "completed_updates": 2430,
        "studies": 3887,
        "epochs": 10,
        "trainable_parameters": 402956,
        "checkpoint_sha256": "c5d53af3d60eace275670c0953cb53df273db4b8daf219355d959718d777e634",
        "prediction_sha256": "a1b068586851acfc175b202152c1708de20daaa7b1a044be510f0a55577173da",
        "inputs_sha256": "a4c5b4910dc818293cdfb6ed59ba92bfa16faa5e0cb2c39c7a5e5b41df877ab0",
        "binding": {"source_git": c.ORIGINAL_SOURCE_GIT},
    }
    torch.save(state, checkpoint)
    t["checkpoint_sha256"] = c.sha256(checkpoint)
    terminal.write_text(json.dumps(t))
    recovery.write_text(json.dumps(r))
    parity.write_text(json.dumps(p))
    return checkpoint, terminal, recovery, parity, state, t, r, p


def test_typed_ranking_export_roundtrip_and_original_loader_rejection(tmp_path):
    checkpoint, terminal, recovery, parity, _, _, _, _ = synthetic(tmp_path)
    output = tmp_path / "ranking.pt"
    receipt = c.export_component(checkpoint, terminal, recovery, parity, output)
    m, provenance = c.load_component(output, receipt["component_sha256"])
    value = torch.load(output, weights_only=True)
    assert provenance["ranking_engine_recovery_sha256"] == c.sha256(recovery)
    assert provenance["original_control_terminal_sha256"] == c.sha256(parity)
    assert "private_rows" not in provenance and "history" not in value
    assert c.predict_context_pair(m, *inputs())[0].shape == (12,)
    with pytest.raises(ValueError):
        original.model_from_component(value)


@pytest.mark.parametrize(
    "bad", ["diagnostic", "uniform", "source", "recovery", "parity", "missing_update", "lr"]
)
def test_final_export_requires_actual_full_ranking_and_both_original_proofs(tmp_path, bad):
    _, _, _, _, s, t, r, p = synthetic(tmp_path)
    s, t, r, p = copy.deepcopy((s, t, r, p))
    if bad == "diagnostic":
        s["completed"] = 4
    elif bad == "uniform":
        s["recipe"]["objective"] = "uniform"
    elif bad == "source":
        s["binding"]["ranking_source_git"] = "a" * 40
    elif bad == "recovery":
        r["roles"]["source_rank"]["all32x12_prediction_bytes_equal"] = False
    elif bad == "parity":
        p["checkpoint_sha256"] = "a" * 64
    elif bad == "missing_update":
        s["history"].pop()
    else:
        s["history"][-1]["lr"] = 0.1
    with pytest.raises(ValueError):
        c.validate_final(s, t, r, p)
