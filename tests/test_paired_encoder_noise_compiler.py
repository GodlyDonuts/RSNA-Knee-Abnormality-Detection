"""New compiler only; synthetic source fixtures do not stand for a GPU test."""

import ast
import base64
import hashlib

import pytest
from test_paired_logit_recovery import payload

from scripts.kaggle import paired_encoder_noise_probe as m
from scripts.kaggle import paired_logit_recovery as paired


def source_fixture(monkeypatch):
    args = payload(monkeypatch)
    recovery, _ = paired.build(*args, permission="synthetic permission")
    monkeypatch.setattr(m, "RECOVERY_SOURCE", hashlib.sha256(recovery.encode()).hexdigest())
    monkeypatch.setattr(m, "provenance", lambda *a: 3)
    training = {
        "kernel_id": 139220032,
        "role": "influence",
        "epoch": 3,
        "source_sha256": "cff5c493ee2832a0672298f1942662474aa3a01d0b421069c4db171ed394b012",
    }
    publication = {"files": {"model.pt": {"bytes": 292870259, "sha256": m.ADAPTED_MODEL}}}
    return recovery, {n: "# synthetic new closure\n" for n in m.EXTRA}, training, publication


def test_probe_is_not_a_recovery_replay_or_optimizer_fit_and_preserves_single_view_objective(
    monkeypatch,
):
    args = source_fixture(monkeypatch)
    source, meta = m.build(*args)
    compile(source, "new-synthetic-noise-bootstrap", "exec")
    values = paired.literals(source)
    worker = base64.b64decode(values["WORKER"], validate=True).decode()
    main = next(
        n for n in ast.parse(worker).body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = {
        n.func.id
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "measure_case" in calls and "fit" not in calls
    assert "range(rank,4,2)" in worker
    assert "reference_draws=16,paired_draws=32" in worker
    assert "validate(state,BINDING['training'],BINDING['publication']" in worker
    assert "model.load_state_dict" in worker and "73136096" in worker
    assert meta["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert meta["worker_sha256"] == hashlib.sha256(worker.encode()).hexdigest()
    assert meta["competition_entries"] == 0 and meta["required_actual_recovery_kernel"] == 139417155
    assert "'loss_averages_views':False" in source
    assert "'optimizer_step_time_or_proven_gradient_bias_accepted':False" in source
    assert "same(full,resumed)" not in source


@pytest.mark.parametrize("bad", ["source", "closure", "epoch", "role"])
def test_wrong_scientific_inputs_cannot_compile_new_gpu_source(monkeypatch, bad):
    recovery, files, training, publication = source_fixture(monkeypatch)
    if bad == "source":
        recovery += "\n# unbound\n"
    elif bad == "closure":
        files.pop(m.EXTRA[0])
    elif bad == "epoch":
        training["epoch"] = 2
    else:
        training["role"] = "uniform"
    with pytest.raises(ValueError):
        m.build(recovery, files, training, publication)
