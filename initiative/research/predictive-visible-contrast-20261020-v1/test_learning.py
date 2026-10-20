import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


controls = module("learning_controls")
sys.modules["learning_controls"] = controls
driver = module("learning_driver")
builder = module("worker_builder")
coordinator = module("learning_once")


def private(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def test_paid_credit_requires_sealed_original_job_and_actual_owned_log(tmp_path):
    primary, logs = tmp_path / "primary", tmp_path / "logs"
    assert controls.primary_paid_logs(primary, logs) == 0
    dispatch = primary / "dispatch"
    source = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
    private(dispatch / "intent.json", {"source_git": source,
        "owner_git": "2a224ae05fc7c2fc78d4d508b6407344ac523a1c",
        "arms": ["C", "B", "C0"], "automatic_retry": False})
    private(primary / "full-admission/binding-C.private.json", {"source": source})
    script = dispatch / "C.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    script.chmod(0o600)
    job = {"GPU_job": "99", "arm": "C", "source_git": source, "node": "evc27",
        "job_name": "rsna-predictive-full2-C", "run_root": "/tmp/rsna-predictive-full-C-20261020-v2",
        "binding_sha256": controls.sha((primary / "full-admission/binding-C.private.json").read_bytes()),
        "script_sha256": controls.sha(script.read_bytes()),
        "stdout": str(logs / "rsna-predictive-full2-C-99.out"),
        "stderr": str(logs / "rsna-predictive-full2-C-99.err")}
    private(dispatch / "C-submitted.json", job)
    private(dispatch / "C-intent.json", {**job, "automatic_retry": False})
    private(dispatch / "C-sbatch.json", {"returncode": 0, "stdout": "99\n"})
    assert controls.primary_paid_logs(primary, logs) == 0
    private(Path(job["stdout"]), {})
    assert controls.primary_paid_logs(primary, logs) == 1
    os.link(job["stdout"], job["stderr"])
    with pytest.raises(AssertionError):
        controls.primary_paid_logs(primary, logs)
    Path(job["stderr"]).unlink()
    private(dispatch / "C-sbatch.json", {"returncode": 0, "stdout": "100\n"})
    with pytest.raises(AssertionError):
        controls.primary_paid_logs(primary, logs)


def test_original_terminal_is_not_replaced_by_transport_success():
    good = "984645|COMPLETED|0:0|evc33\n984690|COMPLETED|0:0|evc33\n984691|COMPLETED|0:0|evc27\n"
    assert len(controls.terminal_jobs(good)) == 3
    for bad in (good.replace("0:0", "1:0", 1), good.replace("COMPLETED", "RUNNING", 1),
                good.replace("984690", "984692"), good + good):
        with pytest.raises(AssertionError):
            controls.terminal_jobs(bad)


def state():
    history = []
    for i in range(200):
        f = i / 2915
        factor = min(1., (i + 1) / (0.05 * 2916)) if f < .05 else (
            .1 + .9 * (1 + math.cos(math.pi * (f - .05) / .95)) / 2)
        history.append({"step": i + 1, "epoch": i // 2 + 1, "studies": 8 if i % 2 == 0 else 4,
            "encoder_updated": i >= 3, "LR": [3e-5 * factor] * 2,
            **dict.fromkeys(("loss", "exam_loss", "density_loss", "encoder_gradient_norm",
                            "head_gradient_norm", "gradient_norm"), 1.)})
    return {"recipe": builder.recipe(), "step": 200, "cohort_size": 12, "epoch_snapshots": False,
            "binding": {"matched_worker": {"role": "normalized"}}, "history": history,
            "model": {"parameter": torch.ones(1)}}


def test_acceptance_requires_full_schedule_peak_encoder_updates_and_finite_state():
    original = state()
    assert len(driver.check_history(original, builder.recipe(), "normalized")) == 200
    for key, bad in (("encoder_updated", False), ("loss", float("nan")), ("studies", 16),
                     ("LR", [3e-5, 3e-5])):
        changed = state()
        changed["history"][-1][key] = bad
        with pytest.raises(AssertionError):
            driver.check_history(changed, builder.recipe(), "normalized")
    original["model"]["parameter"][:] = float("inf")
    with pytest.raises(AssertionError):
        driver.check_history(original, builder.recipe(), "normalized")


def test_dispatch_is_one_ordinary_node_one_hour_and_no_shared_logs():
    script = coordinator.script()
    assert "--nodelist=evc27" in script and "--gres=gpu:nvidia_h100_pcie:1" in script
    assert "--time=01:00:00" in script and "--deadline=2026-10-21T17:15:00" in script
    assert "--output=/tmp/rsna-predictive-contrast-learning-20261020-v1/" in script
    assert "--error=/tmp/rsna-predictive-contrast-learning-20261020-v1/" in script
    assert driver.PHASES == (("control", "full"), ("normalized", "stop1"), ("normalized", "resume"))
