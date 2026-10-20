"""One bounded matched learning comparison; diagnostics remain private/noncompetition."""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from learning_controls import capacity

ROOT = Path("/tmp/rsna-predictive-contrast-learning-20261020-v1")
PHASES = (("control", "full"), ("normalized", "stop1"), ("normalized", "resume"))
MAX_SECONDS = 3600
FINISH_BY = datetime(2026, 10, 21, 17, 15, tzinfo=timezone.utc)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def once(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def check_history(state, recipe, role, steps=200):
    assert state["recipe"] == recipe and state["step"] == steps and state["cohort_size"] == 12
    assert state["epoch_snapshots"] is False and state["binding"]["matched_worker"]["role"] == role
    history = state["history"]
    assert len(history) == steps
    for i, row in enumerate(history):
        fraction = i / 2915
        factor = min(1., (i + 1) / (0.05 * 2916)) if fraction < .05 else (
            .1 + .9 * (1 + math.cos(math.pi * (fraction - .05) / .95)) / 2)
        assert row["step"] == i + 1 and row["epoch"] == i // 2 + 1
        assert row["studies"] == (8 if i % 2 == 0 else 4) and row["encoder_updated"] is (i >= 3)
        assert all(math.isfinite(row[k]) for k in ("loss", "exam_loss", "density_loss",
                   "encoder_gradient_norm", "head_gradient_norm", "gradient_norm"))
        assert row["encoder_gradient_norm"] > 0 and row["head_gradient_norm"] > 0
        assert len(row["LR"]) == 2 and all(math.isclose(x, 3e-5 * factor, rel_tol=1e-12) for x in row["LR"])
    assert all(torch.isfinite(value).all() for value in state["model"].values())
    return history


def main():
    import worker_builder as builder

    started = time.monotonic()
    os.umask(0o077)
    assert os.uname().nodename == "evc27" and os.environ.get("SLURM_JOB_GPUS")
    assert os.environ["SLURM_CPUS_PER_TASK"] == "2" and sys.dont_write_bytecode
    assert torch.cuda.device_count() == 1 and "H100" in torch.cuda.get_device_name(0)
    assert torch.ones(1, device="cuda").item() == 1
    node = subprocess.check_output(["scontrol", "show", "node", "evc27", "--oneliner"], text=True)
    assert re.search(r"(?:^|\s)Gres=gpu:nvidia_h100_pcie:2(?:\([^)]*\))?(?:\s|$)", node)
    assert datetime.now(timezone.utc).timestamp() + MAX_SECONDS <= FINISH_BY.timestamp()
    raw = (ROOT / "binding.private.json").read_bytes()
    binding = json.loads(raw)
    admission = json.loads((ROOT / "prepared.json").read_bytes())
    assert admission["status"] == "actual_contrast_matched_learning_CPU_admitted"
    assert admission["binding_sha256"] == sha(raw) and admission["source_git"] == binding["source_git"]
    assert binding["recipe"] == builder.recipe() and binding["metadata_width"] == 10
    assert binding["candidate_sha256"] == builder.CANDIDATE_SHA
    assert binding["phases"] == [list(x) for x in PHASES] and binding["actual_update_budget"] == 400
    for name, expected in binding["code_sha256"].items():
        assert sha((ROOT / "code" / name).read_bytes()) == expected
    assert str(torch.__version__) == "2.14.0+cu130"
    assert torch.version.git_version == "08187d9e0fba026dc8217405802ab5381dc88d90"
    once(ROOT / "GPU-run-start.json", {"GPU_job": os.environ["SLURM_JOB_ID"],
         "source_git": binding["source_git"], "binding_sha256": sha(raw), "capacity": capacity(ROOT)})
    terminals = []
    for role, phase in PHASES:
        assert not (ROOT / "engineering" / role / phase).exists()
        capacity(ROOT)
        remaining = MAX_SECONDS - (time.monotonic() - started)
        assert remaining > 0
        subprocess.run([sys.executable, "-B", str(ROOT / "code/worker.py"), role, phase], check=True,
                       timeout=remaining, env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        value = json.loads((ROOT / "engineering" / role / phase / "terminal.json").read_bytes())
        assert value["role"] == role and value["phase"] == phase
        assert value["steps"] == (1 if phase == "stop1" else 200) and value["metadata_width"] == 10
        assert value["density_weight"] == .1 and value["original_worker_sha256"] == builder.ORIGINAL_WORKER_SHA
        assert value["fit_study_exposures"] == {"full": 1200, "stop1": 8, "resume": 1192}[phase]
        assert value["encoder_parameters_changed"] is (phase != "stop1") and value["head_parameters_changed"]
        terminals.append(value)
        assert all(x["initial_digests"] == terminals[0]["initial_digests"] for x in terminals)
        capacity(ROOT)
    results = {}
    for role, phase in (("control", "full"), ("normalized", "resume")):
        path = ROOT / "engineering" / role / phase / "final.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        history = check_history(state, builder.recipe(), role)
        assert {k: v for k, v in state["binding"].items() if k != "matched_worker"} == binding
        predictions = (path.parent / "prediction.private.csv").read_bytes()
        values = np.array([[float(x) for x in line.split(",")] for line in predictions.decode().splitlines()])
        assert values.shape == (4, 12) and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        if role == "normalized":
            stopped = torch.load(ROOT / "engineering/normalized/stop1/stopped.pt", map_location="cpu", weights_only=False)
            check_history(stopped, builder.recipe(), role, steps=1)
            assert stopped["history"] == history[:1] and stopped["binding"] == state["binding"]
            del stopped
        results[role] = {"history": history, "checkpoint_bytes": path.stat().st_size,
                         "prediction_sha256": sha(predictions), "terminal": terminals[0 if role == "control" else 2]}
        del state
    once(ROOT / "GPU-completed.json", {
        "status": "actual_matched_visible_contrast_learning_complete", "GPU_job": os.environ["SLURM_JOB_ID"],
        "source_git": binding["source_git"], "binding_sha256": sha(raw), "roles": results,
        "actual_optimizer_updates": 400, "common_initial_digests": terminals[0]["initial_digests"],
        "actual_normalized_stop1_resume199_completed": True,
        "new_GPU_full_versus_resume_equality_test_performed": False,
        "small_engineering_predictions_not_used_for_selection": True,
        "clinical_accuracy_or_automatic_full_promotion": False, "elapsed_seconds": time.monotonic() - started,
        "capacity": capacity(ROOT)})
    print(json.dumps({"status": "actual_matched_visible_contrast_learning_complete",
                      "actual_optimizer_updates": 400, "GPU_job": os.environ["SLURM_JOB_ID"]}), flush=True)


if __name__ == "__main__":
    main()
