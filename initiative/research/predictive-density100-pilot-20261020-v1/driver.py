"""One fresh C100 full/stop/resume pilot; no accuracy gate or automatic promotion."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def check_history(state, binding):
    expected = {
        "arm": "C", "epochs": 12, "effective_batch": 8, "encoder_lr": 3e-5,
        "head_lr": 3e-5, "weight_decay": 0.02, "warmup_fraction": 0.05,
        "end_lr_fraction": 0.1, "density_weight": 100.0, "head_only_updates": 3,
        "clip": 1.0, "seed": 20261020, "schedule_updates": 2916,
    }
    assert state["recipe"] == expected and state["binding"] == binding
    assert state["step"] == 24 and state["cohort_size"] == 12 and state["epoch_snapshots"] is False
    history = state["history"]
    assert [r["step"] for r in history] == list(range(1, 25))
    assert [r["studies"] for r in history] == [8, 4] * 12
    assert [r["encoder_updated"] for r in history] == [False] * 3 + [True] * 21
    for row in history:
        for key in ("loss", "exam_loss", "density_loss", "encoder_gradient_norm",
                    "head_gradient_norm", "gradient_norm"):
            assert math.isfinite(row[key])
        assert row["encoder_gradient_norm"] > 0 and row["head_gradient_norm"] > 0
        assert all(math.isfinite(lr) and 0 < lr <= 3e-5 for lr in row["LR"])
    return history


def main():
    import worker_builder as c

    D = c.ROOT
    os.umask(0o077)
    assert os.uname().nodename == "evc27" and os.environ.get("SLURM_JOB_GPUS")
    c.admit_time(datetime.now(timezone.utc))
    assert torch.cuda.device_count() == 1 and "H100" in torch.cuda.get_device_name(0)
    assert os.environ["SLURM_CPUS_PER_TASK"] == "2"
    assert "Gres=gpu:nvidia_h100_pcie:2" in subprocess.check_output(
        ["scontrol", "show", "node", "evc27", "--oneliner"], text=True,
    )
    binding_bytes = (D / "binding.private.json").read_bytes()
    binding = json.loads(binding_bytes)
    assert c.sha(binding_bytes) == json.loads((D / "prepared.json").read_bytes())["binding_sha256"]
    assert binding["branch"] == "C100" and binding["original_numerical_source_git"] == c.SCIENCE
    for name, expected in binding["code_sha256"].items():
        assert c.sha((D / "code" / name).read_bytes()) == expected
    assert str(torch.__version__) == binding["runtime_versions"]["torch"] == "2.14.0+cu130"
    assert torch.version.git_version == "08187d9e0fba026dc8217405802ab5381dc88d90"
    for name, proof in binding["runtime_files"].items():
        path = c.PORTABLE / name if name != "resolved_python" else c.Path(proof["path"])
        st = path.stat()
        assert [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns] == proof["stat"]
        if name == "resolved_python":
            assert st.st_uid == proof["owner_uid"] and st.st_mode & 0o777 == proof["mode"]
            assert not st.st_mode & 0o022 and st.st_mode & 0o111
        if "sha256" in proof:
            assert c.sha(path.read_bytes()) == proof["sha256"]
    assert not (D / "GPU-run-start.json").exists()

    def storage():
        files = [p for p in D.rglob("*") if p.is_file()]
        checkpoint_bytes = sum(p.stat().st_size for p in files if (D / "engineering") in p.parents)
        control_bytes = sum(p.stat().st_size for p in files if (D / "engineering") not in p.parents)
        assert checkpoint_bytes <= c.CHECKPOINT_CAP and control_bytes <= c.CONTROL_CAP
        c.node_capacity(shutil.disk_usage("/tmp").free, checkpoint_bytes, control_bytes)
        return {"engineering_bytes": checkpoint_bytes, "controls_and_logs_bytes": control_bytes}

    storage()
    c.once(D / "GPU-run-start.json", {
        "GPU_job": os.environ["SLURM_JOB_ID"], "source_git": binding["source_git"],
        "branch": "C100", "full_training_or_competition_candidate": False,
    })
    for phase in ("full", "stop1", "resume"):
        subprocess.run(
            [sys.executable, "-B", str(D / "code/worker.py"), "C", phase], check=True,
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                 "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        )
        storage()
    root = D / "engineering/C"
    complete = torch.load(root / "full/final.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(root / "resume/final.pt", map_location="cpu", weights_only=False)
    equal(complete, resumed)
    history = check_history(complete, binding)
    prediction = (root / "full/prediction.private.csv").read_bytes()
    assert prediction == (root / "resume/prediction.private.csv").read_bytes()
    values = np.loadtxt(root / "full/prediction.private.csv", delimiter=",")
    assert values.shape == (4, 12) and np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    terminal = json.loads((root / "full/terminal.json").read_bytes())
    assert terminal["density_weight"] == 100 and terminal["steps"] == 24
    assert all(math.isfinite(x) for x in terminal["training_only_numerical_ranges"].values())
    original = json.loads((D / "original-engineering.json").read_bytes())
    original_c = next(arm for arm in original["arms"] if arm["arm"] == "C")
    out = {
        "status": "actual_C100_full_stop_resume_engineering_complete",
        "GPU_job": os.environ["SLURM_JOB_ID"], "source_git": binding["source_git"],
        "original_numerical_source_git": c.SCIENCE, "branch": "C100",
        "original_engineering_receipt_sha256": c.ENGINEERING_SHA,
        "objective_measurement_sha256": c.MEASUREMENT_SHA,
        "scientific_decision_sha256": binding["scientific_decision_sha256"],
        "binding_sha256": c.sha(binding_bytes),
        "full_stop_resume_complete_state_and_prediction_bytes_equal": True,
        "private_engineering_prediction_sha256": hashlib.sha256(prediction).hexdigest(),
        "terminal": terminal, "real_optimizer_history": history,
        "engineering_warnings": c.scientific_warnings(history, terminal, original_c),
        "storage": storage(), "new_shared_files": 0,
        "clinical_or_held_out_accuracy_claim": False, "automatic_full_training_promotion": False,
        "primary_B_C_C0_pipeline_unchanged": True,
    }
    c.once(D / "GPU-accepted.json", out)
    storage()
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
