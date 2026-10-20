"""Prepared one-GPU matched C100/G100 engineering run; no dispatch authority."""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/tmp/rsna-predictive-generative-routing-20261020-v1")
PORTABLE = Path("/tmp/rsna-predictive-portable-recovery-20261020-v1")
PILOT = Path("/tmp/rsna-predictive-density100-pilot-20261020-v1/GPU-accepted.json")
SCIENCE = "39e315c65546aa5205518713bc27349ec63e7538"
PILOT_SOURCE = "068a14a75e4d2b3bd83dda7783528956f7a2ca0f"
PILOT_BINDING = "e11c7c7fe5c1fd7cae87df313fb5d42be7128b7906f8d716a486b0857bbdd37a"
BRANCH = "matched_C100_G100_generative_routing"
PHASES = (("C100", "full"), ("G100", "full"), ("G100", "stop1"), ("G100", "resume"))
GIB, MIB = 2**30, 2**20
CHECKPOINT_CAP, CONTROL_CAP, MAX_SECONDS = 6 * GIB, 4 * MIB, 5400


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def recipe():
    return dict(arm="C", epochs=100, effective_batch=8, encoder_lr=3e-5, head_lr=3e-5,
                weight_decay=0.02, warmup_fraction=0.05, end_lr_fraction=0.1,
                density_weight=100.0, head_only_updates=3, clip=1.0,
                seed=20261020, schedule_updates=2916)


def read_reference(ref):
    path = Path(ref["path"])
    before = path.stat()
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and before.st_uid == os.getuid() and not before.st_mode & 0o077
            and before.st_size <= CONTROL_CAP, "Owner-only bounded admission reference required")
    raw = path.read_bytes()
    after = path.stat()
    require(sha(raw) == ref["sha256"] and signature(before) == signature(after),
            "Admission reference changed")
    return json.loads(raw)


def signature(st):
    return [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def once(path, value):
    pending = path.with_name(path.name + ".pending")
    with pending.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(pending, path)
    finally:
        pending.unlink()


def pilot_completed(accounting):
    rows = [row.split("|") for row in accounting.splitlines()
            if row.split("|")[0] == "981620"]
    require(len(rows) == 1 and rows[0][:4] == ["981620", "COMPLETED", "0:0", "evc27"],
            "Original C100 pilot has not actually completed successfully")


def validate_prerequisites(binding, pilot, decision, funding):
    require(binding["branch"] == BRANCH and binding["roles"] == ["C100", "G100"]
            and binding["original_numerical_source_git"] == SCIENCE
            and re.fullmatch("[0-9a-f]{40}", binding["source_git"])
            and binding["recipe"] == recipe()
            and binding["density100_pilot"]["path"] == str(PILOT), "Distinct exact matched binding required")
    require(pilot["status"] == "actual_C100_full_stop_resume_engineering_complete"
            and pilot["GPU_job"] == "981620" and pilot["source_git"] == PILOT_SOURCE
            and pilot["binding_sha256"] == PILOT_BINDING and pilot["branch"] == "C100"
            and pilot["original_numerical_source_git"] == SCIENCE
            and pilot["full_stop_resume_complete_state_and_prediction_bytes_equal"] is True
            and pilot["terminal"]["steps"] == 24 and pilot["terminal"]["density_weight"] == 100
            and pilot["automatic_full_training_promotion"] is False
            and pilot["clinical_or_held_out_accuracy_claim"] is False,
            "Actual original C100 acceptance is a prerequisite")
    history = pilot["real_optimizer_history"]
    require([r["step"] for r in history] == list(range(1, 25))
            and [r["studies"] for r in history] == [8, 4] * 12,
            "Original accepted pilot update history differs")
    require(all(math.isfinite(r[k]) for r in history for k in
                ("loss", "exam_loss", "density_loss", "encoder_gradient_norm",
                 "head_gradient_norm", "gradient_norm")), "Nonfinite original pilot history")
    scheduler = decision["pilot_scheduler"]
    require(scheduler["GPU_job"] == "981620" and scheduler["state"] == "COMPLETED"
            and scheduler["exit_code"] == "0:0" and scheduler["node"] == "evc27",
            "Actual successful pilot scheduler receipt required")
    require(decision["status"] == "accepted_matched_generative_routing_engineering"
            and decision["source_git"] == binding["source_git"]
            and decision["code_sha256"] == binding["code_sha256"]
            and decision["original_numerical_source_git"] == SCIENCE
            and decision["pilot_acceptance_sha256"] == binding["density100_pilot"]["sha256"]
            and decision["pilot_warnings_reviewed"] == pilot["engineering_warnings"]
            and decision["recipe"] == recipe() and decision["roles"] == ["C100", "G100"]
            and decision["phases"] == [list(x) for x in PHASES]
            and decision["actual_optimizer_updates"] == 600
            and decision["maximum_GPU_seconds"] == MAX_SECONDS
            and decision["engineering_predictions_used_for_selection"] is False
            and decision["automatic_recipe_retuning"] is False
            and decision["primary_B_C_C0_C100_unchanged"] is True
            and decision["automatic_full_training_promotion"] is False,
            "Explicit separate root engineering decision required")
    require(funding["status"] == "actual_matched_generative_routing_scope_funded"
            and funding["branch"] == BRANCH and funding["node"] == "evc27"
            and funding["source_git"] == binding["source_git"]
            and funding["new_checkpoint_cap_bytes"] == CHECKPOINT_CAP
            and funding["new_control_and_log_cap_bytes"] == CONTROL_CAP
            and funding["standing_free_reserve_bytes"] == 20 * GIB
            and funding["standing_free_file_reserve"] == 5000
            and funding["prior_C100_pilot_scope_retained"] is True
            and funding["private_descriptor_scale_control_bytes"] >= MIB
            and type(funding["other_unpaid_node_liabilities_bytes"]) is int
            and funding["other_unpaid_node_liabilities_bytes"] >= CHECKPOINT_CAP + CONTROL_CAP + MIB
            and funding["shared_future_bytes"] == 123 * MIB
            and funding["shared_future_files"] == 376
            and funding["new_shared_files"] == funding["new_shared_bytes"] == 0,
            "Fresh separate funding must preserve every earlier obligation")


def topology(node_description):
    gres = re.findall(r"(?:^|\s)Gres=([^\s]+)", node_description)
    require(len(gres) == 1 and re.fullmatch(r"gpu:nvidia_h100_pcie:2(?:\([^)]*\))?", gres[0]),
            "Only an ordinary physical two-H100 evc27 is admitted")


def storage(root, funding):
    files = [p for p in root.rglob("*") if p.is_file()]
    require(all(not p.is_symlink() and p.stat().st_uid == os.getuid() for p in files),
            "Only owned regular pilot output may be credited")
    checkpoints = sum(p.stat().st_size for p in files if root / "engineering" in p.parents)
    controls = sum(p.stat().st_size for p in files if root / "engineering" not in p.parents)
    require(checkpoints <= CHECKPOINT_CAP and controls <= CONTROL_CAP, "New pilot scope exceeded")
    required = (funding["standing_free_reserve_bytes"]
                + funding["other_unpaid_node_liabilities_bytes"]
                + CHECKPOINT_CAP - checkpoints + CONTROL_CAP - controls)
    free = shutil.disk_usage(root).free
    require(free > required, "Fresh free space no longer covers all retained liabilities")
    return dict(engineering_bytes=checkpoints, controls_and_logs_bytes=controls,
                fresh_node_free_bytes=free, required_node_free_bytes=required)


def paid_log_count(jobs, root=Path("/home/sa305415/rsna-knee/kaggle")):
    """Exact existing primary logs; missing files retain their future obligation."""
    require(set(jobs) <= {"B", "C", "C0"}, "Unknown primary log role")
    inodes = set()
    for arm, job in jobs.items():
        name = "rsna-predictive-full2-" + arm
        require(re.fullmatch(r"[0-9]+", job["GPU_job"]) and job["arm"] == arm
                and job["source_git"] == "44cc559e947560a635c34fee7b1f2a3b7be3a241"
                and job["node"] == ("evc27" if arm == "C0" else "evc33")
                and job["job_name"] == name, "Original primary dispatch identity required")
        size = 0
        for field, suffix in (("stdout", "out"), ("stderr", "err")):
            path = root / (name + "-" + job["GPU_job"] + "." + suffix)
            require(job[field] == str(path) and not path.is_symlink(), "Original log path required")
            if not path.exists():
                continue
            st = path.stat()
            require(path.is_file() and st.st_uid == os.getuid() and not st.st_mode & 0o077
                    and (st.st_dev, st.st_ino) not in inodes, "Private distinct original log required")
            size += st.st_size
            inodes.add((st.st_dev, st.st_ino))
        require(size <= 8 * MIB, "Original combined log bound exceeded")
    return len(inodes)


def shared_capacity(text, funding, paid_files=0):
    row = next(r.split() for r in text.splitlines() if r.strip().startswith("/home/sa305415"))
    free_bytes, free_files = (int(row[2]) - int(row[1])) * 1024, int(row[6]) - int(row[5])
    require(type(paid_files) is int and 0 <= paid_files <= 6, "Only six exact primary logs may be credited")
    require(free_bytes > 20 * GIB + funding["shared_future_bytes"]
            and free_files >= 5000 + funding["shared_future_files"] - paid_files,
            "Fresh shared reserve is insufficient")
    return dict(free_bytes=free_bytes, free_files=free_files, new_shared_files=0,
                exact_paid_original_log_files=paid_files, originals_deleted=False)


def equal(a, b):
    require(type(a) is type(b), "Recovery state type differs")
    if isinstance(a, torch.Tensor):
        require(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), "Recovery tensor differs")
    elif isinstance(a, np.ndarray):
        require(a.dtype == b.dtype and np.array_equal(a, b), "Recovery NumPy RNG differs")
    elif isinstance(a, dict):
        require(a.keys() == b.keys(), "Recovery state keys differ")
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        require(len(a) == len(b), "Recovery state length differs")
        for left, right in zip(a, b):
            equal(left, right)
    else:
        require(a == b, "Recovery scalar differs")


def check_history(state, binding, role):
    worker = state["binding"]["matched_worker"]
    require(state["recipe"] == recipe() and state["step"] == 200 and state["cohort_size"] == 12
            and state["epoch_snapshots"] is False and worker["role"] == role
            and {k: v for k, v in state["binding"].items() if k != "matched_worker"} == binding,
            "Exact 200-update source-bound checkpoint required")
    history = state["history"]
    require(len(history) == 200, "Incomplete optimizer history")
    for i, row in enumerate(history):
        fraction, warm = i / 2915, 0.05
        factor = min(1.0, (i + 1) / (warm * 2916)) if fraction < warm else (
            0.1 + 0.9 * (1 + math.cos(math.pi * (fraction - warm) / (1 - warm))) / 2)
        require(row["step"] == i + 1 and row["epoch"] == i // 2 + 1
                and row["studies"] == (8 if i % 2 == 0 else 4)
                and row["encoder_updated"] is (i >= 3), "Original study exposure or warmup changed")
        require(all(math.isfinite(row[k]) for k in ("loss", "exam_loss", "density_loss",
                    "encoder_gradient_norm", "head_gradient_norm", "gradient_norm"))
                and row["encoder_gradient_norm"] > 0 and row["head_gradient_norm"] > 0,
                "Nonfinite or absent real optimizer gradients")
        require(len(row["LR"]) == 2 and all(math.isclose(lr, 3e-5 * factor, rel_tol=1e-12)
                                            for lr in row["LR"]), "Production LR horizon changed")
    require(history[145]["LR"] == [3e-5, 3e-5], "Peak production learning rate was not exercised")
    for group in (state["model"], state["optimizer"]["state"]):
        def finite(item):
            if isinstance(item, torch.Tensor):
                require(bool(torch.isfinite(item).all()), "Nonfinite checkpoint tensor")
            elif isinstance(item, dict):
                for value in item.values():
                    finite(value)
        finite(group)
    return history


def validate_terminal(value, role, phase, builder):
    require(value["role"] == role and value["arm"] == "C" and value["phase"] == phase
            and value["steps"] == (1 if phase == "stop1" else 200)
            and value["seed"] == builder.SEED and value["training_epochs"] == 100
            and value["planned_updates"] == 200 and value["density_weight"] == 100
            and value["production_LR_horizon_updates"] == 2916
            and value["fit_study_exposures"] == {"full": 1200, "stop1": 8, "resume": 1192}[phase]
            and value["exam_evidence_gradient"] is (role == "C100")
            and value["initial_shared_tensor_sha256"] == builder.SHARED_INITIAL_SHA
            and value["original_worker_sha256"] == builder.ORIGINAL_WORKER_SHA
            and value["routing_sha256"] == builder.ROUTING_SHA
            and value["train_sha256"] == builder.NUMERICAL_SHA["rsna/competitive/predictive/train.py"]
            and value["density_sha256"] == builder.NUMERICAL_SHA["rsna/competitive/predictive/density.py"],
            "Exact matched worker terminal required")
    require(all(digest(value[k]) for k in ("initial_model_sha256", "initial_encoder_sha256",
                "initial_head_sha256", "final_encoder_sha256", "final_head_sha256",
                "initial_encoder_parameters_sha256", "initial_head_parameters_sha256",
                "final_encoder_parameters_sha256", "final_head_parameters_sha256")),
            "Actual fresh initialization and final state digests required")
    require(value["encoder_parameters_changed"] is (phase != "stop1")
            and value["head_parameters_changed"] is True
            and (value["initial_encoder_parameters_sha256"] != value["final_encoder_parameters_sha256"])
            is value["encoder_parameters_changed"]
            and value["initial_head_parameters_sha256"] != value["final_head_parameters_sha256"],
            "Real parameter updates were not demonstrated")
    require(all(math.isfinite(x) and x >= 0 for x in value["training_only_numerical_ranges"].values())
            and math.isfinite(value["optimizer_seconds"]) and value["optimizer_seconds"] > 0
            and 0 < value["peak_GPU_bytes"] <= torch.cuda.get_device_properties(0).total_memory,
            "Nonfinite numerical telemetry or invalid allocated VRAM")


def run_phases(root, builder, capacity_check, *, run=subprocess.run, clock=time.monotonic,
               max_seconds=MAX_SECONDS):
    require(0 < max_seconds <= MAX_SECONDS, "Remaining bounded engineering allocation required")
    began = clock()
    terminals = []
    for role, phase in PHASES:
        capacity_check()
        require(not (root / "engineering" / role / phase).exists(), "Existing phase is not retry authority")
        remaining = max_seconds - (clock() - began)
        require(remaining > 0, "Matched engineering allocation exhausted")
        run([sys.executable, "-B", str(root / "code/worker.py"), role, phase], check=True,
            timeout=remaining, env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        capacity_check()
        terminal = json.loads((root / "engineering" / role / phase / "terminal.json").read_bytes())
        validate_terminal(terminal, role, phase, builder)
        terminals.append(terminal)
        require(all(t["initial_model_sha256"] == terminals[0]["initial_model_sha256"]
                    for t in terminals), "Fresh role/phase initialization differs")
    require(clock() - began <= max_seconds, "Matched worker wall time exceeded")
    return terminals


def main():
    import worker_builder as builder

    began = time.monotonic()
    os.umask(0o077)
    require(builder.ROOT == ROOT and os.uname().nodename == "evc27"
            and os.environ.get("SLURM_JOB_GPUS") and os.environ["SLURM_CPUS_PER_TASK"] == "2"
            and torch.cuda.device_count() == 1 and "H100" in torch.cuda.get_device_name(0),
            "Exact one-H100/two-CPU evc27 allocation required")
    topology(subprocess.check_output(["scontrol", "show", "node", "evc27", "--oneliner"], text=True))
    binding_bytes = (ROOT / "binding.private.json").read_bytes()
    binding = json.loads(binding_bytes)
    prepared = json.loads((ROOT / "prepared.json").read_bytes())
    require(prepared["status"] == "actual_matched_generative_routing_CPU_admitted"
            and prepared["binding_sha256"] == sha(binding_bytes)
            and prepared["source_git"] == binding["source_git"], "Actual separate CPU admission required")
    pilot, decision, funding = [read_reference(binding[n]) for n in
                                ("density100_pilot", "engineering_decision", "funding")]
    validate_prerequisites(binding, pilot, decision, funding)
    pilot_completed(subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", "981620",
                    "--format=JobIDRaw,State,ExitCode,NodeList"], text=True))
    require(datetime.now(timezone.utc).timestamp() + MAX_SECONDS
            <= datetime.fromisoformat(decision["finish_by_UTC"]).timestamp(), "Too late for the admitted pilot")
    for name, expected in binding["code_sha256"].items():
        path = ROOT / "code" / name
        require(path.resolve().is_relative_to(ROOT / "code") and sha(path.read_bytes()) == expected,
                "Staged matched code changed")
    require(str(torch.__version__) == binding["runtime_versions"]["torch"] == "2.14.0+cu130"
            and torch.version.git_version == "08187d9e0fba026dc8217405802ab5381dc88d90",
            "Original numerical runtime required")
    for name, proof in binding["runtime_files"].items():
        path = PORTABLE / name if name != "resolved_python" else Path(proof["path"])
        st = path.stat()
        require(signature(st) == proof["stat"], "Original runtime reference changed")
        if name == "resolved_python":
            require(path == Path(sys.executable).resolve() and st.st_uid == proof["owner_uid"]
                    and st.st_mode & 0o777 == proof["mode"] and not st.st_mode & 0o022,
                    "Wrong or writable interpreter")
        if "sha256" in proof:
            require(sha(path.read_bytes()) == proof["sha256"], "Runtime receipt changed")
    require(not (ROOT / "GPU-run-start.json").exists(), "Original run already attempted")
    quota = shared_capacity(subprocess.check_output(
        ["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True), funding,
        paid_log_count(funding.get("original_dispatched_jobs", {})))
    storage(ROOT, funding)
    once(ROOT / "GPU-run-start.json", dict(GPU_job=os.environ["SLURM_JOB_ID"],
         source_git=binding["source_git"], binding_sha256=sha(binding_bytes), branch=BRANCH,
         phases=[list(x) for x in PHASES], maximum_seconds=MAX_SECONDS))
    terminals = run_phases(ROOT, builder, lambda: storage(ROOT, funding),
                          max_seconds=MAX_SECONDS - (time.monotonic() - began))
    accepted = []
    for role in ("C100", "G100"):
        root = ROOT / "engineering" / role
        state = torch.load(root / "full/final.pt", map_location="cpu", weights_only=False)
        history = check_history(state, binding, role)
        terminal = next(t for t in terminals if t["role"] == role and t["phase"] == "full")
        require(all(terminal[k] == v for k, v in state["binding"]["matched_worker"].items()),
                "Checkpoint and terminal fresh initialization bindings differ")
        prediction = (root / "full/prediction.private.csv").read_bytes()
        values = np.loadtxt(root / "full/prediction.private.csv", delimiter=",")
        require(values.shape == (4, 12) and np.isfinite(values).all()
                and ((values >= 0) & (values <= 1)).all(), "Engineering prediction output invalid")
        if role == "G100":
            resumed = torch.load(root / "resume/final.pt", map_location="cpu", weights_only=False)
            equal(state, resumed)
            require(prediction == (root / "resume/prediction.private.csv").read_bytes(),
                    "G100 full/resume prediction bytes differ")
            del resumed
        accepted.append(dict(role=role, real_optimizer_history=history,
                             private_engineering_prediction_sha256=sha(prediction)))
        del state
    elapsed = time.monotonic() - began
    require(elapsed <= MAX_SECONDS, "Matched end-to-end engineering wall time exceeded")
    out = dict(status="actual_matched_C100_G100_through_peak_engineering_complete",
               GPU_job=os.environ["SLURM_JOB_ID"], source_git=binding["source_git"],
               binding_sha256=sha(binding_bytes), original_numerical_source_git=SCIENCE,
               density100_pilot_acceptance_sha256=binding["density100_pilot"]["sha256"],
               engineering_decision_sha256=binding["engineering_decision"]["sha256"],
               funding_sha256=binding["funding"]["sha256"], branch=BRANCH,
               actual_optimizer_updates=600, elapsed_seconds=elapsed, fresh_initialization_equal=True,
               G100_full_stop_resume_complete_state_and_prediction_bytes_equal=True,
               roles=accepted, terminals=terminals, storage=storage(ROOT, funding), shared=quota,
               clinical_or_held_out_accuracy_claim=False, automatic_full_training_promotion=False,
               primary_B_C_C0_C100_pipeline_unchanged=True)
    once(ROOT / "GPU-accepted.json", out)
    storage(ROOT, funding)
    print(json.dumps(out, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
