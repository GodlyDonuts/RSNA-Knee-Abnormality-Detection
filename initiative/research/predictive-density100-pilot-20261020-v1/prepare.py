"""One owning-node CPU admission; reuse original data/runtime without a copy."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def signature(path, *, require_owned=True):
    st = path.stat()
    assert path.is_file() and not path.is_symlink()
    if require_owned:
        assert st.st_uid == os.getuid()
    else:
        assert not st.st_mode & 0o022 and st.st_mode & 0o111
    return [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def main(v):
    assert os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and sys.dont_write_bytecode
    os.umask(0o077)
    ns = {"__name__": "frozen_C100_worker_builder"}
    builder = v["files"]["worker_builder.py"]
    assert hashlib.sha256(builder.encode()).hexdigest() == v["sha256"]["worker_builder.py"]
    exec(compile(builder, "frozen_C100_worker_builder", "exec"), ns)
    D, old, portable = (ns[n] for n in ("ROOT", "ORIGINAL", "PORTABLE"))
    sha = ns["sha"]
    assert not D.exists(), "Existing admission is not retry authority"
    ns["admit_time"](datetime.now(timezone.utc))
    funding = json.loads(v["placement_decision_body"])
    assert sha(v["placement_decision_body"].encode()) == ns["PLACEMENT_SHA"]
    assert funding["destination_existing_unpaid_bytes"] == 205676085248
    assert funding["destination_extra_context_bytes"] == 96 * 2**30
    assert funding["destination_extra_fit_bytes"] == 12 * 2**30
    assert funding["destination_extra_control_bytes"] == 64 * 2**20
    capacity = ns["node_capacity"](shutil.disk_usage("/tmp").free)
    quota = ns["quota_capacity"](subprocess.check_output(
        ["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True,
    ))
    raw_original = (old / "GPU-accepted.json").read_bytes()
    assert sha(raw_original) == ns["ENGINEERING_SHA"]
    assert json.loads(raw_original) == json.loads(v["original_engineering_body"])
    assert sha(v["measurement_body"].encode()) == ns["MEASUREMENT_SHA"]
    measurement = json.loads(v["measurement_body"])
    assert measurement["CPU_job"] == "981494" and measurement["checkpoint_step"] == 24
    assert measurement["status"] == "actual_retained_C24_CPU_objective_balance"
    original_checkpoint_stat = signature(old / "engineering/C/full/final.pt")
    assert original_checkpoint_stat == measurement["checkpoint_stat_before_and_after"]
    predicted_checkpoint_bound = 3 * original_checkpoint_stat[1] + 64 * 2**20
    assert predicted_checkpoint_bound < ns["CHECKPOINT_CAP"]
    decision = json.loads(v["scientific_decision_body"])
    assert sha(v["scientific_decision_body"].encode()) == v["scientific_decision_sha256"]
    assert decision["intervention"] == {
        "architecture": "C", "density_weight": 100.0, "original_density_weight": 0.1,
        "other_scientific_changes": [],
    }
    assert decision["original_numerical_source_git"] == ns["SCIENCE"]
    assert decision["measurement_receipt_sha256"] == ns["MEASUREMENT_SHA"]
    assert decision["original_engineering_receipt_sha256"] == ns["ENGINEERING_SHA"]
    assert decision["automatic_full_training_promotion"] is False
    assert decision["primary_B_C_C0_changed"] is False
    assert decision["maximum_GPU_minutes"] == 30
    assert decision["new_node_checkpoint_cap_bytes"] == ns["CHECKPOINT_CAP"]
    assert decision["new_node_control_and_logs_cap_bytes"] == ns["CONTROL_CAP"]
    original_c = next(a for a in json.loads(raw_original)["arms"] if a["arm"] == "C")
    h = original_c["real_optimizer_history"]
    assert decision["engineering_warning_thresholds"] == {
        "last_six_exam_loss_mean_above": 2 * sum(r["exam_loss"] for r in h[-6:]) / 6,
        "maximum_exam_loss_above": 2 * max(r["exam_loss"] for r in h),
        "maximum_absolute_log_bayes_factor_above": 10 * original_c["terminal"][
            "training_only_numerical_ranges"]["maximum_absolute_log_bayes_factor"],
    }
    old_binding_bytes = (old / "binding.private.json").read_bytes()
    binding = json.loads(old_binding_bytes)
    assert binding["source_git"] == ns["SCIENCE"]
    assert sha(old_binding_bytes) == measurement["binding_sha256"]
    original_worker = (old / "code/worker.py").read_text()
    assert v["files"]["worker.py"] == ns["build"](original_worker)
    numerical = {n: h for n, h in binding["code_sha256"].items() if n.startswith("rsna/")}
    assert {n for n in v["files"] if n.startswith("rsna/")} == set(numerical)
    for name, expected in numerical.items():
        assert sha(v["files"][name].encode()) == expected
        assert (old / "code" / name).read_bytes() == v["files"][name].encode()
    for key in ("targets16", "context_index"):
        assert sha(Path(binding[key + "_private_path"]).read_bytes()) == binding[key + "_sha256"]
    index = json.loads(Path(binding["context_index_private_path"]).read_bytes())
    assert len(index) == 16
    for row in index:
        assert signature(Path(row["path"])) == row["proof"]["stat"]
    parent = Path(binding["parent_weight_private_path"])
    parent_stat = signature(parent)
    assert {"inode": parent_stat[0], "bytes": parent_stat[1], "mtime_ns": parent_stat[2]} \
        == binding["parent_weight_stat"]
    assert sha((portable / "assets/parent_source.private.py").read_bytes()) \
        == binding["parent_source_sha256"]
    runtime_files = {}
    for name in ("runtime-import.private.json", "replica-accepted.json"):
        p = portable / name
        runtime_files[name] = {"sha256": sha(p.read_bytes()), "stat": signature(p)}
    runtime = json.loads((portable / "runtime-import.private.json").read_bytes())
    assert runtime["returncode"] == 0
    versions = json.loads(runtime["stdout"])
    assert versions["torch"] == "2.14.0+cu130" and versions["new_predictive_graph_imported"] is True
    python = (portable / "runtime/bin/python").resolve()
    python_stat = python.stat()
    runtime_files["resolved_python"] = {
        "path": str(python), "stat": signature(python, require_owned=False),
        "owner_uid": python_stat.st_uid, "mode": python_stat.st_mode & 0o777,
    }
    binding.update(
        source_git=v["source_git"], original_numerical_source_git=ns["SCIENCE"],
        original_binding_sha256=sha(old_binding_bytes),
        original_engineering_receipt_sha256=ns["ENGINEERING_SHA"],
        objective_measurement_sha256=ns["MEASUREMENT_SHA"],
        placement_funding_decision_sha256=ns["PLACEMENT_SHA"],
        scientific_decision_sha256=v["scientific_decision_sha256"],
        branch="C100", code_sha256=v["sha256"], original_module_sha256=numerical,
        runtime_files=runtime_files, runtime_versions=versions,
        funding={"node": capacity, "shared": quota, "other_node_future_bytes": ns["OTHER27"],
                 "new_checkpoint_cap_bytes": ns["CHECKPOINT_CAP"],
                 "new_control_and_log_cap_bytes": ns["CONTROL_CAP"]},
        recipe={"arm": "C", "density_weight": 100.0, "epochs": 12, "actual_updates": 24,
                "schedule_updates": 2916, "effective_batch": 8, "head_only_updates": 3,
                "encoder_lr": 3e-5, "head_lr": 3e-5, "side": 320, "context_cap": 24,
                "seed": 20261020},
    )
    assert sum(len(s.encode()) for s in v["files"].values()) < 2 * 2**20
    D.mkdir(mode=0o700)
    for name, source in v["files"].items():
        p = D / "code" / name
        assert p.resolve().is_relative_to((D / "code").resolve())
        assert sha(source.encode()) == v["sha256"][name]
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with p.open("x") as handle:
            handle.write(source)
    ns["once"](D / "binding.private.json", binding)
    ns["once"](D / "original-engineering.json", json.loads(raw_original))
    ns["once"](D / "scientific-decision.json", decision)
    out = {
        "status": "actual_C100_original_contexts_CPU_admitted",
        "CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
        "original_science_git": ns["SCIENCE"], "original_engineering_receipt_sha256": ns["ENGINEERING_SHA"],
        "objective_measurement_sha256": ns["MEASUREMENT_SHA"],
        "binding_sha256": sha((D / "binding.private.json").read_bytes()),
        "node_capacity": capacity, "shared_capacity": quota,
        "context_files_reused_without_copy": 16, "new_source_downloads": 0,
        "three_equal_graph_checkpoint_size_bound_bytes": predicted_checkpoint_bound,
        "new_shared_files": 0, "GPU_dispatched": False,
    }
    ns["once"](D / "prepared.json", out)
    assert sum(p.stat().st_size for p in D.rglob("*") if p.is_file()) < ns["CONTROL_CAP"]
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main(json.load(sys.stdin))
