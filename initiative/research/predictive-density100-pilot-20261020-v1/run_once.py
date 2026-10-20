"""Freeze and admit once, then submit one C100 pilot; uncertain outcomes are not retried."""

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("C100_worker_contract", HERE / "worker.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def frozen(pin, path):
    return subprocess.check_output(["git", "show", pin + ":" + path], cwd=REPO)


def script():
    return f"""#!/bin/bash
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodelist=evc27
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --deadline=2026-10-21T17:15:00
#SBATCH --chdir={C.ROOT}
#SBATCH --job-name=rsna-predictive-C100-pilot
#SBATCH --output={C.ROOT}/GPU-%j.out
#SBATCH --error={C.ROOT}/GPU-%j.err
set -euo pipefail
umask 077
export TZ=UTC PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec {C.PORTABLE}/runtime/bin/python -B -c 'import sys;sys.path[:0]=["{C.ROOT}/code","{C.PORTABLE}/overlay"];exec(compile(open("{C.ROOT}/code/driver.py").read(),"C100_driver","exec"))'
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    assert len(args.source_git) == 40 and set(args.source_git) <= set("0123456789abcdef")
    os.umask(0o077)
    C.admit_time(datetime.now(timezone.utc))
    own = {}
    for name in ("worker.py", "prepare.py", "driver.py", "run_once.py", "README.md", "decision.json"):
        body = frozen(args.source_git, (HERE / name).relative_to(REPO).as_posix())
        assert body == (HERE / name).read_bytes(), "Exact committed controls required"
        own[name] = body.decode()
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", C.SCIENCE, "--", "rsna/competitive/predictive"],
        text=True, cwd=REPO,
    ).splitlines()
    files = {n: frozen(C.SCIENCE, n).decode() for n in names if n.endswith(".py")}
    files.update({"rsna/__init__.py": "", "rsna/competitive/__init__.py": ""})
    worker = frozen(C.SCIENCE,
                    "initiative/research/predictive-stable-warmup-20261020-v1/worker.py.txt").decode()
    files.update({"worker.py": C.build(worker), "worker_builder.py": own["worker.py"],
                  "driver.py": own["driver.py"]})
    def read(path):
        return (REPO / "initiative/research" / path).read_text()

    payload = {
        "source_git": args.source_git, "files": files,
        "sha256": {n: C.sha(body.encode()) for n, body in files.items()},
        "scientific_decision_sha256": C.sha(own["decision.json"].encode()),
        "scientific_decision_body": own["decision.json"],
        "scientific_README_sha256": C.sha(own["README.md"].encode()),
        "placement_decision_body": read("predictive-http-owner-handoff-20261020-v1/decision.accepted.json"),
        "original_engineering_body": read("predictive-stable-warmup-20261020-v1/actual-accepted.json"),
        "measurement_body": read("predictive-objective-balance-20261020-v1/actual-result.json"),
    }
    assert C.sha(payload["placement_decision_body"].encode()) == C.PLACEMENT_SHA
    assert C.sha(payload["original_engineering_body"].encode()) == C.ENGINEERING_SHA
    assert C.sha(payload["measurement_body"].encode()) == C.MEASUREMENT_SHA
    directory = REPO / "artifacts" / HERE.name
    directory.mkdir(mode=0o700, exist_ok=False)
    C.once(directory / "intent.json", {
        "source_git": args.source_git, "maximum_GPU_jobs": 1, "branch": "C100",
        "density_weight": 100, "original_science_git": C.SCIENCE,
        "original_worker_sha256": C.WORKER_SHA,
        "derived_worker_sha256": payload["sha256"]["worker.py"],
        "original_engineering_receipt_sha256": C.ENGINEERING_SHA,
        "objective_measurement_sha256": C.MEASUREMENT_SHA,
        "new_checkpoint_cap_bytes": C.CHECKPOINT_CAP, "new_control_log_cap_bytes": C.CONTROL_CAP,
        "new_shared_files": 0, "automatic_retry": False,
    })
    transport_path = REPO / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    spec = importlib.util.spec_from_file_location("C100_transport", transport_path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    ssh = ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton"]
    command = [
        "srun", "--account=skattel", "--partition=normal", "--qos=normal",
        "--nodelist=evc27", "--ntasks=1", "--cpus-per-task=1", "--gres=none",
        "--mem=1G", "--time=00:05:00", "--immediate=60", "--job-name=rsna-C100-admit",
        "env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python",
        "-B", "-c", transport.COMMAND,
    ]
    result = subprocess.run(
        [*ssh, shlex.join(command)], input=transport.envelope(own["prepare.py"], payload),
        capture_output=True, text=True, timeout=420, check=False,
    )
    C.once(directory / "prepare-transport.private.json",
           {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    assert result.returncode == 0, "Reconcile original admission read-only; never retry"
    admission = json.loads(result.stdout)
    assert admission["status"] == "actual_C100_original_contexts_CPU_admitted"
    assert admission["source_git"] == args.source_git
    C.once(HERE / "actual-prepare.json", admission)
    C.admit_time(datetime.now(timezone.utc))
    body = script()
    C.once(directory / "GPU-intent.json", {
        "script_sha256": C.sha(body.encode()), "source_git": args.source_git,
        "binding_sha256": admission["binding_sha256"], "automatic_retry": False,
        "scheduler_finish_deadline_UTC": C.FINISH_BY.isoformat(),
    })
    result = subprocess.run([*ssh, "TZ=UTC sbatch --parsable"], input=body,
                            capture_output=True, text=True, timeout=60, check=False)
    C.once(directory / "GPU-transport.private.json",
           {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    assert result.returncode == 0, "Uncertain GPU submission: reconcile read-only, never repeat"
    job = result.stdout.strip().split(";")[0]
    assert job.isdigit()
    receipt = {
        "status": "actual_single_C100_pilot_GPU_submitted", "GPU_job": job,
        "source_git": args.source_git, "CPU_admission_job": admission["CPU_job"],
        "node": "evc27", "branch": "C100", "training_studies": 12, "engineering_studies": 4,
        "updates": 24, "density_weight": 100, "runtime_cap_seconds": 1800,
        "binding_sha256": admission["binding_sha256"], "script_sha256": C.sha(body.encode()),
        "scheduler_finish_deadline_UTC": C.FINISH_BY.isoformat(),
        "automatic_full_training_promotion": False, "new_shared_files": 0,
    }
    C.once(HERE / "actual-submit.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
