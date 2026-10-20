"""Admit and submit the single matched contrast pilot; never retry uncertainty."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCIENCE = "39e315c65546aa5205518713bc27349ec63e7538"
ROOT = "/tmp/rsna-predictive-contrast-learning-20261020-v1"
PORTABLE = "/tmp/rsna-predictive-portable-recovery-20261020-v1"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def once(path, value):
    with path.open("x") as out:
        out.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def script():
    return f'''#!/bin/bash
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodelist=evc27
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --deadline=2026-10-21T17:15:00
#SBATCH --chdir={ROOT}
#SBATCH --job-name=rsna-predictive-contrast-pair
#SBATCH --output={ROOT}/GPU-%j.out
#SBATCH --error={ROOT}/GPU-%j.err
set -euo pipefail
umask 077
export TZ=UTC PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec {PORTABLE}/runtime/bin/python -B -c 'import sys;sys.path[:0]=["{ROOT}/code","{PORTABLE}/overlay"];exec(compile(open("{ROOT}/code/learning_driver.py").read(),"contrast_driver","exec"))'
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    assert re.fullmatch("[0-9a-f]{40}", args.source_git)
    assert datetime.now(timezone.utc) < datetime(2026, 10, 21, 16, 15, tzinfo=timezone.utc)
    os.umask(0o077)

    def frozen(pin, path):
        return subprocess.check_output(["git", "show", pin + ":" + path], cwd=REPO)

    own = {}
    for name in ("learning_once.py", "learning_prepare.py", "learning_controls.py", "learning_driver.py",
                 "worker_builder.py", "actual-private-transfer.json", "actual-private-transfer-terminal.json",
                 "learning-decision.json"):
        path = HERE / name
        body = frozen(args.source_git, path.relative_to(REPO).as_posix())
        assert body == path.read_bytes()
        own[name] = body.decode()
    decision = json.loads(own["learning-decision.json"])
    assert decision["actual_update_budget"] == 400 and decision["maximum_GPU_jobs"] == 1
    builder = {"__name__": "frozen_contrast_builder"}
    exec(compile(own["worker_builder.py"], "contrast_builder", "exec"), builder)
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", SCIENCE, "--",
                                     "rsna/competitive/predictive"], text=True, cwd=REPO).splitlines()
    files = {name: frozen(SCIENCE, name).decode() for name in names if name.endswith(".py")}
    files.update({"rsna/__init__.py": "", "rsna/competitive/__init__.py": ""})
    worker = frozen(SCIENCE, "initiative/research/predictive-stable-warmup-20261020-v1/worker.py.txt").decode()
    files.update({name: own[name] for name in ("learning_controls.py", "learning_driver.py", "worker_builder.py")})
    files["worker.py"] = builder["build"](worker)
    payload = {"source_git": args.source_git, "files": files,
               "sha256": {name: sha(body.encode()) for name, body in files.items()},
               "transfer_receipt": json.loads(own["actual-private-transfer.json"])}
    directory = REPO / "artifacts" / (HERE.name + "-learning")
    directory.mkdir(mode=0o700, exist_ok=False)
    once(directory / "intent.json", {"source_git": args.source_git, "code_sha256": payload["sha256"],
         "maximum_GPU_jobs": 1, "actual_update_budget": 400, "automatic_retry": False,
         "decision_sha256": sha(own["learning-decision.json"].encode())})
    spec = importlib.util.spec_from_file_location("contrast_transport", REPO /
        "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    ssh = ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton"]
    argv = ["srun", "--account=skattel", "--partition=normal", "--qos=normal", "--nodelist=evc27",
            "--ntasks=1", "--cpus-per-task=1", "--gres=none", "--mem=1G", "--time=00:05:00",
            "--immediate=60", "--job-name=rsna-contrast-pair-admit", "env", "CUDA_VISIBLE_DEVICES=",
            "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
            "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python", "-B", "-c", transport.COMMAND]
    result = subprocess.run([*ssh, shlex.join(argv)], input=transport.envelope(own["learning_prepare.py"], payload),
                            text=True, capture_output=True, timeout=420, check=False)
    once(directory / "CPU-transport.private.json", {"returncode": result.returncode,
         "stdout": result.stdout, "stderr": result.stderr})
    assert result.returncode == 0, "Reconcile original CPU admission; do not retry"
    actual = json.loads(result.stdout)
    assert actual["status"] == "actual_contrast_matched_learning_CPU_admitted"
    assert actual["source_git"] == args.source_git and actual["actual_update_budget"] == 400
    once(HERE / "actual-learning-prepare.json", actual)
    assert datetime.now(timezone.utc) < datetime(2026, 10, 21, 16, 15, tzinfo=timezone.utc)
    body = script()
    once(directory / "GPU-intent.json", {"source_git": args.source_git, "script_sha256": sha(body.encode()),
         "binding_sha256": actual["binding_sha256"], "automatic_retry": False})
    result = subprocess.run([*ssh, "TZ=UTC sbatch --parsable"], input=body, text=True,
                            capture_output=True, timeout=60, check=False)
    once(directory / "GPU-transport.private.json", {"returncode": result.returncode,
         "stdout": result.stdout, "stderr": result.stderr})
    assert result.returncode == 0, "Reconcile original GPU dispatch; do not retry"
    assert re.fullmatch(r"\d+(?:;[^\s]+)?\s*", result.stdout)
    actual = {"status": "actual_single_matched_contrast_GPU_submitted", "GPU_job": result.stdout.strip().split(";")[0],
        "source_git": args.source_git, "CPU_admission_job": actual["CPU_job"],
        "binding_sha256": actual["binding_sha256"], "script_sha256": sha(body.encode()),
        "node": "evc27", "GPUs": 1, "CPUs": 2, "runtime_cap_seconds": 3600,
        "roles": ["control", "normalized"], "updates_per_role": 200, "actual_update_budget": 400,
        "training_studies": 12, "engineering_prediction_studies": 4, "metadata_width": 10,
        "automatic_full_promotion": False, "new_shared_files": 0,
        "submitted_at_UTC": datetime.now(timezone.utc).isoformat()}
    once(HERE / "actual-learning-submit.json", actual)
    print(json.dumps(actual, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
