"""One read-only CPU objective measurement on retained fitting-engineering inputs."""

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "initiative/research/predictive-stable-warmup-20261020-v1"
ORIGINAL_SHA = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"


def digest(body):
    return hashlib.sha256(body).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    os.umask(0o077)
    if len(args.source_git) != 40 or any(c not in "0123456789abcdef" for c in args.source_git):
        raise ValueError("Exact committed source required")
    path = HERE / "diagnostic.py"
    body = subprocess.check_output(
        ["git", "show", args.source_git + ":" + str(path.relative_to(ROOT))], cwd=ROOT,
    )
    if body != path.read_bytes():
        raise ValueError("Diagnostic differs from its immutable source")
    original = (ORIGINAL / "actual-accepted.json").read_bytes()
    if digest(original) != ORIGINAL_SHA:
        raise ValueError("Original actual engineering receipt differs")
    payload = {
        "source_git": args.source_git,
        "diagnostic_sha256": digest(body),
        "original_engineering_receipt_sha256": ORIGINAL_SHA,
        "original_engineering_receipt": json.loads(original),
        "fitting_indices": [0, 1],
    }
    transport_path = ROOT / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    spec = importlib.util.spec_from_file_location("objective_stdin_transport", transport_path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    argv = [
        "srun", "--account=skattel", "--partition=normal", "--qos=normal",
        "--nodelist=evc27", "--ntasks=1", "--cpus-per-task=2", "--mem=12G",
        "--gres=none", "--time=00:30:00", "--immediate=60",
        "--job-name=rsna-predictive-objective-CPU",
        "env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2",
        "/tmp/rsna-predictive-portable-recovery-20261020-v1/runtime/bin/python",
        "-B", "-c", transport.COMMAND,
    ]
    directory = ROOT / "artifacts" / HERE.name
    directory.mkdir(mode=0o700, exist_ok=False)
    intent = {
        "status": "one_readonly_fitting_objective_measurement_intent",
        "source_git": args.source_git, "diagnostic_sha256": digest(body),
        "original_engineering_receipt_sha256": ORIGINAL_SHA,
        "fitting_indices": [0, 1], "requested_CPUs": 2, "requested_GPU": 0,
        "requested_RAM_GiB": 12, "maximum_seconds": 1800,
        "new_remote_storage_bytes": 0, "new_shared_files": 0,
        "model_or_recipe_mutation": False, "automatic_retry": False,
    }
    (directory / "intent.json").write_text(json.dumps(intent, indent=2) + "\n")
    with (directory / "stdout.private").open("x") as stdout, (directory / "stderr.private").open("x") as stderr:
        result = subprocess.run(
            ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton", shlex.join(argv)],
            input=transport.envelope(body.decode(), payload), text=True,
            stdout=stdout, stderr=stderr, timeout=1920, check=False,
        )
    (directory / "result.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    if result.returncode:
        raise RuntimeError("Original measurement did not complete; reconcile without retry")
    receipt = json.loads((directory / "stdout.private").read_text())
    if (
        receipt["status"] != "actual_retained_C24_CPU_objective_balance"
        or receipt["source_git"] != args.source_git
        or receipt["diagnostic_sha256"] != digest(body)
        or receipt["original_engineering_receipt_sha256"] != ORIGINAL_SHA
        or receipt["remote_files_written"] != 0
        or receipt["optimizer_steps"] != 0
    ):
        raise ValueError("Actual aggregate measurement binding differs")
    with (HERE / "actual-result.json").open("x") as output:
        output.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "CPU_job": receipt["CPU_job"]}), flush=True)


if __name__ == "__main__":
    main()
