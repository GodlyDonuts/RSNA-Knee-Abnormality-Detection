"""Explicit single-use, one-CPU stdin runner; importing does not execute remotely."""

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
PRIOR = HERE.parent / "predictive-objective-balance-20261020-v1"
ENGINEERING = HERE.parent / "predictive-stable-warmup-20261020-v1"
LOADER_PIN = "7b277c4e75a3ee25cdf5247b3dfad93eb5f9eefa"
LOADER_SHA = "36a87becf4a79aeda7f6c491c82ff58c69b779902348f231648b5f00bec3aac0"
PRIOR_SHA = "6f10fc5a21e8149731e1600088e601b97c16be42b0a9e944baa05c67f6920e2a"
ENGINEERING_SHA = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"
STATUS = "actual_frozen_C24_training_only_evidence_attribution"


def digest(body):
    return hashlib.sha256(body).hexdigest()


def git_file(pin, path):
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("Exact committed source required")
    return subprocess.check_output(["git", "show", pin + ":" + str(path.relative_to(ROOT))], cwd=ROOT)


def prepare(source_git):
    bodies = {name: git_file(source_git, HERE / name) for name in ("diagnostic.py", "run_once.py")}
    if any(raw != (HERE / name).read_bytes() for name, raw in bodies.items()):
        raise ValueError("Invoke only the exact frozen diagnostic and runner")
    loader = git_file(LOADER_PIN, PRIOR / "diagnostic.py")
    prior, engineering = (PRIOR / "actual-result.json").read_bytes(), (
        ENGINEERING / "actual-accepted.json").read_bytes()
    if (digest(loader), digest(prior), digest(engineering)) != (
        LOADER_SHA, PRIOR_SHA, ENGINEERING_SHA,
    ):
        raise ValueError("Original loader/actual fitting checkpoint evidence differs")
    payload = {
        "source_git": source_git, "diagnostic_sha256": digest(bodies["diagnostic.py"]),
        "original_loader_source": loader.decode(), "prior_result_text": prior.decode(),
        "original_engineering_receipt_sha256": ENGINEERING_SHA,
        "original_engineering_receipt": json.loads(engineering), "fitting_indices": [0, 1],
    }
    transport_path = ROOT / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    transport = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("attribution_stdin_transport", loader=None))
    exec(compile(git_file(LOADER_PIN, transport_path), "original_stdin_transport", "exec"),
         transport.__dict__)
    return bodies, payload, transport


def command(socket, transport_command):
    argv = [
        "srun", "--account=skattel", "--partition=normal", "--qos=normal",
        "--nodelist=evc27", "--ntasks=1", "--cpus-per-task=1", "--mem=12G",
        "--gres=none", "--time=00:30:00", "--immediate=60", "--chdir=/tmp",
        "--job-name=rsna-predictive-attribution-CPU",
        "env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "/tmp/rsna-predictive-portable-recovery-20261020-v1/runtime/bin/python",
        "-B", "-c", transport_command,
    ]
    return ["ssh", "-S", socket, "-o", "BatchMode=yes", "newton", shlex.join(argv)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    os.umask(0o077)
    bodies, payload, transport = prepare(args.source_git)
    directory = ROOT / "artifacts" / HERE.name
    directory.mkdir(mode=0o700, exist_ok=False)
    intent = {
        "status": "single_frozen_C24_training_attribution_intent",
        "source_git": args.source_git,
        "source_sha256": {name: digest(raw) for name, raw in bodies.items()},
        "original_loader_sha256": LOADER_SHA, "prior_actual_receipt_sha256": PRIOR_SHA,
        "original_engineering_receipt_sha256": ENGINEERING_SHA,
        "fitting_indices": [0, 1], "requested_CPUs": 1, "requested_GPU": 0,
        "requested_RAM_GiB": 12, "maximum_seconds": 1800,
        "new_remote_storage_bytes": 0, "new_shared_files": 0,
        "optimizer_steps": 0, "model_or_recipe_mutation": False,
        "covariance_or_temperature_sweep": False, "automatic_retry": False,
    }
    (directory / "intent.json").write_text(json.dumps(intent, indent=2) + "\n")
    try:
        with (directory / "stdout.private").open("x") as out, (
            directory / "stderr.private"
        ).open("x") as err:
            result = subprocess.run(
                command(args.socket, transport.COMMAND),
                input=transport.envelope(bodies["diagnostic.py"].decode(), payload),
                text=True, stdout=out, stderr=err, timeout=1920, check=False,
            )
    except BaseException as error:
        (directory / "transport-error.json").write_text(
            json.dumps({"error_type": type(error).__name__, "automatic_retry": False}) + "\n")
        raise
    (directory / "result.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    if result.returncode:
        raise RuntimeError("Single read-only measurement failed; retain intent, no retry")
    receipt = json.loads((directory / "stdout.private").read_text())
    if (
        receipt["status"] != STATUS or receipt["source_git"] != args.source_git
        or receipt["diagnostic_sha256"] != digest(bodies["diagnostic.py"])
        or receipt["previous_actual_receipt_sha256"] != PRIOR_SHA
        or receipt["original_engineering_receipt_sha256"] != ENGINEERING_SHA
        or receipt["remote_files_written"] != 0 or receipt["optimizer_steps"] != 0
        or receipt["requested_CPUs"] != 1 or receipt["gradients_computed"] is not False
        or receipt["checkpoint_model_state_unchanged"] is not True
        or [case["fitting_ordinal"] for case in receipt["cases"]] != [0, 1]
    ):
        raise ValueError("Actual frozen-checkpoint attribution binding differs")
    with (HERE / "actual-result.json").open("x") as out:
        out.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "CPU_job": receipt["CPU_job"]}), flush=True)


if __name__ == "__main__":
    main()
