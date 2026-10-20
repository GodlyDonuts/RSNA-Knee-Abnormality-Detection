"""Run one bounded read-only CPU measurement, preserving uncertain outcomes."""

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


def digest(body):
    return hashlib.sha256(body).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    os.umask(0o077)
    assert len(args.source_git) == 40 and all(c in "0123456789abcdef" for c in args.source_git)
    paths = [HERE / "benchmark.py", ROOT / "initiative/research/predictive-final-serving-20261020-v1/descriptor_cache.py"]
    bodies = []
    for path in paths:
        body = subprocess.check_output(["git", "show", args.source_git + ":" + str(path.relative_to(ROOT))], cwd=ROOT)
        assert body == path.read_bytes()
        bodies.append(body)
    prepared = json.loads((ROOT / "initiative/research/predictive-full-fitting-20261020-v2/actual-cpu-prepared.json").read_text())
    source_hashes = {name: sha for name, sha in prepared["code_sha256"].items() if name.startswith("rsna/")}
    payload = {
        "source_git": args.source_git,
        "scientific_modules": source_hashes,
        "benchmark_sha256": digest(bodies[0]),
        "descriptor_helper": bodies[1].decode(),
        "descriptor_helper_sha256": digest(bodies[1]),
    }
    transport_path = ROOT / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    spec = importlib.util.spec_from_file_location("stdin_transport", transport_path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    argv = [
        "srun", "--account=skattel", "--partition=normal", "--qos=normal",
        "--nodelist=evc33", "--ntasks=1", "--cpus-per-task=2", "--mem=8G",
        "--gres=none", "--time=00:10:00", "--immediate=60",
        "--job-name=rsna-predictive-realinput-CPU",
        "env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python",
        "-B", "-c", transport.COMMAND,
    ]
    directory = ROOT / "artifacts" / HERE.name
    directory.mkdir(mode=0o700, exist_ok=False)
    intent = {
        "status": "one_readonly_CPU_measurement_intent",
        "source_git": args.source_git,
        "benchmark_sha256": payload["benchmark_sha256"],
        "descriptor_helper_sha256": payload["descriptor_helper_sha256"],
        "requested_CPUs": 2, "requested_GPU": 0, "requested_RAM_GiB": 8,
        "maximum_seconds": 600, "remote_files_or_shared_files_written": 0,
        "new_data_or_models_or_scientific_changes": False,
    }
    (directory / "intent.json").write_text(json.dumps(intent, indent=2) + "\n")
    with (directory / "stdout.private").open("x") as stdout, (directory / "stderr.private").open("x") as stderr:
        result = subprocess.run(
            ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton", shlex.join(argv)],
            input=transport.envelope(bodies[0].decode(), payload), text=True,
            stdout=stdout, stderr=stderr, timeout=720, check=False,
        )
    (directory / "result.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    if result.returncode:
        raise RuntimeError("CPU measurement did not complete; preserve original private logs, no retry")
    receipt = json.loads((directory / "stdout.private").read_text())
    assert receipt["status"] == "actual_retained16_MRI_descriptor_once_CPU_comparison"
    assert receipt["source_git"] == args.source_git and receipt["studies"] == 16
    assert receipt["all_complete_contexts_bitwise_equal"]
    assert receipt["benchmark_sha256"] == payload["benchmark_sha256"]
    assert not receipt["remote_files_written"] and not receipt["source_downloads"]
    with (HERE / "actual-result.json").open("x") as output:
        output.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "CPU_job": receipt["CPU_job"],
                      "paired_render_ratio": receipt["ratio_of_paired_render_sums"]}), flush=True)


if __name__ == "__main__":
    main()
