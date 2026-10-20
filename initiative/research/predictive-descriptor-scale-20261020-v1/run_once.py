"""Single original twelve-case scale collection; uncertainty never permits retry."""

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODULE = ROOT / "rsna/competitive/predictive/descriptor_normalization.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    if len(args.source_git) != 40 or set(args.source_git) - set("0123456789abcdef"):
        raise ValueError("Exact committed source required")
    os.umask(0o077)
    sources = {}
    for path in (HERE / "collect.py", HERE / "run_once.py", MODULE):
        source = subprocess.check_output(
            ["git", "show", args.source_git + ":" + str(path.relative_to(ROOT))], cwd=ROOT,
        )
        if source != path.read_bytes():
            raise ValueError("Collection source differs from its immutable Git binding")
        sources[path.name] = source
    module_sha = hashlib.sha256(sources[MODULE.name]).hexdigest()
    payload = {
        "source_git": args.source_git, "normalizer_sha256": module_sha,
        "normalizer_source": sources[MODULE.name].decode(),
    }
    spec = importlib.util.spec_from_file_location(
        "descriptor_scale_stdin",
        ROOT / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py",
    )
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    argv = [
        "srun", "--account=skattel", "--partition=normal", "--qos=normal", "--nodelist=evc27",
        "--ntasks=1", "--cpus-per-task=1", "--gres=none", "--mem=4G", "--time=00:10:00",
        "--immediate=60", "--job-name=rsna-predictive-descriptor-scale",
        "env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "/tmp/rsna-predictive-portable-recovery-20261020-v1/runtime/bin/python",
        "-B", "-c", transport.COMMAND,
    ]
    journal = ROOT / "artifacts" / HERE.name
    journal.mkdir(mode=0o700, exist_ok=False)
    intent = {
        "status": "single_twelve_case_fitting_scale_collection_intent", **{
            k: payload[k] for k in ("source_git", "normalizer_sha256")},
        "collector_sha256": hashlib.sha256(sources["collect.py"]).hexdigest(),
        "CPU": 1, "GPU": 0, "maximum_seconds": 600,
        "new_node_bytes_cap": 2**20, "new_shared_bytes": 0, "new_shared_files": 0,
        "model_or_labels_changed": False, "automatic_retry": False,
    }
    (journal / "intent.json").write_text(json.dumps(intent, indent=2) + "\n")
    with (journal / "stdout.private").open("x") as out, (journal / "stderr.private").open("x") as err:
        result = subprocess.run(
            ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "newton", shlex.join(argv)],
            input=transport.envelope(sources["collect.py"].decode(), payload),
            text=True, stdout=out, stderr=err, timeout=720, check=False,
        )
    (journal / "returned.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    if result.returncode:
        raise RuntimeError("Original collection failed; reconcile it read-only before any successor")
    receipt = json.loads((journal / "stdout.private").read_text())
    if (
        receipt["status"] != "actual_twelve_fitting_case_descriptor_scale_engineering_only"
        or receipt["source_git"] != args.source_git or receipt["normalizer_code_sha256"] != module_sha
        or receipt["fitting_studies"] != 12 or receipt["excluded_engineering_studies"] != 4
        or receipt["optimizer_steps"] != 0 or receipt["GPU_dispatch_or_full_fit_authority"]
        or receipt["clinical_labels_models_images_loaded"]
        or not receipt["exact_private_artifact_readback_and_binding_passed"]
    ):
        raise ValueError("Actual scale receipt is incomplete or differs")
    with (HERE / "actual-result.json").open("x") as output:
        output.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in (
        "status", "CPU_job", "fitting_studies", "canonical_regions",
        "coordinate_SD_range", "floored_coordinates", "elapsed_seconds",
    )}), flush=True)


if __name__ == "__main__":
    main()
