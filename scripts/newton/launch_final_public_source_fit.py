"""Launch the unchanged, admitted final public-source trainer on owning-node inputs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.newton.probe_final_public_source_recovery import engineering_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_admission(root):
    cpu = json.loads((root / "input-accepted.json").read_text())
    source = json.loads((root / "source.json").read_text())
    cache = json.loads((root / "full-cache-transfer-verified.json").read_text())
    recovery = json.loads((root / "T4-recovery-accepted.json").read_text())["actual_recovery"]
    values = json.loads((root / "final-config.json").read_text())
    engineering_config(values)  # Validate full recipe; never use its reduced returned budget.
    if (
        cpu["status"] != "actual_final_public_source_CPU_roster_and_16MRI_inputs_verified"
        or cpu["source_git"] != "1bcd6c0e2e2a47d6894d5c545d6950e0c6f0d5b9"
        or (cpu["training_studies"], cpu["outer_studies"]) != (3403, 0)
        or sha(root / "final-config.json") != cpu["config_sha256"]
        or sha(root / "overlay.json") != cpu["overlay_sha256"]
        or cache["members"] != 37812
        or cache["bytes"] != 140626483351
        or cache["registry_sha256"]
        != "73d06c86900f03e8571fb68a8d341333c547ab984c7a4317387754949b8ab770"
        or cache["all_original_member_size_SHA_verified_during_transfer"] is not True
        or recovery["status"] != "actual_single_T4_final_public_source_recovery_accepted"
        or recovery["science_git"] != cpu["source_git"]
        or recovery["CPU_receipt_sha256"]
        != hashlib.sha256((json.dumps(cpu, sort_keys=True, indent=2) + "\n").encode()).hexdigest()
        or recovery["exact_complete_state_scaler_RNG_recovery"] is not True
        or recovery["amp_scaler"]["_growth_tracker"] != 4
    ):
        raise ValueError("Actual original final input/cache/numerical admission required")
    for name, digest in source["files"].items():
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or sha(path) != digest
        ):
            raise ValueError("Frozen scientific source differs")
    inp = json.loads((root / "trainer-inputs.json").read_text())
    inp["cache"] = str(root / "cache")
    return values, inp


def command(root, inp):
    args = [
        sys.executable,
        "-B",
        str(root / "code/scripts/newton/enforce_deterministic_runtime.py"),
        "--module",
        "rsna.competitive.train",
        "--config",
        str(root / "final-config.json"),
        "--device",
        "cuda",
        "--fitting-supervision",
        str(root / "overlay.json"),
        "--output",
        str(root / "production"),
    ]
    for key, value in inp.items():
        args.extend(("--" + key, value))
    return args


def run(root):
    if os.uname().nodename != "evc40" or os.environ.get("SLURM_CPUS_PER_TASK") != "2":
        raise ValueError("Exact one-H100 two-CPU evc40 placement required")
    _, inp = validate_admission(root)
    if (root / "production").exists() or (root / "production-run-start.json").exists():
        raise ValueError("Existing production requires reconciliation, not restart")
    import torch

    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise ValueError("One allocated H100 required")
    torch.ones(1, device="cuda").sum().item()
    with (root / "production-run-start.json").open("x") as f:
        json.dump(
            {
                "job_id": os.environ["SLURM_JOB_ID"],
                "science_git": "1bcd6c0e2e2a47d6894d5c545d6950e0c6f0d5b9",
                "training_studies": 3403,
                "epochs": 10,
                "updates": 4260,
                "GPU": torch.cuda.get_device_name(0),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "automatic_retry": False,
            },
            f,
        )
    with (
        (root / "production.stdout").open("x") as out,
        (root / "production.stderr").open("x") as err,
    ):
        result = subprocess.run(
            command(root, inp), cwd=root / "code", env=os.environ.copy(), stdout=out, stderr=err
        )
    with (root / "production-terminal.json").open("x") as f:
        json.dump({"job_id": os.environ["SLURM_JOB_ID"], "returncode": result.returncode}, f)
    if result.returncode:
        raise ValueError("Original production failed; preserve and reconcile")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args().root)
