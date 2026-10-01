"""After both source lanes succeed, admit and restore the full cohort without GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path("/home/sa305415/rsna-knee")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path, record):
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_terminal_lanes(output):
    rows = {
        parts[0]: parts[1:] for line in output.splitlines() if len(parts := line.split("|")) == 3
    }
    if any(rows.get(job) != ["COMPLETED", "0:0"] for job in ("866426", "866427")):
        raise ValueError("both_exact_source_lanes_must_succeed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control
    if (
        control.is_symlink()
        or not control.resolve().is_relative_to(BASE.resolve())
        or control.resolve() == BASE.resolve()
    ):
        raise ValueError("private_assembly_control_required")
    job = os.environ.get("SLURM_JOB_ID", "")
    if (
        not job.isdecimal()
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_JOB_GPUS")
    ):
        raise ValueError("one_cpu_zero_gpu_required")
    launch = json.loads((control / "launch.json").read_text())
    for name, record in launch["files"].items():
        path = control / name
        if (
            Path(name).name != name
            or path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or digest(path) != record["sha256"]
        ):
            raise ValueError("pinned_assembly_source_changed")
    if digest(Path(__file__)) != launch["files"]["runner.py"]["sha256"]:
        raise ValueError("pinned_assembly_runner_required")
    require_terminal_lanes(
        subprocess.check_output(
            [
                "sacct",
                "-j",
                "866426,866427",
                "--noheader",
                "--parsable2",
                "--format=JobID,State,ExitCode",
            ],
            text=True,
        )
    )
    if (control / "run-start.json").exists() or (control / "registry.json").exists():
        raise FileExistsError("existing_assembly_requires_readonly_review")
    quota = (
        subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
        .splitlines()[-1]
        .split()
    )
    used, soft, files, file_soft = [int(quota[i]) for i in (1, 2, 5, 6)]
    if (soft - used) * 1024 - launch[
        "remaining_shared_obligations_bytes"
    ] < 20 * 1024**3 or file_soft - files - 30 < 5000:
        raise ValueError("native_production_shared_reserve_required")
    payload = json.load(sys.stdin)
    if set(payload) != {"token"} or not payload["token"].startswith("hf_"):
        raise ValueError("private_credential_stdin_required")
    save_new(
        control / "run-start.json",
        {
            "job_id": job,
            "source_git": launch["source_git"],
            "cpus": 1,
            "gpus": 0,
            "node": os.uname().nodename,
            "launch_sha256": digest(control / "launch.json"),
            "automatic_retry": False,
        },
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "KAGGLE_API_TOKEN", "PYTHONPATH"):
        env.pop(key, None)
    runtime = None
    scratch = None
    phase = "complete_registry"
    try:
        subprocess.run(
            [
                "/usr/bin/python3.12",
                "-B",
                str(control / "assemble.py"),
                "registry",
                "--manifest",
                launch["manifest"],
                "--first-control",
                launch["first_control"],
                "--batch-control",
                launch["batch_control"],
                "--output",
                str(control / "registry.json"),
            ],
            env=env,
            check=True,
        )
        registry = json.loads((control / "registry.json").read_text())
        needed = sum(
            sum(v["bytes"] for v in s["members"].values()) for s in registry["shards"]
        ) + max(s["files"]["cache-side-384.tar"]["bytes"] for s in registry["shards"])
        if (
            shutil.disk_usage("/tmp").free < needed + 6322691335 + 20 * 1024**3
            or Path("/tmp").stat().st_dev == BASE.stat().st_dev
        ):
            raise ValueError("full_native_node_local_capacity_required")
        scratch = Path(tempfile.mkdtemp(prefix="rsna-native384-cohort-" + job + "-", dir="/tmp"))
        save_new(
            scratch / ".native384-cohort-owner.json",
            {
                "kind": "native384-cohort-owned-v1",
                "job_id": job,
                "uid": os.getuid(),
                "registry_sha256": digest(control / "registry.json"),
            },
        )
        save_new(
            control / "scratch.json",
            {
                "path": str(scratch),
                "job_id": job,
                "registry_sha256": digest(control / "registry.json"),
                "node": os.uname().nodename,
                "training_cache_retained": True,
            },
        )
        env["HF_HOME"] = str(scratch / "hf")
        phase = "accepted_environment"
        helper = Path(launch["environment_helper"])
        for p, row in launch["runtime_helpers"].items():
            if digest(Path(p)) != row["sha256"]:
                raise ValueError("pinned_environment_helper_changed")
        result = json.loads(
            subprocess.check_output(
                [
                    "/usr/bin/python3.12",
                    "-B",
                    str(helper),
                    "extract",
                    "--archive",
                    launch["environment_archive"],
                    "--sha256",
                    launch["environment_sha256"],
                    "--expected-files",
                    "40403",
                    "--expected-bytes",
                    "6322691335",
                    "--scratch-base",
                    "/tmp",
                    "--shared-python",
                    "/usr/bin/python3.12",
                    "--min-free-gib",
                    "20",
                ],
                cwd=helper.parent,
                env=env,
                text=True,
            )
        )
        runtime = Path(result["runtime_root"])
        phase = "full_native384_restoration"
        subprocess.run(
            [
                str(runtime / "environment/bin/python"),
                "-B",
                str(control / "assemble.py"),
                "restore",
                "--registry",
                str(control / "registry.json"),
                "--registry-sha256",
                digest(control / "registry.json"),
                "--output",
                str(scratch / "restored"),
            ],
            env=env,
            input=json.dumps(payload),
            text=True,
            check=True,
        )
        accepted = json.loads((scratch / "restored/accepted.json").read_text())
        if (
            accepted.get("status") != "full_native384_cohort_bytes_metadata_restored"
            or accepted.get("job_id") != job
            or accepted.get("node") != os.uname().nodename
            or accepted.get("members") != 37812
        ):
            raise ValueError("actual_full_native_restore_acceptance_required")
        save_new(
            control / "accepted.json",
            {
                **accepted,
                "registry_file_sha256": digest(control / "registry.json"),
                "source_git": launch["source_git"],
                "owned_root": str(scratch),
                "environment_archive_sha256": launch["environment_sha256"],
                "gpu_reserved": False,
                "fresh_training_or_accuracy_claim": False,
            },
        )
        print(
            json.dumps(
                {
                    "status": "full_native384_cpu_assembly_accepted",
                    "job_id": job,
                    "gpu_reserved": False,
                }
            ),
            flush=True,
        )
    except BaseException as error:
        save_new(
            control / "failure.json",
            {
                "job_id": job,
                "phase": phase,
                "error_type": type(error).__name__,
                "partial_cache_retained": str(scratch) if scratch else None,
                "automatic_retry": False,
            },
        )
        raise
    finally:
        payload.clear()
        if runtime is not None:
            helper = Path(launch["environment_helper"])
            subprocess.run(
                ["/usr/bin/python3.12", "-B", str(helper), "cleanup", "--root", str(runtime)],
                cwd=helper.parent,
                env=env,
                check=True,
            )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(
            json.dumps(
                {
                    "status": "native_assembly_runner_failed",
                    "error_type": type(error).__name__,
                    "automatic_retry": False,
                }
            ),
            flush=True,
        )
        raise SystemExit(2) from None
