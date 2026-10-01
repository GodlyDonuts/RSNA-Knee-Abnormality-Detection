"""Run one original-source cache worker and preserve it before reclaiming scratch.

The scheduler supplies one CPU and no GPU. Credentials arrive once on stdin;
only the two finite child processes receive them, also through stdin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCIENCE = "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
PLAN_SHA = "44e6775c5413e42d8744ead0f7d7bb0de258893a10867123e47fbd770f84793f"
WORKER_SHA = "62a879a7c97d118163b3e9769f18f691e9a971f7ee3cf0133c5130e9ad5fb173"
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


def owned_marker(job, driver_hash, uid):
    if not re.fullmatch(r"[1-9][0-9]*", job) or not re.fullmatch(r"[0-9a-f]{64}", driver_hash):
        raise ValueError("real_job_and_driver_binding_required")
    return {
        "kind": "native-cache-owned-scratch-v1",
        "job_id": job,
        "driver_sha256": driver_hash,
        "uid": uid,
    }


def require_private_preservation(root, shard, selection_sha, verified, execution):
    """Cleanup is authorized only by an exact complete private download receipt."""
    if (
        verified.get("status") != "private_native_shard_full_download_size_sha_verified"
        or verified.get("repo_id") != "GodlyDonuts/RSNA-Data"
        or verified.get("shard") != shard
        or verified.get("execution_provenance") != execution
        or verified.get("prefix")
        != (
            "experiments/dinov3-v1/native-original384-v1/selection-"
            + selection_sha
            + f"/shard-{shard:03d}"
        )
        or not re.fullmatch(r"[0-9a-f]{40}", verified.get("revision", ""))
        or set(verified.get("files", {}))
        != {
            "cache-side-256.tar",
            "cache-side-384.tar",
            "cache-verified.json",
            "series-proofs.jsonl",
            "attribution.json",
            "manifest.json",
        }
    ):
        raise ValueError("complete_private_preservation_required")
    if any(
        type(record.get("bytes")) is not int
        or record["bytes"] <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))
        for record in verified["files"].values()
    ):
        raise ValueError("complete_remote_byte_bindings_required")
    for name in (
        "cache-side-256.tar",
        "cache-side-384.tar",
        "cache-verified.json",
        "series-proofs.jsonl",
    ):
        p = root / name
        record = verified["files"][name]
        if (
            p.is_symlink()
            or not p.is_file()
            or p.stat().st_size != record["bytes"]
            or digest(p) != record["sha256"]
        ):
            raise ValueError("local_preserved_payload_changed")


def remove_owned(root, expected, home_device):
    marker = root / ".native-cache-owner.json"
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.stat().st_uid != expected["uid"]
        or root.stat().st_dev == home_device
        or not root.name.startswith("rsna-native-shard-" + expected["job_id"] + "-")
        or marker.is_symlink()
        or json.loads(marker.read_text()) != expected
    ):
        raise ValueError("owned_node_local_scratch_required")
    shutil.rmtree(root)


def child_command(python, checkout, target, args):
    return [
        str(python),
        "-B",
        "-c",
        "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);runpy.run_path(p,run_name='__main__')",
        str(checkout),
        str(target),
        *map(str, args),
    ]


def require_control(control, base):
    """Allow the cluster's mapped home ancestor, but never a linked control leaf."""
    if (
        control.is_symlink()
        or control.resolve() == base.resolve()
        or not control.resolve().is_relative_to(base.resolve())
    ):
        raise ValueError("private_cluster_control_required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control
    job = os.environ.get("SLURM_JOB_ID", "")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("one_cpu_zero_gpu_required")
    require_control(control, BASE)
    plan_path = control / "launch.json"
    plan = json.loads(plan_path.read_text())
    binding = owned_marker(job, digest(Path(__file__)), os.getuid())
    if (
        plan["scientific_source_git"] != SCIENCE
        or plan["files"]["worker.py"]["sha256"] != WORKER_SHA
    ):
        raise ValueError("pinned_worker_and_science_required")
    for name, record in plan["files"].items():
        path = control / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.is_symlink()
            or digest(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError("staged_launch_file_changed")
    if digest(control / "manifest.json") != PLAN_SHA:
        raise ValueError("full_plan_changed")
    science = json.loads((control / "science-binding.json").read_text())
    checkout = Path(science["checkout"])
    if (
        subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        != SCIENCE
    ):
        raise ValueError("scientific_checkout_changed")
    if list((checkout / "rsna").rglob("*.pyc")) or list((checkout / "rsna").rglob("*.so")):
        raise ValueError("unbound_scientific_cached_or_native_code")
    for name, record in science["files"].items():
        if digest(checkout / name) != record["sha256"]:
            raise ValueError("scientific_source_changed")
    for name, record in plan["runtime_helpers"].items():
        if digest(Path(name)) != record["sha256"]:
            raise ValueError("runtime_helper_changed")
    shard = json.loads((control / f"shard-{plan['shard']:03d}.json").read_text())
    for name, record in shard["input_bindings"].items():
        path = Path(plan["input_root"]) / name
        if (
            path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or digest(path) != record["sha256"]
        ):
            raise ValueError("accepted_input_changed")
    quota = (
        subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
        .splitlines()[-1]
        .split()
    )
    used, soft, files, file_soft = [int(quota[i]) for i in (1, 2, 5, 6)]
    projected = (soft - used) * 1024 - plan["remaining_shared_obligations_bytes"]
    if projected < 20 * 1024**3 or file_soft - files - 100 < 5000:
        raise ValueError("shared_reserve_not_met")
    payload = json.load(sys.stdin)
    if (
        set(payload) != {"token"}
        or not isinstance(payload["token"], str)
        or not payload["token"].startswith("hf_")
    ):
        raise ValueError("exact_private_token_stdin_required")
    if (control / "publication").exists() or (control / "run-start.json").exists():
        raise FileExistsError("existing_attempt_requires_readonly_review")
    save_new(
        control / "run-start.json",
        {
            **binding,
            "node": os.uname().nodename,
            "pid": os.getpid(),
            "launch_sha256": digest(plan_path),
            "projected_remaining_gib": projected / 1024**3,
            "gpu_reserved": False,
        },
    )
    scratch = Path(tempfile.mkdtemp(prefix="rsna-native-shard-" + job + "-", dir="/tmp"))
    if scratch.stat().st_dev == BASE.stat().st_dev:
        raise ValueError("node_local_disk_required")
    save_new(scratch / ".native-cache-owner.json", binding)
    save_new(
        control / "scratch.json",
        {"path": str(scratch), "owner": binding, "preserve_until_private_verification": True},
    )
    runtime_root = None
    phase = "environment"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HOME": str(scratch / "hf"),
            "HF_HUB_DISABLE_XET": "1",
            "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    env.pop("HF_TOKEN", None)
    env.pop("PYTHONPATH", None)
    helper = Path(plan["environment_helper"])
    try:
        request = [
            "/usr/bin/python3.12",
            str(helper),
            "extract",
            "--archive",
            plan["environment_archive"],
            "--sha256",
            plan["environment_sha256"],
            "--expected-files",
            "40403",
            "--expected-bytes",
            "6322691335",
            "--scratch-base",
            "/tmp",
            "--shared-python",
            "/usr/bin/python3.12",
            "--min-free-gib",
            "40",
        ]
        runtime = json.loads(
            subprocess.check_output(request, cwd=helper.parent, env=env, text=True)
        )
        runtime_root = Path(runtime["runtime_root"])
        python = runtime_root / "environment/bin/python"
        phase = "original_source_worker"
        command = child_command(
            python,
            checkout,
            control / "worker.py",
            [
                "--root",
                plan["input_root"],
                "--plan-manifest",
                control / "manifest.json",
                "--shard",
                plan["shard"],
                "--reference-archive",
                plan["reference_archive"],
                "--output",
                scratch / "paired",
                "--science-binding",
                control / "science-binding.json",
            ],
        )
        subprocess.run(
            command, cwd=checkout, env=env, input=json.dumps(payload), text=True, check=True
        )
        phase = "private_preservation"
        command = child_command(
            python,
            checkout,
            control / "publisher.py",
            [
                "--root",
                scratch / "paired",
                "--plan-manifest",
                control / "manifest.json",
                "--shard",
                plan["shard"],
                "--journal",
                control / "publication",
            ],
        )
        subprocess.run(
            command, cwd=checkout, env=env, input=json.dumps(payload), text=True, check=True
        )
        verified = json.loads((control / "publication/verified.json").read_text())
        report = json.loads((scratch / "paired/cache-verified.json").read_text())
        require_private_preservation(
            scratch / "paired",
            plan["shard"],
            shard["selection_sha256"],
            verified,
            report["execution_provenance"],
        )
        for name in ("cache-verified.json", "series-proofs.jsonl"):
            with (control / name).open("xb") as destination:
                destination.write((scratch / "paired" / name).read_bytes())
        save_new(
            control / "private-preservation-accepted.json",
            {
                "status": "paired_native_shard_privately_verified",
                "job_id": job,
                "shard": plan["shard"],
                "revision": verified["revision"],
                "studies": report["studies"],
                "series": report["series"],
                "slices": report["slices"],
                "publication_receipt_sha256": digest(control / "publication/verified.json"),
                "model_training_or_accuracy_claim": False,
            },
        )
        phase = "owned_cleanup"
        remove_owned(scratch, binding, BASE.stat().st_dev)
        save_new(
            control / "result.json",
            {
                "status": "original_source_paired_shard_and_private_preservation_complete",
                "job_id": job,
                "shard": plan["shard"],
                "revision": verified["revision"],
                "scratch_reclaimed_after_full_private_verification": True,
            },
        )
        print(
            json.dumps(
                {
                    "status": "original_source_paired_shard_and_private_preservation_complete",
                    "shard": plan["shard"],
                    "job_id": job,
                }
            ),
            flush=True,
        )
    except BaseException as error:
        save_new(
            control / "failure.json",
            {
                "phase": phase,
                "error_type": type(error).__name__,
                "job_id": job,
                "owned_scratch_preserved": str(scratch),
                "automatic_retry": False,
            },
        )
        raise
    finally:
        payload.clear()
        if runtime_root is not None:
            subprocess.run(
                ["/usr/bin/python3.12", str(helper), "cleanup", "--root", str(runtime_root)],
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
                    "status": "native_shard_run_failed",
                    "error_type": type(error).__name__,
                    "automatic_retry": False,
                }
            ),
            flush=True,
        )
        raise SystemExit(2) from None
