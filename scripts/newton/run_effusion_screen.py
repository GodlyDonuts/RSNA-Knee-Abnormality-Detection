"""Run the admitted fixed Effusion contrast; no scheduler retry or scientific edits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path("/home/sa305415/rsna-knee")
PARENT = BASE / "effusion-supervision-engineering-20261003-v1"
RECOVERY = PARENT / "recovery-v2"
ROUTE = Path("/tmp/rsna-effusion-native-route-evc32-20261004-v1")
CONTROL = Path("/tmp/rsna-effusion-screen-evc32-20261004-v1")
SOURCE_SHA = "53ff55e329e4c697870bf68679cebee80d5369566f017e483c2818006ad3c9a3"
GPU_SHA = "be201fc89f16ede13f379d854018f4d7dc5ca9a55a26a528081418eea87632ef"
OVERLAY_SHA = "936e3e2af058726fe3572b7e4ad1680c44f2fd82ad6e9840e2607a1b090ff018"
REGISTRY_SHA = "73d06c86900f03e8571fb68a8d341333c547ab984c7a4317387754949b8ab770"


def sha(path):
    import hashlib

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fresh_step_environment(environment):
    """Discard staging-job CPU masks; Slurm assigns each new step's own cores."""
    result = environment.copy()
    for key in (
        "SLURM_CPU_BIND",
        "SLURM_CPU_BIND_LIST",
        "SLURM_CPU_BIND_TYPE",
        "SLURM_CPU_BIND_VERBOSE",
    ):
        result.pop(key, None)
    return result


def accept_copy(record):
    required = {
        "status": "private_original_native384_copy_fully_byte_verified",
        "job_id": "869745",
        "source_node": "evc31",
        "destination_node": "evc32",
        "members": 37812,
        "bytes": 140626483351,
        "registry_sha256": REGISTRY_SHA,
        "original_cache_retained": True,
        "GPU_reserved": False,
        "target_values_read": False,
        "clinical_accuracy_evidence": False,
    }
    if any(type(record.get(k)) is not type(v) or record[k] != v for k, v in required.items()):
        raise ValueError("Actual complete original cache placement is required")
    cache = Path(record["cache_root"])
    if (
        not str(cache).startswith("/tmp/rsna-effusion-native384-evc32-route-869745-")
        or cache.parts[-2:] != ("restored", "cache")
        or ".." in cache.parts
    ):
        raise ValueError("Exact private owning-node cache path required")
    return cache


def verify_control():
    if os.uname().nodename.split(".")[0] != "evc32":
        raise ValueError("Admitted owning node required")
    if CONTROL.is_symlink() or CONTROL.stat().st_uid != os.getuid():
        raise ValueError("Owner-only node-local control required")
    manifest = json.loads((CONTROL / "files.json").read_text())
    for name, row in manifest["files"].items():
        path = CONTROL / name
        if path.is_symlink() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Immutable launch control changed")
    if sha(RECOVERY / "source.json") != SOURCE_SHA or sha(PARENT / "overlay.json") != OVERLAY_SHA:
        raise ValueError("Actual accepted source or supervision changed")
    if sha(RECOVERY / "recovery-accepted.json") != GPU_SHA:
        raise ValueError("Original actual numerical recovery changed")
    copy = json.loads((ROUTE / "copy-accepted.json").read_text())
    cache = accept_copy(copy)
    if sha(ROUTE / "copy-accepted.json") != manifest["copy_receipt_sha256"]:
        raise ValueError("Original placement acceptance changed")
    if shutil.disk_usage("/tmp").free < (20 + 7 + 3) * 2**30 + 32 * 2**20:
        raise ValueError("Runtime/checkpoint/control/reserve funding exhausted")
    return manifest, cache


def verify_cache_stats(cache, *, original=None):
    helper = BASE / "production-launch-native384-a32-c9-v1/native384_admission.py"
    if sha(helper) != "81060696c0fb4312d04e8f0a68b2c0eb515db743938af723a31f58e7ed962124":
        raise ValueError("Original cache admission helper changed")
    registry = BASE / "native384-assembly-c9-v1/registry.json"
    if sha(registry) != REGISTRY_SHA:
        raise ValueError("Original cache registry changed")
    module = load_module(helper, "effusion_original_cache")
    members = module.validate_registry(json.loads(registry.read_text()))
    actual = {}
    for name, row in members.items():
        path = cache.parent / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError("Byte-verified original cache member changed")
        actual[name] = module.signature(path)
    if original is not None and actual != original:
        raise ValueError("Original cache stat inventory changed since placement")
    return actual


def prepare():
    """One zero-GPU CPU admission/extraction after the complete alternate copy."""
    manifest, cache = verify_control()
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("One CPU, zero GPU input preparation required")
    terminal = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-j",
            "869300,869745,869722",
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode",
        ],
        text=True,
    )
    if set(terminal.splitlines()) != {
        "869300|COMPLETED|0:0",
        "869745|COMPLETED|0:0",
        "869722|COMPLETED|0:0",
    }:
        raise ValueError("Actual recovery, complete copy and assigned driver acceptance required")
    source = json.loads((RECOVERY / "source.json").read_text())
    if sha(RECOVERY / "source.tar") != source["source_tar_sha256"]:
        raise ValueError("Accepted numerical source archive changed")
    expected = source["files"]["scripts/newton/launch_effusion_engineering.py"]["sha256"]
    if sha(RECOVERY / "launch_effusion_engineering.py") != expected:
        raise ValueError("Original safe source unpacker changed")
    loader = load_module(RECOVERY / "launch_effusion_engineering.py", "effusion_original_loader")
    code = CONTROL / "code"
    code.mkdir(mode=0o700)
    loader.unpack_source(RECOVERY / "source.tar", code, source["files"])
    save(CONTROL / "cache-stats-private.json", verify_cache_stats(cache))
    native = load_module(
        BASE / "production-launch-native384-a32-c9-v1/native384_admission.py",
        "effusion_shared_assets",
    )
    prior = json.loads(
        (BASE / "production-launch-native384-a32-c9-v1/cpu-preflight.json").read_text()
    )
    native.verify_assets(
        BASE / "production-launch-native384-a32-c9-v1/config.json",
        BASE / "native384-assembly-c9-v1",
        prior=prior["scientific_assets"],
    )
    runtime = json.loads(
        subprocess.check_output(
            [
                "/usr/bin/python3.12",
                "-B",
                str(code / "scripts/newton/runtime_environment.py"),
                "extract",
                "--archive",
                str(
                    BASE
                    / "kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar"
                ),
                "--sha256",
                "89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338",
                "--expected-files",
                "40403",
                "--expected-bytes",
                "6322691335",
                "--scratch-base",
                "/tmp",
                "--shared-python",
                "/usr/bin/python3.12",
                "--min-free-gib",
                "23",
            ],
            text=True,
        )
    )
    save(CONTROL / "runtime.json", runtime)
    save(
        CONTROL / "CPU-prepared.json",
        {
            "status": "exact_inputs_runtime_source_prepared_on_driver_accepted_node",
            "job_id": os.environ["SLURM_JOB_ID"],
            "node": "evc32",
            "copy_job": "869745",
            "driver_probe_job": "869722",
            "source_manifest_sha256": SOURCE_SHA,
            "source_files": len(source["files"]),
            "cache_stats_sha256": sha(CONTROL / "cache-stats-private.json"),
            "clinical_accuracy_evidence": False,
            "model_or_optimizer_run": False,
        },
    )


def arm(role):
    import torch

    manifest, cache = verify_control()
    if os.environ.get("SLURM_CPUS_PER_TASK") != "2" or torch.cuda.device_count() != 1:
        raise ValueError("One scheduler-isolated H100 and two CPUs per independent arm required")
    if "H100" not in torch.cuda.get_device_name(0) or not torch.cuda.is_bf16_supported():
        raise ValueError("Actual H100 BF16 device required")
    verify_cache_stats(
        cache, original=json.loads((CONTROL / "cache-stats-private.json").read_text())
    )
    code = CONTROL / "code"
    source = json.loads((RECOVERY / "source.json").read_text())
    for name, row in source["files"].items():
        if sha(code / name) != row["sha256"]:
            raise ValueError("Actual accepted trainer source changed")
    inputs = BASE / "code-stopfix-1f486765"
    command = [
        sys.executable,
        "-B",
        str(code / "scripts/newton/enforce_deterministic_runtime.py"),
        "--module",
        "rsna.competitive.train",
        "--config",
        str(CONTROL / (role + "-screen.json")),
        "--studies",
        str(inputs / "initiative/releases/report-supervision-v1/studies.parquet"),
        "--series",
        str(
            inputs
            / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
        ),
        "--cache",
        str(cache),
        "--folds",
        str(inputs / "initiative/experiments/competitive-v1/folds"),
        "--artifact-root",
        str(inputs),
        "--fitting-supervision",
        str(PARENT / "overlay.json"),
        "--device",
        "cuda",
        "--output",
        str(CONTROL / role),
    ]
    save(CONTROL / (role + "-trainer-intent.json"), {"command": command, "automatic_retry": False})
    subprocess.run(command, cwd=code, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "arm"))
    parser.add_argument("--role", choices=("control", "candidate"))
    args = parser.parse_args()
    os.umask(0o077)
    if args.phase == "prepare":
        prepare()
    elif args.role:
        arm(args.role)
    else:
        raise ValueError("Independent role required")


if __name__ == "__main__":
    main()
