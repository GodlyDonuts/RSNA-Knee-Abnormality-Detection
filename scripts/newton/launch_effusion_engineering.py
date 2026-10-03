"""One immutable Slurm admission/recovery allocation; no dispatch or downloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unpack_source(archive, root, expected):
    seen = set()
    with tarfile.open(archive, "r:") as source:
        for member in source:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or str(name) != member.name:
                raise ValueError("Unsafe source path")
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("Unexpected or duplicate source member")
            row = expected[member.name]
            if member.size != row["bytes"]:
                raise ValueError("Source size differs")
            path = root / member.name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with source.extractfile(member) as stream, path.open("xb") as out:
                shutil.copyfileobj(stream, out)
            if sha(path) != row["sha256"]:
                raise ValueError("Source SHA differs")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Incomplete original source")


def verify_original_inputs():
    base = Path("/home/sa305415/rsna-knee")
    original = base / "production-launch-native384-a32-c9-v1"
    module = original / "native384_admission.py"
    if sha(module) != "81060696c0fb4312d04e8f0a68b2c0eb515db743938af723a31f58e7ed962124":
        raise ValueError("Original native admission changed")
    spec = importlib.util.spec_from_file_location("original_native_admission", module)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    prior = json.loads((original / "cpu-preflight.json").read_text())
    assembly = base / "native384-assembly-c9-v1"
    _, _, members, restored = m.assembly_inputs(assembly)
    m.verify_assets(original / "config.json", assembly, prior=prior["scientific_assets"])
    signatures = m.verify_members(restored, members, prior=prior["signatures"])
    receipt = base / "native384-all-development-20261002-v2/inputs-accepted.json"
    if sha(receipt) != "dc70ab477aad8db956ea276ef656508f7cab3860586a1916f723ceab44800961":
        raise ValueError("Original whole-byte input proof changed")
    proof = json.loads(receipt.read_text())
    if m.object_sha(signatures) != proof["signatures_sha256"]:
        raise ValueError("Original complete cache inventory changed")
    actual = subprocess.check_output(
        [
            "sacct",
            "-X",
            "-j",
            proof["job_id"],
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode",
        ],
        text=True,
    ).strip()
    if actual != proof["job_id"] + "|COMPLETED|0:0":
        raise ValueError("Original complete admission not terminal accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--phase", choices=("inputs", "recovery"), required=True)
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdecimal() or os.uname().nodename.split(".")[0] != "evc31":
        raise ValueError("Actual owning-node allocation required")
    expected_cpus = "1" if args.phase == "inputs" else "2"
    if os.environ.get("SLURM_CPUS_PER_TASK") != expected_cpus or (
        args.phase == "inputs" and os.environ.get("SLURM_JOB_GPUS")
    ):
        raise ValueError("Exact minimum CPU/GPU allocation required")
    for name, row in json.loads((args.control / "control-files.json").read_text())["files"].items():
        path = args.control / name
        if path.is_symlink() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Immutable control changed")
    if args.phase == "recovery":
        ticket = json.loads((args.control / "gpu-admission.json").read_text())
        cpu_path = args.control / "inputs-accepted.json"
        if sha(cpu_path) != ticket["cpu_receipt_sha256"]:
            raise ValueError("Original accepted CPU receipt changed")
        cpu = json.loads(cpu_path.read_text())
        terminal = subprocess.check_output(
            [
                "sacct",
                "-X",
                "-j",
                cpu["job_id"],
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State,ExitCode",
            ],
            text=True,
        ).strip()
        if terminal != cpu["job_id"] + "|COMPLETED|0:0":
            raise ValueError("Actual CPU terminal acceptance required before GPU")
    manifest = json.loads((args.control / "source.json").read_text())
    if sha(args.control / "source.tar") != manifest["source_tar_sha256"]:
        raise ValueError("Scientific source archive changed")
    if shutil.disk_usage("/tmp").free < (20 + 7 + 3) * 2**30:
        raise ValueError("Node-local reserve after runtime/recovery liabilities required")
    environment = None
    with tempfile.TemporaryDirectory(
        prefix="rsna-effusion-code-" + job + "-", dir="/tmp"
    ) as temporary:
        code = Path(temporary)
        if code.stat().st_dev == args.control.stat().st_dev:
            raise ValueError("Source must remain node-local")
        unpack_source(args.control / "source.tar", code, manifest["files"])
        os.environ.update(
            PYTHONPATH=str(code),
            PYTHONNOUSERSITE="1",
            PYTHONDONTWRITEBYTECODE="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS=expected_cpus,
            MKL_NUM_THREADS=expected_cpus,
            OPENBLAS_NUM_THREADS=expected_cpus,
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        if args.phase == "inputs":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        helper = code / "scripts/newton/runtime_environment.py"
        try:
            record = json.loads(
                subprocess.check_output(
                    [
                        "/usr/bin/python3.12",
                        "-B",
                        str(helper),
                        "extract",
                        "--archive",
                        "/home/sa305415/rsna-knee/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar",
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
            environment = Path(record["runtime_root"])
            os.environ["TRITON_CACHE_DIR"] = str(environment / "triton-cache")
            # Reuse original full-byte/stat proofs; never duplicate bulk cache transfers.
            subprocess.run(
                [
                    record["python"],
                    "-B",
                    "-c",
                    "from scripts.newton.launch_effusion_engineering import verify_original_inputs; verify_original_inputs()",
                ],
                cwd=code,
                check=True,
                timeout=300,
            )
            command = [
                record["python"],
                "-B",
                str(code / "scripts/newton/probe_effusion_engineering.py"),
                "--control",
                str(args.control),
                "--phase",
                args.phase,
                "--output",
                "/tmp/rsna-effusion-supervision-engineering-20261003-v1",
            ]
            subprocess.run(command, cwd=code, check=True, timeout=5400)
        finally:
            if environment is not None:
                subprocess.run(
                    [
                        "/usr/bin/python3.12",
                        "-B",
                        str(helper),
                        "cleanup",
                        "--root",
                        str(environment),
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()
