"""Materialize pinned diagnostic source/runtime on node-local disk; use zero GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Source member escapes node-local root")
            if member.isdir():
                continue
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("Unexpected source member")
            row = expected[member.name]
            if member.size != row["bytes"]:
                raise ValueError("Source member size differs")
            path = root / member.name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with source.extractfile(member) as stream, path.open("xb") as out:
                out.write(stream.read())
            if sha(path) != row["sha256"]:
                raise ValueError("Materialized scientific source differs")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Materialized source is incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control
    manifest = json.loads((control / "source.json").read_text())
    job = os.environ.get("SLURM_JOB_ID", "")
    if (
        not job.isdecimal()
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_JOB_GPUS")
    ):
        raise ValueError("A real one-CPU/zero-GPU allocation is required")
    if sha(control / "source.tar") != manifest["source_tar_sha256"]:
        raise ValueError("Pinned scientific source archive changed")
    proof_path, request_path = map(Path, (manifest["proof_path"], manifest["request_path"]))
    if (
        sha(proof_path) != manifest["proof_sha256"]
        or sha(request_path) != manifest["request_sha256"]
    ):
        raise ValueError("Original full-byte proof or training geometry request changed")
    proof = json.loads(proof_path.read_text())
    for name, row in proof["assets"].items():
        path = Path(name)
        stat = path.stat()
        if (
            path.is_symlink()
            or {
                "dev": stat.st_dev,
                "ino": stat.st_ino,
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
            != row["stat"]
        ):
            raise ValueError("An accepted original asset was replaced or modified")
    os.environ.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    environment = None
    with tempfile.TemporaryDirectory(
        prefix="rsna-relation-cpu-" + job + "-", dir="/tmp"
    ) as temporary:
        code = Path(temporary)
        if code.stat().st_dev == Path.home().stat().st_dev:
            raise ValueError("Scientific source must use node-local disk")
        unpack_source(control / "source.tar", code, manifest["files"])
        os.environ["PYTHONPATH"] = str(code)
        helper = code / "scripts/newton/runtime_environment.py"
        try:
            archive = manifest["environment_archive"]
            extracted = json.loads(
                subprocess.check_output(
                    [
                        "/usr/bin/python3.12",
                        "-B",
                        str(helper),
                        "extract",
                        "--archive",
                        archive,
                        "--sha256",
                        proof["assets"][archive]["sha256"],
                        "--expected-files",
                        "40403",
                        "--expected-bytes",
                        "6322691335",
                        "--scratch-base",
                        "/tmp",
                        "--min-free-gib",
                        "20",
                        "--shared-python",
                        "/usr/bin/python3.12",
                    ],
                    text=True,
                    timeout=900,
                )
            )
            environment = Path(extracted["runtime_root"])
            status = {
                "status": "pinned_source_and_node_local_runtime_materialized",
                "source_git": manifest["source_git"],
                "job_id": job,
                "cpus": 1,
                "gpus": 0,
                "environment": extracted,
                "source_file_count": len(manifest["files"]),
            }
            (control / "runtime-materialized.json").write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n"
            )
            command = [
                str(environment / "environment/bin/python"),
                "-B",
                str(code / "scripts/newton/probe_real_slice_relations.py"),
            ]
            paths = {
                "config": code
                / "initiative/experiments/competitive-v1/slice-relations-v1/physical-pilot-config.json",
                "request": request_path,
                "proof": proof_path,
                "run": Path(manifest["run_path"]),
                "archive": Path(manifest["cache_archive"]),
                "series": Path(manifest["series_path"]),
                "output": control / "result",
            }
            for name, path in paths.items():
                command += ["--" + name, str(path)]
            subprocess.run(command, check=True, cwd=code, timeout=2700)
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
