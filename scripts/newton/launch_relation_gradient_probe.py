"""Immutable CPU admission / H100 measurement wrapper; never trains production."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def signature(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular immutable source asset is required")
    stat = path.stat()
    return dict(
        dev=stat.st_dev,
        ino=stat.st_ino,
        bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
    )


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
                raise ValueError("Unexpected diagnostic source member")
            row = expected[member.name]
            if member.size != row["bytes"]:
                raise ValueError("Diagnostic source member size differs")
            path = root / member.name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with source.extractfile(member) as stream, path.open("xb") as out:
                shutil.copyfileobj(stream, out, 1024**2)
            if sha(path) != row["sha256"]:
                raise ValueError("Materialized diagnostic source differs")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Materialized diagnostic source is incomplete")


def materialize_cache(archive, cache, request, series, proof, run):
    """Copy only sixteen training studies to node-local disk, never to Lustre."""
    if signature(archive) != proof["assets"][str(archive)]["stat"]:
        raise ValueError("Previously whole-byte-verified cache changed")
    uids = sorted(request["training_uids"])
    digest = hashlib.sha256(
        json.dumps(uids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        len(set(uids)) != len(uids)
        or len(uids) != run["training_studies"]
        or digest != request["training_uids_sha256"]
        or digest != run["binding"]["training_uids_sha256"]
        or sha(series) != request["series_table_sha256"]
    ):
        raise ValueError("Diagnostic request differs from original training source")
    selected = set(uids[:16])
    if len(selected) != 16:
        raise ValueError("Exactly sixteen training studies are required")
    expected, represented = set(), set()
    with series.open(newline="") as handle:
        for row in csv.DictReader(handle):
            uid, sid = row["StudyInstanceUID"], row["SeriesInstanceUID"]
            if uid not in selected:
                continue
            if any(
                not value or "/" in value or "\\" in value or value in {".", ".."}
                for value in (uid, sid)
            ):
                raise ValueError("Series identity escapes training cache")
            stem = f"cache/train_series/{uid}/{sid}"
            names = {stem + ".npz", stem + ".json"}
            if expected & names:
                raise ValueError("Duplicate training series")
            expected.update(names)
            represented.add(uid)
    if represented != selected or cache.exists():
        raise ValueError("Missing training series or preexisting materialization")
    if shutil.disk_usage(cache.parent).free < 22 * 1024**3:
        raise OSError("Node-local cache must preserve20GiB plus its2GiB bound")
    cache.mkdir(mode=0o700)
    records, total = {}, 0
    with tarfile.open(archive, "r:") as source:
        for member in source:
            if member.name not in expected:
                continue
            if (
                not member.isfile()
                or member.name in records
                or member.size < 1
                or member.size > (2 * 1024**2 if member.name.endswith(".json") else 256 * 1024**2)
            ):
                raise ValueError("Invalid original training cache member")
            total += member.size
            if total > 2 * 1024**3:
                raise OSError("Selected training cache exceeds its node-local bound")
            relative = member.name.removeprefix("cache/")
            path = cache / relative
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with source.extractfile(member) as stream, path.open("xb") as out:
                shutil.copyfileobj(stream, out, 1024**2)
            records[member.name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
            if records[member.name]["bytes"] != member.size:
                raise ValueError("Incomplete node-local cache copy")
    if records.keys() != expected or signature(archive) != proof["assets"][str(archive)]["stat"]:
        raise ValueError("Missing original training cache members or archive changed")
    return {
        "training_studies": len(selected),
        "members": records,
        "bytes": total,
        "large_artifacts_written_to_shared_storage": False,
    }


def save_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu-input", "gpu-measure"), required=True)
    parser.add_argument("--cpu-proof-sha256")
    args = parser.parse_args()
    control, mode = args.control, args.mode
    manifest = json.loads((control / "source.json").read_text())
    job = os.environ.get("SLURM_JOB_ID", "")
    cpu = mode == "cpu-input"
    if (
        not job.isdecimal()
        or os.environ.get("SLURM_CPUS_PER_TASK") != ("1" if cpu else "2")
        or bool(os.environ.get("SLURM_JOB_GPUS")) == cpu
    ):
        raise ValueError("Actual one-CPU/zero-GPU or two-CPU/one-H100 allocation required")
    output = control / ("cpu-input-verified.json" if cpu else "gradient-measurement.json")
    if output.exists() or (control / f"runtime-{mode}.json").exists():
        raise FileExistsError("Diagnostic mode was attempted; reconcile original state")
    if sha(control / "source.tar") != manifest["source_tar_sha256"]:
        raise ValueError("Pinned diagnostic source archive changed")
    for key in ("proof", "request", "run", "series", "studies"):
        if sha(Path(manifest[key + "_path"])) != manifest[key + "_sha256"]:
            raise ValueError("An exact diagnostic input was replaced or modified")
    proof = json.loads(Path(manifest["proof_path"]).read_text())
    for name, row in proof["assets"].items():
        if signature(Path(name)) != row["stat"]:
            raise ValueError("An accepted original source/runtime/weight asset changed")
    if not cpu and (
        args.cpu_proof_sha256 is None
        or sha(control / "cpu-input-verified.json") != args.cpu_proof_sha256
    ):
        raise ValueError("GPU launch needs the exact completed CPU input proof")
    os.environ.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="1" if cpu else "2",
        MKL_NUM_THREADS="1" if cpu else "2",
        OPENBLAS_NUM_THREADS="1" if cpu else "2",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    environment = None
    with tempfile.TemporaryDirectory(
        prefix=f"rsna-relation-gradient-{job}-", dir="/tmp"
    ) as temporary:
        code = Path(temporary)
        if code.stat().st_dev == control.stat().st_dev:
            raise ValueError("Source/cache/runtime must be materialized on node-local disk")
        unpack_source(control / "source.tar", code, manifest["files"])
        os.environ["PYTHONPATH"] = str(code)
        cache = code / "diagnostic-cache"
        copied = materialize_cache(
            Path(manifest["cache_archive"]),
            cache,
            json.loads(Path(manifest["request_path"]).read_text()),
            Path(manifest["series_path"]),
            proof,
            json.loads(Path(manifest["run_path"]).read_text()),
        )
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
            save_new(
                control / f"runtime-{mode}.json",
                {
                    "status": "exact_source_original_training_cache_and_runtime_materialized",
                    "source_git": manifest["source_git"],
                    "job_id": job,
                    "mode": mode,
                    "cpus": 1 if cpu else 2,
                    "gpus": 0 if cpu else 1,
                    "environment": extracted,
                    "source_file_count": len(manifest["files"]),
                    "cache": copied,
                },
            )
            command = [
                str(environment / "environment/bin/python"),
                "-B",
                str(code / "scripts/newton/measure_relation_gradients.py"),
            ]
            paths = {
                "config-root": code / "initiative/experiments/competitive-v1/slice-relations-v1",
                "proof": manifest["proof_path"],
                "request": manifest["request_path"],
                "run": manifest["run_path"],
                "studies": manifest["studies_path"],
                "artifact-root": manifest["artifact_root"],
                "folds": manifest["folds_path"],
                "series": manifest["series_path"],
                "archive": manifest["cache_archive"],
                "cache": cache,
                "output": output,
                "source-manifest": control / "source.json",
            }
            for name, path in paths.items():
                command += ["--" + name, str(path)]
            command += (
                ["--cpu-input-only"]
                if cpu
                else ["--cpu-acceptance", str(control / "cpu-input-verified.json")]
            )
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
