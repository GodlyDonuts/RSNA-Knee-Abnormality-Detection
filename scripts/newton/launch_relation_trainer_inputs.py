"""Immutable CPU-only 32-study input wrapper; no model, optimizer or GPU launch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_fixture_reference(fixture, geometry, run):
    if (
        fixture["training_uids"] != sorted(geometry["training_uids"])[:16]
        or fixture["full_training_uids_sha256"] != geometry["training_uids_sha256"]
        or fixture["full_training_uids_sha256"] != run["binding"]["training_uids_sha256"]
        or fixture["full_outer_uids_sha256"] != run["binding"]["outer_uids_sha256"]
        or fixture["release_binding"] != run["binding"]["release"]
    ):
        raise ValueError(
            "Fixture differs from the original accepted training/outer/release binding"
        )


def copy_fixture(archive, cache, fixture, series, reference, helper):
    if (
        len(fixture["training_uids"]) != 16
        or len(fixture["outer_uids"]) != 16
        or len(set(fixture["training_uids"] + fixture["outer_uids"])) != 32
        or fixture.get("diagnostic_only") is not True
        or fixture.get("clinical_selection_allowed") is not False
        or helper.signature(archive) != reference["assets"][str(archive)]["stat"]
    ):
        raise ValueError("Exact disjoint32-study diagnostic fixture and original archive required")
    uids = set(fixture["training_uids"] + fixture["outer_uids"])
    expected, represented = set(), set()
    with series.open(newline="") as handle:
        for row in csv.DictReader(handle):
            uid, sid = row["StudyInstanceUID"], row["SeriesInstanceUID"]
            if uid not in uids:
                continue
            if any(not x or "/" in x or "\\" in x or x in {".", ".."} for x in (uid, sid)):
                raise ValueError("Fixture identity escapes cache root")
            stem = f"cache/train_series/{uid}/{sid}"
            names = {stem + ".json", stem + ".npz"}
            if expected & names:
                raise ValueError("Duplicate fixture series")
            expected.update(names)
            represented.add(uid)
    if represented != uids or cache.exists():
        raise ValueError("Incomplete fixture series or existing cache materialization")
    if shutil.disk_usage(cache.parent).free < 31 * 1024**3:
        raise OSError("Preserve20GiB plus4GiB cache and7GiB runtime on node-local disk")
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
                raise ValueError("Invalid original fixture cache member")
            total += member.size
            if total > 4 * 1024**3:
                raise OSError("Recovery fixture exceeds its node-local4GiB bound")
            path = cache / member.name.removeprefix("cache/")
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with source.extractfile(member) as stream, path.open("xb") as out:
                shutil.copyfileobj(stream, out, 1024**2)
            if path.stat().st_size != member.size:
                raise ValueError("Incomplete original cache copy")
            records[member.name] = {"bytes": member.size, "sha256": sha(path)}
    if (
        records.keys() != expected
        or helper.signature(archive) != reference["assets"][str(archive)]["stat"]
    ):
        raise ValueError("Missing original fixture members or changed archive")
    return {"studies": 32, "bytes": total, "members": records, "shared_pixel_bytes_written": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control
    if (
        not os.environ.get("SLURM_JOB_ID", "").isdecimal()
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_JOB_GPUS")
    ):
        raise ValueError("Actual one-CPU/zero-GPU Slurm allocation required")
    source = json.loads((control / "source.json").read_text())
    if (
        sha(Path(__file__))
        != source["files"]["scripts/newton/launch_relation_trainer_inputs.py"]["sha256"]
    ):
        raise ValueError("CPU bootstrap differs from the frozen source pack")
    reference = json.loads(Path(source["reference_manifest_path"]).read_text())
    for key, path in {
        "source_tar_sha256": control / "source.tar",
        "reference_manifest_sha256": Path(source["reference_manifest_path"]),
        "fixture_request_sha256": control / "fixture-request.json",
        "original_CPU_acceptance_sha256": Path(source["original_CPU_acceptance_path"]),
    }.items():
        if sha(path) != source[key]:
            raise ValueError("Exact recovery source/input proof changed")
    for key in ("proof", "request", "run", "series", "studies"):
        if sha(Path(reference[key + "_path"])) != reference[key + "_sha256"]:
            raise ValueError("Original accepted input source changed")
    if any((control / name).exists() for name in ("runtime.json", "cpu-input-verified.json")):
        raise FileExistsError("Existing CPU attempt; reconcile its original state")
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
    proof = json.loads(Path(reference["proof_path"]).read_text())
    fixture = json.loads((control / "fixture-request.json").read_text())
    verify_fixture_reference(
        fixture,
        json.loads(Path(reference["request_path"]).read_text()),
        json.loads(Path(reference["run_path"]).read_text()),
    )
    with tempfile.TemporaryDirectory(
        prefix="rsna-relation-trainer-inputs-", dir="/tmp"
    ) as temporary:
        code = Path(temporary)
        if code.stat().st_dev == control.stat().st_dev:
            raise ValueError("Large source/cache/runtime must use node-local disk")
        # Bootstrap using the independently hashed original stdlib helper.
        original = Path(source["reference_control"]) / "launch_gradient.py"
        if (
            sha(original)
            != reference["files"]["scripts/newton/launch_relation_gradient_probe.py"]["sha256"]
        ):
            raise ValueError("Original materialization helper changed")
        spec = importlib.util.spec_from_file_location("original_materializer", original)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        for name, row in proof["assets"].items():
            if helper.signature(Path(name)) != row["stat"]:
                raise ValueError("Original byte-verified cache/runtime/weight asset changed")
        helper.unpack_source(control / "source.tar", code, source["files"])
        os.environ["PYTHONPATH"] = str(code)
        cache = code / "diagnostic-cache"
        copied = copy_fixture(
            Path(reference["cache_archive"]),
            cache,
            fixture,
            Path(reference["series_path"]),
            proof,
            helper,
        )
        environment = None
        runtime_helper = code / "scripts/newton/runtime_environment.py"
        try:
            archive = reference["environment_archive"]
            extracted = json.loads(
                subprocess.check_output(
                    [
                        "/usr/bin/python3.12",
                        "-B",
                        str(runtime_helper),
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
            helper.save_new(
                control / "runtime.json",
                {
                    "status": "exact_32_study_source_runtime_and_cache_materialized",
                    "source_git": source["source_git"],
                    "reference_source_git": reference["source_git"],
                    "job_id": os.environ["SLURM_JOB_ID"],
                    "cpus": 1,
                    "gpus": 0,
                    "environment": extracted,
                    "cache": copied,
                },
            )
            command = [
                str(environment / "environment/bin/python"),
                "-B",
                str(code / "scripts/newton/admit_relation_trainer_inputs.py"),
            ]
            paths = {
                key: reference[key + "_path"]
                for key in ("proof", "request", "run", "studies", "series", "folds")
            }
            paths.update(
                {
                    "config-root": code
                    / "initiative/experiments/competitive-v1/slice-relations-v1",
                    "artifact-root": reference["artifact_root"],
                    "cache": cache,
                    "archive": reference["cache_archive"],
                    "output": control / "cpu-input-verified.json",
                    "source-manifest": source["reference_manifest_path"],
                    "recovery-source-manifest": control / "source.json",
                    "fixture-request": control / "fixture-request.json",
                    "cpu-acceptance": source["original_CPU_acceptance_path"],
                }
            )
            for name, path in paths.items():
                command += ["--" + name, str(path)]
            subprocess.run(command, cwd=code, check=True, timeout=2700)
        finally:
            if environment is not None:
                subprocess.run(
                    [
                        "/usr/bin/python3.12",
                        "-B",
                        str(runtime_helper),
                        "cleanup",
                        "--root",
                        str(environment),
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()
