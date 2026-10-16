"""Real H100 recovery for the separately bound fitting-derived positive loss.

No training-only acceptance is inferred from the old N3 recovery. This invokes
fresh full/stop1/resume processes of the actual new numerical engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def validate_admission(value: dict) -> None:
    from rsna.competitive.compact_loss import ENGINE, validate_policy

    if (
        value.get("status") != "full_fitting_positive_weight_policy_CPU_admitted"
        or value.get("source_git") != "f5af4f238929652f9f482c3741ee09d799fa9cc2"
        or value.get("node") != "evc33"
        or value.get("fitting_count") != 2247
        or value.get("outer_count") != 1156
        or value.get("engine") != ENGINE
        or value.get("source_targets_masks_and_folds_unchanged") is not True
        or value.get("engineering16_uses_identical_full2247_weights") is not True
        or value.get("encoder_optimizer_or_GPU_run") is not False
        or value.get("images_decoded") != 0
    ):
        raise ValueError("Actual full-fitting positive-weight CPU admission required")
    validate_policy(value["policy"])
    if value["policy"]["fitting_count"] != 2247:
        raise ValueError("Full2247 loss policy required, never engineering-subset estimates")


def verify_source(source: Path, manifest: dict) -> None:
    """Match the immutable scientific archive, including any unexpected additions."""
    files = manifest.get("files", {})
    if any(p.is_symlink() for p in (source / "rsna").rglob("*")):
        raise ValueError("CPU-admitted scientific source contains a symlink")
    actual = {
        str(p.relative_to(source))
        for p in (source / "rsna").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if (
        manifest.get("source_git") != "f5af4f238929652f9f482c3741ee09d799fa9cc2"
        or manifest.get("archive_sha256")
        != "31be7ae74691682733b1055b01b3041b29d7e4ab147a2b05907632888b994b07"
        or not files
        or set(files) != actual
    ):
        raise ValueError("Exact actual CPU-admitted scientific source manifest required")
    for name, expected in files.items():
        p = source / name
        if (
            p.is_symlink()
            or not p.resolve().is_relative_to((source / "rsna").resolve())
            or hashlib.sha256(p.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("CPU-admitted scientific source changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "artifact-root",
        "cache",
        "overlay",
        "weights",
        "input-admission",
        "source-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-admission-sha256", required=True)
    args = parser.parse_args()

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    if digest(args.input_admission) != args.expected_admission_sha256:
        raise ValueError("Exact original actual input receipt required")
    admitted = json.loads(args.input_admission.read_text())
    if (
        Path(admitted["source"]) != args.source
        or args.cache != Path("/tmp/rsna-compact-native-evc33-relocation-20261016-v1/cache")
        or args.source != Path("/tmp/rsna-compact-positive-weight-inputs-20261016-v1/code")
        or args.output != args.source.parent / "recovery-v1"
        or os.uname().nodename != admitted["node"]
        or args.output.exists()
    ):
        raise ValueError("Original owning-node inputs and unique recovery output required")
    env = dict(
        os.environ,
        PYTHONPATH=str(args.source)
        + ":/tmp/rsna-compact-native-n3-assets-20261016-v1/runtime-overlay",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CPATH="/tmp/rsna-python312-devel-20261016-v1/usr/include/python3.12",
        TRITON_CACHE_DIR=str(args.output / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(args.output / "inductor"),
    )
    os.environ.update(env)
    sys.path[:0] = env["PYTHONPATH"].split(":")
    import torch
    from run_compact_bag_recovery import compare

    from rsna.competitive.checkpoint import atomic_json, file_sha256
    from rsna.competitive.compact_loss import ENGINE

    if (
        torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or (torch.cuda.get_device_name(0) != "NVIDIA H100 PCIe")
    ):
        raise ValueError("Exactly one assigned original-recipe BF16 H100 required")
    if torch.ones(8, device="cuda").sum().item() != 8:
        raise ValueError("Allocated CUDA arithmetic failed")
    verify_source(args.source, json.loads(args.source_manifest.read_text()))
    validate_admission(admitted)
    if (
        hashlib.sha256((args.input_admission.parent / "loss-policy.json").read_bytes()).hexdigest()
        != admitted["policy_sha256"]
    ):
        raise ValueError("Actual original full-fitting loss-policy bytes changed")
    args.output.mkdir(mode=0o700)
    base = [
        "--studies",
        str(args.artifact_root / "initiative/releases/report-supervision-v1/studies.parquet"),
        "--series",
        str(
            args.artifact_root
            / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
        ),
        "--folds",
        str(args.artifact_root / "initiative/experiments/competitive-v1/folds"),
        "--artifact-root",
        str(args.artifact_root),
        "--cache",
        str(args.cache),
        "--fitting-supervision",
        str(args.overlay),
        "--weights",
        str(args.weights),
        "--engineering",
        "--device",
        "cuda",
        "--loss-policy",
        "positive-weighted",
    ]
    records = []
    started = time.monotonic()
    for phase, directory, flags, expected in (
        ("full", "full", [], 0),
        ("stop1", "interrupted", ["--stop-after-updates", "1"], 75),
        ("resume", "interrupted", ["--resume"], 0),
    ):
        command = [
            sys.executable,
            "-B",
            "-m",
            "rsna.competitive.compact_train",
            *base,
            "--output",
            str(args.output / directory),
            *flags,
        ]
        with (
            (args.output / f"{phase}.stdout").open("x") as out,
            (args.output / f"{phase}.stderr").open("x") as err,
        ):
            result = subprocess.run(
                command, env=env, cwd=args.source, stdout=out, stderr=err, timeout=900
            )
        row = {
            "phase": phase,
            "returncode": result.returncode,
            "stdout_sha256": file_sha256(args.output / f"{phase}.stdout"),
            "stderr_sha256": file_sha256(args.output / f"{phase}.stderr"),
        }
        atomic_json(args.output / f"{phase}.raw-exit.json", row)
        records.append(row)
        if result.returncode != expected:
            raise ValueError("Actual numerical-engine phase failed; preserve original logs")
    comparison = compare(args.output / "full", args.output / "interrupted", args.weights)
    state = torch.load(args.output / "full/checkpoint.pt", map_location="cpu", weights_only=True)
    original = json.loads(
        Path(
            "/tmp/rsna-compact-native-n3-production-evc33-20261016-v1/production/run.json"
        ).read_text()
    )["binding"]
    if (
        state["binding"]["engine"] != ENGINE
        or state["binding"]["loss_policy"] != admitted["policy"]
        or state["binding"]["initial_head_sha256"] != original["initial_head_sha256"]
        or state["binding"]["initialization_sha256"] != original["initialization_sha256"]
    ):
        raise ValueError(
            "New arm does not bind exact full-fitting loss policy and matched initial learned state"
        )
    accepted = {
        "status": "actual_positive_weight_H100_recovery_accepted",
        "source_git": admitted["source_git"],
        "node": os.uname().nodename,
        "job_id": os.environ["SLURM_JOB_ID"],
        "GPU": torch.cuda.get_device_name(0),
        "input_admission_sha256": args.expected_admission_sha256,
        "loss_policy_sha256": admitted["policy_sha256"],
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "engine": ENGINE,
        "initial_head_and_official_encoder_match_original_N3": True,
        "phases": records,
        "comparison": comparison,
        "elapsed_seconds": time.monotonic() - started,
        "diagnostic_only": True,
        "clinical_accuracy_claim": False,
    }
    atomic_json(args.output / "accepted.json", accepted)
    print(json.dumps(accepted), flush=True)


if __name__ == "__main__":
    main()
