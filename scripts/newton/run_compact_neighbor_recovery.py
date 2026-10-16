"""Real H100 recovery for the separately bound matched-center donor contrast.

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
    if (
        value.get("status") != "matched_source_neighbor_inputs_admitted"
        or value.get("source_git") != "3803b717b69db6c5e89e11724c4daee15490cbe2"
        or value.get("node") != "evc33"
        or value.get("fitting_count") != 2247
        or value.get("outer_count") != 1156
        or value.get("center_channel_bit_equal") is not True
        or value.get("original_fitting_outer_release_masks_unchanged") is not True
        or value.get("donor_windows_changed", 0) <= 0
        or value.get("GPU_used") is not False
    ):
        raise ValueError("Actual matched-source-neighbor native input admission required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "artifact-root",
        "cache",
        "overlay",
        "weights",
        "input-admission",
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
    validate_admission(admitted)
    if (
        Path(admitted["source"]) != args.source
        or Path(admitted["cache"]) != args.cache
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
    from rsna.competitive.compact_source_neighbors import ENGINE, POLICY

    if (
        torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or (torch.cuda.get_device_name(0) != "NVIDIA H100 PCIe")
    ):
        raise ValueError("Exactly one assigned original-recipe BF16 H100 required")
    if torch.ones(8, device="cuda").sum().item() != 8:
        raise ValueError("Allocated CUDA arithmetic failed")
    proof_path = args.input_admission.parent / "inputs/inputs.json"
    if file_sha256(proof_path) != admitted["input_receipt_sha256"]:
        raise ValueError("Original CPU tensor/source proof changed")
    proof = json.loads(proof_path.read_text())
    if admitted["input_policy"] != POLICY or proof["binding"]["input_policy"] != POLICY:
        raise ValueError("Source-donor policy differs")
    for name, expected in proof["binding"]["source_sha256"].items():
        if file_sha256(args.source / name) != expected:
            raise ValueError("CPU-admitted numerical source changed")
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
        "--input-policy",
        "matched-source-neighbors",
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
        or state["binding"]["input_policy"] != POLICY
        or state["binding"]["initial_head_sha256"] != original["initial_head_sha256"]
        or state["binding"]["initialization_sha256"] != original["initialization_sha256"]
    ):
        raise ValueError("New arm does not bind exact policy and matched initial learned state")
    accepted = {
        "status": "actual_matched_source_neighbor_H100_recovery_accepted",
        "source_git": admitted["source_git"],
        "node": os.uname().nodename,
        "job_id": os.environ["SLURM_JOB_ID"],
        "GPU": torch.cuda.get_device_name(0),
        "input_admission_sha256": args.expected_admission_sha256,
        "input_policy": POLICY,
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
