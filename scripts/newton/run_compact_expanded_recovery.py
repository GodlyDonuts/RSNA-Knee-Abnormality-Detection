"""One real-H100 full/stop1/resume check for the admitted2592-population engine.

Uses eight old plus eight added fitting studies and16 unchanged outer engineering
studies. No tiny diagnostic metric can select a recipe or become a submission.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "source-manifest",
        "artifact-root",
        "input-admission",
        "weights",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    if (
        os.uname().nodename != "evc33"
        or args.output.exists()
        or sha(args.source_manifest) != args.source_manifest_sha256
    ):
        raise ValueError(
            "Exact original owning-node source manifest and fresh recovery output required"
        )
    manifest = json.loads(args.source_manifest.read_text())
    actual = {
        str(p.relative_to(args.source))
        for p in (args.source / "rsna").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if actual != set(manifest["files"]) or any(
        sha(args.source / n) != h for n, h in manifest["files"].items()
    ):
        raise ValueError("Exact complete frozen numerical source required")
    env = dict(
        os.environ,
        PYTHONPATH=str(args.source)
        + ":/tmp/rsna-compact-native-n3-assets-20261016-v1/runtime-overlay",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
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

    from rsna.competitive.checkpoint import atomic_json
    from rsna.competitive.compact_expanded_train import ADMISSION_SHA, validate_admission

    if sha(args.input_admission) != ADMISSION_SHA:
        raise ValueError("Original actual CPU input receipt differs")
    admitted = json.loads(args.input_admission.read_text())
    validate_admission(admitted)
    if (
        torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != "NVIDIA H100 PCIe"
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Exactly one assigned original-recipe BF16 H100 required")
    if torch.ones(8, device="cuda").sum().item() != 8:
        raise ValueError("Actual assigned CUDA arithmetic failed")
    args.output.mkdir(mode=0o700)
    header = {
        "GPU": torch.cuda.get_device_name(0),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "node": os.uname().nodename,
        "job_id": os.environ["SLURM_JOB_ID"],
        "input_admission_sha256": ADMISSION_SHA,
        "source_git": manifest["source_git"],
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_admission_engineering_counts": {"old_fitting": 8, "added_fitting": 8, "outer": 16},
    }
    atomic_json(args.output / "header.json", header)
    base = [
        sys.executable,
        "-B",
        "-m",
        "rsna.competitive.compact_expanded_train",
        "--artifact-root",
        str(args.artifact_root),
        "--input-admission",
        str(args.input_admission),
        "--weights",
        str(args.weights),
        "--engineering",
    ]
    records = []
    started = time.monotonic()
    for phase, directory, flags, expected in [
        ("full", "full", [], 0),
        ("stop1", "interrupted", ["--stop-after-updates", "1"], 75),
        ("resume", "interrupted", ["--resume"], 0),
    ]:
        with (
            (args.output / (phase + ".stdout")).open("x") as out,
            (args.output / (phase + ".stderr")).open("x") as err,
        ):
            result = subprocess.run(
                [*base, "--output", str(args.output / directory), *flags],
                env=env,
                cwd=args.source,
                stdout=out,
                stderr=err,
                timeout=1050,
            )
        row = {
            "phase": phase,
            "returncode": result.returncode,
            "stdout_sha256": sha(args.output / (phase + ".stdout")),
            "stderr_sha256": sha(args.output / (phase + ".stderr")),
        }
        atomic_json(args.output / (phase + ".raw-exit.json"), row)
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
        state["binding"]["initial_head_sha256"] != original["initial_head_sha256"]
        or state["binding"]["initialization_sha256"] != original["initialization_sha256"]
        or state["binding"]["fitting_population"]["population"] != admitted["population"]
    ):
        raise ValueError("Expanded population or exact matched initial model differs")
    accepted = {
        "status": "actual_expanded2592_H100_recovery_accepted",
        **header,
        "comparison": comparison,
        "phases": records,
        "initial_head_and_official_encoder_match_original_N3": True,
        "elapsed_seconds": time.monotonic() - started,
        "full_fitting_count": 2592,
        "full_outer_count": 1156,
        "full_optimizer_updates": 5184,
        "diagnostic_only": True,
        "clinical_accuracy_claim": False,
    }
    atomic_json(args.output / "accepted.json", accepted)
    print(json.dumps(accepted), flush=True)


if __name__ == "__main__":
    main()
