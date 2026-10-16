"""Bounded real-MRI full/stop1/resume acceptance for the unchanged compact trainer.

Prepared wrapper only; root dispatches once after an actual endpoint input receipt.
No competition mutations or production fit. Tiny metrics are never promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def exact(a, b) -> None:
    import torch

    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError("Decoded recovery tensor differs")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError("Decoded recovery dictionary differs")
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("Decoded recovery sequence differs")
        for left, right in zip(a, b, strict=True):
            exact(left, right)
    elif type(a) is not type(b) or a != b:
        raise ValueError("Decoded recovery scalar differs")


def compare(full: Path, resumed: Path, weights: Path) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from safetensors.torch import load_file

    from rsna.competitive.checkpoint import file_sha256
    from rsna.schema import TARGETS

    a = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
    b = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=True)
    exact(a, b)
    if (
        a["optimizer_steps"] != 4
        or a["epoch"] != 2
        or a["cursor"] != 0
        or a["scheduler"]["last_epoch"] != 4
        or a["binding"]["fitting_count"] != 16
        or a["binding"]["diagnostic_only"] is not True
        or len(a["history"]) != 4
        or any(
            row["studies"] != 8
            or not math.isfinite(row["gradient_norm"])
            or not math.isfinite(row["loss"])
            for row in a["history"]
        )
    ):
        raise ValueError("Unexpected engineering scope, update history or counters")
    original = load_file(str(weights))
    counters = {k: int(v) for k, v in a["model"].items() if k.endswith("num_batches_tracked")}
    if len(counters) != 25 or any(
        count != int(original[key.removeprefix("encoder.")]) + 32 for key, count in counters.items()
    ):
        raise ValueError("BatchNorm did not update exactly once per fitting bag")
    left, right = full / "outer_predictions.csv", resumed / "outer_predictions.csv"
    if left.read_bytes() != right.read_bytes():
        raise ValueError("Original final16x12 prediction CSV bytes differ")
    frame = pd.read_csv(left)
    values = frame[list(TARGETS)].to_numpy()
    roster_hash = hashlib.sha256(json.dumps(sorted(frame.study_uid)).encode()).hexdigest()
    if (
        list(frame) != ["study_uid", *TARGETS]
        or frame.shape != (16, 13)
        or frame.study_uid.duplicated().any()
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or roster_hash != a["binding"]["outer_uids_sha256"]
    ):
        raise ValueError("Engineering outer prediction roster/format differs")
    return {
        "complete_model_BN_AdamW_OneCycle_counters_sampler_allRNG_history_equal": True,
        "actual_optimizer_steps": 4,
        "fitting_bag_forwards": 32,
        "BN_counters_checked": 25,
        "exact_final16x12_CSV_equal": True,
        "prediction_sha256": file_sha256(left),
        "diagnostic_only": True,
        "clinical_accuracy_claim": False,
    }


def main() -> None:
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
    parser.add_argument("--remaining-seconds", type=int, default=4800)
    args = parser.parse_args()
    if args.output.exists() or not str(args.output.resolve()).startswith("/tmp/rsna-"):
        raise ValueError("Distinct owner-private node-local recovery output required")
    if (
        hashlib.sha256(args.input_admission.read_bytes()).hexdigest()
        != args.expected_admission_sha256
    ):
        raise ValueError("Actual endpoint input admission differs")
    admitted = json.loads(args.input_admission.read_text())
    if (
        admitted["status"] != "original_native_inputs_and_runtime_admitted_on_distinct_evc33"
        or admitted["fitting_count"] != 2247
        or admitted["outer_count"] != 1156
        or admitted["same_original32_input_tensor_sha256"]
        != "183097af0095317c7a85069e73c8b2c654e39cfc12cdbc7104ff3139a30bfe9b"
        or admitted["source_git"] != "dcfff319cb7bd7bbae679a20da2b1b9268e8d680"
        or Path(admitted["cache"]) != args.cache
        or Path(admitted["source"]) != args.source
        or os.uname().nodename != admitted["node"]
    ):
        raise ValueError("Actual endpoint source, native tensors or scope differs")
    args.output.mkdir(mode=0o700)
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

    from rsna.competitive.checkpoint import atomic_json, file_sha256

    if (
        torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or ("H100" not in torch.cuda.get_device_name(0))
    ):
        raise ValueError("One actually assigned BF16 H100 required")
    if torch.ones(8, device="cuda").sum().item() != 8:
        raise ValueError("Allocated CUDA arithmetic failed")
    proof = json.loads((args.input_admission.parent / "inputs/inputs.json").read_text())
    for name, digest in proof["binding"]["source_sha256"].items():
        if file_sha256(args.source / name) != digest:
            raise ValueError("Frozen CPU-admitted numerical source changed")
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
    ]
    deadline = time.monotonic() + args.remaining_seconds
    phases = []
    for phase, folder, extra, expected_code in (
        ("full", "full", [], 0),
        ("stop1", "interrupted", ["--stop-after-updates", "1"], 75),
        ("resume", "interrupted", ["--resume"], 0),
    ):
        telemetry = args.output / f"{phase}-runtime.json"
        child = (
            "import atexit,json,runpy,time,torch;from pathlib import Path;"
            f"p=Path({str(telemetry)!r});"
            "atexit.register(lambda:p.write_text(json.dumps({'cuda_peak_bytes':torch.cuda.max_memory_allocated(),"
            "'GPU':torch.cuda.get_device_name(0)})));"
            "runpy.run_module('rsna.competitive.compact_train',run_name='__main__')"
        )
        started = time.monotonic()
        with (
            (args.output / f"{phase}.stdout").open("x") as out,
            (args.output / f"{phase}.stderr").open("x") as err,
        ):
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    child,
                    *base,
                    "--output",
                    str(args.output / folder),
                    *extra,
                ],
                env=env,
                cwd=args.source,
                stdout=out,
                stderr=err,
                timeout=max(1, deadline - time.monotonic()),
            )
        row = {
            "phase": phase,
            "actual_integer_returncode": result.returncode,
            "expected_integer_returncode": expected_code,
            "elapsed_seconds": time.monotonic() - started,
            "stdout_sha256": file_sha256(args.output / f"{phase}.stdout"),
            "stderr_sha256": file_sha256(args.output / f"{phase}.stderr"),
        }
        phases.append(row)
        atomic_json(args.output / f"{phase}-raw-exit.json", row)
        if result.returncode != expected_code:
            raise ValueError(
                f"Original {phase} child returned {result.returncode}, expected {expected_code}"
            )
    acceptance = compare(args.output / "full", args.output / "interrupted", args.weights)
    peak = [
        json.loads((args.output / f"{p['phase']}-runtime.json").read_text())["cuda_peak_bytes"]
        for p in phases
    ]
    if not all(type(p) is int and p > 0 for p in peak):
        raise ValueError("Actual CUDA peak memory telemetry missing")
    atomic_json(
        args.output / "accepted.json",
        {
            **acceptance,
            "job_id": os.environ["SLURM_JOB_ID"],
            "node": os.uname().nodename,
            "source_git": admitted["source_git"],
            "input_admission_sha256": args.expected_admission_sha256,
            "GPU": torch.cuda.get_device_name(0),
            "phases": phases,
            "actual_cuda_peak_bytes": peak,
            "production_dispatched": False,
            "competition_submitted": False,
        },
    )
    print(
        json.dumps({"status": "actual_compact_trainer_GPU_recovery_accepted", **acceptance}),
        flush=True,
    )


if __name__ == "__main__":
    main()
