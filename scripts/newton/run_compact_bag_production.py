"""Run the fixed16 compact recipe after actual input and real-trainer acceptance.

This wrapper reserves no hardware and publishes nothing. Root submits it once.
Signals request a complete optimizer-boundary stop in the unchanged trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def verify_acceptance(admission: dict, recovery: dict) -> None:
    if (
        admission.get("status") != "original_native_inputs_and_runtime_admitted_on_distinct_evc33"
        or admission.get("source_git") != "dcfff319cb7bd7bbae679a20da2b1b9268e8d680"
        or admission.get("fitting_count") != 2247
        or admission.get("outer_count") != 1156
        or admission.get("same_original32_input_tensor_sha256")
        != "183097af0095317c7a85069e73c8b2c654e39cfc12cdbc7104ff3139a30bfe9b"
        or recovery.get("complete_model_BN_AdamW_OneCycle_counters_sampler_allRNG_history_equal")
        is not True
        or recovery.get("exact_final16x12_CSV_equal") is not True
        or recovery.get("actual_optimizer_steps") != 4
        or recovery.get("BN_counters_checked") != 25
        or recovery.get("fitting_bag_forwards") != 32
        or recovery.get("diagnostic_only") is not True
        or recovery.get("source_git") != admission.get("source_git")
        or recovery.get("node") != admission.get("node")
        or not recovery.get("actual_cuda_peak_bytes")
        or len(recovery["actual_cuda_peak_bytes"]) != 3
        or not all(type(v) is int and v > 0 for v in recovery["actual_cuda_peak_bytes"])
    ):
        raise ValueError("Actual original input/complete real-MRI trainer recovery required")
    phases = recovery.get("phases", [])
    if [p.get("phase") for p in phases] != ["full", "stop1", "resume"] or [
        p.get("actual_integer_returncode") for p in phases
    ] != [0, 75, 0]:
        raise ValueError("All three actual recovery child outcomes are required")


def forward_signal(process, signum: int) -> None:
    """The child creates its own process group; never signal unrelated processes."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass  # The owned child exited between poll and signal delivery.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "input-admission",
        "recovery",
        "source",
        "artifact-root",
        "cache",
        "overlay",
        "weights",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--input-admission-sha256", required=True)
    parser.add_argument("--recovery-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [
        str(args.source),
        "/tmp/rsna-compact-native-n3-assets-20261016-v1/runtime-overlay",
    ]
    from rsna.competitive.checkpoint import atomic_json, file_sha256, run_lock

    if (
        file_sha256(args.input_admission) != args.input_admission_sha256
        or file_sha256(args.recovery) != args.recovery_sha256
    ):
        raise ValueError("Actual acceptance receipt bytes differ")
    admitted = json.loads(args.input_admission.read_text())
    recovered = json.loads(args.recovery.read_text())
    verify_acceptance(admitted, recovered)
    if recovered["input_admission_sha256"] != args.input_admission_sha256:
        raise ValueError("Recovery did not use this exact input admission")
    if os.uname().nodename != admitted["node"] or (
        Path(admitted["cache"]) != args.cache or Path(admitted["source"]) != args.source
    ):
        raise ValueError("Production endpoint differs from the accepted original")
    proof = json.loads((args.input_admission.parent / "inputs/inputs.json").read_text())
    for name, expected in proof["binding"]["source_sha256"].items():
        if file_sha256(args.source / name) != expected:
            raise ValueError("Accepted immutable numerical source changed")
    if not str(args.output.resolve()).startswith("/tmp/rsna-"):
        raise ValueError("Owner-private node-local production output required")
    if not args.resume and args.output.exists():
        raise ValueError("Distinct fresh production output required")
    if args.resume and not (args.output / "production/checkpoint.pt").is_file():
        raise ValueError("Resume requires an actual original production checkpoint")
    if shutil.disk_usage("/tmp").free < 396100843459 + (20 + 24) * 2**30:
        raise OSError("Original liabilities, full production and reserve must remain funded")
    args.output.mkdir(mode=0o700, exist_ok=args.resume)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(sys.path[:2]),
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
    import torch

    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("One actually allocated BF16 CUDA device required")
    if torch.cuda.get_device_name(0) != recovered["GPU"]:
        raise ValueError("Production GPU differs from the actual accepted recovery device")
    if torch.ones(8, device="cuda").sum().item() != 8:
        raise ValueError("Actual allocated CUDA arithmetic failed")
    job = os.environ["SLURM_JOB_ID"]
    with run_lock(args.output / "wrapper-lock"):
        active = args.output / "active-job.json"
        if args.resume:
            old = json.loads(active.read_text())["job_id"]
            states = subprocess.check_output(
                ["sacct", "-X", "-j", old, "-n", "--format=State"], text=True
            ).split()
            if (
                old == job
                or not states
                or any(
                    not s.startswith(
                        (
                            "COMPLETED",
                            "CANCELLED",
                            "FAILED",
                            "TIMEOUT",
                            "OUT_OF_MEMORY",
                            "NODE_FAIL",
                            "PREEMPTED",
                        )
                    )
                    for s in states
                )
            ):
                raise ValueError("Resume requires actual terminal state of the original job")
        invocation = args.output / "invocations" / job
        invocation.mkdir(parents=True, mode=0o700)
        atomic_json(active, {"job_id": job, "resume": args.resume})
        start = {
            "job_id": job,
            "source_git": admitted["source_git"],
            "input_admission_sha256": args.input_admission_sha256,
            "actual_recovery_sha256": args.recovery_sha256,
            "GPU": torch.cuda.get_device_name(0),
            "fixed_epochs": 16,
            "fixed_updates": 4496,
            "fitting_studies": 2247,
            "outer_studies": 1156,
            "fresh_ImageNet_initialization": not args.resume,
            "resume": args.resume,
            "new_shared_bytes_files": 0,
            "production_and_preservation_cap_bytes": 24 * 2**30,
        }
        atomic_json(invocation / "start.json", start)
        command = [
            sys.executable,
            "-B",
            "-m",
            "rsna.competitive.compact_train",
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
            "--output",
            str(args.output / "production"),
            "--device",
            "cuda",
        ]
        if args.resume:
            command.append("--resume")
        started = time.monotonic()
        with (invocation / "stdout").open("x") as out, (invocation / "stderr").open("x") as err:
            process = subprocess.Popen(
                command, env=env, cwd=args.source, stdout=out, stderr=err, start_new_session=True
            )
            for sig in (signal.SIGUSR1, signal.SIGTERM):
                signal.signal(sig, lambda signum, frame: forward_signal(process, signum))
            code = process.wait()
        terminal = {
            **start,
            "actual_integer_child_returncode": code,
            "elapsed_seconds": time.monotonic() - started,
            "stdout_sha256": file_sha256(invocation / "stdout"),
            "stderr_sha256": file_sha256(invocation / "stderr"),
        }
        if code == 0:
            result = json.loads((args.output / "production/result.json").read_text())
            if (
                result["status"] != "fixed_final_fit_complete"
                or result["optimizer_steps"] != 4496
                or result["diagnostic_only"] is not False
            ):
                raise ValueError("Fixed-final production scope/count differs")
            terminal.update(
                status="fixed_final_compact_production_complete",
                result_sha256=file_sha256(args.output / "production/result.json"),
            )
        elif code == 75:
            state = torch.load(
                args.output / "production/checkpoint.pt", map_location="cpu", weights_only=True
            )
            if not state["optimizer_boundary"] or not 0 < state["optimizer_steps"] < 4496:
                raise ValueError("Stop did not preserve a complete original production boundary")
            terminal.update(
                status="stopped_complete_optimizer_boundary",
                optimizer_steps=state["optimizer_steps"],
            )
        else:
            terminal["status"] = "original_production_child_failed"
        atomic_json(invocation / "terminal.json", terminal)
        print(json.dumps(terminal), flush=True)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
