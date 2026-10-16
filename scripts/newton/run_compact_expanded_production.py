"""Run the distinct2592-study fixed16 recipe after actual original H100 recovery.

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

SOURCE_GIT = "a1ca238526dad75b2a5cea46edba44fd95c71938"
SOURCE_MANIFEST_SHA = "bb550e783816611fc409a636e92495ff7524bd0f21553e66e19c8f65ba244ee9"
INPUT_SHA = "235538195accbac10369779324f10c3fbd7eea42981c95a7164144a17591c3de"
REQUIRED_FREE = (
    581021415363 + 2 * 2**30 + (2 + 16 + 64 + 32 + 256 + 32) * 2**20 + 4 * 2**30 + 2 * 2**30
)


def verify_acceptance(admission: dict, recovery: dict) -> None:
    from rsna.competitive.compact_expanded_train import validate_admission

    validate_admission(admission)
    if (
        recovery.get("status") != "actual_expanded2592_H100_recovery_accepted"
        or recovery.get("job_id") != "933382"
        or recovery.get("input_admission_sha256") != INPUT_SHA
        or recovery.get("source_git") != SOURCE_GIT
        or recovery.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA
        or recovery.get("node") != admission.get("node")
        or recovery.get("GPU") != "NVIDIA H100 PCIe"
        or recovery.get("source_admission_engineering_counts")
        != {"old_fitting": 8, "added_fitting": 8, "outer": 16}
        or recovery.get("initial_head_and_official_encoder_match_original_N3") is not True
        or recovery.get("full_fitting_count") != 2592
        or recovery.get("full_outer_count") != 1156
        or recovery.get("full_optimizer_updates") != 5184
        or recovery.get("diagnostic_only") is not True
        or recovery.get("clinical_accuracy_claim") is not False
    ):
        raise ValueError("Original actual2592 inputs and exact real-H100 recovery933382 required")
    c = recovery.get("comparison", {})
    if (
        c.get("complete_model_BN_AdamW_OneCycle_counters_sampler_allRNG_history_equal") is not True
        or c.get("exact_final16x12_CSV_equal") is not True
        or c.get("actual_optimizer_steps") != 4
        or c.get("BN_counters_checked") != 25
        or c.get("fitting_bag_forwards") != 32
        or c.get("clinical_accuracy_claim") is not False
    ):
        raise ValueError("Complete actual expanded model/BN/optimizer/RNG/CSV recovery required")
    phases = recovery.get("phases", [])
    if [x.get("phase") for x in phases] != ["full", "stop1", "resume"] or [
        x.get("returncode") for x in phases
    ] != [0, 75, 0]:
        raise ValueError("All original expanded numerical recovery phases required")


def verify_source(source: Path, manifest: dict) -> None:
    from rsna.competitive.checkpoint import file_sha256

    actual = {
        str(p.relative_to(source))
        for p in (source / "rsna").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if manifest.get("source_git") != SOURCE_GIT or actual != set(manifest.get("files", {})):
        raise ValueError("Complete original immutable expanded source membership required")
    for name, expected in manifest["files"].items():
        p = source / name
        if p.is_symlink() or not p.resolve().is_relative_to(source.resolve()):
            raise ValueError("Original regular source member required")
        if file_sha256(p) != expected:
            raise ValueError("Original numerical source changed")


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
        "source-manifest",
        "funding",
        "weights",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--input-admission-sha256", required=True)
    parser.add_argument("--recovery-sha256", required=True)
    parser.add_argument("--funding-sha256", required=True)
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
    if (
        os.uname().nodename != "evc33"
        or args.source != Path("/tmp/rsna-compact-expanded2592-engine-20261016-v1/code")
        or file_sha256(args.source_manifest) != SOURCE_MANIFEST_SHA
        or args.input_admission_sha256 != INPUT_SHA
        or file_sha256(args.funding) != args.funding_sha256
    ):
        raise ValueError("Exact original expanded source/receipt/funding endpoint required")
    funding = json.loads(args.funding.read_text())
    if (
        funding.get("required_free_bytes_including_all_liabilities") != REQUIRED_FREE
        or funding.get("production_and_publication_cap_bytes") != 16 * 2**30
        or funding.get("actual_recovery_sha256") != args.recovery_sha256
        or funding.get("source_git") != SOURCE_GIT
        or funding.get("deletion_credit_bytes") != 0
        or funding.get("new_shared_bytes_files") != 0
    ):
        raise ValueError("Exact newly reviewed recovery and retained production funding required")
    verify_source(args.source, json.loads(args.source_manifest.read_text()))
    if not str(args.output.resolve()).startswith("/tmp/rsna-"):
        raise ValueError("Owner-private node-local production output required")
    if not args.resume and args.output.exists():
        raise ValueError("Distinct fresh production output required")
    if args.resume and not (args.output / "production/checkpoint.pt").is_file():
        raise ValueError("Resume requires an actual original production checkpoint")
    if shutil.disk_usage("/tmp").free < REQUIRED_FREE:
        raise OSError("All current retained liabilities and reserve must remain funded")
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
            "source_git": SOURCE_GIT,
            "input_admission_sha256": args.input_admission_sha256,
            "actual_recovery_sha256": args.recovery_sha256,
            "funding_sha256": args.funding_sha256,
            "GPU": torch.cuda.get_device_name(0),
            "fixed_epochs": 16,
            "fixed_updates": 5184,
            "fitting_studies": 2592,
            "outer_studies": 1156,
            "fresh_ImageNet_initialization": not args.resume,
            "resume": args.resume,
            "passes_match_baseline_but_optimizer_compute_does_not": True,
            "new_shared_bytes_files": 0,
            "production_and_preservation_cap_bytes": 16 * 2**30,
        }
        atomic_json(invocation / "start.json", start)
        command = [
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
            "--output",
            str(args.output / "production"),
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
                or result["optimizer_steps"] != 5184
                or result["diagnostic_only"] is not False
            ):
                raise ValueError("Fixed-final production scope/count differs")
            terminal.update(
                status="fixed_final_expanded2592_production_complete",
                result_sha256=file_sha256(args.output / "production/result.json"),
            )
        elif code == 75:
            state = torch.load(
                args.output / "production/checkpoint.pt", map_location="cpu", weights_only=True
            )
            if not state["optimizer_boundary"] or not 0 < state["optimizer_steps"] < 5184:
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
