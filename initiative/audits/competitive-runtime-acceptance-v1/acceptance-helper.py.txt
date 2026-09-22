"""Replay real-release training, interrupted recovery and optional public examples.

Run from the repository root. No credentials, absolute cache paths, reports or
external URLs are copied into the portable acceptance receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def equal(first, second) -> bool:
    if type(first) is not type(second):
        return False
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(equal(a, b) for a, b in zip(first, second))
    return first == second


def run_child(command: list[str], handle, timeout: int = 600) -> int:
    """Bound one diagnostic command and reap its whole owned process group."""
    process = subprocess.Popen(
        command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
    )

    def interrupted(signum, _frame):
        raise InterruptedError(f"Diagnostic acceptance interrupted by signal {signum}")

    previous = {
        number: signal.signal(number, interrupted)
        for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)
    }
    try:
        return process.wait(timeout=timeout)
    finally:
        # Repeated scheduler signals must not interrupt cleanup. A launcher may
        # exit normally before a descendant, so clean the group on every path.
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        try:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/competitive/cpu-acceptance.json")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-public-examples", action="store_true")
    args = parser.parse_args()
    config = args.config
    config_values = json.loads(config.read_text())
    if config_values.get("diagnostic_study_limit") is None or config_values.get("epochs") != 2:
        raise ValueError("Acceptance requires an explicit diagnostic subset and two epochs")
    if (
        not args.skip_public_examples
        and config_values["encoder"]["architecture"] != "diagnostic_tiny"
    ):
        raise ValueError("Public examples in this helper require the tiny diagnostic encoder")
    args.output.mkdir(parents=True, exist_ok=False)
    series = Path("initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv")
    started = time.monotonic()
    commands = []

    def execute(label: str, command: list[str], expected: int) -> None:
        if shutil.disk_usage(args.output).free < 20 * 1024**3:
            raise OSError("Local disk reserve would be violated")
        before = time.monotonic()
        with (args.output / f"{label}.log").open("w") as handle:
            returncode = run_child(command, handle)
        record = {
            "label": label,
            "returncode": returncode,
            "expected_returncode": expected,
            "elapsed_seconds": time.monotonic() - before,
        }
        commands.append(record)
        print(json.dumps(record), flush=True)
        if returncode != expected:
            raise RuntimeError(f"Acceptance command failed: {label}; inspect local log")

    training = [sys.executable]
    if args.world_size == 2:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        training += [
            "-m",
            "torch.distributed.run",
            "--rdzv-backend",
            "static",
            "--master-addr",
            "127.0.0.1",
            "--master-port",
            str(port),
            "--nproc-per-node",
            "2",
            "--max-restarts",
            "0",
        ]
    training += [
        "-m",
        "rsna.competitive.train",
        "--config",
        str(config),
        "--studies",
        "initiative/releases/report-supervision-v1/studies.parquet",
        "--series",
        str(series),
        "--cache",
        str(args.cache),
        "--folds",
        "initiative/experiments/competitive-v1/folds",
        "--artifact-root",
        ".",
        "--device",
        args.device,
    ]
    continuous, resumed = args.output / "continuous", args.output / "resumed"
    execute("continuous", training + ["--output", str(continuous)], 0)
    # torchrun wraps a worker's deliberate75 exit as a ChildFailedError/exit1.
    # Check the actual committed checkpoint and recorded worker exit, too.
    execute(
        "interrupted",
        training + ["--output", str(resumed), "--stop-after-updates", "1"],
        75 if args.world_size == 1 else 1,
    )
    invocation = json.loads((resumed / "invocation.json").read_text())
    if invocation.get("exit_code") != 75:
        raise ValueError("Interrupted worker did not report deliberate recovery exit75")
    partial = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=True)
    if partial["epoch"] != 0 or partial["next_update"] != 1 or not partial["optimizer_boundary"]:
        raise ValueError("Interruption did not occur at the requested within-epoch boundary")
    shutil.copy2(resumed / "checkpoint.pt", args.output / "interrupted-checkpoint.pt")
    execute(
        "resumed",
        training + ["--output", str(resumed), "--resume", str(resumed / "checkpoint.pt")],
        0,
    )
    left = torch.load(continuous / "checkpoint.pt", map_location="cpu", weights_only=True)
    right = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=True)
    if not equal(left, right):
        raise ValueError("Continuous/resumed checkpoint trees differ")
    for directory in (
        (continuous, resumed) if config_values.get("submission_every_epochs") == 2 else ()
    ):
        snapshot = directory / "epoch-checkpoints" / "epoch-002.pt"
        metadata = json.loads(snapshot.with_suffix(".json").read_text())
        if not equal(torch.load(snapshot, map_location="cpu", weights_only=True), left):
            raise ValueError("Scheduled epoch snapshot differs from the completed checkpoint")
        if metadata["submission_eligible"] or not metadata["diagnostic_only"]:
            raise ValueError("Diagnostic snapshot was incorrectly admitted for submission")
    if (continuous / "outer_predictions.csv").read_bytes() != (
        resumed / "outer_predictions.csv"
    ).read_bytes():
        raise ValueError("Continuous/resumed outer predictions differ")
    if not args.skip_public_examples:
        execute(
            "public-examples",
            [
                sys.executable,
                "-m",
                "rsna.competitive.infer",
                "--checkpoint",
                str(continuous / "checkpoint.pt"),
                "--encoder-spec",
                "configs/competitive/diagnostic-encoder.json",
                "--series",
                "initiative/source/test_series.csv",
                "--template",
                "initiative/source/sample_submission.csv",
                "--cache",
                str(args.cache),
                "--device",
                "cpu",
                "--allow-diagnostic",
                "--output",
                str(args.output / "submission.csv"),
            ],
            0,
        )
    result = json.loads((continuous / "result.json").read_text())
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "author_agent": "/root",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "real accepted release, explicit diagnostic subset, no competitive performance claim",
        "world_size": args.world_size,
        "device_type": args.device,
        "encoder_architecture": config_values["encoder"]["architecture"],
        "commands": commands,
        "elapsed_seconds": time.monotonic() - started,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_binding": left["binding"],
        "full_checkpoint_tree_exact_after_mid_epoch_resume": True,
        "outer_prediction_bytes_exact_after_resume": True,
        "scheduled_epoch_snapshot_exact_and_submission_ineligible": (
            True if config_values.get("submission_every_epochs") == 2 else None
        ),
        "training_studies": result["training_studies"],
        "outer_studies": result["outer_studies"],
        "optimizer_steps": result["optimizer_steps"],
        "public_example_predictions": 0 if args.skip_public_examples else 3,
        "gold_or_reserved_confirmation_predictions": 0,
        "competitive_score_established": False,
        "gpu_diagnostic_performed": args.device == "cuda",
        "remaining_free_bytes": shutil.disk_usage(args.output).free,
    }
    (args.output / "acceptance.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "world_size": args.world_size,
                "elapsed_seconds": receipt["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
