"""Finite private two-device diagnostic launcher; no competition submission."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(args):
    if not 0 < args.remaining_seconds <= 14400:
        raise ValueError("Four-hour diagnostic envelope required")
    args.output.mkdir(mode=0o700)
    workers = []
    handles = []
    started = time.monotonic()
    try:
        for index, role in enumerate(("control", "candidate")):
            command = [
                sys.executable,
                "-B",
                str(args.worker),
                "--control",
                str(args.control),
                "--cache",
                str(args.cache),
                "--cpu-acceptance",
                str(args.cpu_acceptance),
                "--cpu-acceptance-sha256",
                args.cpu_acceptance_sha256,
                "--encoder-spec",
                str(args.encoder_spec),
                "--output",
                str(args.output / role),
                "--role",
                role,
                "--device-index",
                str(index),
                "--remaining-seconds",
                str(args.remaining_seconds),
            ]
            out = (args.output / (role + ".stdout")).open("x")
            err = (args.output / (role + ".stderr")).open("x")
            handles.extend((out, err))
            env = os.environ.copy()
            env.update(
                PYTHONPATH=str(args.control / "code"),
                PYTHONNOUSERSITE="1",
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            # Inherit the notebook's bounded process group: grandchildren are
            # terminated as well if a role or the whole deadline fails.
            workers.append(subprocess.Popen(command, env=env, stdout=out, stderr=err))
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None, 0) for p in workers):
                raise ValueError("An actual T4 trainer role failed; original logs retained")
            if time.monotonic() - started >= args.remaining_seconds:
                raise TimeoutError("Private diagnostic whole-pipeline deadline expired")
            time.sleep(1)
        if any(p.returncode != 0 for p in workers):
            raise ValueError("Actual trainer recovery did not complete")
        results = {
            r: json.loads((args.output / r / "recovery-accepted.json").read_text())
            for r in ("control", "candidate")
        }
        if any(
            v["status"] != "actual_private_T4_patch_witness_canonical_trainer_recovery_accepted"
            or v["exact_complete_model_base_AdamW_LR_cursors_all_RNG_and_CSV_recovery"] is not True
            or v["optimizer_updates"] != 4
            or v["competition_candidate"] is not False
            for v in results.values()
        ):
            raise ValueError("Both exact actual four-update recovery receipts required")
        record = {
            "status": "actual_private_two_T4_patch_witness_canonical_trainer_recovery_accepted",
            "results": results,
            "CPU_receipt_sha256": args.cpu_acceptance_sha256,
            "elapsed_seconds": time.monotonic() - started,
            "diagnostic_only": True,
            "clinical_accuracy_evidence": False,
            "competition_submission_created": False,
        }
        (args.output / "recovery-accepted.json").write_text(
            json.dumps(record, sort_keys=True, indent=2) + "\n"
        )
        Path("/kaggle/working/witness-T4-recovery-accepted.json").write_text(
            json.dumps(record, sort_keys=True, indent=2) + "\n"
        )
        print(json.dumps(record), flush=True)
        return record
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
        for p in workers:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for handle in handles:
            handle.close()


def main():
    p = argparse.ArgumentParser()
    for name in ("control", "cache", "cpu-acceptance", "encoder-spec", "output", "worker"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cpu-acceptance-sha256", required=True)
    p.add_argument("--remaining-seconds", type=float, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
