"""Run a finite, prebound sequence of independent original-source cache shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path("/home/sa305415/rsna-knee")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def validate_lane(plan, lane):
    if lane not in (0, 1) or plan.get("shards_by_lane") != {
        "0": list(range(1, 75, 2)),
        "1": list(range(2, 75, 2)),
    }:
        raise ValueError("exact_two_lane_cohort_partition_required")
    units = plan["units"]
    if set(units) != {str(i) for i in range(1, 75)}:
        raise ValueError("all_remaining_unique_shard_controls_required")
    controls = [u["control"] for u in units.values()]
    if len(set(controls)) != 74:
        raise ValueError("distinct_shard_controls_required")
    return plan["shards_by_lane"][str(lane)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--lane", type=int, required=True)
    args = p.parse_args()
    root = args.control
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job or os.environ.get("SLURM_CPUS_PER_TASK") != "1" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("one_cpu_zero_gpu_required")
    if (
        root.is_symlink()
        or not root.resolve().is_relative_to(BASE.resolve())
        or root.resolve() == BASE.resolve()
    ):
        raise ValueError("private_batch_control_required")
    plan = json.loads((root / "batch-plan.json").read_text())
    shards = validate_lane(plan, args.lane)
    if digest(Path(__file__)) != plan["lane_driver_sha256"]:
        raise ValueError("pinned_lane_driver_required")
    lane = root / f"lane-{args.lane}"
    lane.mkdir(mode=0o700, exist_ok=True)
    if (lane / "run-start.json").exists():
        raise FileExistsError("existing_lane_requires_readonly_reconciliation")
    for unit in plan["units"].values():
        control = Path(unit["control"])
        if (
            control.is_symlink()
            or not control.resolve().is_relative_to(root.resolve())
            or digest(control / "launch.json") != unit["launch_sha256"]
        ):
            raise ValueError("prebound_unit_control_changed")
    payload = json.load(sys.stdin)
    if (
        set(payload) != {"token"}
        or not isinstance(payload["token"], str)
        or not payload["token"].startswith("hf_")
    ):
        raise ValueError("private_credential_stdin_required")
    save(
        lane / "run-start.json",
        {
            "job_id": job,
            "lane": args.lane,
            "source_git": plan["source_git"],
            "batch_plan_sha256": digest(root / "batch-plan.json"),
            "cpus": 1,
            "gpus": 0,
            "shards": shards,
            "automatic_retry": False,
        },
    )
    current = None
    try:
        for current in shards:
            unit = plan["units"][str(current)]
            control = Path(unit["control"])
            launch = json.loads((control / "launch.json").read_text())
            runner = Path(launch["files"]["runner.py"].get("path", str(control / "runner.py")))
            if (
                launch["shard"] != current
                or runner.is_symlink()
                or not runner.resolve().is_relative_to(root.resolve())
                or digest(runner) != launch["files"]["runner.py"]["sha256"]
            ):
                raise ValueError("unit_runner_or_shard_changed")
            print(
                json.dumps({"lane": args.lane, "shard_start": current, "job_id": job}), flush=True
            )
            subprocess.run(
                ["/usr/bin/python3.12", str(runner), "--control", str(control)],
                input=json.dumps(payload),
                text=True,
                check=True,
            )
            result = json.loads((control / "result.json").read_text())
            if (
                result.get("status")
                != "original_source_paired_shard_and_private_preservation_complete"
                or result.get("shard") != current
                or result.get("job_id") != job
                or result.get("scratch_reclaimed_after_full_private_verification") is not True
            ):
                raise ValueError("exact_completed_shard_result_required")
            save(lane / f"shard-{current:03d}-complete.json", result)
        save(
            lane / "complete.json",
            {
                "status": "all_assigned_shards_privately_preserved",
                "job_id": job,
                "lane": args.lane,
                "shards": shards,
                "full_cohort_complete": False,
            },
        )
    except BaseException as error:
        save(
            lane / "failure.json",
            {
                "job_id": job,
                "lane": args.lane,
                "shard": current,
                "error_type": type(error).__name__,
                "automatic_retry": False,
                "completed_shards_retained": True,
            },
        )
        raise
    finally:
        payload.clear()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(
            json.dumps(
                {
                    "status": "native_cache_lane_failed",
                    "error_type": type(error).__name__,
                    "automatic_retry": False,
                }
            ),
            flush=True,
        )
        raise SystemExit(2) from None
