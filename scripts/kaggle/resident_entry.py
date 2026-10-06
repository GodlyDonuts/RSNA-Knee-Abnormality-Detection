"""Offline dual-device entry retaining the original cache and CSV contracts."""

from __future__ import annotations

import json
import time

if __package__:
    from . import chunked_infer
    from .resident_pool import ResidentPool
else:
    import chunked_infer
    from resident_pool import ResidentPool


def run(args, *, device_workers=2, remaining_seconds=28800, pool_factory=ResidentPool):
    if args.precision != "fp32" or args.workers != 0 or args.device != "cuda":
        raise ValueError("Initial offline resident entry requires CUDA/FP32/no data workers")
    if not args.resilient_series or not args.recover_slices or not args.isolate_study_errors:
        raise ValueError("Resident entry must retain complete MRI recovery handling")
    if args.baseline_epoch_one or args.allow_diagnostic:
        raise ValueError("This resident entry admits completed competition weights only")
    if device_workers != 2 or not 0 < remaining_seconds <= 28800:
        raise ValueError("Two-device bounded eight-hour resident envelope required")
    args.single_decode = True
    pool = pool_factory(workers=2, timeout_seconds=remaining_seconds)
    started = time.monotonic()
    try:
        result = chunked_infer.run(args, invoke=pool.invoke)
    finally:
        pool.close()
    result["resident_engine"] = "two_device_exact_FP32_v1"
    result["resident_engine_elapsed_seconds"] = time.monotonic() - started
    result["resident_cpu_threads_per_worker"] = 1
    result["resident_total_budget_seconds"] = remaining_seconds
    result["hidden_test_runtime_validated"] = False
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    return result


def main():
    parser = chunked_infer.argument_parser()
    parser.add_argument("--resident-remaining-seconds", type=float, default=28800)
    parser.add_argument("--resident-workers", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args,
                device_workers=args.resident_workers,
                remaining_seconds=args.resident_remaining_seconds,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
