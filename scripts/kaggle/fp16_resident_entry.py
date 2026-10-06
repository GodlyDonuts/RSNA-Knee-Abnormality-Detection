"""Separate offline FP16 performance-probe entry; no competition admission."""

from __future__ import annotations

import json
import time

if __package__:
    from . import chunked_infer
    from .resident_pool import ResidentPool
else:
    import chunked_infer
    from resident_pool import ResidentPool


def run(args, *, remaining_seconds=28800, pool_factory=ResidentPool):
    if args.precision != "fp16" or args.device != "cuda" or args.workers != 0:
        raise ValueError("Explicit CUDA FP16/no-data-worker performance probe required")
    if (
        args.baseline_epoch_one
        or args.allow_diagnostic
        or not args.resilient_series
        or not args.recover_slices
        or not args.isolate_study_errors
    ):
        raise ValueError("Completed original weights and unchanged MRI recovery required")
    if not 0 < remaining_seconds <= 28800:
        raise ValueError("Bounded eight-hour performance-probe envelope required")
    args.single_decode = True
    pool = pool_factory(workers=2, timeout_seconds=remaining_seconds, performance_fp16_probe=True)
    start = time.monotonic()
    try:
        result = chunked_infer.run(args, invoke=pool.invoke)
    finally:
        pool.close()
    result.update(
        resident_engine="two_device_FP16_performance_probe_v1",
        resident_engine_elapsed_seconds=time.monotonic() - start,
        hidden_test_runtime_validated=False,
        competition_submission_authorized=False,
        performance_probe_only=True,
        FP32_prediction_agreement_measured=False,
    )
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = chunked_infer.argument_parser()
    for action in parser._actions:
        if action.dest == "precision":
            action.choices = tuple(action.choices) + ("fp16",)
    parser.add_argument("--resident-remaining-seconds", type=float, default=28800)
    args = parser.parse_args()
    print(json.dumps(run(args, remaining_seconds=args.resident_remaining_seconds)))
