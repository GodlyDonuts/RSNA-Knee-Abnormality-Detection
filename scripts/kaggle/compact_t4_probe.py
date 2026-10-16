"""Offline two-T4 compact inference engineering probe; never writes submission.csv.

Uses a source-bound diagnostic checkpoint and private unlabeled MRI window
fixture. No optimizer, training, labels, competition entry, or accuracy metric.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from rsna.competitive.compact_bag import candidate_windows
from rsna.competitive.compact_infer import load_model
from rsna.competitive.compact_mc import predict_both


def validate_fixture(fixture: dict) -> int:
    if set(fixture) != {"windows", "series_mask", "slice_mask", "study_uid"}:
        raise ValueError("Only unlabeled normalized windows and synthetic identity allowed")
    if fixture["study_uid"] != "engineering-fixture-v1":
        raise ValueError("Fixture cannot contain original case identifiers")
    windows = fixture["windows"]
    count = len(candidate_windows(fixture["series_mask"], fixture["slice_mask"]))
    if (
        windows.shape != (count, 3, 384, 384)
        or windows.dtype != torch.float32
        or not torch.isfinite(windows).all()
        or windows.min() < -1
        or windows.max() > 1
    ):
        raise ValueError("Exact normalized native384 fixture required")
    return count


def compare(actual, expected, tolerance: float) -> float:
    a = torch.as_tensor(actual, dtype=torch.float32).cpu()
    b = torch.as_tensor(expected, dtype=torch.float32).cpu()
    if (
        a.shape != (12,)
        or b.shape != (12,)
        or not torch.isfinite(a).all()
        or not torch.isfinite(b).all()
    ):
        raise ValueError("Finite twelve-finding probabilities required")
    if a.min() < 0 or a.max() > 1 or b.min() < 0 or b.max() > 1:
        raise ValueError("Probability bounds required")
    error = float((a - b).abs().max())
    if error > tolerance:
        raise ValueError("Target precision exceeds predeclared probability tolerance")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.name == "submission.csv" or args.output.exists():
        raise ValueError("Distinct diagnostic receipt only")
    torch.set_num_threads(2)
    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(i) for i in range(2)
    ):
        raise ValueError("Actual two-T4 target required")
    fixture = torch.load(args.bundle / "fixture.pt", map_location="cpu", weights_only=True)
    count = validate_fixture(fixture)
    reference = json.loads((args.bundle / "reference.json").read_text())
    from rsna.competitive.checkpoint import file_sha256

    if file_sha256(args.bundle / "fixture.pt") != reference["fixture_sha256"]:
        raise ValueError("Original fixture bytes changed")
    devices = []
    for index in range(2):
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
        model, policy = load_model(
            args.bundle / "checkpoint.pt",
            reference["checkpoint_sha256"],
            device=device,
            precision="float32",
            allow_diagnostic=True,
        )
        if policy["diagnostic_only"] is not True or policy["competition_eligible"] is not False:
            raise ValueError("Only original engineering weights qualify this probe")
        windows = fixture["windows"].to(device)
        torch.cuda.reset_peak_memory_stats(index)
        outcomes = {}
        for precision in ["float32", "float16"]:
            model.precision = precision
            started = time.monotonic()
            value = predict_both(
                model,
                windows,
                fixture["series_mask"],
                fixture["slice_mask"],
                study_uid=fixture["study_uid"],
                chunk_size=12,
            )
            torch.cuda.synchronize(index)
            tolerance = 0.0003 if precision == "float32" else 0.003
            outcomes[precision] = {
                "full_max_error_vs_original_CPU": compare(
                    value["full"], reference["full"], tolerance
                ),
                "MC128_max_error_vs_original_CPU": compare(
                    value["mc128"], reference["mc128"], tolerance
                ),
                "elapsed_seconds": time.monotonic() - started,
            }
        model.precision = "float16"
        # Worst allowed window count; repeated fixture windows are timing inputs,
        # never additional studies or evidence of accuracy.
        timing_windows = windows.repeat((84 + count - 1) // count, 1, 1, 1)[:84]
        timings = []
        with torch.inference_mode():
            for repeat in range(5):
                torch.cuda.synchronize(index)
                started = time.monotonic()
                logits = model(timing_windows, chunk_size=12)
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite timing output")
                torch.cuda.synchronize(index)
                elapsed = time.monotonic() - started
                if repeat:
                    timings.append(elapsed)
        peak = torch.cuda.max_memory_allocated(index)
        if peak > 15 * 2**30:
            raise ValueError("Insufficient T4 memory margin")
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "precision_checks": outcomes,
                "FP16_84_window_seconds": timings,
                "FP16_84_window_median_seconds": statistics.median(timings),
                "peak_allocated_bytes": peak,
            }
        )
        del logits, timing_windows, windows, model, value
        torch.cuda.empty_cache()
    receipt = {
        "status": "actual_two_T4_compact_precision_and_encoder_timing_accepted",
        "diagnostic_only": True,
        "competition_eligible": False,
        "accuracy_claim": False,
        "hidden_scale_end_to_end_runtime_qualified": False,
        "fixture_windows": count,
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "fixture_sha256": reference["fixture_sha256"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": devices,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
