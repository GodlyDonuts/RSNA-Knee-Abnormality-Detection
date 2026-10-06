"""Private fitting-image throughput/equivalence probe, never a competition entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import chunked_infer
import numpy as np
import pandas as pd
import torch
from resident_pool import ResidentPool

from rsna.schema import TARGETS


def main():
    parser = argparse.ArgumentParser()
    for name in ("bundle", "encoder-spec", "competition", "roster", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--study-count", type=int, choices=(32, 1300), default=32)
    args = parser.parse_args()
    started = time.monotonic()
    if torch.cuda.device_count() != 2 or any(
        "T4" not in torch.cuda.get_device_name(i) for i in range(2)
    ):
        raise ValueError("Actual two-T4 performance execution required")
    roster = json.loads(args.roster.read_text())
    if len(roster) != args.study_count or len(set(roster)) != args.study_count:
        raise ValueError("Exact admitted fitting-only roster required")
    series = pd.read_csv(args.competition / "train_series.csv")
    if not set(roster) <= set(series.StudyInstanceUID):
        raise ValueError("Admitted fitting roster missing from original MRI source")
    args.output.mkdir(mode=0o700)
    source = args.output / "source"
    (source / "test_series").mkdir(parents=True)
    # Read original mounted bytes through a private alias; no MRI bytes copied.
    for uid in roster:
        (source / "test_series" / uid).symlink_to(
            args.competition / "train_series" / uid, target_is_directory=True
        )
    subset = series.loc[series.StudyInstanceUID.isin(roster)].copy()
    sp = args.output / "series.csv"
    subset.to_csv(sp, index=False)
    template = pd.DataFrame({"StudyInstanceUID": roster})
    for target in TARGETS:
        template[target] = 0.5
    tp = args.output / "roster.csv"
    template.to_csv(tp, index=False)
    options = SimpleNamespace(
        checkpoint=args.bundle / "checkpoint.pt",
        encoder_spec=args.encoder_spec,
        encoder_config=None,
        series=sp,
        template=tp,
        test_studies=None,
        cache_root=args.output / "cache",
        output=args.output / "benchmark-predictions.csv",
        dicom_root=source,
        source_revision="private-fitting-runtime-v1",
        chunk_size=8,
        workers=0,
        device="cuda",
        precision="fp32",
        allow_diagnostic=False,
        baseline_epoch_one=False,
        fitting_admission=None,
        resilient_series=True,
        fallback_on_unusable_study=True,
        recover_slices=True,
        validate_model_geometry=True,
        isolate_study_errors=True,
        single_decode=True,
        cache_side=json.loads((args.bundle / "bundle-manifest.json").read_text())[
            "inference_policy"
        ]["cache_side"],
        max_series=6,
        series_sampling="selected",
    )
    pool = ResidentPool(workers=2, timeout_seconds=28800)
    comparison = {}

    def invoke(command, **kwargs):
        if comparison:
            return pool.invoke(command, **kwargs)
        reference_command = list(command)
        template_path = Path(command[command.index("--template") + 1])
        original_template = pd.read_csv(template_path).head(3)
        reference_template = args.output / "reference-roster.csv"
        reference_output = args.output / "reference-predictions.csv"
        original_template.to_csv(reference_template, index=False)
        reference_command[reference_command.index("--template") + 1] = str(reference_template)
        reference_command[reference_command.index("--output") + 1] = str(reference_output)
        before = time.monotonic()
        subprocess.run(reference_command, **kwargs)
        comparison["reference_seconds"] = time.monotonic() - before
        result = pool.invoke(command, **kwargs)
        expected = pd.read_csv(reference_output).set_index("StudyInstanceUID")
        actual = (
            pd.read_csv(Path(command[command.index("--output") + 1]))
            .set_index("StudyInstanceUID")
            .loc[expected.index]
        )
        equal = np.array_equal(actual[list(TARGETS)].to_numpy(), expected[list(TARGETS)].to_numpy())
        if not equal:
            raise ValueError("Actual MRI resident predictions differ from original FP32 inference")
        comparison.update(
            studies=3, exact_prediction_values_equal=True, prepared_MRI_cache_shared=True
        )
        return result

    try:
        metrics = chunked_infer.run(options, invoke=invoke)
    finally:
        pool.close()
    if sum(c.get("fallback_studies", 0) for c in metrics["chunks"]):
        raise ValueError("Performance roster has unusable studies; cannot admit clean throughput")
    predictions = pd.read_csv(options.output)
    all_finite = np.isfinite(predictions[list(TARGETS)].to_numpy()).all()
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    shards = []
    for path in args.output.glob("cache/chunk-*/attempt-*/submission.metrics.json"):
        value = json.loads(path.read_text())
        shards.extend(value.get("resident_shards", []))
    if not shards:
        raise ValueError("Actual worker/shard measurements absent")
    by_device = {i: [r for r in shards if r["resident_device_index"] == i] for i in (0, 1)}
    if any(
        not rows or any(r["resident_model_load_count"] != 1 for r in rows)
        for rows in by_device.values()
    ):
        raise ValueError("Both devices must load exactly once")
    peaks = {
        str(i): max(r["GPU_peak_allocated_bytes"] for r in rows) for i, rows in by_device.items()
    }
    upper_rss = rss + sum(
        max(r["host_peak_rss_bytes"] for r in rows) for rows in by_device.values()
    )
    pipeline_seconds = metrics["elapsed_seconds"] - comparison["reference_seconds"]
    record = {
        "status": "actual_private_fitting_T4_throughput_observed",
        "studies": args.study_count,
        "device_names": [torch.cuda.get_device_name(i) for i in (0, 1)],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "elapsed_seconds": time.monotonic() - started,
        "pipeline_seconds_excluding_reference": pipeline_seconds,
        "studies_per_second": args.study_count / pipeline_seconds,
        "original_FP32_comparison": comparison,
        "GPU_peak_allocated_bytes": peaks,
        "conservative_process_tree_peak_rss_upper_bytes": upper_rss,
        "CPU_threads_per_worker": 1,
        "data_workers": 0,
        "checkpoint_loads_per_device": 1,
        "fallback_studies": 0,
        "predictions_finite": bool(all_finite),
        "precision": "fp32",
        "fitting_UIDs_sha256": hashlib.sha256(
            json.dumps(sorted(roster), separators=(",", ":")).encode()
        ).hexdigest(),
        "labels_read": False,
        "protected_gold_or_confirmation_used": False,
        "competition_submission_eligible": False,
        "competition_submission_created": False,
        "full_hidden_runtime_admitted": False,
        "private_fitting_scale_studies": args.study_count,
        "kaggle_score": None,
    }
    if not all_finite or upper_rss > 29 * 2**30 or any(v > 16 * 2**30 for v in peaks.values()):
        raise ValueError("Actual runtime memory or output envelope violated")
    (args.output / "runtime-benchmark.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n"
    )
    Path("/kaggle/working/runtime-benchmark.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
