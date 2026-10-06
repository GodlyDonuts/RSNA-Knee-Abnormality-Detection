"""CPU-only original32 MRI preparation equivalence and timing; no labels/models."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

if __package__:
    from .chunked_infer import prepare_usable_series
    from .parallel_preparation import ParallelPreparation
else:
    from chunked_infer import prepare_usable_series
    from parallel_preparation import ParallelPreparation


def cache_files(root):
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                value = hashlib.file_digest(stream, "sha256").hexdigest()
            result[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": value,
            }
    if not result:
        raise ValueError("Actual original MRI cache files required")
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("competition", "roster", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 0:
        raise ValueError("CPU-only numerical input gate required")
    torch.set_num_threads(2)
    roster = json.loads(args.roster.read_text())
    if len(roster) != 32 or len(set(roster)) != 32:
        raise ValueError("Original32 distinct fitting studies required")
    if shutil.disk_usage(args.output.parent).free < 9 * 2**30:
        raise ValueError("Two3GiB caches plus3GiB reserve required")
    args.output.mkdir(mode=0o700)
    alias = args.output / "source"
    (alias / "test_series").mkdir(parents=True)
    for uid in roster:
        (alias / "test_series" / uid).symlink_to(
            args.competition / "train_series" / uid, target_is_directory=True
        )
    series = pd.read_csv(args.competition / "train_series.csv")
    frame = series.loc[series.StudyInstanceUID.isin(roster)].copy()
    if set(frame.StudyInstanceUID) != set(roster):
        raise ValueError("Original fitting source metadata absent")
    upper_cache = sum(
        len(
            list(
                (alias / "test_series" / row.StudyInstanceUID / row.SeriesInstanceUID).glob("*.dcm")
            )
        )
        * (384 * 384 * 2 + 4096)
        + 65536
        for row in frame.itertuples(index=False)
    )
    if upper_cache > 3 * 2**30:
        raise ValueError("Original entire-series cache upper bound exceeds each funded3GiB cap")
    options = dict(
        dicom_root=alias,
        cache_side=384,
        max_series=6,
        source_revision="private-fitting-runtime-v1",
        allow_unusable_studies=True,
        recover_slices=True,
        validate_geometry=True,
        single_decode=True,
        series_sampling="selected",
    )
    start = time.monotonic()
    selected, receipt = prepare_usable_series(
        frame, roster, cache=args.output / "serial", **options
    )
    serial_seconds = time.monotonic() - start
    if receipt["unusable_study_ids"]:
        raise ValueError("Original equivalence roster contains unusable MRI")
    serial_files = cache_files(args.output / "serial")
    if sum(row["bytes"] for row in serial_files.values()) > 3 * 2**30:
        raise ValueError("Serial cache cap exceeded")
    start = time.monotonic()
    with ParallelPreparation() as worker:
        actual, observed = worker.prepare(frame, roster, cache=args.output / "parallel", **options)
    parallel_seconds = time.monotonic() - start
    pd.testing.assert_frame_equal(actual, selected)
    if observed != receipt or cache_files(args.output / "parallel") != serial_files:
        raise ValueError("Actual parallel MRI cache bytes/recovery/selection differ")
    peak = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        + 2 * resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    peak *= 1 if sys.platform == "darwin" else 1024
    if peak > 30 * 2**30:
        raise ValueError("CPU source process-tree memory bound exceeded")
    result = {
        "status": "actual_original32_serial_parallel_preparation_bytes_exactly_equal",
        "studies": 32,
        "selected_series": len(selected),
        "cache_members": len(serial_files),
        "one_cache_bytes": sum(row["bytes"] for row in serial_files.values()),
        "cache_member_records_sha256": hashlib.sha256(
            json.dumps(serial_files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "fitting_UIDs_sha256": hashlib.sha256(
            json.dumps(sorted(roster), separators=(",", ":")).encode()
        ).hexdigest(),
        "serial_CPU_threads": 2,
        "parallel_CPU_workers": 2,
        "threads_per_parallel_worker": 1,
        "serial_seconds": serial_seconds,
        "parallel_seconds_including_spawn": parallel_seconds,
        "observed_ratio_serial_over_parallel": serial_seconds / parallel_seconds,
        "conservative_process_tree_peak_rss_upper_bytes": peak,
        "MRI_labels_or_protected_cohort_read": False,
        "model_encoder_GPU_or_optimizer_executed": False,
        "clinical_accuracy_or_hidden_runtime_accepted": False,
        "competition_submission": False,
    }
    Path("/kaggle/working/parallel-preparation-accepted.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    main()
