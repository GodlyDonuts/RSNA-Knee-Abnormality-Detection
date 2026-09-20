#!/usr/bin/env python3
"""Run the rules-compliant RSNA training pipeline after feature extraction.

The driver is restartable: immutable intermediate tables are reused, and the
ablation runner already skips completed checkpoints.  It uses reports and
external public models only during training; the final image-only inference
path is implemented separately in :mod:`kaggle_offline_infer`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def cache_ready(data_root: Path, cache_dir: Path) -> tuple[bool, int, int]:
    expected = len(pd.read_csv(data_root / "train.csv", usecols=["StudyInstanceUID"]))
    cached = len(list(cache_dir.glob("*.pt"))) if cache_dir.exists() else 0
    index_path = cache_dir / "train_cache_index.csv"
    if not index_path.exists():
        return False, cached, expected
    index = pd.read_csv(index_path)
    complete = (
        cached == expected
        and len(index) == expected
        and index["StudyInstanceUID"].nunique() == expected
        and index["cache_file"].map(lambda path: Path(path).is_file()).all()
    )
    return bool(complete), cached, expected


def wait_for_complete_cache(
    data_root: Path,
    cache_dir: Path,
    poll_seconds: int,
    timeout_hours: float,
) -> None:
    deadline = time.monotonic() + timeout_hours * 3600
    while True:
        ready, cached, expected = cache_ready(data_root, cache_dir)
        print(f"cache {cached}/{expected}; ready={ready}", flush=True)
        if ready:
            return
        if not poll_seconds:
            raise RuntimeError("feature cache is incomplete")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the complete feature cache")
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/rsna-knee"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/workspace/cache/dinov2-base/train")
    )
    parser.add_argument(
        "--summary-cache-dir",
        type=Path,
        default=Path("/workspace/cache/dinov2-base/train-summary"),
    )
    parser.add_argument("--labels-dir", type=Path, default=Path("/workspace/labels"))
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("/workspace/runs/dinov2_alibi")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--candidate-seeds", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-timeout-hours", type=float, default=12.0)
    parser.add_argument("--skip-nli", action="store_true")
    parser.add_argument(
        "--nli-model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )
    parser.add_argument("--nli-batch-size", type=int, default=32)
    parser.add_argument("--nli-local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    args.labels_dir.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    wait_for_complete_cache(
        args.data_root,
        args.cache_dir,
        args.poll_seconds,
        args.wait_timeout_hours,
    )

    cache_index = args.cache_dir / "train_cache_index.csv"
    summary_cache_index = args.summary_cache_dir / "train_cache_index.csv"
    raw_targets = args.labels_dir / "raw_targets.csv"
    folded_targets = args.labels_dir / "folded_targets.csv"
    folded_reports = args.labels_dir / "folded_reports.csv"
    nli_scores = args.labels_dir / "report_nli.csv"
    final_targets = args.labels_dir / "train_targets.csv"

    if not summary_cache_index.exists():
        run(
            [
                sys.executable,
                str(here / "derive_summary_cache.py"),
                "--cache-index",
                str(cache_index),
                "--output",
                str(args.summary_cache_dir),
            ]
        )

    if not raw_targets.exists():
        run(
            [
                sys.executable,
                str(here / "report_teacher.py"),
                "--train-csv",
                str(args.data_root / "train.csv"),
                "--output",
                str(raw_targets),
            ]
        )
    if not folded_targets.exists():
        run(
            [
                sys.executable,
                str(here / "folds.py"),
                "--cache-index",
                str(cache_index),
                "--labels-csv",
                str(raw_targets),
                "--output",
                str(folded_targets),
                "--folds",
                str(args.folds),
                "--candidate-seeds",
                str(args.candidate_seeds),
            ]
        )
    if not folded_reports.exists():
        run(
            [
                sys.executable,
                str(here / "prepare_folded_reports.py"),
                "--train-csv",
                str(args.data_root / "train.csv"),
                "--folds-csv",
                str(folded_targets),
                "--output",
                str(folded_reports),
            ]
        )
    if not args.skip_nli and not nli_scores.exists():
        command = [
            sys.executable,
            str(here / "score_reports_nli.py"),
            "--train-csv",
            str(folded_reports),
            "--output",
            str(nli_scores),
            "--model-name",
            args.nli_model,
            "--batch-size",
            str(args.nli_batch_size),
        ]
        if args.nli_local_files_only:
            command.append("--local-files-only")
        run(command)
    if not final_targets.exists():
        command = [
            sys.executable,
            str(here / "report_teacher.py"),
            "--train-csv",
            str(folded_reports),
            "--output",
            str(final_targets),
        ]
        if not args.skip_nli:
            command.extend(["--nli-csv", str(nli_scores)])
        run(command)

    run(
        [
            sys.executable,
            str(here / "run_ablation.py"),
            "--cache-index",
            str(summary_cache_index),
            "--labels-csv",
            str(final_targets),
            "--output",
            str(args.runs_dir),
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--folds",
            str(args.folds),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
        ]
    )
    run([sys.executable, str(here / "summarize_ablation.py"), str(args.runs_dir)])
    (args.runs_dir / "pipeline_complete.json").write_text(
        json.dumps(
            {
                "cache_index": str(cache_index),
                "summary_cache_index": str(summary_cache_index),
                "labels": str(final_targets),
                "seeds": args.seeds,
                "folds": args.folds,
                "reports_used_at_inference": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("rules-compliant baseline pipeline complete", flush=True)


if __name__ == "__main__":
    main()
