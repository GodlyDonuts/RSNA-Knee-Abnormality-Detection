"""Bound temporary test-cache size while using the exact trained inference code."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from rsna.infer import write_submission
from rsna.schema import TARGETS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace, invoke=None) -> dict:
    baseline = getattr(args, "baseline_epoch_one", False)
    if baseline and args.allow_diagnostic:
        raise ValueError("A completed training baseline cannot be diagnostic")
    if args.chunk_size < 1 or args.workers < 0:
        raise ValueError("Chunk size and worker count must be valid")
    if args.cache_root.exists() or args.cache_root.is_symlink():
        raise ValueError("Chunk scratch must be new")
    if args.output.exists():
        raise ValueError("Refusing to replace an existing submission")
    template = pd.read_csv(args.template)
    series = pd.read_csv(args.series)
    if list(template.columns) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("Official template column order differs")
    if template.empty or template.StudyInstanceUID.isna().any():
        raise ValueError("Official test template is empty or has missing identities")
    if template.StudyInstanceUID.duplicated().any():
        raise ValueError("Duplicate test studies")
    if not {"StudyInstanceUID", "SeriesInstanceUID"} <= set(series):
        raise ValueError("Test series metadata lacks required identities")
    if series.SeriesInstanceUID.isna().any() or series.SeriesInstanceUID.duplicated().any():
        raise ValueError("Missing or duplicate test series identities")
    if set(template.StudyInstanceUID) - set(series.StudyInstanceUID):
        raise ValueError("A test study has no series")

    checkpoint_sha = sha256(args.checkpoint)
    args.cache_root.mkdir(parents=True, mode=0o700)
    invoke = invoke or subprocess.run
    started = time.monotonic()
    predictions = []
    receipts = []
    common = None
    for start in range(0, len(template), args.chunk_size):
        part = template.iloc[start : start + args.chunk_size].copy()
        ids = set(part.StudyInstanceUID)
        selected_series = series[series.StudyInstanceUID.isin(ids)]
        chunk = args.cache_root / f"chunk-{start // args.chunk_size:04d}"
        chunk.mkdir(mode=0o700)
        template_path = chunk / "sample_submission.csv"
        series_path = chunk / "test_series.csv"
        cache = chunk / "cache"
        output = chunk / "submission.csv"
        part.to_csv(template_path, index=False)
        selected_series.to_csv(series_path, index=False)
        command = [
            sys.executable,
            *(
                [str(Path(__file__).with_name("baseline_infer.py"))]
                if baseline
                else ["-m", "rsna.competitive.infer"]
            ),
            "--checkpoint",
            str(args.checkpoint),
            "--encoder-config",
            str(args.encoder_config),
            "--series",
            str(series_path),
            "--template",
            str(template_path),
            "--cache",
            str(cache),
            "--output",
            str(output),
            "--workers",
            str(args.workers),
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--source-revision",
            args.source_revision,
        ]
        if baseline:
            command.extend(
                (
                    "--baseline-sidecar",
                    str(args.baseline_sidecar),
                    "--baseline-proof",
                    str(args.baseline_proof),
                )
            )
        else:
            command.append("--scheduled-epoch")
        if args.dicom_root is not None:
            command.extend(("--dicom-root", str(args.dicom_root)))
        if args.allow_diagnostic:
            command.append("--allow-diagnostic")
        invoke(command, check=True)
        metrics = json.loads(output.with_suffix(".metrics.json").read_text())
        expected_inputs = {
            "series_sha256": sha256(series_path),
            "template_sha256": sha256(template_path),
            "metadata_binding": "separate_test_inputs_not_checkpoint_bound",
        }
        if (
            metrics.get("status") != "passed"
            or metrics.get("checkpoint_sha256") != checkpoint_sha
            or metrics.get("studies") != len(part)
            or metrics.get("diagnostic_only") is not args.allow_diagnostic
            or metrics.get("submission_sha256") != sha256(output)
            or metrics.get("test_inputs") != expected_inputs
            or metrics.get("scheduled_epoch_authorized") is not (not baseline)
            or (baseline and metrics.get("checkpoint_epoch") != 1)
            or metrics.get("initial_pretrained_weights_read") is not False
            or metrics.get("cache_source_revision") != args.source_revision
            or metrics.get("inference_precision") != args.precision
            or metrics.get("device_type") != args.device
        ):
            raise ValueError("Chunk inference receipt differs from its bound inputs")
        comparable = {
            key: metrics[key]
            for key in (
                "training_release_binding",
                "cache_version",
                "cache_side",
                "encoder",
                "training_precision",
                "inference_precision",
                "device_type",
                "checkpoint_epoch",
            )
        }
        if common is None:
            common = comparable
        elif comparable != common:
            raise ValueError("Chunk model or runtime provenance differs")
        frame = pd.read_csv(output)
        if list(frame.columns) != ["StudyInstanceUID", *TARGETS]:
            raise ValueError("Chunk output column order differs")
        if frame.StudyInstanceUID.tolist() != part.StudyInstanceUID.tolist():
            raise ValueError("Chunk output identities or order differ")
        predictions.append(frame)
        cache_bytes = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
        receipts.append(
            {
                "first_row": start,
                "studies": len(part),
                "submission_sha256": sha256(output),
                "cache_bytes_before_cleanup": cache_bytes,
                "elapsed_seconds": metrics["elapsed_seconds"],
            }
        )
        if cache.exists():
            if cache.is_symlink():
                raise ValueError("Chunk cache became a symlink")
            shutil.rmtree(cache)
    assert common is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(pd.concat(predictions, ignore_index=True), template, args.output)
    result = {
        "schema_version": 1,
        "status": "passed",
        "diagnostic_only": args.allow_diagnostic,
        "studies": len(template),
        "checkpoint_sha256": checkpoint_sha,
        "submission_sha256": sha256(args.output),
        "test_inputs": {
            "series_sha256": sha256(args.series),
            "template_sha256": sha256(args.template),
            "metadata_binding": "separate_test_inputs_not_checkpoint_bound",
        },
        "chunk_size": args.chunk_size,
        "chunk_count": len(receipts),
        "peak_chunk_cache_bytes": max(row["cache_bytes_before_cleanup"] for row in receipts),
        "chunks": receipts,
        "elapsed_seconds": time.monotonic() - started,
        "hidden_test_runtime_validated": False,
        **common,
    }
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "encoder-config", "series", "template", "cache-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dicom-root", type=Path)
    parser.add_argument("--source-revision", default="kaggle-runtime-test")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("checkpoint", "fp32", "bf16"), default="fp32")
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--baseline-epoch-one", action="store_true")
    parser.add_argument("--baseline-sidecar", type=Path)
    parser.add_argument("--baseline-proof", type=Path)
    args = parser.parse_args()
    if args.baseline_epoch_one and (args.baseline_sidecar is None or args.baseline_proof is None):
        parser.error("Epoch-1 baseline requires its byte-bound sidecar and replay proof")
    if not args.baseline_epoch_one and (args.baseline_sidecar or args.baseline_proof):
        parser.error("Baseline evidence supplied without baseline mode")
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
