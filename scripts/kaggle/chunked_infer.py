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

from rsna.data import PLANES, select_series
from rsna.infer import write_submission
from rsna.preprocess import Source, cache_series
from rsna.schema import TARGETS

# Only image/geometry failures may exclude a series. Cache integrity, disk,
# environment, and model failures still stop inference.
DATA_ERROR_PREFIXES = (
    "DICOM UID mismatch:",
    "The number of bytes of pixel data is less than expected",
    "Only single-frame grayscale slices are supported",
    "Unsupported photometric interpretation:",
    "Invalid DICOM rescale parameters",
    "Non-finite pixel values",
    "Non-finite DICOM geometry",
    "Malformed DICOM position/orientation",
    "Invalid DICOM pixel spacing",
    "Invalid image orientation vectors",
    "Neither geometry nor unique InstanceNumber can order series",
    "Mixed pixel matrix dimensions within series",
    "Mixed photometric interpretation within series",
    "Constant-intensity series",
    "Mixed pixel spacing within series",
)
DICOM_ATTRIBUTE_ERROR_PREFIXES = (
    "'FileDataset' object has no attribute '",
    "The dataset has no 'Pixel Data'",
    "Missing required element:",
    "Unable to decode the pixel data as the dataset's 'file_meta' has no "
    "(0002,0010) 'Transfer Syntax UID' element",
)
DICOM_NOT_IMPLEMENTED_PREFIXES = (
    "Unable to decode the pixel data as a (0002,0010) 'Transfer Syntax UID' value of ",
    "No pixel data decoders have been implemented for ",
    "The data type '",
)
DICOM_DECODER_RUNTIME_PREFIX = (
    "Unable to decode as exceptions were raised by all available plugins:"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_cache_contract(checkpoint: Path, *, max_series: int, cache_side: int) -> None:
    """Read the saved training contract without changing its bound model code."""
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    binding = saved["binding"]
    if binding["config"].get("max_series") != max_series or binding.get("cache_side") != cache_side:
        raise ValueError("Series limit/cache resolution differs from the trained checkpoint")
    del saved


def prepare_usable_series(
    frame: pd.DataFrame,
    study_ids: list[str],
    *,
    dicom_root: Path,
    cache: Path,
    cache_side: int,
    max_series: int,
    source_revision: str,
    allow_unusable_studies: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Cache only useful MRI series; use an alternate when source pixels are invalid."""
    from pydicom.errors import InvalidDicomError

    source = Source(dicom_root)
    usable_ids: set[str] = set()
    unusable_study_ids: list[str] = []
    failures = {"missing_dicoms": 0, "invalid_dicoms": 0, "invalid_series": 0}
    for study_uid in study_ids:
        rows = frame.loc[frame.StudyInstanceUID == study_uid].to_dict("records")
        ordered = select_series(rows, len(rows), False)
        attempted: set[str] = set()
        accepted: list[dict] = []

        def attempt(row: dict) -> bool:
            sid = row["SeriesInstanceUID"]
            attempted.add(sid)
            folder = dicom_root / "test_series" / study_uid / sid
            paths = sorted(str(path.relative_to(dicom_root)) for path in folder.glob("*.dcm"))
            if not paths:
                failures["missing_dicoms"] += 1
                return False
            try:
                cache_series(source, paths, cache, cache_side, source_revision)
            except InvalidDicomError:
                failures["invalid_dicoms"] += 1
                return False
            except AttributeError as exc:
                if not str(exc).startswith(DICOM_ATTRIBUTE_ERROR_PREFIXES):
                    raise
                failures["invalid_dicoms"] += 1
                return False
            except NotImplementedError as exc:
                if not str(exc).startswith(DICOM_NOT_IMPLEMENTED_PREFIXES):
                    raise
                failures["invalid_dicoms"] += 1
                return False
            except RuntimeError as exc:
                if not str(exc).startswith(DICOM_DECODER_RUNTIME_PREFIX):
                    raise
                failures["invalid_dicoms"] += 1
                return False
            except ValueError as exc:
                if not str(exc).startswith(DATA_ERROR_PREFIXES):
                    raise
                failures["invalid_series"] += 1
                return False
            usable_ids.add(sid)
            accepted.append(row)
            return True

        # Match the trained sampler's plane-first selection. If its preferred
        # series is corrupt, seek another of that plane before filling slots.
        for plane in PLANES:
            for row in ordered:
                if row["Anatomical_Plane"] != plane or row["SeriesInstanceUID"] in attempted:
                    continue
                if attempt(row):
                    break
        for row in ordered:
            if len(accepted) >= max_series:
                break
            if row["SeriesInstanceUID"] not in attempted:
                attempt(row)
        if not accepted:
            if not allow_unusable_studies:
                raise ValueError("No decodable MRI series for a test study")
            unusable_study_ids.append(study_uid)
            continue
        if len(accepted) > max_series:
            raise ValueError("Prepared more series than the trained model accepts")
    selected = frame.loc[frame.SeriesInstanceUID.isin(usable_ids)].copy()
    if set(selected.StudyInstanceUID) != set(study_ids) - set(unusable_study_ids):
        raise ValueError("Prepared MRI series do not cover the test study chunk")
    receipt = {"selected_series": len(selected), "skipped_series": failures}
    if allow_unusable_studies:
        receipt["unusable_study_ids"] = unusable_study_ids
    return selected, receipt


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
    test_studies = getattr(args, "test_studies", None)
    sample = pd.read_csv(args.template) if args.template is not None else None
    if sample is not None and list(sample.columns) != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("Official sample submission column order differs")
    if test_studies is not None:
        roster = pd.read_csv(test_studies)
        if "StudyInstanceUID" not in roster.columns:
            raise ValueError("Official test.csv lacks study identities")
        template = pd.DataFrame({"StudyInstanceUID": roster.StudyInstanceUID})
        for target in TARGETS:
            template[target] = 0.5
    elif sample is not None:
        template = sample
    else:
        raise ValueError("A test.csv roster or submission template is required")
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
    fallback_unusable = getattr(args, "fallback_on_unusable_study", False)
    if set(template.StudyInstanceUID) - set(series.StudyInstanceUID) and not fallback_unusable:
        raise ValueError("A test study has no series")

    checkpoint_sha = sha256(args.checkpoint)
    resilient = getattr(args, "resilient_series", False)
    if fallback_unusable and not resilient:
        raise ValueError("Unusable-study fallback requires resilient MRI preparation")
    if resilient:
        if (
            args.dicom_root is None
            or type(args.cache_side) is not int
            or type(args.max_series) is not int
            or args.cache_side < 16
            or args.max_series < 3
        ):
            raise ValueError("Resilient MRI preparation needs source, cache side and series limit")
        verify_cache_contract(
            args.checkpoint, max_series=args.max_series, cache_side=args.cache_side
        )
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
        preparation = None
        if resilient:
            selected_series, preparation = prepare_usable_series(
                selected_series,
                part.StudyInstanceUID.tolist(),
                dicom_root=args.dicom_root,
                cache=cache,
                cache_side=args.cache_side,
                max_series=args.max_series,
                source_revision=args.source_revision,
                allow_unusable_studies=fallback_unusable,
            )
        unusable = set(preparation.get("unusable_study_ids", [])) if preparation else set()
        if unusable - ids:
            raise ValueError("Unusable-study receipt contains an unrelated identity")
        part_for_model = part.loc[~part.StudyInstanceUID.isin(unusable)].copy()
        if part_for_model.empty:
            frame = part.copy()
            for target in TARGETS:
                frame[target] = 0.5
            predictions.append(frame)
            receipts.append(
                {
                    "first_row": start,
                    "studies": len(part),
                    "inferred_studies": 0,
                    "fallback_studies": len(part),
                    "submission_sha256": None,
                    "cache_bytes_before_cleanup": 0,
                    "elapsed_seconds": 0.0,
                    "series_preparation": preparation,
                }
            )
            if cache.exists():
                if cache.is_symlink():
                    raise ValueError("Chunk cache became a symlink")
                shutil.rmtree(cache)
            continue
        part_for_model.to_csv(template_path, index=False)
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
        if args.dicom_root is not None and not resilient:
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
            or metrics.get("studies") != len(part_for_model)
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
        if frame.StudyInstanceUID.tolist() != part_for_model.StudyInstanceUID.tolist():
            raise ValueError("Chunk output identities or order differ")
        if unusable:
            neutral = part.loc[part.StudyInstanceUID.isin(unusable)].copy()
            for target in TARGETS:
                neutral[target] = 0.5
            frame = pd.concat([frame, neutral], ignore_index=True)
            if frame.StudyInstanceUID.duplicated().any() or set(frame.StudyInstanceUID) != ids:
                raise ValueError("Merged MRI and fallback identities differ")
            frame = frame.set_index("StudyInstanceUID").loc[part.StudyInstanceUID].reset_index()
        predictions.append(frame)
        cache_bytes = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())
        receipts.append(
            {
                "first_row": start,
                "studies": len(part),
                **(
                    {"inferred_studies": len(part_for_model), "fallback_studies": len(unusable)}
                    if fallback_unusable
                    else {}
                ),
                "submission_sha256": sha256(output),
                "cache_bytes_before_cleanup": cache_bytes,
                "elapsed_seconds": metrics["elapsed_seconds"],
                **({"series_preparation": preparation} if preparation is not None else {}),
            }
        )
        if cache.exists():
            if cache.is_symlink():
                raise ValueError("Chunk cache became a symlink")
            shutil.rmtree(cache)
    if common is None:
        raise ValueError("No predictions from MRI-backed studies")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(pd.concat(predictions, ignore_index=True), template, args.output)
    result = {
        "schema_version": 1,
        "status": "passed",
        "diagnostic_only": args.allow_diagnostic,
        "studies": len(template),
        "checkpoint_sha256": checkpoint_sha,
        "submission_sha256": sha256(args.output),
        "test_inputs": (
            {
                "series_sha256": sha256(args.series),
                "test_studies_sha256": sha256(test_studies),
                "sample_template_sha256": sha256(args.template) if args.template else None,
                "metadata_binding": "test_csv_authoritative_not_checkpoint_bound",
            }
            if test_studies is not None
            else {
                "series_sha256": sha256(args.series),
                "template_sha256": sha256(args.template),
                "metadata_binding": "separate_test_inputs_not_checkpoint_bound",
            }
        ),
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
    for name in ("checkpoint", "encoder-config", "series", "cache-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--test-studies", type=Path)
    parser.add_argument("--template", type=Path)
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
    parser.add_argument("--resilient-series", action="store_true")
    parser.add_argument("--fallback-on-unusable-study", action="store_true")
    parser.add_argument("--cache-side", type=int)
    parser.add_argument("--max-series", type=int)
    args = parser.parse_args()
    if args.baseline_epoch_one and (args.baseline_sidecar is None or args.baseline_proof is None):
        parser.error("Epoch-1 baseline requires its byte-bound sidecar and replay proof")
    if not args.baseline_epoch_one and (args.baseline_sidecar or args.baseline_proof):
        parser.error("Baseline evidence supplied without baseline mode")
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
