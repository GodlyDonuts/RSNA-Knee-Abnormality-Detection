from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna.data import StudyDataset
from rsna.model import StudyNet
from rsna.preprocess import Source, cache_series
from rsna.schema import CACHE_VERSION, TARGETS
from rsna.train import predict


def load_inference_metadata(
    checkpoint: dict,
    series_path: Path,
    studies_path: Path | None,
    template_path: Path | None,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict]:
    """Bind held-out evaluation to the exact training inputs before preprocessing.

    New test metadata cannot match the training series table. Template mode
    records its separate hashes without claiming checkpoint-input equality.
    """
    if (studies_path is None) == (template_path is None):
        raise ValueError("Provide exactly one test template or labeled study manifest")
    series_bytes = series_path.read_bytes()
    series_hash = hashlib.sha256(series_bytes).hexdigest()
    bindings = {"series_sha256": series_hash}
    template = None
    if studies_path is not None:
        manifest_bytes = studies_path.read_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        for name, actual in [("series_sha256", series_hash), ("manifest_sha256", manifest_hash)]:
            expected = checkpoint.get(name)
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)
            ):
                raise ValueError(f"Checkpoint lacks a valid {name}; held-out inference is unbound")
            if actual != expected:
                raise ValueError(f"Held-out inference {name} differs from checkpoint")
        revision = checkpoint.get("cache_source_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("Checkpoint lacks cache source revision for held-out inference")
        # Parse the same immutable byte snapshots that were hashed above.
        manifest = pd.read_parquet(io.BytesIO(manifest_bytes))
        studies = manifest.loc[manifest.split == split].copy()
        bindings.update(
            manifest_sha256=manifest_hash,
            metadata_binding="exact_checkpoint_series_and_manifest",
        )
    else:
        assert template_path is not None
        template_bytes = template_path.read_bytes()
        template = pd.read_csv(io.BytesIO(template_bytes))
        studies = template[["StudyInstanceUID"]].rename(columns={"StudyInstanceUID": "study_uid"})
        bindings.update(
            template_sha256=hashlib.sha256(template_bytes).hexdigest(),
            metadata_binding="separate_test_inputs_not_checkpoint_bound",
        )
    series = pd.read_csv(io.BytesIO(series_bytes))
    return series, studies, template, bindings


def write_submission(predictions: pd.DataFrame, template: pd.DataFrame, output: Path) -> None:
    columns = ["StudyInstanceUID", *TARGETS]
    if list(template.columns) != columns or list(predictions.columns) != columns:
        raise ValueError("Submission columns differ from the competition template")
    if (
        predictions.StudyInstanceUID.duplicated().any()
        or template.StudyInstanceUID.duplicated().any()
    ):
        raise ValueError("Duplicate prediction/template studies")
    if set(predictions.StudyInstanceUID) != set(template.StudyInstanceUID):
        raise ValueError("Prediction studies differ from template")
    ordered = predictions.set_index("StudyInstanceUID").loc[template.StudyInstanceUID].reset_index()
    if not ordered[list(TARGETS)].ge(0).all().all() or not ordered[list(TARGETS)].le(1).all().all():
        raise ValueError("Predictions must be finite probabilities in [0,1]")
    ordered.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--studies", type=Path)
    parser.add_argument("--split", default="gold_evaluation")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--dicom-root", type=Path)
    parser.add_argument("--source-revision", default="kaggle-runtime-test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.monotonic()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["targets"] != list(TARGETS):
        raise ValueError("Checkpoint target order mismatch")
    if checkpoint.get("cache_version") != CACHE_VERSION:
        raise ValueError("Checkpoint cache transform version differs")
    series, studies, template, bindings = load_inference_metadata(
        checkpoint, args.series, args.studies, args.template, args.split
    )
    raw_split = "test_series" if template is not None else "train_series"
    if (
        args.studies
        and args.dicom_root
        and args.source_revision != checkpoint["cache_source_revision"]
    ):
        raise ValueError("Held-out preprocessing source revision differs from checkpoint")
    if args.dicom_root:
        source = Source(args.dicom_root)
        for uid in studies.study_uid:
            for series_uid in series.loc[series.StudyInstanceUID == uid, "SeriesInstanceUID"]:
                folder = args.dicom_root / raw_split / uid / series_uid
                paths = sorted(str(p.relative_to(args.dicom_root)) for p in folder.glob("*.dcm"))
                if not paths:
                    raise ValueError(f"No DICOMs in {folder}")
                cache_series(
                    source, paths, args.cache, checkpoint["cache_side"], args.source_revision
                )
    preprocessing_seconds = time.monotonic() - started
    dataset = StudyDataset(studies, series, args.cache, raw_split=raw_split, **checkpoint["config"])
    if dataset.cache_side != checkpoint["cache_side"]:
        raise ValueError("Inference cache resolution differs from training")
    if args.studies and dataset.source_revision != checkpoint["cache_source_revision"]:
        raise ValueError("Held-out inference cache source revision differs from checkpoint")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    model = StudyNet(checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["model"])
    predictions, metrics = predict(model, loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if template is not None:
        write_submission(predictions, template, args.output)
    else:
        predictions.to_csv(args.output, index=False)
    elapsed = time.monotonic() - started
    metrics.update(
        {
            "studies": len(studies),
            "elapsed_seconds": elapsed,
            "preprocessing_seconds": preprocessing_seconds,
            "inference_studies_per_second": len(studies)
            / max(elapsed - preprocessing_seconds, 1e-9),
            "evaluation_source": "sample_or_hidden_test" if args.template else args.split,
            "cache_version": CACHE_VERSION,
            "cache_side": dataset.cache_side,
            "cache_source_revision": dataset.source_revision,
            **bindings,
        }
    )
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
