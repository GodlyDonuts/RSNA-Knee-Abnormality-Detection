from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna.data import StudyDataset
from rsna.model import StudyNet
from rsna.preprocess import Source, cache_series
from rsna.schema import TARGETS
from rsna.train import predict


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
    series = pd.read_csv(args.series)
    if args.template:
        template = pd.read_csv(args.template)
        studies = template[["StudyInstanceUID"]].rename(columns={"StudyInstanceUID": "study_uid"})
        raw_split = "test_series"
    elif args.studies:
        manifest = pd.read_parquet(args.studies)
        studies = manifest[manifest.split == args.split]
        raw_split = "train_series"
    else:
        raise ValueError("Provide a test submission template or a labeled study manifest")
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
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    model = StudyNet(checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["model"])
    predictions, metrics = predict(model, loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.template:
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
        }
    )
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
