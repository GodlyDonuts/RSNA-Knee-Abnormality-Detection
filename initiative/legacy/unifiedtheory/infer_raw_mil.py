#!/usr/bin/env python3
"""Run one raw-DICOM specialist family on the hidden test studies."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .constants import TARGETS
    from .raw_mil import (
        RawStudyDataset,
        RawStudyMILModel,
        SliceFeatureBackbone,
        build_study_manifest,
        collate_raw_studies,
        normalize_pixels,
    )
except ImportError:
    from constants import TARGETS
    from raw_mil import (
        RawStudyDataset,
        RawStudyMILModel,
        SliceFeatureBackbone,
        build_study_manifest,
        collate_raw_studies,
        normalize_pixels,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", help="local DINO config directory override")
    parser.add_argument("--slices-per-plane", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoder-batch-size", type=int, default=24)
    parser.add_argument("--ensemble", choices=("rank", "mean"), default="rank")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    # These are checkpoints produced by train_raw_mil.py and attached by the
    # user as an offline Kaggle dataset. Path-valued arguments in early pilot
    # checkpoints require the compatibility fallback.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def build_model(
    checkpoint: dict[str, Any], model_name: str | None, encoder_batch_size: int
) -> RawStudyMILModel:
    saved = checkpoint["args"]
    backbone_name = str(saved["backbone"])
    configured_name = model_name or saved.get("model_name")
    backbone = SliceFeatureBackbone(
        backbone_name,
        configured_name,
        int(saved.get("trainable_blocks", 0)),
        True,
        False,
        None,
    )
    model = RawStudyMILModel(
        backbone,
        len(TARGETS),
        pool=str(saved.get("pool", "max")),
        encoder_batch_size=encoder_batch_size,
        topk=int(saved.get("topk", 3)),
        report_dim=int(saved.get("report_dim", 0)),
    )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {
        "plane_target_bias.weight",
        "fluid_target_bias.weight",
        "fatsat_target_bias.weight",
    }
    if set(missing).difference(allowed_missing) or unexpected:
        raise ValueError(
            f"checkpoint/model mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def main() -> None:
    args = parse_args()
    checkpoint_paths = sorted(
        {Path(path) for pattern in args.checkpoint_glob for path in glob.glob(pattern)}
    )
    if not checkpoint_paths:
        raise ValueError("checkpoint globs resolved to no files")
    checkpoints = [load_checkpoint(path) for path in checkpoint_paths]
    signatures = {
        (str(row["args"]["backbone"]), int(row["args"]["image_size"]))
        for row in checkpoints
    }
    if len(signatures) != 1:
        raise ValueError(f"one raw inference family must share backbone and resolution: {signatures}")
    _, image_size = next(iter(signatures))
    sample = pd.read_csv(args.sample_submission, dtype={"StudyInstanceUID": str})
    required = {"StudyInstanceUID", *TARGETS}
    if missing := required.difference(sample.columns):
        raise ValueError(f"sample submission is missing {sorted(missing)}")
    manifest = build_study_manifest(
        args.data_root / "test_series.csv", args.data_root, "test"
    )
    dataset = RawStudyDataset(
        sample,
        manifest,
        TARGETS,
        image_size,
        args.slices_per_plane,
        float(checkpoints[0]["args"].get("crop_mm", 160.0)),
        False,
        int(checkpoints[0]["args"].get("max_series_per_plane", 1)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        collate_fn=collate_raw_studies,
    )
    device = torch.device(args.device)
    models = [
        build_model(checkpoint, args.model_name, args.encoder_batch_size).to(device).eval()
        for checkpoint in checkpoints
    ]
    member_predictions: list[list[np.ndarray]] = [[] for _ in models]
    uids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            uids.extend(batch["uid"])
            pixels = normalize_pixels(batch["pixels"].to(device, non_blocking=True))
            plane = batch["plane"].to(device, non_blocking=True)
            study_index = batch["study_index"].to(device, non_blocking=True)
            for index, model in enumerate(models):
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    logits = model(
                        pixels,
                        plane,
                        batch["fluid"].to(device, non_blocking=True),
                        batch["fatsat"].to(device, non_blocking=True),
                        study_index,
                        batch["num_studies"],
                    )
                member_predictions[index].append(torch.sigmoid(logits).float().cpu().numpy())
    members = np.stack([np.concatenate(rows) for rows in member_predictions])
    if args.ensemble == "rank":
        ranked = np.empty_like(members)
        for member in range(members.shape[0]):
            for target in range(members.shape[2]):
                ranked[member, :, target] = rank01(members[member, :, target])
        prediction = ranked.mean(axis=0)
    else:
        prediction = members.mean(axis=0)
    if uids != sample["StudyInstanceUID"].astype(str).tolist():
        raise ValueError("inference study order does not match sample submission")
    output = sample.copy()
    output[TARGETS] = prediction
    if not np.isfinite(output[TARGETS].to_numpy(float)).all():
        raise ValueError("raw MIL inference produced non-finite probabilities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"wrote {args.output} from {len(models)} checkpoints")


if __name__ == "__main__":
    main()
