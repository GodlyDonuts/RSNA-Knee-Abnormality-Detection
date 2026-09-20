#!/usr/bin/env python3
"""Ensemble cached-feature checkpoints and write a Kaggle submission."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .constants import TARGETS
    from .data import (
        FeatureStudyDataset,
        collate_studies,
        model_inputs,
        move_batch,
        patch_model_inputs,
    )
    from .model import KneeAlibiModel, KneeModelConfig
    from .patch_model import PatchKneeAlibiModel, PatchKneeModelConfig
except ImportError:
    from constants import TARGETS
    from data import FeatureStudyDataset, collate_studies, model_inputs, move_batch, patch_model_inputs
    from model import KneeAlibiModel, KneeModelConfig
    from patch_model import PatchKneeAlibiModel, PatchKneeModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ensemble", choices=("probability", "rank"), default="rank")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def rank01(values: np.ndarray) -> np.ndarray:
    """Tie-aware ranks in [0, 1], appropriate for an AUC objective."""

    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


@torch.inference_mode()
def predict_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("targets") != TARGETS:
        raise ValueError(f"target order mismatch in {checkpoint_path}")
    model_type = checkpoint.get("model_type", "summary")
    if model_type == "patch":
        model = PatchKneeAlibiModel(PatchKneeModelConfig(**checkpoint["model_config"]))
    else:
        model = KneeAlibiModel(KneeModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    probabilities: list[np.ndarray] = []
    uids: list[str] = []
    for batch in loader:
        uids.extend(batch["uid"])
        batch = move_batch(batch, device)
        inputs = patch_model_inputs(batch) if model_type == "patch" else model_inputs(batch)
        logits = model(**inputs)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    return uids, np.concatenate(probabilities)


def main() -> None:
    args = parse_args()
    index = pd.read_csv(args.cache_index)
    checkpoint_types = []
    for path in args.checkpoints:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        checkpoint_types.append(checkpoint.get("model_type", "summary"))
    dataset = FeatureStudyDataset(
        index,
        TARGETS,
        include_patch_features="patch" in checkpoint_types,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        collate_fn=collate_studies,
    )
    device = torch.device(args.device)
    member_predictions: list[np.ndarray] = []
    canonical_uids: list[str] | None = None
    for path in args.checkpoints:
        uids, probabilities = predict_checkpoint(path, loader, device)
        if canonical_uids is None:
            canonical_uids = uids
        elif uids != canonical_uids:
            raise RuntimeError("checkpoint inference order changed")
        member_predictions.append(probabilities)

    stack = np.stack(member_predictions)
    if args.ensemble == "rank":
        ranked = np.empty_like(stack, dtype=np.float64)
        for member in range(stack.shape[0]):
            for target in range(stack.shape[2]):
                ranked[member, :, target] = rank01(stack[member, :, target])
        prediction = ranked.mean(axis=0)
    else:
        prediction = stack.mean(axis=0)

    predicted = pd.DataFrame({"StudyInstanceUID": canonical_uids})
    for index_target, target in enumerate(TARGETS):
        predicted[target] = prediction[:, index_target]
    sample = pd.read_csv(args.sample_submission)
    id_column = "StudyInstanceUID" if "StudyInstanceUID" in sample else sample.columns[0]
    result = sample[[id_column]].merge(
        predicted,
        left_on=id_column,
        right_on="StudyInstanceUID",
        how="left",
        validate="one_to_one",
    )
    if id_column != "StudyInstanceUID":
        result = result.drop(columns=["StudyInstanceUID"]).rename(columns={id_column: sample.columns[0]})
    if result[TARGETS].isna().any().any():
        missing = int(result[TARGETS].isna().any(axis=1).sum())
        raise ValueError(f"{missing} sample-submission studies have no cached prediction")
    result[sample.columns].to_csv(args.output, index=False)
    print(f"wrote {args.output} from {len(args.checkpoints)} checkpoint(s)")


if __name__ == "__main__":
    main()
