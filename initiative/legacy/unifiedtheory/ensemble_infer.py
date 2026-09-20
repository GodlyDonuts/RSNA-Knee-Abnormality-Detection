#!/usr/bin/env python3
"""Apply a target-wise OOF-fitted blend to groups of fold checkpoints."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from .constants import TARGETS
    from .data import FeatureStudyDataset, collate_studies
    from .infer import predict_checkpoint, rank01
except ImportError:
    from constants import TARGETS
    from data import FeatureStudyDataset, collate_studies
    from infer import predict_checkpoint, rank01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--member", action="append", required=True, help="NAME=CHECKPOINT_GLOB")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blend = json.loads(args.blend.read_text())
    specs = {}
    for spec in args.member:
        if "=" not in spec:
            raise ValueError("member must be NAME=GLOB")
        name, pattern = spec.split("=", 1)
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
        if not paths:
            raise ValueError(f"member {name!r} resolved to no checkpoints")
        specs[name] = paths
    if list(specs) != blend["members"]:
        raise ValueError(f"member order/names must be exactly {blend['members']}")

    index = pd.read_csv(args.cache_index)
    dataset = FeatureStudyDataset(index, TARGETS)
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
    member_predictions = []
    canonical_uids = None
    for name, checkpoints in specs.items():
        folds = []
        for checkpoint in checkpoints:
            uids, prediction = predict_checkpoint(checkpoint, loader, device)
            if canonical_uids is None:
                canonical_uids = uids
            elif uids != canonical_uids:
                raise RuntimeError("checkpoint inference order changed")
            folds.append(prediction)
        member_predictions.append(np.mean(folds, axis=0))
        print(f"predicted {name} from {len(checkpoints)} fold checkpoint(s)", flush=True)
    stack = np.stack(member_predictions)  # [M,N,T]
    ranked = np.empty_like(stack, dtype=float)
    for member in range(stack.shape[0]):
        for target in range(stack.shape[2]):
            ranked[member, :, target] = rank01(stack[member, :, target])
    prediction = np.empty((stack.shape[1], stack.shape[2]), dtype=float)
    for target_index, target in enumerate(TARGETS):
        weight = np.asarray(blend["weights"][target], dtype=float)
        if weight.shape != (stack.shape[0],) or not np.isclose(weight.sum(), 1):
            raise ValueError(f"invalid blend weights for {target}")
        prediction[:, target_index] = ranked[:, :, target_index].T @ weight

    predicted = pd.DataFrame({"StudyInstanceUID": canonical_uids})
    for target_index, target in enumerate(TARGETS):
        predicted[target] = prediction[:, target_index]
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
        raise ValueError("sample submission contains studies with no prediction")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result[sample.columns].to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
