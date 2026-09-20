#!/usr/bin/env python3
"""Select a conservative equal-rank model subset with nested OOF auditing.

Raw families enter from the fixed pilot audit and registry.  A required anchor
member is always retained.  Model subsets are selected globally (not per
target) on the training folds and evaluated on each untouched fold.  The raw
extension is promoted only when that nested macro gain clears the declared
threshold; otherwise the emitted blend artifact contains the anchor alone.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--anchor", required=True, help="required NAME=OOF_GLOB")
    parser.add_argument(
        "--anchor-checkpoint-glob",
        action="append",
        default=[],
        help="checkpoint glob(s) implementing the anchor OOF member",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-nested-gain", type=float, default=0.005)
    parser.add_argument("--fold-column", default="fold")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_member(name: str, paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError(f"member {name!r} resolved to no OOF files")
    frames = [pd.read_csv(path, dtype={"StudyInstanceUID": str}) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    required = {"StudyInstanceUID", *TARGETS}
    if missing := required.difference(frame.columns):
        raise ValueError(f"member {name!r} is missing {sorted(missing)}")
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"member {name!r} contains duplicate OOF studies")
    return frame[["StudyInstanceUID", *TARGETS]]


def parse_anchor(specification: str) -> tuple[str, list[Path]]:
    if "=" not in specification:
        raise ValueError("anchor must be NAME=OOF_GLOB")
    name, pattern = specification.split("=", 1)
    return name, [Path(path) for path in sorted(glob.glob(pattern))]


def promoted_paths(
    audit: dict[str, Any], registry: dict[str, Any]
) -> list[tuple[str, list[Path]]]:
    promoted = audit.get("promoted")
    families = registry.get("families")
    if not isinstance(promoted, list) or not isinstance(families, dict):
        raise ValueError("invalid audit or registry")
    loaded = []
    for name in promoted:
        if name not in families:
            raise ValueError(f"promoted family {name!r} is absent from registry")
        family = families[name]
        output = Path(family["output"])
        pattern = str(family["oof_pattern"])
        folds = sorted(output.glob(pattern.replace("{fold}", "*")))
        loaded.append((name, folds))
    return loaded


def checkpoint_from_oof(path: Path) -> Path:
    if not path.name.endswith("_oof.csv"):
        raise ValueError(f"cannot derive checkpoint from OOF source {path}")
    return path.with_name(path.name.removesuffix("_oof.csv") + ".pt")


def auc(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(score)
    return (
        float(roc_auc_score(y[valid], score[valid]))
        if np.unique(y[valid]).size == 2
        else float("nan")
    )


def subsets(member_count: int) -> list[tuple[int, ...]]:
    # Index zero is the required anchor.  Prefer fewer members on exact ties.
    return [
        (0, *choice)
        for size in range(member_count)
        for choice in itertools.combinations(range(1, member_count), size)
    ]


def subset_macro(
    ranked: dict[str, np.ndarray],
    labels: pd.DataFrame,
    row_mask: np.ndarray,
    subset: tuple[int, ...],
) -> float:
    scores = []
    for target in TARGETS:
        y = pd.to_numeric(labels[target], errors="coerce").to_numpy(float)
        gold = f"{target}__gold"
        valid = row_mask.copy()
        if gold in labels:
            valid &= pd.to_numeric(labels[gold], errors="coerce").fillna(0).to_numpy() > 0
        prediction = ranked[target][:, subset].mean(axis=1)
        value = auc(y[valid], prediction[valid])
        if np.isfinite(value):
            scores.append(value)
    return float(np.mean(scores)) if scores else float("nan")


def best_subset(
    ranked: dict[str, np.ndarray],
    labels: pd.DataFrame,
    row_mask: np.ndarray,
    candidates: list[tuple[int, ...]],
) -> tuple[tuple[int, ...], float]:
    scored = [(candidate, subset_macro(ranked, labels, row_mask, candidate)) for candidate in candidates]
    finite = [row for row in scored if np.isfinite(row[1])]
    if not finite:
        raise ValueError("no candidate subset has a defined AUC")
    return max(finite, key=lambda row: (row[1], -len(row[0]), tuple(-x for x in row[0])))


def main() -> None:
    args = parse_args()
    audit, registry = read_json(args.audit), read_json(args.registry)
    anchor_name, anchor_paths = parse_anchor(args.anchor)
    specifications = [(anchor_name, anchor_paths), *promoted_paths(audit, registry)]
    names = [name for name, _ in specifications]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate ensemble member names: {names}")
    members = [(name, load_member(name, paths), paths) for name, paths in specifications]

    labels = pd.read_csv(args.labels_csv, dtype={"StudyInstanceUID": str})
    if args.fold_column not in labels:
        raise ValueError(f"labels have no {args.fold_column!r} column")
    gold = [column for column in labels if column.endswith("__gold")]
    merged = labels[["StudyInstanceUID", args.fold_column, *TARGETS, *gold]].copy()
    for name, frame, _ in members:
        renamed = frame.rename(columns={target: f"{name}::{target}" for target in TARGETS})
        merged = merged.merge(renamed, on="StudyInstanceUID", how="inner", validate="one_to_one")
    if not len(merged):
        raise ValueError("OOF members and labels have no shared studies")

    ranked: dict[str, np.ndarray] = {}
    for target in TARGETS:
        columns = []
        for name in names:
            values = pd.to_numeric(merged[f"{name}::{target}"], errors="coerce")
            columns.append(values.rank(method="average", pct=True).to_numpy(float))
        ranked[target] = np.column_stack(columns)

    choices = subsets(len(names))
    folds = sorted(pd.unique(merged[args.fold_column]))
    nested_rows = []
    for held_fold in folds:
        train = (merged[args.fold_column].to_numpy() != held_fold)
        test = ~train
        selected, training_macro = best_subset(ranked, merged, train, choices)
        anchor = (0,)
        for target in TARGETS:
            y = pd.to_numeric(merged[target], errors="coerce").to_numpy(float)
            valid = test.copy()
            marker = f"{target}__gold"
            if marker in merged:
                valid &= pd.to_numeric(merged[marker], errors="coerce").fillna(0).to_numpy() > 0
            selected_auc = auc(y[valid], ranked[target][:, selected].mean(axis=1)[valid])
            anchor_auc = auc(y[valid], ranked[target][:, anchor].mean(axis=1)[valid])
            nested_rows.append(
                {
                    "fold": int(held_fold),
                    "target": target,
                    "training_members": [names[index] for index in selected],
                    "training_macro_auc": training_macro,
                    "selected_auc": selected_auc,
                    "anchor_auc": anchor_auc,
                }
            )

    nested = pd.DataFrame(nested_rows)
    selected_nested = float(nested["selected_auc"].mean())
    anchor_nested = float(nested["anchor_auc"].mean())
    nested_gain = selected_nested - anchor_nested
    all_rows = np.ones(len(merged), dtype=bool)
    full_subset, full_macro = best_subset(ranked, merged, all_rows, choices)
    promote = bool(len(names) > 1 and nested_gain >= args.minimum_nested_gain)
    final_subset = full_subset if promote else (0,)
    final_names = [names[index] for index in final_subset]
    weight = 1.0 / len(final_names)
    anchor_checkpoints = [
        Path(path)
        for pattern in args.anchor_checkpoint_glob
        for path in sorted(glob.glob(pattern))
    ]
    if args.anchor_checkpoint_glob and not anchor_checkpoints:
        raise ValueError("anchor checkpoint globs resolved to no files")
    checkpoint_source_files = {
        name: [
            str(path)
            for path in (
                anchor_checkpoints
                if name == anchor_name and anchor_checkpoints
                else [checkpoint_from_oof(source) for source in paths]
            )
        ]
        for name, _, paths in members
    }
    artifact = {
        "schema_version": 1,
        "method": "nested globally-selected equal-rank subset",
        "members": final_names,
        "weights": {target: [weight] * len(final_names) for target in TARGETS},
        "anchor": anchor_name,
        "available_members": names,
        "full_oof_selected_members": [names[index] for index in full_subset],
        "full_oof_macro_auc": full_macro,
        "nested_selected_macro_auc": selected_nested,
        "nested_anchor_macro_auc": anchor_nested,
        "nested_gain": nested_gain,
        "minimum_nested_gain": args.minimum_nested_gain,
        "raw_extension_promoted": promote,
        "source_files": {name: [str(path) for path in paths] for name, _, paths in members},
        "checkpoint_source_files": checkpoint_source_files,
        "fold_results": nested_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=True) + "\n")
    nested.to_csv(args.output.with_suffix(".fold_targets.csv"), index=False)
    print(json.dumps({key: artifact[key] for key in (
        "members", "nested_selected_macro_auc", "nested_anchor_macro_auc",
        "nested_gain", "raw_extension_promoted"
    )}, indent=2))


if __name__ == "__main__":
    main()
