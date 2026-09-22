"""Deterministic training-only grouped experiment overlays; accepted data stay immutable."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsna.release import load_verified_release, validate_checkpoint_binding
from rsna.schema import TARGETS

ALGORITHM = "iterative_group_presence_stratification_v1"
DEFAULT_SEED = 20260922
OVERLAY_COLUMNS = ["study_uid", "leakage_group", "fold"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Competitive folds: " + message)


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def training_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate identities across the full manifest; inspect labels only inside train.

    Public production entry points must first call load_verified_release. This
    pure helper also supports synthetic tests and cannot attest external evidence.
    """
    required = {"study_uid", "leakage_group", "split", *TARGETS}
    required.update(t + "__mask" for t in TARGETS)
    _require(manifest.columns.is_unique and required <= set(manifest), "missing/duplicate columns")
    _require(
        not manifest.empty and not manifest.study_uid.duplicated().any(), "empty/duplicate UID"
    )
    for column in ("study_uid", "leakage_group"):
        _require(
            bool(manifest[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all()),
            "invalid " + column,
        )
    _require(
        bool(
            manifest.split.isin(["train", "weak_validation", "gold_evaluation", "quarantine"]).all()
        ),
        "unknown split",
    )
    selected = manifest.loc[manifest.split.eq("train")].sort_values("study_uid").copy()
    _require(not selected.empty, "empty train split")
    outside = set(manifest.loc[~manifest.split.eq("train"), "leakage_group"])
    _require(not set(selected.leakage_group) & outside, "train group crosses accepted split")
    for target in TARGETS:
        mask = selected[target + "__mask"]
        _require(bool(mask.isin([0, 1]).all()), "invalid train mask")
        _require(
            bool(selected.loc[mask.eq(1), target].isin([0, 1]).all()), "nonbinary observed label"
        )
        if target + "__provenance" in selected:
            _require(
                not bool(selected[target + "__provenance"].eq("official_gold").any()),
                "official gold provenance in train",
            )
    return selected.reset_index(drop=True)


def training_fingerprint(manifest: pd.DataFrame) -> str:
    train = training_rows(manifest)
    records = []
    for row in train.to_dict("records"):
        records.append(
            [row["study_uid"], row["leakage_group"]]
            + [int(row[t]) if row[t + "__mask"] == 1 else None for t in TARGETS]
        )
    return _digest(records)


def membership_hash(overlay: pd.DataFrame) -> str:
    rows = overlay.sort_values("study_uid")[OVERLAY_COLUMNS].to_dict("records")
    return _digest(rows)


def validate_overlay(manifest: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    train = training_rows(manifest)
    _require(
        list(overlay.columns) == OVERLAY_COLUMNS and not overlay.study_uid.duplicated().any(),
        "overlay columns or duplicate UID",
    )
    _require(
        set(overlay.study_uid) == set(train.study_uid), "missing/extra/nontrain overlay membership"
    )
    _require(
        bool(
            overlay.fold.map(
                lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, bool)
            ).all()
        ),
        "fold must be integer",
    )
    _require(bool(overlay.fold.ge(0).all()), "negative fold")
    expected = train.set_index("study_uid").leakage_group
    _require(
        bool(
            overlay.set_index("study_uid")
            .leakage_group.eq(expected.reindex(overlay.study_uid))
            .all()
        ),
        "overlay group differs from release",
    )
    _require(
        bool(overlay.groupby("leakage_group").fold.nunique().eq(1).all()), "group crosses folds"
    )
    folds = sorted(overlay.fold.unique().tolist())
    _require(
        len(folds) >= 2 and folds == list(range(len(folds))), "noncontiguous/insufficient folds"
    )
    return overlay.sort_values("study_uid").reset_index(drop=True)


def _counts(train: pd.DataFrame, overlay: pd.DataFrame) -> list[dict]:
    joined = train.merge(overlay[["study_uid", "fold"]], on="study_uid", validate="one_to_one")
    result = []
    for fold in sorted(overlay.fold.unique().tolist()):
        rows = joined.loc[joined.fold.eq(fold)]
        for target in TARGETS:
            observed = rows[target + "__mask"].eq(1)
            positive = observed & rows[target].eq(1)
            negative = observed & rows[target].eq(0)
            result.append(
                {
                    "fold": int(fold),
                    "target": target,
                    "studies": len(rows),
                    "components": int(rows.leakage_group.nunique()),
                    "positive": int(positive.sum()),
                    "negative": int(negative.sum()),
                    "masked": int((~observed).sum()),
                    "positive_components": int(rows.loc[positive, "leakage_group"].nunique()),
                    "negative_components": int(rows.loc[negative, "leakage_group"].nunique()),
                    "masked_components": int(rows.loc[~observed, "leakage_group"].nunique()),
                }
            )
    return result


def build_folds(
    manifest: pd.DataFrame, n_folds: int = 3, seed: int = DEFAULT_SEED
) -> tuple[pd.DataFrame, dict]:
    """Assign every accepted train row, including fully masked rows, as whole groups.

    Rare positive AND negative component indicators are iteratively balanced.
    This is a deterministic heuristic, not an optimal multilabel partition solver;
    it fails closed if sufficient global support nevertheless has a missing fold.
    """
    _require(type(n_folds) is int and n_folds >= 2, "fold count must be at least two")
    _require(type(seed) is int and 0 <= seed < 2**32, "invalid seed")
    train = training_rows(manifest)
    groups = sorted(train.leakage_group.unique())
    _require(len(groups) >= n_folds, "fewer groups than folds")
    group_index = {g: i for i, g in enumerate(groups)}
    study_group = train.leakage_group.map(group_index).to_numpy(dtype=int)
    counts = np.zeros((len(groups), 2 * len(TARGETS)), dtype=np.int64)
    for j, target in enumerate(TARGETS):
        mask = train[target + "__mask"].eq(1).to_numpy()
        for value in (0, 1):
            keep = mask & train[target].eq(value).to_numpy()
            np.add.at(counts[:, 2 * j + value], study_group[keep], 1)
    presence = (counts > 0).astype(np.int64)
    totals = presence.sum(axis=0)
    sizes = np.bincount(study_group, minlength=len(groups))
    # Stable identifier hashes break ties without depending on input row order.
    hashes = [hashlib.sha256(f"{seed}:{g}".encode()).hexdigest() for g in groups]
    rarity = (presence / np.maximum(totals, 1)).sum(axis=1)
    remaining = np.ones(len(groups), dtype=bool)
    remaining_classes = totals.copy()
    fold_classes = np.zeros((n_folds, counts.shape[1]), dtype=np.int64)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    assignment = np.full(len(groups), -1, dtype=np.int64)
    while remaining.any():
        supported = np.flatnonzero(remaining_classes > 0)
        chosen = (
            min(supported, key=lambda j: (remaining_classes[j], totals[j], j))
            if len(supported)
            else None
        )
        candidates = (
            np.flatnonzero(remaining & (presence[:, chosen] > 0))
            if chosen is not None
            else np.flatnonzero(remaining)
        )
        index = min(candidates, key=lambda i: (-rarity[i], -sizes[i], hashes[i], groups[i]))
        targets = np.maximum(totals / n_folds, 1.0)
        scores = []
        for fold in range(n_folds):
            before = (fold_classes[fold] - totals / n_folds) / targets
            after = (fold_classes[fold] + presence[index] - totals / n_folds) / targets
            score = float(np.sum(after**2 - before**2))
            tie = hashlib.sha256(f"{seed}:{groups[index]}:{fold}".encode()).hexdigest()
            scores.append(
                (
                    fold_classes[fold, chosen] if chosen is not None else 0,
                    score,
                    fold_sizes[fold],
                    tie,
                    fold,
                )
            )
        selected = min(scores)[-1]
        assignment[index] = selected
        fold_classes[selected] += presence[index]
        fold_sizes[selected] += sizes[index]
        remaining[index] = False
        remaining_classes -= presence[index]
    # Deterministically repair any absent supported class, preserving donor coverage.
    for _ in range(n_folds * counts.shape[1]):
        missing = [
            (f, j)
            for f in range(n_folds)
            for j in range(counts.shape[1])
            if totals[j] >= n_folds and fold_classes[f, j] == 0
        ]
        if not missing:
            break
        moved = False
        for destination, class_index in missing:
            donors = []
            for i in np.flatnonzero(presence[:, class_index]):
                donor = assignment[i]
                after = fold_classes[donor] - presence[i]
                protected = (totals >= n_folds) & (fold_classes[donor] > 0)
                if (
                    donor != destination
                    and np.all(after[protected] > 0)
                    and np.sum(assignment == donor) > 1
                ):
                    donors.append(i)
            if donors:
                i = min(donors, key=lambda x: (sizes[x], hashes[x], groups[x]))
                donor = assignment[i]
                assignment[i] = destination
                fold_classes[donor] -= presence[i]
                fold_classes[destination] += presence[i]
                moved = True
                break
        _require(moved, "cannot ensure supported class coverage without losing another class")
    _require(
        bool(np.all(fold_classes[:, totals >= n_folds] > 0)), "supported class missing from fold"
    )
    overlay = train[["study_uid", "leakage_group"]].copy()
    overlay["fold"] = assignment[study_group]
    overlay = validate_overlay(manifest, overlay)
    summary = {
        "schema_version": 1,
        "algorithm": ALGORITHM,
        "seed": seed,
        "n_folds": n_folds,
        "scope": "accepted_train_only_all_rows_including_fully_masked",
        "target_order": list(TARGETS),
        "n_studies": len(train),
        "n_components": len(groups),
        "training_input_sha256": training_fingerprint(manifest),
        "membership_sha256": membership_hash(overlay),
        "counts": _counts(train, overlay),
        "limitations": "Conservative leakage components are not verified patients; masked values are ignored. Component P/N/masked counts may overlap within a group.",
    }
    return overlay, summary


def write_folds(
    output: Path,
    manifest: pd.DataFrame,
    release_binding: dict,
    *,
    n_folds: int = 3,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Persist a new overlay after the caller's complete load_verified_release pass."""
    validate_checkpoint_binding(release_binding)
    overlay, summary = build_folds(manifest, n_folds, seed)
    raw = overlay.to_csv(index=False, lineterminator="\n").encode()
    summary.update(
        release_binding=release_binding,
        source_manifest_sha256=release_binding["manifest_sha256"],
        release_binding_sha256=release_binding["binding_sha256"],
        fold_csv_sha256=hashlib.sha256(raw).hexdigest(),
    )
    output.mkdir(parents=True, exist_ok=True)
    _require(
        not (output / "fold.csv").exists() and not (output / "folds.json").exists(),
        "existing overlay is immutable",
    )
    with (output / "fold.csv").open("xb") as stream:
        stream.write(raw)
    with (output / "folds.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def load_folds(path: Path, manifest: pd.DataFrame, release_binding: dict) -> pd.DataFrame:
    """Validate exact source, membership, masks, seed, class counts and algorithm replay."""
    validate_checkpoint_binding(release_binding)
    directory = path if path.is_dir() else path.parent
    _require(path.is_dir() or path.name in {"fold.csv", "folds.json"}, "unexpected overlay path")
    metadata = json.loads((directory / "folds.json").read_text())
    raw = (directory / "fold.csv").read_bytes()
    _require(metadata.get("release_binding") == release_binding, "incompatible release binding")
    _require(
        metadata.get("source_manifest_sha256") == release_binding["manifest_sha256"],
        "manifest hash mismatch",
    )
    _require(
        metadata.get("release_binding_sha256") == release_binding["binding_sha256"],
        "binding hash mismatch",
    )
    _require(
        metadata.get("fold_csv_sha256") == hashlib.sha256(raw).hexdigest(),
        "overlay file hash mismatch",
    )
    overlay = pd.read_csv(io.BytesIO(raw), dtype={"study_uid": str, "leakage_group": str})
    overlay = validate_overlay(manifest, overlay)
    expected, summary = build_folds(manifest, metadata["n_folds"], metadata["seed"])
    _require(
        all(metadata.get(k) == v for k, v in summary.items()), "overlay summary/source mismatch"
    )
    _require(overlay.equals(expected), "overlay membership differs from deterministic replay")
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()
    manifest, binding = load_verified_release(args.studies, args.artifact_root)
    summary = write_folds(args.output, manifest, binding, n_folds=args.folds, seed=args.seed)
    load_folds(args.output, manifest, binding)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "n_studies",
                    "n_components",
                    "membership_sha256",
                    "release_binding_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
