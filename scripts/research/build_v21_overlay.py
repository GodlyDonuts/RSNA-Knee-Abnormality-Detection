"""Build the v2.1 severity re-review overlay for one fold.

The v2.1 adjudication (`label-source-ablation-v1/v21_adjudication.csv`)
recounts v2-negative cells on the five severity targets whose evidence names
the finding at any severity (effusion, Baker's cyst, compartment-named
degenerative evidence). This emits a bound ``label_source_v1`` overlay:
v2 decided states plus the adjudicated positive flips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.supervision import roster_sha
from rsna.competitive.train import select_development
from rsna.schema import TARGETS

STATE_VALUES = {"positive": 1.0, "negative": 0.0}


def build(
    studies: Path,
    folds: Path,
    fold: int,
    adjudication: Path,
    output: Path,
    *,
    negative_flips: bool = False,
) -> dict:
    manifest = pd.read_parquet(studies)
    fold_frame = pd.read_csv(folds)
    adj = pd.read_csv(adjudication)
    required = {"study_uid", "target", "v21_call"}
    if not required <= set(adj.columns):
        raise ValueError("Adjudication file must carry study_uid, target, v21_call")
    if not adj.v21_call.isin({"positive", "negative"}).all():
        raise ValueError("v21_call must be positive or negative")
    if adj.duplicated(["study_uid", "target"]).any():
        raise ValueError("Duplicate adjudicated cell")
    flipped = {(r.study_uid, r.target) for r in adj.itertuples() if r.v21_call == "positive"}
    negated = {(r.study_uid, r.target) for r in adj.itertuples() if r.v21_call == "negative"}

    train, outer = select_development(manifest, fold_frame, fold, "outer_fold")
    entries, applied, applied_neg = [], 0, 0
    for _, row in train.iterrows():
        labels, mask = {}, {}
        for target in TARGETS:
            if (row.study_uid, target) in flipped:
                labels[target] = 1.0
                mask[target] = 1
                applied += 1
            elif negative_flips and (row.study_uid, target) in negated:
                labels[target] = 0.0
                mask[target] = 1
                applied_neg += 1
            else:
                state = row[f"{target}__state"]
                if state in STATE_VALUES:
                    labels[target] = STATE_VALUES[state]
                    mask[target] = 1
                else:
                    labels[target] = 0.0
                    mask[target] = 0
        entries.append({"study_uid": row.study_uid, "labels": labels, "mask": mask})

    artifact = {
        "schema_version": 1,
        "status": "label_source_experiment_only",
        "source": {
            "name": "v21pn_severity_rereview" if negative_flips else "v21_severity_rereview",
            "csv_sha256": file_sha256(adjudication),
            "revision": "report-supervision-v1",
        },
        "release_sha256": file_sha256(studies),
        "folds_sha256": file_sha256(folds),
        "fit_scope": "outer_fold",
        "fold": fold,
        "fitting_uid_sha256": roster_sha(train),
        "outer_uid_sha256": roster_sha(outer),
        "study_count": len(entries),
        "studies": entries,
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "output": str(output),
        "overlay_sha256": file_sha256(output),
        "fitting_studies": len(entries),
        "positive_flips": applied,
        "negative_flips": applied_neg,
        "supervised_cells": sum(sum(e["mask"].values()) for e in entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--negative-flips",
        action="store_true",
        help="Also unmask adjudicated-negative cells as explicit 0 labels "
        "(v2.1pn arm: positive + negative flips).",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.studies,
                args.folds,
                args.fold,
                args.adjudication,
                args.output,
                negative_flips=args.negative_flips,
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
