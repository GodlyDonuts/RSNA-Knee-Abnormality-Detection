"""Build a per-target blended label-source overlay for one outer fold.

Combines two bound sources cell-by-cell: v2 review states on the targets
where v2's semantics measured better (binary-masked baseline semantics —
positive -> 1.0, negative -> 0.0, uncertain/not_mentioned -> masked), and the
public `llm_labels_v4_blend` probability on the severity-qualified targets
where its recall measured better (Medial OA, Lateral OA, Effusion, Baker's).

The per-target assignment was informed by the 58-study gold arbitration
(GOLD-ARBITRATION.md): this is an explicitly gold-informed upper-bound probe
under `label_source_experiment_only` status. Its result cannot be arbitrated
on gold or any gold-derived split — it answers "how much does a per-target
best-source policy buy", not "which source should ship".
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from rsna.competitive.train import select_development
from rsna.schema import TARGETS

STATES = {"positive", "negative", "uncertain", "not_mentioned"}
PUBLIC_TARGETS = {"Medial OA", "Lateral OA", "Effusion", "Baker's"}
V2_TARGETS = [t for t in TARGETS if t not in PUBLIC_TARGETS]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roster_sha(uids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(uids), separators=(",", ":")).encode()).hexdigest()


def blended_cells(studies: pd.DataFrame, public: pd.DataFrame, uid: str) -> tuple[dict, dict]:
    row = studies.loc[uid]
    prow = public.loc[uid]
    labels, mask = {}, {}
    for target in TARGETS:
        if target in PUBLIC_TARGETS:
            labels[target], mask[target] = float(prow[target]), 1
            continue
        state = row[f"{target}__state"]
        if state not in STATES:
            raise ValueError(f"Unknown v2 state {state!r} for {uid} / {target}")
        if state == "positive":
            labels[target], mask[target] = 1.0, 1
        elif state == "negative":
            labels[target], mask[target] = 0.0, 1
        else:
            labels[target], mask[target] = 0.0, 0
    return labels, mask


def build(
    studies: Path,
    public: Path,
    folds: Path,
    fold: int,
    output: Path,
) -> dict:
    manifest = pd.read_parquet(studies)
    if manifest.study_uid.duplicated().any():
        raise ValueError("Duplicate studies in release manifest")
    missing = [f"{t}__state" for t in TARGETS if f"{t}__state" not in manifest.columns]
    if missing:
        raise ValueError(f"Release manifest lacks graded state columns: {missing}")
    pub = pd.read_csv(public)
    uid_col = "StudyInstanceUID" if "StudyInstanceUID" in pub.columns else "study_uid"
    if pub[uid_col].duplicated().any():
        raise ValueError("Duplicate studies in public label source")
    pub = pub.set_index(uid_col)
    missing_pub = [t for t in TARGETS if t not in pub.columns]
    if missing_pub:
        raise ValueError(f"Public label source lacks targets: {missing_pub}")
    fold_frame = pd.read_csv(folds)
    by_uid = manifest.set_index("study_uid")
    train, outer = select_development(manifest, fold_frame, fold, "outer_fold")
    entries = []
    skipped_public = 0
    for uid in train.study_uid:
        if uid not in pub.index:
            skipped_public += 1
            continue
        labels, mask = blended_cells(by_uid, pub, uid)
        entries.append({"study_uid": uid, "labels": labels, "mask": mask})
    artifact = {
        "schema_version": 1,
        "status": "label_source_experiment_only",
        "source": {
            "name": "blended_v2plus_public_severity_v1",
            "csv_sha256": file_sha256(public),
            "release_sha256": file_sha256(studies),
            "public_targets": sorted(PUBLIC_TARGETS),
            "v2_targets": V2_TARGETS,
        },
        "release_sha256": file_sha256(studies),
        "folds_sha256": file_sha256(folds),
        "fit_scope": "outer_fold",
        "fold": fold,
        "fitting_uid_sha256": roster_sha(train.study_uid.tolist()),
        "outer_uid_sha256": roster_sha(outer.study_uid.tolist()),
        "study_count": len(entries),
        "studies": entries,
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
    supervised = sum(1 for e in entries for t in TARGETS if e["mask"][t] == 1)
    return {
        "output": str(output),
        "overlay_sha256": file_sha256(output),
        "fitting_studies": len(entries),
        "skipped_fitting_studies_missing_public": skipped_public,
        "supervised_cells": supervised,
        "total_cells": len(entries) * len(TARGETS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
