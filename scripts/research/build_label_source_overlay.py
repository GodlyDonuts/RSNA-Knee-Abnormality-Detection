"""Build a bound label-source overlay for one outer fold.

Emits a label_source_v1 artifact covering every fitting study so the trainer's
fitting-supervision mechanism can swap the complete label table while the
release, folds and evaluation frame stay immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from rsna.competitive.train import select_development
from rsna.schema import TARGETS

ID_COLUMN = "StudyInstanceUID"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roster_sha(uids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(uids), separators=(",", ":")).encode()).hexdigest()


def build(
    studies: Path,
    folds: Path,
    fold: int,
    source: Path,
    output: Path,
    source_name: str,
    source_revision: str,
) -> dict:
    manifest = pd.read_parquet(studies)
    fold_frame = pd.read_csv(folds)
    source_csv_sha = file_sha256(source)
    labels = pd.read_csv(source)
    if list(labels.columns) != [ID_COLUMN, *TARGETS]:
        raise ValueError("Source columns must be the competition UID plus all 12 targets")
    if labels[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate studies in label source")
    values = labels[labels[ID_COLUMN].isin(manifest.study_uid)].set_index(ID_COLUMN)
    train, outer = select_development(manifest, fold_frame, fold, "outer_fold")
    entries = []
    for uid in train.study_uid:
        if uid in values.index:
            row = values.loc[uid]
            study_labels = {t: float(row[t]) for t in TARGETS}
            if not all(0.0 <= v <= 1.0 for v in study_labels.values()):
                raise ValueError(f"Source label outside [0,1] for study {uid}")
            study_mask = {t: 1 for t in TARGETS}
        else:
            study_labels = {t: 0.0 for t in TARGETS}
            study_mask = {t: 0 for t in TARGETS}
        entries.append({"study_uid": uid, "labels": study_labels, "mask": study_mask})
    artifact = {
        "schema_version": 1,
        "status": "label_source_experiment_only",
        "source": {
            "name": source_name,
            "csv_sha256": source_csv_sha,
            "revision": source_revision,
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
    return {
        "output": str(output),
        "overlay_sha256": file_sha256(output),
        "fitting_studies": len(entries),
        "masked_out_studies": sum(1 for e in entries if not any(e["mask"].values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
