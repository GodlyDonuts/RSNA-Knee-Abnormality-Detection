"""Build a graded-target overlay for one outer fold from v2 review states.

Emits a label_source_v1-compatible artifact so the existing bound admission
path can carry ordinal supervision: positive -> 1.0, negative -> 0.0,
uncertain -> `--uncertain` (default 0.5), not_mentioned -> masked. Compared
with the binary baseline this is a strictly paired contrast on how the v2
`uncertain` state is fed to the fitting objective: masked (baseline) vs
supervised at an intermediate value (this overlay).
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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roster_sha(uids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(uids), separators=(",", ":")).encode()).hexdigest()


def graded_cells(studies: pd.DataFrame, uid: str, uncertain: float) -> tuple[dict, dict]:
    row = studies.loc[uid]
    labels, mask = {}, {}
    for target in TARGETS:
        state = row[f"{target}__state"]
        if state not in STATES:
            raise ValueError(f"Unknown v2 state {state!r} for {uid} / {target}")
        if state == "positive":
            labels[target], mask[target] = 1.0, 1
        elif state == "negative":
            labels[target], mask[target] = 0.0, 1
        elif state == "uncertain":
            labels[target], mask[target] = float(uncertain), 1
        else:
            labels[target], mask[target] = 0.0, 0
    return labels, mask


def build(
    studies: Path,
    folds: Path,
    fold: int,
    output: Path,
    uncertain: float,
) -> dict:
    if not 0.0 <= uncertain <= 1.0:
        raise ValueError("Uncertain-cell target must lie in [0,1]")
    manifest = pd.read_parquet(studies)
    if manifest.study_uid.duplicated().any():
        raise ValueError("Duplicate studies in release manifest")
    missing = [f"{t}__state" for t in TARGETS if f"{t}__state" not in manifest.columns]
    if missing:
        raise ValueError(f"Release manifest lacks graded state columns: {missing}")
    fold_frame = pd.read_csv(folds)
    by_uid = manifest.set_index("study_uid")
    train, outer = select_development(manifest, fold_frame, fold, "outer_fold")
    entries = []
    for uid in train.study_uid:
        labels, mask = graded_cells(by_uid, uid, uncertain)
        entries.append({"study_uid": uid, "labels": labels, "mask": mask})
    artifact = {
        "schema_version": 1,
        "status": "label_source_experiment_only",
        "source": {
            "name": f"v2_graded_states_u{uncertain:g}",
            "csv_sha256": file_sha256(studies),
            "revision": "report-supervision-v1",
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
    supervised_uncertain = sum(
        1
        for e in entries
        for t in TARGETS
        if e["mask"][t] == 1 and e["labels"][t] == float(uncertain)
    )
    return {
        "output": str(output),
        "overlay_sha256": file_sha256(output),
        "fitting_studies": len(entries),
        "uncertain_target": float(uncertain),
        "uncertain_cells_supervised": supervised_uncertain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--uncertain", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
