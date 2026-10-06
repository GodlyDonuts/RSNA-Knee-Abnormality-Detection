"""Build a bound canonical-side (laterality) artifact for one outer fold.

Derives each study's imaged side from the series-header census: the median
lateral ``ImagePositionPatient`` offset of its sagittal series (validated at
97.5-99.1% agreement with DICOM laterality tags in
``initiative/audits/series-headers-census-v1``). Studies that are ambiguous,
bilateral, or missing geometry emit ``side: null`` and are never mirrored.

Emits a canonical_side_v1 artifact covering exactly the train+outer roster so
the trainer's canonical-sides mechanism can canonicalize every series to one
side. Evaluation frames stay immutable — the same table binds both.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

from rsna.competitive.train import select_development

AMBIGUOUS_MM = 20.0


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roster_sha(uids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(uids), separators=(",", ":")).encode()).hexdigest()


def derive_sides(census: Path) -> pd.DataFrame:
    records = [json.loads(line) for line in census.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(records)
    frame = frame[frame.parse_ok]
    frame["x"] = frame.ImagePositionPatient.apply(
        lambda value: ast.literal_eval(value)[0] if value else float("nan")
    )
    sagittal = frame[(frame.anatomical_plane == "Sagittal")].dropna(subset=["x"])
    stats = sagittal.groupby("study_uid").x.agg(["median", "count"])

    def bilateral(group: pd.Series) -> bool:
        values = group.dropna()
        return bool((values > AMBIGUOUS_MM).any() and (values < -AMBIGUOUS_MM).any())

    spans = frame.groupby("study_uid").x.agg(bilateral).rename("bilateral")
    sides = stats.join(spans)
    sides["side"] = None
    resolved = (sides["median"].abs() >= AMBIGUOUS_MM) & ~sides["bilateral"]
    sides.loc[resolved, "side"] = ["L" if x > 0 else "R" for x in sides.loc[resolved, "median"]]
    return sides[["side", "median", "bilateral"]].rename(columns={"median": "median_x_mm"})


def build(
    studies: Path,
    folds: Path,
    fold: int,
    census: Path,
    canonical: str,
    output: Path,
) -> dict:
    if canonical not in ("L", "R"):
        raise ValueError("Canonical side must be L or R")
    manifest = pd.read_parquet(studies)
    fold_frame = pd.read_csv(folds)
    derived = derive_sides(census)
    train, outer = select_development(manifest, fold_frame, fold, "outer_fold")
    roster = pd.concat([train[["study_uid"]], outer[["study_uid"]]])
    sides = {
        uid: (derived.at[uid, "side"] if uid in derived.index else None) for uid in roster.study_uid
    }
    artifact = {
        "schema_version": 1,
        "status": "canonical_side_experiment_only",
        "method": "median sagittal ImagePositionPatient lateral offset",
        "ambiguous_threshold_mm": AMBIGUOUS_MM,
        "canonical_side": canonical,
        "release_sha256": file_sha256(studies),
        "folds_sha256": file_sha256(folds),
        "fit_scope": "outer_fold",
        "fold": fold,
        "fitting_uid_sha256": roster_sha(train.study_uid.tolist()),
        "outer_uid_sha256": roster_sha(outer.study_uid.tolist()),
        "study_count": len(sides),
        "sides": sides,
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
    mirrored = sum(1 for side in sides.values() if side is not None and side != canonical)
    return {
        "output": str(output),
        "artifact_sha256": file_sha256(output),
        "studies": len(sides),
        "mirrored_studies": mirrored,
        "unresolved_side_studies": sum(1 for side in sides.values() if side is None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--canonical", default="R", choices=("L", "R"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
