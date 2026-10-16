"""Add correspondence-eligible fitting rows while keeping the outer endpoint fixed.

Accepted release masks are not edited. A separately bound public-source overlay
supplies the new rows; every old fitting cell is preserved exactly.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsna.competitive.label_source import load_fitting_label_source
from rsna.competitive.supervision import roster_sha
from rsna.competitive.train import select_development
from rsna.schema import TARGETS

VERSION = "fitting_report_allowed_expansion_v1"


def select_population(studies: pd.DataFrame, folds: pd.DataFrame, fold: int = 0):
    baseline, outer = select_development(studies, folds, fold)
    joined = studies.loc[studies.split.eq("train")].merge(
        folds[["study_uid", "fold"]], on="study_uid", validate="one_to_one"
    )
    masks = joined[[t + "__mask" for t in TARGETS]]
    if not masks.isin([0, 1]).all().all():
        raise ValueError("Original binary masks required")
    if joined.groupby("leakage_group").fold.nunique().gt(1).any():
        raise ValueError("Leakage groups cannot cross folds")
    extra = (
        joined.loc[
            joined.fold.ne(fold)
            & joined.report_supervision_allowed.eq(True)  # noqa: E712
            & masks.sum(axis=1).eq(0)
        ]
        .sort_values("study_uid")
        .reset_index(drop=True)
    )
    if extra.empty or not baseline.report_supervision_allowed.eq(True).all():  # noqa: E712
        raise ValueError("Eligible additional rows and safe baseline correspondence required")
    fitting = pd.concat([baseline, extra], ignore_index=True).sort_values("study_uid")
    fitting = fitting.reset_index(drop=True)
    excluded = joined.loc[joined.fold.eq(fold)]
    if (
        fitting.study_uid.duplicated().any()
        or set(fitting.study_uid) & set(excluded.study_uid)
        or set(fitting.leakage_group) & set(excluded.leakage_group)
    ):
        raise ValueError("Expanded fitting overlaps the complete outer fold")
    proof = {
        "version": VERSION,
        "fold": fold,
        "baseline_fitting_count": len(baseline),
        "additional_fitting_count": len(extra),
        "expanded_fitting_count": len(fitting),
        "baseline_fitting_uid_sha256": roster_sha(baseline),
        "additional_fitting_uid_sha256": roster_sha(extra),
        "expanded_fitting_uid_sha256": roster_sha(fitting),
        "outer_endpoint_count": len(outer),
        "outer_endpoint_uid_sha256": roster_sha(outer),
        "excluded_all_fold_studies": len(excluded),
        "excluded_all_fold_uid_sha256": roster_sha(excluded),
        "accepted_masks_labels_and_outer_unchanged": True,
    }
    return fitting, outer, proof


def selected_source(raw: bytes, fitting_uids: set[str]) -> pd.DataFrame:
    """Parse numeric target values only after the exact fitting UID filter.

    Keeping original text cells for pandas' CSV parser preserves the source
    conversion used by the existing overlay builder, including decimal rounding.
    """
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    header = next(reader)
    if header != ["StudyInstanceUID", *TARGETS]:
        raise ValueError("Exact original twelve-target CSV header required")
    selected = io.StringIO()
    writer = csv.writer(selected)
    writer.writerow(header)
    seen, retained = set(), set()
    for row in reader:
        if len(row) != 13 or not row[0] or row[0] in seen:
            raise ValueError("Complete distinct source identities required")
        seen.add(row[0])
        if row[0] in fitting_uids:
            writer.writerow(row)
            retained.add(row[0])
    if retained != fitting_uids:
        raise ValueError("Every expanded fitting row must have source coverage")
    frame = pd.read_csv(
        io.StringIO(selected.getvalue()), dtype={"StudyInstanceUID": str}
    ).set_index("StudyInstanceUID")
    values = frame[list(TARGETS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Observed fitting source probabilities must be finite in [0,1]")
    return frame


def build_overlay(
    studies: pd.DataFrame,
    folds: pd.DataFrame,
    old_overlay: Path,
    old_overlay_sha256: str,
    source_csv: Path,
    source_csv_sha256: str,
    output: Path,
    *,
    release_sha256: str,
    folds_sha256: str,
    fold: int = 0,
) -> dict:
    baseline, original_outer = select_development(studies, folds, fold)
    fitting, outer, population = select_population(studies, folds, fold)
    pd.testing.assert_frame_equal(original_outer, outer)
    old, _ = load_fitting_label_source(
        baseline,
        outer,
        old_overlay,
        old_overlay_sha256,
        release_sha256=release_sha256,
        folds_sha256=folds_sha256,
        fold=fold,
    )
    raw = source_csv.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source_csv_sha256:
        raise ValueError("Exact already-admitted public CSV required")
    source = selected_source(raw, set(fitting.study_uid))
    previous = json.loads(old_overlay.read_text())
    if previous["source"]["csv_sha256"] != source_csv_sha256:
        raise ValueError("Population expansion cannot silently change label source")
    old_entries = {e["study_uid"]: e for e in previous["studies"]}
    for _, row in old.iterrows():
        for target in TARGETS:
            if row[target + "__mask"]:
                actual = np.float64(row[target]).tobytes()
                expected = np.float64(source.loc[row.study_uid, target]).tobytes()
                if actual != expected:
                    raise ValueError("Existing observed fitting values differ from original source")
    entries = []
    for uid in fitting.study_uid:
        if uid in old_entries:
            entries.append(old_entries[uid])
        else:
            entries.append(
                {
                    "study_uid": uid,
                    "labels": {t: float(source.loc[uid, t]) for t in TARGETS},
                    "mask": {t: 1 for t in TARGETS},
                }
            )
    artifact = {
        "schema_version": 1,
        "status": "label_source_experiment_only",
        "fit_scope": "outer_fold",
        "fold": fold,
        "source": previous["source"],
        "release_sha256": release_sha256,
        "folds_sha256": folds_sha256,
        "fitting_uid_sha256": roster_sha(fitting),
        "outer_uid_sha256": roster_sha(outer),
        "study_count": len(entries),
        "population_expansion": {**population, "baseline_overlay_sha256": old_overlay_sha256},
        "studies": entries,
    }
    with output.open("x") as handle:
        handle.write(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    loaded, _ = load_fitting_label_source(
        fitting,
        outer,
        output,
        digest,
        release_sha256=release_sha256,
        folds_sha256=folds_sha256,
        fold=fold,
    )
    retained = loaded.loc[loaded.study_uid.isin(old.study_uid)].reset_index(drop=True)
    pd.testing.assert_frame_equal(retained, old)
    return {**population, "overlay_sha256": digest, "source_csv_sha256": source_csv_sha256}
