"""Plan additional fitting-only native MRI cache without changing accepted labels.

Retains the existing fold exclusion and report-to-image correspondence flags.
This prepares pixels only; source-label quality and a new training recipe need
separate decisions. Original observed-development cache plans are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from rsna.schema import TARGETS
from scripts.newton.native_cache_shards import (
    GIB,
    INVENTORY,
    INVENTORY_GZ_SHA,
    INVENTORY_SHA,
    SCIENCE,
    SIDES,
    SOURCE,
    canonical_sha,
    digest,
    effective_inventory,
    pack_studies,
    save_new,
)


def select_additional(studies: pd.DataFrame, folds: pd.DataFrame, fold: int = 0):
    training = studies.loc[studies.split.eq("train")].copy()
    if (
        training.study_uid.duplicated().any()
        or folds.study_uid.duplicated().any()
        or set(training.study_uid) != set(folds.study_uid)
        or fold not in set(folds.fold)
    ):
        raise ValueError("Exact distinct training fold membership required")
    joined = training.merge(folds[["study_uid", "fold"]], on="study_uid", validate="one_to_one")
    if joined.groupby("leakage_group").fold.nunique().gt(1).any():
        raise ValueError("Leakage groups cannot cross folds")
    forbidden = set(studies.loc[studies.split.ne("train"), "leakage_group"])
    if set(joined.leakage_group) & forbidden:
        raise ValueError("Training groups overlap a protected split")
    masks = joined[[target + "__mask" for target in TARGETS]]
    if not masks.isin([0, 1]).all().all():
        raise ValueError("Binary accepted masks required")
    selected = joined.loc[
        joined.fold.ne(fold)
        & joined.report_supervision_allowed.eq(True)  # noqa: E712
        & masks.sum(axis=1).eq(0)
    ].sort_values("study_uid")
    if selected.empty:
        raise ValueError("No eligible additional fitting studies")
    return selected.reset_index(drop=True), joined.loc[joined.fold.eq(fold)].copy()


def cohort(root: Path):
    reference = json.loads(
        (root / "initiative/audits/native-pilot-cache384-v1/selection.json").read_text()
    )
    bindings = dict(reference["input_bindings"])
    for name, record in bindings.items():
        path = root / name
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Original accepted input changed")
    bindings[INVENTORY] = {"bytes": (root / INVENTORY).stat().st_size, "sha256": INVENTORY_GZ_SHA}
    studies = pd.read_parquet(
        root / "initiative/releases/report-supervision-v1/studies.parquet",
        columns=["study_uid", "split", "leakage_group", "report_supervision_allowed"]
        + [target + "__mask" for target in TARGETS],
    )
    folds = pd.read_csv(root / "initiative/experiments/competitive-v1/folds/fold.csv")
    selected, excluded = select_additional(studies, folds)
    if len(selected) != 345 or len(excluded) != 1301:
        raise ValueError("Actual additional345/all-fold0-1301 membership changed")
    uids = set(selected.study_uid)
    series = pd.read_csv(
        root / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
    )
    series = series.loc[series.StudyInstanceUID.isin(uids)]
    index = pd.read_parquet(
        root / "initiative/audits/full-image-recovery-v1/audit/series_cache_index.parquet"
    )
    joined = series.merge(
        index.loc[index.study_uid.isin(uids) & index.raw_split.eq("train_series")],
        left_on=["StudyInstanceUID", "SeriesInstanceUID"],
        right_on=["study_uid", "series_uid"],
        validate="one_to_one",
    )
    if len(joined) != len(series) or set(joined.study_uid) != uids or len(joined) != 1769:
        raise ValueError("Complete additional-study accepted-series reference required")
    accepted = json.loads(
        (
            root
            / "initiative/audits/full-image-recovery-v1/train-selection/series_selection_plan.json"
        ).read_text()
    )
    if accepted["source_revision"] != SOURCE:
        raise ValueError("Pinned corrected original source required")
    parents = {row["series_uid"]: row for row in accepted["series"]}
    inventory = effective_inventory(root)
    counts = (
        inventory.loc[inventory.series_uid.isin(joined.series_uid)]
        .groupby(["study_uid", "series_uid"], sort=False)
        .agg(n_slices=("path", "size"), source_bytes=("bytes", "sum"))
    )
    rows = []
    for row in joined.itertuples(index=False):
        parent = parents[row.series_uid]
        source = counts.loc[(row.study_uid, row.series_uid)]
        if (
            not parent["retained"]
            or parent["study_uid"] != row.study_uid
            or int(source.n_slices) != int(row.n_slices)
        ):
            raise ValueError("Original source/reference identity differs")
        rows.append(
            {
                "study_uid": row.study_uid,
                "series_uid": row.series_uid,
                "plane": row.Anatomical_Plane,
                "fluid_sensitive": int(row.Fluid_Sensitive),
                "fat_suppression": int(row.Fat_Suppression),
                "selection_tier": int(row.Selection_Tier),
                "cache_path": row.cache_path,
                "cache_bytes": int(row.cache_bytes),
                "cache_sha256": row.cache_sha256,
                "cache_metadata_bytes": parent["cache_metadata"]["bytes"],
                "cache_metadata_sha256": parent["cache_metadata"]["sha256"],
                "n_slices": int(row.n_slices),
                "source_bytes": int(source.source_bytes),
            }
        )
    return rows, {
        "input_bindings": bindings,
        "inventory_parquet_sha256": INVENTORY_SHA,
        "training_uids": sorted(uids),
        "outer_uids": [],
        "excluded_all_fold0_roster_sha256": hashlib.sha256(
            json.dumps(sorted(excluded.study_uid), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "excluded_all_fold0_studies": len(excluded),
        "correspondence_exclusions_preserved": True,
        "new_training_or_label_admission": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh private additional-cohort plan required")
    rows, bindings = cohort(args.root)
    shards = pack_studies(rows, 4 * GIB)
    plan = {
        "schema_version": 1,
        "status": "additional_fitting_only_native_cache_planned_not_built",
        "scientific_source_git": SCIENCE,
        "source_revision": SOURCE,
        "repo_id": "GodlyDonuts/RSNA-Data",
        "cache_sides": list(SIDES),
        "study_count": 345,
        "series_count": len(rows),
        "slices": sum(row["n_slices"] for row in rows),
        "source_bytes": sum(row["source_bytes"] for row in rows),
        "target_values_or_reports_read": False,
        **bindings,
    }
    plan["selection_sha256"] = canonical_sha({**plan, "series": rows})
    args.output.mkdir(parents=True, mode=0o700)
    records = {}
    for shard in shards:
        name = f"shard-{shard['shard']:03d}.json"
        save_new(args.output / name, {**plan, **shard})
        records[name] = {
            "bytes": (args.output / name).stat().st_size,
            "sha256": digest(args.output / name),
            "owned_cache_archive_scratch_bound_bytes": shard[
                "owned_cache_archive_scratch_bound_bytes"
            ],
        }
    save_new(args.output / "manifest.json", {**plan, "shards": records})
    print(
        json.dumps(
            {
                k: plan[k]
                for k in [
                    "status",
                    "study_count",
                    "series_count",
                    "slices",
                    "source_bytes",
                    "selection_sha256",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
