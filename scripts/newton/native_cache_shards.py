"""Plan bounded, study-complete original-pixel cache shards without reading targets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from rsna.schema import TARGETS

SCIENCE = "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
SOURCE = "0b5e410288232f570c69ba4f44723356e6ae6ab7"
INVENTORY = "initiative/audits/final-release-technical-inputs-v1/inventory_effective.parquet.gz"
INVENTORY_GZ_SHA = "33065e6a60b878e5eda90ff2ffe865b5778cdcfa668adab23f2fd65bccdde240"
INVENTORY_SHA = "68d202eb951babc9365eddd2f7a1219201cd7b0905f04afdace785ad0e079537"
SIDES = (256, 384)
GIB = 1024**3


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def effective_inventory(root: Path) -> pd.DataFrame:
    path = root / INVENTORY
    if digest(path) != INVENTORY_GZ_SHA:
        raise ValueError("effective_inventory_compressed_binding_changed")
    with gzip.open(path, "rb") as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != INVENTORY_SHA:
        raise ValueError("effective_inventory_binding_changed")
    frame = pd.read_parquet(io.BytesIO(data))
    if frame.path.duplicated().any() or len(frame) != 819635:
        raise ValueError("effective_inventory_membership_changed")
    return frame


def bound_series(row: dict) -> dict:
    """Compression/header/metadata/tar overhead bounds; never assume compression gain."""
    count = row["n_slices"]
    if not isinstance(count, int) or count < 1:
        raise ValueError("positive_source_slice_count_required")
    bounds = {}
    for side in SIDES:
        raw = count * side * side * 2
        # zlib overhead is small even for incompressible data. The explicit
        # extra MiB and 2KiB per slice also cover ZIP, provenance JSON and tar.
        bounds[str(side)] = raw + (raw + 999) // 1000 + 1024**2 + 2048 * count
    return {**row, "cache_archive_upper_bytes_by_side": bounds}


def pack_studies(rows: list[dict], archive_bound_bytes: int) -> list[dict]:
    if archive_bound_bytes < GIB:
        raise ValueError("shard_bound_must_be_at_least_one_gib")
    identities = {(r["study_uid"], r["series_uid"]) for r in rows}
    if not rows or len(identities) != len(rows):
        raise ValueError("nonempty_unique_series_required")
    studies: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: (r["study_uid"], r["series_uid"])):
        studies.setdefault(row["study_uid"], []).append(bound_series(row))
    groups: list[list[dict]] = []
    group: list[dict] = []
    size = 20480
    for members in studies.values():
        needed = sum(sum(r["cache_archive_upper_bytes_by_side"].values()) for r in members)
        if needed + 20480 > archive_bound_bytes:
            raise ValueError("single_study_exceeds_shard_bound_no_silent_split")
        if group and size + needed > archive_bound_bytes:
            groups.append(group)
            group, size = [], 20480
        group.extend(members)
        size += needed
    if group:
        groups.append(group)
    shards = []
    for number, members in enumerate(groups):
        bounds = {
            str(side): 10240
            + sum(r["cache_archive_upper_bytes_by_side"][str(side)] for r in members)
            for side in SIDES
        }
        pair = sum(bounds.values())
        shards.append(
            {
                "shard": number,
                "study_uids": sorted({r["study_uid"] for r in members}),
                "series": members,
                "series_count": len(members),
                "slices": sum(r["n_slices"] for r in members),
                "source_bytes": sum(r["source_bytes"] for r in members),
                "largest_series_source_bytes": max(r["source_bytes"] for r in members),
                "archive_upper_bytes_by_side": bounds,
                "paired_archive_upper_bytes": pair,
                # Retain both arrays and archives, then stream verification bytes.
                "owned_cache_archive_scratch_bound_bytes": 2 * pair + GIB,
                "regular_members_per_side": 2 * len(members),
            }
        )
    return shards


def cohort(root: Path) -> tuple[list[dict], dict]:
    reference = json.loads(
        (root / "initiative/audits/native-pilot-cache384-v1/selection.json").read_text()
    )
    bindings = dict(reference["input_bindings"])
    for name, row in bindings.items():
        path = root / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("accepted_selection_input_changed")
    bindings[INVENTORY] = {"bytes": (root / INVENTORY).stat().st_size, "sha256": INVENTORY_GZ_SHA}
    masks = [target + "__mask" for target in TARGETS]
    studies = pd.read_parquet(
        root / "initiative/releases/report-supervision-v1/studies.parquet",
        columns=["study_uid", "split", *masks],
    )
    folds = pd.read_csv(
        root / "initiative/experiments/competitive-v1/folds/fold.csv",
        dtype={"study_uid": str, "leakage_group": str},
    )
    studies = studies.loc[studies.split.eq("train") & studies[masks].any(axis=1)]
    studies = studies.merge(folds[["study_uid", "fold"]], on="study_uid", validate="one_to_one")
    if len(studies) != 3403 or int(studies.fold.eq(0).sum()) != 1156:
        raise ValueError("development_membership_changed")
    uids = set(studies.study_uid)
    series = pd.read_csv(
        root / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    series = series.loc[series.StudyInstanceUID.isin(uids)]
    index = pd.read_parquet(
        root / "initiative/audits/full-image-recovery-v1/audit/series_cache_index.parquet"
    )
    selected = series.merge(
        index.loc[index.study_uid.isin(uids) & index.raw_split.eq("train_series")],
        left_on=["StudyInstanceUID", "SeriesInstanceUID"],
        right_on=["study_uid", "series_uid"],
        validate="one_to_one",
    )
    accepted = json.loads(
        (
            root
            / "initiative/audits/full-image-recovery-v1/train-selection/series_selection_plan.json"
        ).read_text()
    )
    if accepted["source_revision"] != SOURCE:
        raise ValueError("accepted_source_revision_changed")
    parents = {r["series_uid"]: r for r in accepted["series"]}
    inventory = effective_inventory(root)
    inventory = inventory.loc[inventory.series_uid.isin(set(selected.series_uid))]
    counts = inventory.groupby(["study_uid", "series_uid"], sort=False).agg(
        n_slices=("path", "size"), source_bytes=("bytes", "sum")
    )
    rows = []
    for r in selected.itertuples(index=False):
        parent = parents[r.series_uid]
        if not parent["retained"] or parent["study_uid"] != r.study_uid:
            raise ValueError("unaccepted_series_in_shard_plan")
        item = counts.loc[(r.study_uid, r.series_uid)]
        if int(item.n_slices) != int(r.n_slices):
            raise ValueError("effective_source_slice_membership_changed")
        rows.append(
            {
                "study_uid": r.study_uid,
                "series_uid": r.series_uid,
                "plane": r.Anatomical_Plane,
                "fluid_sensitive": int(r.Fluid_Sensitive),
                "fat_suppression": int(r.Fat_Suppression),
                "selection_tier": int(r.Selection_Tier),
                "cache_path": r.cache_path,
                "cache_bytes": int(r.cache_bytes),
                "cache_sha256": r.cache_sha256,
                "cache_metadata_bytes": parent["cache_metadata"]["bytes"],
                "cache_metadata_sha256": parent["cache_metadata"]["sha256"],
                "n_slices": int(r.n_slices),
                "source_bytes": int(item.source_bytes),
            }
        )
    if len(rows) != 18906 or sum(r["n_slices"] for r in rows) != 643501:
        raise ValueError("full_eligible_series_membership_changed")
    return rows, {
        "input_bindings": bindings,
        "inventory_parquet_sha256": INVENTORY_SHA,
        "training_uids": sorted(studies.loc[studies.fold.ne(0), "study_uid"]),
        "outer_uids": sorted(studies.loc[studies.fold.eq(0), "study_uid"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-shard-gib", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("distinct_shard_plan_output_required")
    rows, bindings = cohort(args.root)
    shards = pack_studies(rows, args.paired_shard_gib * GIB)
    plan = {
        "schema_version": 1,
        "status": "full_native_cache_shards_metadata_prepared_not_built",
        "scientific_source_git": SCIENCE,
        "source_revision": SOURCE,
        "repo_id": "GodlyDonuts/RSNA-Data",
        "cache_sides": list(SIDES),
        "study_count": 3403,
        "series_count": len(rows),
        "slices": sum(r["n_slices"] for r in rows),
        "source_bytes": sum(r["source_bytes"] for r in rows),
        "shard_count": len(shards),
        "paired_shard_upper_bytes": args.paired_shard_gib * GIB,
        "target_values_or_reports_read": False,
        "gold_or_confirmation_labels_read": False,
        "accepted_cache_labels_masks_folds_mutated": False,
        "full_native_cache_built": False,
        **bindings,
    }
    plan["selection_sha256"] = canonical_sha({**plan, "series": rows})
    args.output.mkdir(mode=0o700, parents=True)
    records = {}
    for shard in shards:
        name = f"shard-{shard['shard']:03d}.json"
        save_new(args.output / name, {**plan, **shard})
        records[name] = {
            "bytes": (args.output / name).stat().st_size,
            "sha256": digest(args.output / name),
            "studies": len(shard["study_uids"]),
            "series": shard["series_count"],
            "slices": shard["slices"],
            "paired_archive_upper_bytes": shard["paired_archive_upper_bytes"],
            "owned_cache_archive_scratch_bound_bytes": shard[
                "owned_cache_archive_scratch_bound_bytes"
            ],
        }
    save_new(args.output / "manifest.json", {**plan, "shards": records})
    print(
        json.dumps(
            {
                k: plan[k]
                for k in (
                    "status",
                    "study_count",
                    "series_count",
                    "slices",
                    "source_bytes",
                    "shard_count",
                    "selection_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
