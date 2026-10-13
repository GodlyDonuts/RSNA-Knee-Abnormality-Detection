"""Prepared owning-node fitting-only input comparison: public Pilkwang v15 vs current inputs.

NOT RUN by the authoring session. Requires evc31, CPU only, the exact v15 source bytes,
the private Pilkwang manifest and m_e5427d6c21.pt (hash-checked, not loaded). Pixels and
study IDs stay on the node; stdout/--out receive only sanitized aggregates and provenance.
Fitting labels the current dataset attaches to each item are discarded unread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.research.public_dinov2_input_reference import (  # noqa: E402
    compare,
    contract,
    provenance,
    reference,
)

MAX_ROSTER = 64
ROSTER_SALT = "public-dinov2-input-parity-v1"
CURRENT_INPUTS = {"slices": 16, "side": 384, "max_series": 6}
RAW_SPLIT = "train_series"


def select_roster(study_uids: pd.Series, size: int, salt: str = ROSTER_SALT) -> list[str]:
    """Outcome-free bounded roster: ranks fitting UIDs by salted SHA256 only."""
    if not 0 < size <= MAX_ROSTER:
        raise ValueError(f"Roster size must be in 1..{MAX_ROSTER}")
    uids = sorted(set(map(str, study_uids)))
    ranked = sorted(uids, key=lambda u: hashlib.sha256(f"{salt}:{u}".encode()).hexdigest())
    return ranked[:size]


def member_pixel_group(manifest: bytes, checkpoint: str = provenance.CHECKPOINT) -> dict:
    """The checkpoint's own manifest entry: its `pixel_group` and model config, verbatim."""
    members = [m for m in json.loads(manifest).get("members", []) if m.get("file") == checkpoint]
    if len(members) != 1:
        raise ValueError(f"Manifest must list {checkpoint} exactly once")
    group = members[0]["pixel_group"]
    return {
        "pixel_group": json.loads(group) if isinstance(group, str) else dict(group),
        "config": dict(members[0].get("config", {})),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def header_field_mm(study_dir: Path) -> list[float]:
    fields = []
    for series_dir in sorted(p for p in study_dir.iterdir() if p.is_dir()):
        first = next(iter(sorted(series_dir.glob("*.dcm"))), None)
        if first is None:
            continue
        ds = pydicom.dcmread(first, stop_before_pixels=True)
        spacing = [float(v) for v in getattr(ds, "PixelSpacing", [])]
        fields.append(compare.physical_field_mm((int(ds.Rows), int(ds.Columns)), spacing))
    return fields


def distinct_channels(images: torch.Tensor, mask: torch.Tensor) -> list[int]:
    """Distinct consecutive slice planes per occupied slot (band padding repeats the last)."""
    return [
        1 + int((images[s, 1:] != images[s, :-1]).flatten(1).any(1).sum())
        for s in range(images.shape[0])
        if mask[s]
    ]


def run(args: argparse.Namespace) -> dict:
    if os.uname().nodename != "evc31" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Original owning-node CPU-only execution required")
    torch.set_num_threads(1)
    if file_sha256(Path(args.checkpoint)) != provenance.CHECKPOINT_SHA256:
        raise ValueError("Checkpoint bytes differ from the audited public member")
    manifest = Path(args.member_manifest).read_bytes()
    member = member_pixel_group(manifest)
    namespace, extraction = reference.load(Path(args.source).read_bytes())
    reference.configure(namespace, member["pixel_group"])

    from rsna.competitive.data import CompetitiveStudyDataset
    from rsna.competitive.folds import load_folds
    from rsna.competitive.train import select_development
    from rsna.data import select_series
    from rsna.release import load_verified_release

    frame, binding = load_verified_release(Path(args.studies), Path(args.artifact_root))
    folds = load_folds(Path(args.folds), frame, binding)
    fitting, outer = select_development(frame, folds, 0)
    if (len(fitting), len(outer)) != (2247, 1156):
        raise ValueError("Original fitting/evaluation population changed")
    roster = select_roster(fitting.study_uid, args.roster_size)
    if set(roster) & set(map(str, outer.study_uid)):
        raise ValueError("Roster crosses the evaluation boundary")
    series = pd.read_csv(args.series)
    plane_map = dict(zip(series.SeriesInstanceUID.astype(str), series.Anatomical_Plane))
    rows = fitting[fitting.study_uid.astype(str).isin(roster)].copy()
    dataset = CompetitiveStudyDataset(
        rows.reset_index(drop=True), series, Path(args.cache), **CURRENT_INPUTS
    )
    keys = (
        "current_field_mm",
        "public_field_mm",
        "current_slots",
        "public_slots",
        "slot_series_overlap",
        "current_slices",
        "public_distinct_slices",
        "public_geometry_ordered_fraction",
        "current_min",
        "current_max",
        "public_min",
        "public_max",
        "public_mean",
    )
    stats: dict[str, list[float]] = {k: [] for k in keys}
    counts: dict[str, int] = {}
    slot_hits = [0] * contract.RECIPE["slots"].value
    refusals: dict[str, int] = {}
    for index, uid in enumerate(map(str, dataset.studies.study_uid)):
        study_dir = Path(args.dicom_root) / RAW_SPLIT / uid
        try:
            current = dataset[index]
            public = reference.render_study(namespace, study_dir, plane_map, RAW_SPLIT)
            contract.validate_rendering(public["images"], public["slot_mask"])
        except Exception as error:  # counted by type only; no IDs or messages leave the node
            refusals[type(error).__name__] = refusals.get(type(error).__name__, 0) + 1
            continue
        fields = header_field_mm(study_dir)
        stats["current_field_mm"].append(float(np.nanmedian(fields)) if fields else np.nan)
        for header in filter(None, public["slot_header"]):
            stats["public_field_mm"].append(reference.public_field_mm(namespace, *header))
        mask = public["slot_mask"]
        stats["current_slots"].append(int(current["series_mask"].sum()))
        stats["public_slots"].append(int(mask.sum()))
        for slot, present in enumerate(mask.tolist()):
            slot_hits[slot] += int(present)
        chosen = select_series(dataset.series[uid], CURRENT_INPUTS["max_series"], False)
        current_series = {str(e["SeriesInstanceUID"]) for e in chosen}
        stats["slot_series_overlap"].append(len(set(filter(None, public["slot_series"])) & current_series))
        public_series = [s for s in public["slot_series"] if s]
        counts["public_duplicate_series_slots"] = counts.get("public_duplicate_series_slots", 0) + (
            len(public_series) - len(set(public_series))
        )
        stats["current_slices"].append(int(current["slice_mask"].sum()))
        stats["public_distinct_slices"].extend(distinct_channels(public["images"], mask))
        ordered = public["geometry_ordered"]
        if ordered and ordered[1]:
            stats["public_geometry_ordered_fraction"].append(ordered[0] / ordered[1])
        source = public["laterality_source"] or "none"
        counts[f"laterality_{source}"] = counts.get(f"laterality_{source}", 0) + 1
        if public["laterality"] == "R":
            counts["laterality_right_mirrored"] = counts.get("laterality_right_mirrored", 0) + 1
        cur = compare.tensor_stats(current["images"].float())
        pub = compare.tensor_stats(public["images"].float(), mask)
        for side, values in (("current", cur), ("public", pub)):
            for key in ("min", "max"):
                stats[f"{side}_{key}"].append(values.get(key, np.nan))
        stats["public_mean"].append(pub.get("mean", np.nan))
        del current
    report = {
        "roster_size": len(roster),
        "roster_rule": {"salt": ROSTER_SALT, "population": "fold0 fitting only", "outcomes": 0},
        "provenance": {
            **extraction,
            "checkpoint": provenance.CHECKPOINT,
            "checkpoint_sha256": provenance.CHECKPOINT_SHA256,
            "checkpoint_loaded": False,
            "member_manifest_sha256": provenance.sha256(manifest),
            "member_pixel_group": member["pixel_group"],
            "member_config": member["config"],
            "release_sha256": binding["manifest_sha256"],
            "current_inputs": CURRENT_INPUTS,
            "public_recipe": {k: f.value for k, f in contract.RECIPE.items()},
        },
        "fov_mm": {
            "public_rendered_per_slot": compare.summarize(stats["public_field_mm"]),
            "current_full_field": compare.summarize(stats["current_field_mm"]),
        },
        "slot_occupancy": {
            **{
                k: compare.summarize(stats[k])
                for k in ("current_slots", "public_slots", "slot_series_overlap")
            },
            "public_per_slot": dict(zip([s[0] for s in namespace["SLOTS"]], slot_hits)),
        },
        "channel_coverage": {
            k: compare.summarize(stats[k]) for k in ("current_slices", "public_distinct_slices")
        },
        "order": {
            "public_geometry_ordered_fraction": compare.summarize(
                stats["public_geometry_ordered_fraction"]
            ),
        },
        "comparison_counts": counts,
        "tensor": {
            k: compare.summarize(v) for k, v in stats.items() if k.endswith(("min", "max", "mean"))
        },
        "refusals": refusals,
    }
    return compare.sanitize(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("source", "member-manifest", "checkpoint", "studies", "artifact-root", "folds")
    for name in names + ("series", "cache", "dicom-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--roster-size", type=int, default=32)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(run(args), indent=2, sort_keys=True, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
