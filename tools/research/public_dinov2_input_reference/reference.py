"""Callable Pilkwang v15 renderer: exact reviewed source nodes plus explicit adapters.

Original work: pilkwang/rsna-knee-baseline-v1 (Kaggle notebook version 15) by Pilkwang.
Every pixel decision (header probe, series annotation, slot choice, laterality, slice
order, band sampling, 130 mm crop, percentile window, resize, mirroring, cache layout and
TTA windows) is executed by the original functions. This module only supplies what the
notebook got from its top level: the metadata frame `walk` built, the `SLOT_SCHEME`
choice, `log`, the time origin, and an unset `ORDER_CACHE`.
"""

from __future__ import annotations

import gc
import os
import re
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from . import provenance

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = ROOT / "initiative/audits/public-dinov2-source-handoff-20261013-v1/source.notebook.json"

# Manifest fields shared by the downloaded Pilkwang members (public-solution-intelligence
# README); the fingerprint receipt independently binds img/group. `rules` and `slots` are
# per-member manifest fields not retained in Git and must come from the private manifest.
MANIFEST_BOUND_PIXEL_GROUP = {"img": 336, "group": 3, "slices": 12, "crop_mm": 130.0, "band": [0.2, 0.8]}
SLOT_SCHEMES = ("SLOTS_RECOVERED", "SLOTS_PUBLIC")


def adapters(messages: list[str] | None = None) -> dict[str, Any]:
    sink = [] if messages is None else messages
    return {
        "log": sink.append,
        "T0": time.time(),
        "time": time,
        "gc": gc,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        # Original reads RSNA_ORDER_CACHE; unset on the scored platform, so no order file.
        "ORDER_CACHE": None,
        "Path": provenance.Refused("Path (ORDER_CACHE branch)"),
        "pd": provenance.MetadataPandas,
    }


def load(raw: bytes | None = None, messages: list[str] | None = None) -> tuple[dict, dict]:
    raw = SOURCE_PATH.read_bytes() if raw is None else raw
    namespace, receipt = provenance.load_reference(raw, adapters=adapters(messages))
    return namespace, receipt


def configure(namespace: dict, pixel_group: dict) -> None:
    """Apply one member's manifest pixel group through the original `adopt_config_globals`.

    The notebook picks `SLOTS` from the SLOT_SCHEME environment variable before the
    member is read and then refuses a mismatch; choosing the scheme whose names equal
    the member's recorded slots is the only setting under which that check passes.
    """
    names = list(pixel_group["slots"])
    scheme = next((s for s in SLOT_SCHEMES if [x[0] for x in namespace[s]] == names), None)
    if scheme is None:
        raise ValueError("Member slots match neither original slot scheme")
    namespace["SLOTS"] = namespace[scheme]
    namespace["N_SLOT"] = len(namespace["SLOTS"])
    namespace["adopt_config_globals"](pixel_group)


def header_frame(namespace: dict, study_dirs: Sequence[Path], split: str) -> pd.DataFrame:
    """The frame original `walk(split)` builds, restricted to the given study directories.

    Same item tuple, scandir order and `probe` call; threads only change wall time since
    `pool.map` preserves order. Directory names are the study/series UIDs, as on Kaggle.
    """
    items = []
    for study in study_dirs:
        for series in os.scandir(study):
            if series.is_dir():
                items.append((split, Path(study).name, series.name, series.path))
    if not items:
        columns = ["split", "StudyInstanceUID", "SeriesInstanceUID", "dir", "files", "n_slices"]
        return pd.DataFrame(columns=columns + namespace["HDR_TAGS"])
    return pd.DataFrame([namespace["probe"](item) for item in items])


ORDERED_LOG = re.compile(r"ordered (\d+)/(\d+) by geometry")
LATERALITY_LOG = re.compile(r"(\d+) from the tag, (\d+) from geometry, (\d+) unresolved")


def _counts(pattern: re.Pattern, messages: list[str]) -> tuple[int, ...] | None:
    found = [pattern.search(m) for m in messages]
    hits = [tuple(map(int, f.groups())) for f in found if f]
    return hits[-1] if hits else None


def public_field_mm(namespace: dict, px: float, rows: int, columns: int) -> float:
    """Physical in-plane extent `read_slot` keeps: the 130 mm crop or the full field."""
    want = int(round(namespace["CROP_MM"] / px))
    if 16 < want < min(rows, columns):
        return 2 * (want // 2) * px
    return max(rows, columns) * px


def render_original(
    namespace: dict, study_dirs: Sequence[Path], plane_map: dict, split: str = "train_series"
) -> dict:
    """Original multi-study path: annotate -> pick_slots -> lat_of -> build_cache."""
    sink = getattr(namespace["log"], "__self__", [])
    start = len(sink)
    frame = namespace["annotate"](header_frame(namespace, study_dirs, split))
    slot_map = namespace["pick_slots"](frame, plane_map)
    lat_map = namespace["lat_of"](frame, "reference ")
    studies, cache, mask = namespace["build_cache"](slot_map, plane_map, lat_map, "reference")
    messages = list(sink[start:])
    return {
        "studies": studies,
        "cache": cache,
        "mask": mask,
        "slots": slot_map,
        "lat": lat_map,
        "geometry_ordered": _counts(ORDERED_LOG, messages),
        "laterality_counts": _counts(LATERALITY_LOG, messages),
    }


def render_study(
    namespace: dict, study_dir: Path, plane_map: dict, split: str = "train_series"
) -> dict:
    """One study -> original cache row `[N_SLOT, CACHE_SLICES, IMG, IMG]` uint8 and mask."""
    out = render_original(namespace, [study_dir], plane_map, split)
    if out["studies"] != [Path(study_dir).name]:
        return {
            "images": torch.zeros(
                namespace["N_SLOT"], namespace["CACHE_SLICES"], namespace["IMG"],
                namespace["IMG"], dtype=torch.uint8,
            ),
            "slot_mask": torch.zeros(namespace["N_SLOT"], dtype=torch.bool),
            "slot_series": [None] * namespace["N_SLOT"],
            "slice_order": [None] * namespace["N_SLOT"],
            "laterality": None,
            "slot_header": [None] * namespace["N_SLOT"],
            "geometry_ordered": out["geometry_ordered"],
            "laterality_source": None,
        }
    chosen = out["slots"][out["studies"][0]]
    names = [slot[0] for slot in namespace["SLOTS"]]
    return {
        "images": torch.from_numpy(out["cache"][0]),
        "slot_mask": torch.from_numpy(out["mask"][0]).bool(),
        "slot_series": [str(chosen[n]["SeriesInstanceUID"]) if n in chosen else None for n in names],
        "slice_order": [list(chosen[n]["ordered"]) if n in chosen else None for n in names],
        "laterality": out["lat"].get(out["studies"][0]),
        "slot_header": [
            (float(chosen[n]["px"]), int(chosen[n]["Rows"]), int(chosen[n]["Columns"]))
            if n in chosen
            else None
            for n in names
        ],
        "geometry_ordered": out["geometry_ordered"],
        "laterality_source": _laterality_source(out["laterality_counts"]),
    }


def _laterality_source(counts: tuple[int, ...] | None) -> str | None:
    if counts is None:
        return None
    return ("tag", "geometry", "unresolved")[counts.index(1)] if sum(counts) == 1 else None


def windows(namespace: dict, images: torch.Tensor) -> torch.Tensor:
    """Original overlapping TTA windows `[n_window, N_SLOT, GROUP, IMG, IMG]`."""
    group = namespace["GROUP"]
    starts = namespace["window_starts"](images.shape[1], group)
    return torch.stack([images[:, s : s + group] for s in starts])
