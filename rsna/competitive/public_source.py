"""Accepted-series source binding for the public ordered-triplet cache producer.

The unfiltered original source holds series outside the accepted native series set.
The original renderer probes and annotates every series directory of a study, and its
slot choice and laterality depend on that frame, so unaccepted series are excluded
*before* the header probe: each study is staged as a private view holding only its
accepted series directories, in the original directory order. No pixels, labels or
paths leave the owning node; receipts carry identities, counts and bytes only.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ACCEPTED_SOURCE_REVISION = "0b5e410288232f570c69ba4f44723356e6ae6ab7"
HISTORICAL_SOURCE_REVISION = "52f995f4568e36bfa5fbbaf69a89af41226b4f5d"
SERIES_COLUMNS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
)
PLANES = frozenset({"Sagittal", "Coronal", "Axial"})
UID_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,127}")


def _safe_uid(uid: object) -> bool:
    return isinstance(uid, str) and bool(UID_PATTERN.fullmatch(uid)) and ".." not in uid


@dataclass(frozen=True)
class AcceptedSeries:
    sha256: str
    series_by_study: Mapping[str, frozenset[str]]
    plane_map: Mapping[str, str]


def load_accepted_series(
    path: Path,
    *,
    sha256: str,
    series_count: int | None = None,
    study_count: int | None = None,
) -> AcceptedSeries:
    """Read the accepted series CSV once and bind it to its exact bytes and counts."""
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("Accepted series CSV differs from its bound SHA256")
    frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
    if tuple(frame.columns) != SERIES_COLUMNS or frame.empty:
        raise ValueError("Accepted series CSV must have exactly the source series columns")
    studies, series = frame.StudyInstanceUID, frame.SeriesInstanceUID
    if (
        not all(map(_safe_uid, studies))
        or not all(map(_safe_uid, series))
        or series.duplicated().any()
        or not frame.Anatomical_Plane.isin(PLANES).all()
    ):
        raise ValueError("Accepted series UIDs must be safe, unique and plane-annotated")
    grouped = {uid: frozenset(g) for uid, g in series.groupby(studies)}
    if (series_count is not None and len(frame) != series_count) or (
        study_count is not None and len(grouped) != study_count
    ):
        raise ValueError("Accepted series CSV differs from its bound series/study counts")
    return AcceptedSeries(sha256, grouped, dict(zip(series, frame.Anatomical_Plane)))


def require_roster(accepted: AcceptedSeries, fitting: Collection[str], evaluation: Collection[str]):
    """Disjoint unique rosters, every study holding at least one accepted series."""
    fitting_list, evaluation_list = list(fitting), list(evaluation)
    declared = set(fitting_list) | set(evaluation_list)
    if (
        not fitting_list
        or len(set(fitting_list)) != len(fitting_list)
        or len(set(evaluation_list)) != len(evaluation_list)
        or set(fitting_list) & set(evaluation_list)
        or not all(map(_safe_uid, declared))
    ):
        raise ValueError("Public source rosters must be safe, disjoint, unique and nonempty")
    if not declared <= set(accepted.series_by_study):
        raise ValueError("Roster studies are absent from the accepted series CSV")


def validate_source_revision(
    study_uid: str,
    revision: str,
    *,
    accepted: AcceptedSeries,
    retained_uids: Collection[str] = (),
    corrected_series: Collection[str] = (),
) -> None:
    """Newly acquired bytes come from the corrected release; retained ones stay reused.

    A retained study may keep its historical revision only when none of its accepted
    series is one whose official bytes were corrected between the two revisions.
    """
    if revision == ACCEPTED_SOURCE_REVISION:
        return
    if revision != HISTORICAL_SOURCE_REVISION or study_uid not in retained_uids:
        raise ValueError("Newly acquired source bytes must use the corrected release revision")
    if accepted.series_by_study.get(study_uid, frozenset()) & set(corrected_series):
        raise ValueError("Retained historical bytes include an officially corrected series")


def _series_files(series: Path) -> tuple[int, int]:
    files, size = 0, 0
    for entry in os.scandir(series):
        if not entry.name.endswith(".dcm"):
            continue
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError("Original source files must be regular files, not links")
        files, size = files + 1, size + entry.stat(follow_symlinks=False).st_size
    if not files:
        raise ValueError("Accepted source series has no DICOM files")
    return files, size


def stage_accepted_study(
    source_study: Path,
    view_root: Path,
    accepted: AcceptedSeries,
    *,
    source_revision: str,
    max_bytes: int,
    retained_uids: Collection[str] = (),
    corrected_series: Collection[str] = (),
) -> tuple[Path, dict]:
    """Expose only one study's accepted series to the renderer; returns (view, receipt).

    Every check (identity, revision, links, completeness, byte cap) runs before the view
    is created. The view is created exclusively and holds only directory links; it is
    refused if it cannot reproduce the original directory order the renderer would see.
    """
    source_study = Path(source_study)
    uid = source_study.name
    if not _safe_uid(uid) or source_study.is_symlink() or not source_study.is_dir():
        raise ValueError("Original study directory must be a real, safely named directory")
    wanted = accepted.series_by_study.get(uid)
    if not wanted:
        raise ValueError("Study is absent from the accepted series CSV")
    validate_source_revision(
        uid,
        source_revision,
        accepted=accepted,
        retained_uids=retained_uids,
        corrected_series=corrected_series,
    )
    order, excluded, series, total = [], 0, [], 0
    for entry in os.scandir(source_study):
        if entry.is_symlink():
            raise ValueError("Original study directory must not contain links")
        if not entry.is_dir(follow_symlinks=False):
            continue
        if entry.name not in wanted:
            excluded += 1
            continue
        files, size = _series_files(Path(entry.path))
        order.append(entry.name)
        series.append({"series_uid": entry.name, "files": files, "bytes": size})
        total += size
    if set(order) != wanted:
        raise ValueError("Accepted source series are missing from the original study")
    if type(max_bytes) is not int or total > max_bytes:
        raise ValueError("Original study exceeds its explicit source byte bound")
    view_root = Path(view_root)
    if view_root.is_symlink() or not view_root.is_dir():
        raise ValueError("Private view root must be an existing real directory")
    view = view_root / uid
    view.mkdir()
    real = Path(os.path.realpath(source_study))
    try:
        for name in order:
            os.symlink(real / name, view / name, target_is_directory=True)
        seen = [e.name for e in os.scandir(view) if e.is_dir()]
        if seen != order:
            raise ValueError("Filtered view cannot preserve the original series order")
    except BaseException:
        for link in view.iterdir():
            if link.is_symlink():
                link.unlink()
        view.rmdir()
        raise
    receipt = {
        "study_uid": uid,
        "source_revision": source_revision,
        "accepted_series_sha256": accepted.sha256,
        "series": series,
        "excluded_series": excluded,
        "source_bytes": total,
    }
    return view, receipt
