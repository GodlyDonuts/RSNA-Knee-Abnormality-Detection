"""Accepted-series source binding before the public renderer's header probe."""

import hashlib
import os

import pandas as pd
import pytest
import torch

from rsna.competitive import public_source as ps
from tools.research.public_dinov2_input_reference import synthetic

LEFT = synthetic.MIRROR["left"]
EXTRA = synthetic.SeriesSpec("Sagittal", "PD FS", slices=30, laterality="L", seed=9)
STUDY = "1.2.826.0.1.7"


def write(root, extra=True):
    for index, spec in enumerate(LEFT, 1):
        synthetic.write_series(root, STUDY, index, spec)
    if extra:
        synthetic.write_series(root, STUDY, len(LEFT) + 1, EXTRA)
    return root / STUDY


def series_csv(path, rows=None):
    rows = rows or [
        (STUDY, f"{STUDY}.{i}", "1", "1", spec.plane) for i, spec in enumerate(LEFT, 1)
    ]
    pd.DataFrame(rows, columns=ps.SERIES_COLUMNS).to_csv(path, index=False)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def accepted(tmp_path, **counts):
    path, digest = series_csv(tmp_path / "accepted.csv")
    return ps.load_accepted_series(path, sha256=digest, **counts)


def stage(tmp_path, source, value, **changes):
    view_root = tmp_path / "view"
    view_root.mkdir(exist_ok=True)
    values = dict(source_revision=ps.ACCEPTED_SOURCE_REVISION, max_bytes=10**9) | changes
    return ps.stage_accepted_study(source, view_root, value, **values)


def test_filter_precedes_header_probe_and_matches_clean_original_render(tmp_path):
    from test_public_dinov2_input_reference import configured

    from tools.research.public_dinov2_input_reference import reference
    from tools.research.public_dinov2_input_reference.dino_adapter import render_inputs

    ns, _ = configured(reference.SOURCE_PATH.read_bytes())
    value = accepted(tmp_path, series_count=2, study_count=1)
    source = write(tmp_path / "source")
    clean = write(tmp_path / "clean", extra=False)
    unfiltered_map = value.plane_map | {f"{STUDY}.3": "Sagittal"}
    view, receipt = stage(tmp_path, source, value)
    assert receipt["excluded_series"] == 1 and len(receipt["series"]) == 2
    assert receipt["source_bytes"] == sum(s["bytes"] for s in receipt["series"])
    labels = (torch.zeros(12), torch.zeros(12))
    filtered = render_inputs(ns, view, unfiltered_map, *labels)
    expected = render_inputs(ns, clean, value.plane_map, *labels)
    raw = render_inputs(ns, source, unfiltered_map, *labels)
    for key in ("images", "series_mask", "features", "positions"):
        assert torch.equal(filtered[key], expected[key])
    assert not torch.equal(raw["images"], expected["images"])
    assert sorted(os.listdir(source)) == [f"{STUDY}.{i}" for i in (1, 2, 3)]


@pytest.mark.parametrize("change", ["hash", "columns", "duplicate", "unsafe", "plane", "count"])
def test_accepted_series_csv_binding_refused(tmp_path, change):
    rows = [(STUDY, f"{STUDY}.1", "1", "1", "Sagittal"), (STUDY, f"{STUDY}.2", "1", "1", "Axial")]
    if change == "duplicate":
        rows[1] = rows[0]
    elif change == "unsafe":
        rows[1] = (STUDY, "../escape", "1", "1", "Axial")
    elif change == "plane":
        rows[1] = (STUDY, f"{STUDY}.2", "1", "1", "Oblique")
    path, digest = series_csv(tmp_path / "s.csv", rows)
    if change == "columns":
        frame = pd.read_csv(path).drop(columns="Fat_Suppression")
        frame.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    kwargs = {"series_count": 3} if change == "count" else {}
    with pytest.raises(ValueError):
        ps.load_accepted_series(path, sha256="0" * 64 if change == "hash" else digest, **kwargs)


def test_roster_must_be_disjoint_unique_and_covered(tmp_path):
    value = accepted(tmp_path)
    ps.require_roster(value, [STUDY], [])
    for fitting, evaluation in (([STUDY], [STUDY]), ([STUDY, STUDY], []), ([], [STUDY])):
        with pytest.raises(ValueError):
            ps.require_roster(value, fitting, evaluation)
    with pytest.raises(ValueError, match="absent"):
        ps.require_roster(value, [STUDY], ["1.2.3"])


def test_new_bytes_need_corrected_revision_and_retained_packet_is_reused(tmp_path):
    value = accepted(tmp_path)
    check = ps.validate_source_revision
    check(STUDY, ps.ACCEPTED_SOURCE_REVISION, accepted=value)
    check(STUDY, ps.HISTORICAL_SOURCE_REVISION, accepted=value, retained_uids={STUDY})
    with pytest.raises(ValueError, match="corrected release"):
        check(STUDY, ps.HISTORICAL_SOURCE_REVISION, accepted=value)
    with pytest.raises(ValueError, match="corrected release"):
        check(STUDY, "f" * 40, accepted=value, retained_uids={STUDY})
    with pytest.raises(ValueError, match="officially corrected"):
        check(
            STUDY,
            ps.HISTORICAL_SOURCE_REVISION,
            accepted=value,
            retained_uids={STUDY},
            corrected_series={f"{STUDY}.2"},
        )


@pytest.mark.parametrize(
    "change", ["file_link", "series_link", "missing", "empty", "cap", "revision", "absent_study"]
)
def test_unsafe_or_incomplete_source_refused_before_any_view(tmp_path, change):
    value = accepted(tmp_path)
    source = write(tmp_path / "source")
    changes = {}
    if change == "file_link":
        outside = tmp_path / "outside.dcm"
        outside.write_bytes(b"x")
        os.symlink(outside, source / f"{STUDY}.1" / "im-link.dcm")
    elif change == "series_link":
        os.symlink(source / f"{STUDY}.1", source / "linked")
    elif change == "missing":
        for path in (source / f"{STUDY}.2").iterdir():
            path.unlink()
        (source / f"{STUDY}.2").rmdir()
    elif change == "empty":
        for path in (source / f"{STUDY}.2").iterdir():
            path.unlink()
    elif change == "cap":
        changes["max_bytes"] = 1
    elif change == "revision":
        changes["source_revision"] = ps.HISTORICAL_SOURCE_REVISION
    elif change == "absent_study":
        source = source.rename(source.parent / "1.2.826.0.1.8")
    with pytest.raises(ValueError):
        stage(tmp_path, source, value, **changes)
    assert not (tmp_path / "view" / source.name).exists()


def test_existing_view_never_overwritten_and_links_stay_contained(tmp_path):
    value = accepted(tmp_path)
    source = write(tmp_path / "source")
    view, _ = stage(tmp_path, source, value)
    real = os.path.realpath(source)
    assert all(os.path.dirname(os.path.realpath(link)) == real for link in view.iterdir())
    with pytest.raises(FileExistsError):
        stage(tmp_path, source, value)
    with pytest.raises(ValueError, match="real directory"):
        ps.stage_accepted_study(
            source,
            tmp_path / "missing-root",
            value,
            source_revision=ps.ACCEPTED_SOURCE_REVISION,
            max_bytes=10**9,
        )
