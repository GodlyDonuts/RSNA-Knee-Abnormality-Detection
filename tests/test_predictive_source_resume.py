from __future__ import annotations

import hashlib
import io
from contextlib import contextmanager
from pathlib import Path

import pytest

from rsna.competitive.predictive import source_resume as sr

UID = "1.2.3"
SERIES = "1.2.4"
REVISION = "a" * 40
TERMINAL = {"job_id": "123", "state": "TIMEOUT", "sha256": "b" * 64}


@pytest.fixture
def case(tmp_path: Path) -> dict:
    data, destination = tmp_path / "original", tmp_path / "destination"
    data.mkdir(mode=0o700)
    destination.mkdir(mode=0o700)
    series_csv = ",".join(sr.SERIES_COLUMNS) + f"\n{UID},{SERIES},1,0,sagittal\n"
    (data / "train.csv").write_text("StudyInstanceUID\n" + UID + "\n")
    (data / "train_series.csv").write_text(series_csv)
    bodies = [b"complete original DICOM", b"partial original DICOM", b"missing original DICOM"]
    rows = []
    for index, body in enumerate(bodies):
        sop = f"1.2.5.{index}"
        path = f"train_series/{UID}/{SERIES}/{sop}.dcm"
        rows.append({
            "study_uid": UID, "series_uid": SERIES, "sop_uid": sop,
            "split": "train_series", "path": path, "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        })
        if index < 2:
            target = data / path
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target.write_bytes(body if index == 0 else body[:7])
    return dict(data=data, destination=destination, rows=rows, bodies=bodies, series_csv=series_csv)


def plan(case: dict, **changes) -> sr.ResumePlan:
    arguments = dict(
        uid=UID, rows=case["rows"], series_csv=case["series_csv"],
        revision=REVISION, cap=1 << 20,
    )
    arguments.update(changes)
    return sr.reconcile(case["data"], **arguments)


def opener(case: dict, calls: list, *, change=None):
    @contextmanager
    def open_one(row, offset):
        index = next(i for i, candidate in enumerate(case["rows"]) if row == candidate)
        calls.append((index, offset))
        body = case["bodies"][index]
        headers = {"Content-Length": str(len(body) - offset)}
        if offset:
            headers["Content-Range"] = f"bytes {offset}-{len(body) - 1}/{len(body)}"
        response = dict(
            stream=io.BytesIO(body[offset:]), status=206 if offset else 200,
            headers=headers, revision=REVISION, path=row["path"],
        )
        if change:
            change(response, index, offset)
        yield sr.SourceResponse(**response)
    return open_one


def test_read_only_reconcile_then_reuses_complete_and_only_fetches_suffix_missing(case):
    paths = [case["data"] / row["path"] for row in case["rows"]]
    originals = [(p.read_bytes(), p.stat()) if p.exists() else None for p in paths]
    p = plan(case)
    assert [m.retained_size for m in p.members] == [len(case["bodies"][0]), 7, 0]
    assert [(p.read_bytes(), p.stat()) if p.exists() else None for p in paths] == originals
    calls = []
    data, proof = sr.resume_missing(
        case["destination"], p, opener=opener(case, calls),
        terminal_receipt=TERMINAL, workers=2,
    )
    assert sorted(calls) == [(1, 7), (2, 0)]
    for row, body in zip(case["rows"], case["bodies"], strict=True):
        assert (data / row["path"]).read_bytes() == body
    assert (data / case["rows"][0]["path"]).stat().st_ino == paths[0].stat().st_ino
    assert paths[1].read_bytes() == originals[1][0]
    assert not paths[2].exists()
    assert proof["fetched_bytes"] == len(case["bodies"][1]) - 7 + len(case["bodies"][2])
    assert proof["automatic_retry"] is False
    assert (data.parent / "source-proof.private.json").stat().st_mode & 0o077 == 0


def test_all_complete_requires_no_network(case):
    for row, body in zip(case["rows"], case["bodies"], strict=True):
        (case["data"] / row["path"]).write_bytes(body)

    def forbidden(*args):
        raise AssertionError("A complete original must never be fetched")

    _, proof = sr.resume_missing(
        case["destination"], plan(case), opener=forbidden, terminal_receipt=TERMINAL,
    )
    assert proof["fetched_bytes"] == 0
    assert all(f["action"] == "linked_complete" for f in proof["files"])


@pytest.mark.parametrize("kind", [
    "status", "range_start", "range_total", "length", "encoding", "revision", "path",
    "short", "long", "changed",
])
def test_rejects_bad_suffix_or_changed_bytes_and_preserves_original(case, kind):
    old = case["data"] / case["rows"][1]["path"]
    prefix = old.read_bytes()
    calls = []

    def change(response, index, offset):
        if index != 1:
            return
        if kind == "status":
            response["status"] = 200
        elif kind == "range_start":
            response["headers"]["Content-Range"] = "bytes 0-21/22"
        elif kind == "range_total":
            response["headers"]["Content-Range"] = "bytes 7-21/23"
        elif kind == "length":
            response["headers"]["Content-Length"] = "999"
        elif kind == "encoding":
            response["headers"]["Content-Encoding"] = "gzip"
        elif kind in {"revision", "path"}:
            response[kind] = "different"
        else:
            suffix = case["bodies"][index][offset:]
            if kind == "short":
                suffix = suffix[:-1]
            elif kind == "long":
                suffix += b"x"
            else:
                suffix = b"x" * len(suffix)
            response["stream"] = io.BytesIO(suffix)

    with pytest.raises(ValueError):
        sr.resume_missing(
            case["destination"], plan(case), opener=opener(case, calls, change=change),
            terminal_receipt=TERMINAL,
        )
    assert calls == [(1, 7)]
    assert old.read_bytes() == prefix
    failures = list(case["destination"].glob("*/source-failure.private.json"))
    assert len(failures) == 1
    assert "different" not in failures[0].read_text()


def test_bad_retained_prefix_fails_complete_hash_without_altering_original(case):
    path = case["data"] / case["rows"][1]["path"]
    path.write_bytes(b"changed")
    calls = []
    with pytest.raises(ValueError, match="size/SHA"):
        sr.resume_missing(
            case["destination"], plan(case), opener=opener(case, calls),
            terminal_receipt=TERMINAL,
        )
    assert calls == [(1, 7)]
    assert path.read_bytes() == b"changed"


@pytest.mark.parametrize("kind", ["path", "uid", "duplicate", "cap", "metadata", "sha", "symlink"])
def test_reconciliation_rejects_changed_contract_or_unsafe_original(case, kind):
    if kind == "path":
        case["rows"][0]["path"] = "../outside.dcm"
    elif kind == "uid":
        case["rows"][0]["study_uid"] = "1.9"
    elif kind == "duplicate":
        case["rows"].append(dict(case["rows"][0]))
    elif kind == "metadata":
        (case["data"] / "train.csv").write_text("wrong")
    elif kind == "sha":
        (case["data"] / case["rows"][0]["path"]).write_bytes(b"x" * len(case["bodies"][0]))
    elif kind == "symlink":
        path = case["data"] / case["rows"][1]["path"]
        path.unlink()
        path.symlink_to(case["data"] / case["rows"][0]["path"])
    with pytest.raises(ValueError):
        plan(case, **({"cap": 128} if kind == "cap" else {}))
    assert list(case["destination"].iterdir()) == []


def test_changed_original_after_reconciliation_refused_before_network(case):
    p = plan(case)
    (case["data"] / case["rows"][1]["path"]).write_bytes(b"changed")
    calls = []
    with pytest.raises(ValueError, match="after reconciliation"):
        sr.resume_missing(
            case["destination"], p, opener=opener(case, calls), terminal_receipt=TERMINAL,
        )
    assert calls == []
    assert list(case["destination"].iterdir()) == []


def test_empty_retained_file_uses_full_missing_request_and_preserves_empty_original(case):
    original = case["data"] / case["rows"][1]["path"]
    original.write_bytes(b"")
    calls = []
    data, proof = sr.resume_missing(
        case["destination"], plan(case), opener=opener(case, calls), terminal_receipt=TERMINAL,
    )
    assert calls == [(1, 0), (2, 0)]
    assert original.read_bytes() == b""
    assert (data / case["rows"][1]["path"]).read_bytes() == case["bodies"][1]
    total = sum(p.stat().st_size for p in data.parent.rglob("*") if p.is_file())
    assert total <= proof["source_cap_bytes"]


def test_no_fallback_transfer_when_complete_file_cannot_be_hardlinked(case, monkeypatch):
    calls = []

    def refuse_link(*args, **kwargs):
        raise OSError("different filesystem")

    monkeypatch.setattr(sr.os, "link", refuse_link)
    with pytest.raises(OSError):
        sr.resume_missing(
            case["destination"], plan(case), opener=opener(case, calls),
            terminal_receipt=TERMINAL,
        )
    assert calls == []
    assert len(list(case["destination"].glob("*/source-failure.private.json"))) == 1


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "UNKNOWN"])
def test_live_or_uncertain_source_job_is_not_continuation_authority(case, state):
    calls = []
    with pytest.raises(ValueError, match="terminal"):
        sr.resume_missing(
            case["destination"], plan(case), opener=opener(case, calls),
            terminal_receipt={**TERMINAL, "state": state},
        )
    assert not calls and not list(case["destination"].iterdir())
