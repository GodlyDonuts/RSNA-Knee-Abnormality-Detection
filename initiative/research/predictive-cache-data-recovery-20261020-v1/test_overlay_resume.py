"""Exercise real missing-only continuation with synthetic private source bytes."""

import hashlib
import importlib.util
import io
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from rsna.competitive.predictive import source_resume as sr

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("source_correction", HERE / "source_correction.py")
correction = importlib.util.module_from_spec(spec)
sys.modules["source_correction"] = correction
spec.loader.exec_module(correction)
spec = importlib.util.spec_from_file_location("overlay_resume_tested", HERE / "overlay_resume.py")
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)
TERMINAL = {"state": "FAILED", "job_id": "980960", "sha256": "a" * 64}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    original, output = tmp_path / "original", tmp_path / "output"
    original.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    uid, series = "1.2.3", "1.2.4"
    csv = ",".join(sr.SERIES_COLUMNS) + f"\r\n{uid},{series},1,0,sagittal\r\n"
    (original / "train.csv").write_bytes(f"StudyInstanceUID\n{uid}\n".encode())
    (original / "train_series.csv").write_bytes(csv.encode())
    bodies = [b"original short member", b"unchanged complete", b"partial original", b"missing original"]
    rows = []
    for i, body in enumerate(bodies):
        sop = f"1.2.5.{i}"
        row = dict(split="train_series", study_uid=uid, series_uid=series, sop_uid=sop,
                   path=f"train_series/{uid}/{series}/{sop}.dcm", bytes=len(body),
                   sha256=hashlib.sha256(body).hexdigest())
        rows.append(row)
        path = original / row["path"]
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if i != 3:
            path.write_bytes(body if i < 2 else body[:5])
    corrected = b"the exact authoritative full source member"
    after = {**rows[0], "bytes": len(corrected), "sha256": hashlib.sha256(corrected).hexdigest()}
    changes = {rows[0]["path"]: {"before": rows[0], "after": after}}
    # Historical real plan/hash validation is independently covered by the
    # neighbouring tests; these invented MRI-free bytes exercise transport reuse.
    monkeypatch.setattr(correction, "validate", lambda *args: changes)
    resident = tmp_path / "resident"
    resident.write_bytes(corrected)
    plan = sr.reconcile(original, uid=uid, rows=rows, series_csv=csv,
                        revision=correction.ORIGINAL_REVISION, cap=1 << 20)
    calls = []

    @contextmanager
    def opener(row, offset):
        index = next(i for i, x in enumerate(rows) if x == row)
        assert index != 0, "Corrected resident bytes must never require a network request"
        calls.append((index, offset))
        body = bodies[index]
        headers = {"Content-Length": str(len(body) - offset)}
        if offset:
            headers["Content-Range"] = f"bytes {offset}-{len(body)-1}/{len(body)}"
        yield sr.SourceResponse(io.BytesIO(body[offset:]), 206 if offset else 200,
                                headers, correction.ORIGINAL_REVISION, row["path"])

    args = dict(plan_body=b"validated plan", receipt_body=b"validated receipt",
                resident_paths={rows[0]["path"]: resident}, opener=opener,
                terminal_receipt=TERMINAL, workers=2)
    return dict(original=original, output=output, plan=plan, args=args, rows=rows,
                bodies=bodies, corrected=corrected, resident=resident, calls=calls)


def test_exact_overlay_reuses_complete_and_fetches_only_missing_suffix(setup, monkeypatch):
    s = setup
    original_accept = sr._accept_response
    accepted = []

    def check(response, plan, member):
        accepted.append((response.path, response.revision, plan.revision))
        original_accept(response, plan, member)

    monkeypatch.setattr(sr, "_accept_response", check)
    data, proof = overlay.resume(s["output"], s["plan"], **s["args"])
    assert sr._accept_response is check  # No global monkeypatch by the helper.
    assert sorted(s["calls"]) == [(2, 5), (3, 0)]
    assert (data / s["rows"][0]["path"]).read_bytes() == s["corrected"]
    assert (s["original"] / s["rows"][0]["path"]).read_bytes() == s["bodies"][0]
    assert (data / s["rows"][1]["path"]).stat().st_ino == (
        s["original"] / s["rows"][1]["path"]).stat().st_ino
    assert (s["original"] / s["rows"][2]["path"]).read_bytes() == s["bodies"][2][:5]
    assert not (s["original"] / s["rows"][3]["path"]).exists()
    assert (s["rows"][0]["path"], correction.CORRECTED_REVISION,
            correction.CORRECTED_REVISION) in accepted
    assert proof["fetched_bytes"] == len(s["bodies"][2]) - 5 + len(s["bodies"][3])
    assert proof["resident_authoritative_bytes"] == len(s["corrected"])
    assert proof["original_source_table_sha256"] == s["plan"].source_table_sha256
    assert proof["source_table_sha256"] != s["plan"].source_table_sha256
    assert proof == json.loads((data.parent / "source-proof.private.json").read_bytes())
    assert proof["source_correction_lineage"][0]["before"] == s["rows"][0]


def test_wrong_resident_bytes_rejected_before_new_output_or_network(setup):
    s = setup
    s["resident"].write_bytes(b"x" * len(s["corrected"]))
    with pytest.raises(ValueError, match="Resident authoritative"):
        overlay.resume(s["output"], s["plan"], **s["args"])
    assert not list(s["output"].iterdir()) and not s["calls"]


def test_changed_original_after_reconciliation_remains_rejected(setup):
    s = setup
    path = s["original"] / s["rows"][0]["path"]
    path.write_bytes(b"changed original bytes")
    with pytest.raises(ValueError, match="Original source changed"):
        overlay.resume(s["output"], s["plan"], **s["args"])
    assert not list(s["output"].iterdir()) and not s["calls"]


def test_corrected_size_is_included_in_total_cap_before_output(setup):
    s = setup
    minimum = sum(m.size for m in s["plan"].members) + s["plan"].control_reserve
    minimum += sum(len(body) for _, body in sr._metadata(s["plan"].uid, s["plan"].series_csv))
    with pytest.raises(ValueError, match="exceeds original cap"):
        overlay.resume(s["output"], replace(s["plan"], cap=minimum), **s["args"])
    assert not list(s["output"].iterdir()) and not s["calls"]


def test_missing_defective_original_uses_resident_without_fetching_bad_source(setup):
    s = setup
    (s["original"] / s["rows"][0]["path"]).unlink()
    plan = sr.reconcile(s["original"], uid=s["plan"].uid, rows=s["rows"],
                        series_csv=s["plan"].series_csv, revision=correction.ORIGINAL_REVISION,
                        cap=s["plan"].cap)
    data, _ = overlay.resume(s["output"], plan, **s["args"])
    assert (data / s["rows"][0]["path"]).read_bytes() == s["corrected"]
    assert sorted(s["calls"]) == [(2, 5), (3, 0)]


def test_fresh_resident_response_keeps_true_revision_and_exact_effective_row(setup):
    s = setup
    row = {**s["rows"][0], "bytes": len(s["corrected"]),
           "sha256": hashlib.sha256(s["corrected"]).hexdigest()}
    with overlay.resident_response(
        row, s["args"]["resident_paths"], b"plan", b"receipt",
    ) as response:
        assert response.revision == correction.CORRECTED_REVISION
        assert response.status == 200 and response.stream.read() == s["corrected"]
    with pytest.raises(ValueError, match="Exact effective member"):
        with overlay.resident_response(
            s["rows"][0], s["args"]["resident_paths"], b"plan", b"receipt",
        ):
            raise AssertionError("Before payload cannot be supplied as corrected source")
    assert not s["calls"] and not list(s["output"].iterdir())
