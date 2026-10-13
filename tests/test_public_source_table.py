from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from rsna.audit import inventory_fingerprint
from rsna.competitive import public_source_table as pst
from rsna.competitive.public_input import canonical_sha256, roster_sha256

ROOT = Path(__file__).resolve().parents[1]
REAL_PLAN = ROOT / "initiative/audits/source-pixel-representation-v1/source_correction_plan.json"
REAL_RECEIPT = (
    ROOT / "initiative/audits/source-pixel-representation-v1/hf-source-correction-receipt.json"
)


def row(split, study, series, sop, size=100, digest="a"):
    return {
        "split": split,
        "study_uid": study,
        "series_uid": series,
        "sop_uid": sop,
        "path": f"{split}/{study}/{series}/{sop}.dcm",
        "sha256": hashlib.sha256(f"{sop}{digest}".encode()).hexdigest(),
        "bytes": size,
        "storage": "lfs",
    }


def build(**overrides):
    fitting, evaluation = ["1.1", "1.2"], ["1.3"]
    rows, accepted = [], []
    for study in [*fitting, *evaluation]:
        for series in ("1", "2"):
            accepted.append({"StudyInstanceUID": study, "SeriesInstanceUID": f"{study}.{series}"})
            rows += [
                row("train_series", study, f"{study}.{series}", f"{study}.{series}.{i}")
                for i in (1, 2)
            ]
        rows.append(row("train_series", study, f"{study}.9", f"{study}.9.1"))  # unaccepted
    rows.append(row("train_series", "2.1", "2.1.1", "2.1.1.1"))  # outside cohort
    rows.append(row("test_series", "3.1", "3.1.1", "3.1.1.1"))
    accepted.append({"StudyInstanceUID": "2.1", "SeriesInstanceUID": "2.1.1"})
    inventory = pd.DataFrame(rows, columns=pst.COLUMNS)
    changes = []
    for index in (0, 5):  # one fitting and one development... both in cohort
        before = dict(rows[index])
        after = {
            **before,
            "sha256": hashlib.sha256(b"official" + bytes([index])).hexdigest(),
            "bytes": 150,
        }
        changes.append(
            {k: before[k] for k in ("path", "study_uid", "series_uid", "sop_uid")}
            | {"before": before, "after": after}
        )
    plan = {
        "schema_version": 1,
        "status": "accepted_exact_authoritative_source_replacement",
        "original_pinned_revision_preserved": True,
        "decoder_changed": False,
        "baseline_source_revision": pst.HISTORICAL_SOURCE_REVISION,
        "publication_parent_revision": "8" * 40,
        "baseline_inventory_fingerprint_sha256": inventory_fingerprint(inventory),
        "n_changed_source_files": 2,
        "changes": changes,
    }
    plan.update(overrides.pop("plan", {}))
    plan_bytes = json.dumps(plan).encode()
    receipt = {
        "status": "verified_private_hf_source_correction",
        "remote_objects_verified": True,
        "source_correction_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "baseline_source_revision": pst.HISTORICAL_SOURCE_REVISION,
        "corrected_source_revision": pst.CORRECTED_SOURCE_REVISION,
        "changed_source_paths": [change["path"] for change in changes],
    }
    expected = {
        "fitting_studies": 2,
        "development_studies": 1,
        "fitting_roster_sha256": roster_sha256(fitting),
        "development_roster_sha256": roster_sha256(evaluation),
        "source_series": 6,
        "excluded_source_series": 3,
        "source_files": 12,
        "source_bytes": 12 * 100 + 2 * 50,
        "corrected_source_files": 2,
        "corrected_source_byte_delta": 100,
    }
    case = {
        "inventory": inventory,
        "accepted_series": pd.DataFrame(accepted),
        "fitting_uids": fitting,
        "evaluation_uids": evaluation,
        "plan_bytes": plan_bytes,
        "receipt": receipt,
        "source_revision": pst.CORRECTED_SOURCE_REVISION,
        "expected": expected,
        "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan_sha256": canonical_sha256(json.loads(plan_bytes)),
    }
    case.update(overrides)
    return case, changes


def test_overlays_corrected_objects_and_excludes_unaccepted_series():
    case, changes = build()
    objects, summary = pst.accepted_source_objects(**case)
    assert sorted(objects) == ["1.1", "1.2", "1.3"]
    records = {r["path"]: r for rows in objects.values() for r in rows}
    assert len(records) == 12 and not any(".9." in path for path in records)
    for change in changes:
        assert records[change["path"]] == change["after"]
        assert records[change["path"]]["sha256"] != change["before"]["sha256"]
    assert all(set(r) == set(pst.COLUMNS) for r in records.values())
    assert summary["source_revision"] == pst.CORRECTED_SOURCE_REVISION
    assert (summary["source_files"], summary["source_bytes"]) == (12, 1300)
    text = json.dumps(summary)
    assert not any(uid in text for uid in ["1.1.1", "1.2", "1.3.2"])
    assert summary["object_table_sha256"] == canonical_sha256(objects)


@pytest.mark.parametrize("revision", [pst.HISTORICAL_SOURCE_REVISION, "c" * 40])
def test_requires_corrected_revision(revision):
    case, _ = build(source_revision=revision)
    with pytest.raises(ValueError, match="revision"):
        pst.accepted_source_objects(**case)


@pytest.mark.parametrize(
    "fitting,evaluation",
    [
        (["1.1", "1.2"], ["1.2"]),
        (["1.1", "1.1"], ["1.3"]),
        (["1.1"], ["1.3"]),
        (["1.1", "1.4"], ["1.3"]),
        (["1.1", "1.3"], ["1.2"]),
    ],
)
def test_rejects_roster_mismatch(fitting, evaluation):
    case, _ = build(fitting_uids=fitting, evaluation_uids=evaluation)
    with pytest.raises(ValueError, match="Rosters"):
        pst.accepted_source_objects(**case)


@pytest.mark.parametrize("uid", ["../1.1", "1/1", "", "..", "1.1\x00", ".1"])
def test_rejects_unsafe_uids(uid):
    case, _ = build(fitting_uids=["1.1", uid])
    with pytest.raises(ValueError):
        pst.accepted_source_objects(**case)


def test_rejects_plan_tamper_and_receipt_mismatch():
    case, _ = build()
    tampered = dict(
        case, plan_bytes=case["plan_bytes"].replace(b'"bytes": 150', b'"bytes": 151', 1)
    )
    with pytest.raises(ValueError, match="accepted plan"):
        pst.accepted_source_objects(**tampered)
    for key, value in [
        ("corrected_source_revision", pst.HISTORICAL_SOURCE_REVISION),
        ("source_correction_plan_sha256", "0" * 64),
        ("remote_objects_verified", False),
    ]:
        receipt = dict(case["receipt"], **{key: value})
        with pytest.raises(ValueError, match="receipt"):
            pst.accepted_source_objects(**dict(case, receipt=receipt))
    receipt = dict(
        case["receipt"], changed_source_paths=case["receipt"]["changed_source_paths"][:1]
    )
    with pytest.raises(ValueError, match="membership"):
        pst.accepted_source_objects(**dict(case, receipt=receipt))


def test_rejects_correction_identity_change():
    case, changes = build()
    plan = json.loads(case["plan_bytes"])
    plan["changes"][0]["after"]["sop_uid"] = "other"
    data = json.dumps(plan).encode()
    receipt = dict(case["receipt"], source_correction_plan_sha256=hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="only exact"):
        pst.accepted_source_objects(
            **dict(
                case,
                plan_bytes=data,
                receipt=receipt,
                plan_file_sha256=hashlib.sha256(data).hexdigest(),
                plan_sha256=canonical_sha256(plan),
            )
        )


def test_rejects_duplicate_json_fields():
    case, _ = build()
    data = b'{"schema_version": 1, "schema_version": 1}'
    with pytest.raises(ValueError, match="Duplicate"):
        pst.load_correction(
            data, case["receipt"], plan_file_sha256=hashlib.sha256(data).hexdigest()
        )


def test_rejects_already_corrected_or_changed_inventory():
    case, changes = build()
    inventory = case["inventory"].copy()
    inventory.loc[inventory.path == changes[0]["path"], ["sha256", "bytes"]] = [
        changes[0]["after"]["sha256"],
        150,
    ]
    with pytest.raises(ValueError, match="baseline"):
        pst.accepted_source_objects(**dict(case, inventory=inventory))
    shuffled = case["inventory"][list(reversed(pst.COLUMNS))]
    with pytest.raises(ValueError, match="schema"):
        pst.accepted_source_objects(**dict(case, inventory=shuffled))


def test_rejects_malicious_inventory_paths():
    case, _ = build()
    for path in ["../train_series/x.dcm", "/abs/x.dcm", "train_series/1.1/1.1.1/../../x.dcm"]:
        inventory = case["inventory"].copy()
        inventory.loc[1, "path"] = path
        with pytest.raises(ValueError):
            pst.accepted_source_objects(**dict(case, inventory=inventory))


def test_rejects_series_and_total_mismatches():
    case, _ = build()
    accepted = pd.concat(
        [
            case["accepted_series"],
            pd.DataFrame([{"StudyInstanceUID": "1.2", "SeriesInstanceUID": "1.1.1"}]),
        ]
    )
    with pytest.raises(ValueError, match="unique"):
        pst.accepted_source_objects(**dict(case, accepted_series=accepted))
    swapped = case["accepted_series"].copy()
    swapped.loc[0, "StudyInstanceUID"], swapped.loc[2, "StudyInstanceUID"] = "1.2", "1.1"
    with pytest.raises(ValueError, match="different study"):
        pst.accepted_source_objects(**dict(case, accepted_series=swapped))
    missing = case["accepted_series"][case["accepted_series"].SeriesInstanceUID != "1.3.1"]
    with pytest.raises(ValueError):
        pst.accepted_source_objects(**dict(case, accepted_series=missing))
    for key, value in [
        ("source_series", 7),
        ("excluded_source_series", 2),
        ("source_files", 13),
        ("source_bytes", 1200),
        ("corrected_source_files", 1),
        ("corrected_source_byte_delta", 0),
    ]:
        expected = dict(case["expected"], **{key: value})
        with pytest.raises(ValueError):
            pst.accepted_source_objects(**dict(case, expected=expected))


def test_correction_outside_cohort_is_not_counted():
    case, changes = build()
    plan = json.loads(case["plan_bytes"])
    outside = row("train_series", "2.1", "2.1.1", "2.1.1.1")
    plan["changes"][1] = {k: outside[k] for k in ("path", "study_uid", "series_uid", "sop_uid")} | {
        "before": outside,
        "after": {**outside, "sha256": "f" * 64, "bytes": 150},
    }
    data = json.dumps(plan).encode()
    receipt = dict(
        case["receipt"],
        source_correction_plan_sha256=hashlib.sha256(data).hexdigest(),
        changed_source_paths=[c["path"] for c in plan["changes"]],
    )
    common = dict(
        case,
        plan_bytes=data,
        receipt=receipt,
        plan_file_sha256=hashlib.sha256(data).hexdigest(),
        plan_sha256=canonical_sha256(plan),
    )
    with pytest.raises(ValueError, match="bound totals"):
        pst.accepted_source_objects(**common)
    expected = dict(
        case["expected"],
        corrected_source_files=1,
        corrected_source_byte_delta=50,
        source_bytes=1250,
    )
    _, summary = pst.accepted_source_objects(**dict(common, expected=expected))
    assert summary["corrected_source_files"] == 1


def test_real_correction_plan_binds_corrected_revision():
    plan_bytes = REAL_PLAN.read_bytes()
    receipt = json.loads(REAL_RECEIPT.read_text())
    changes = pst.load_correction(plan_bytes, receipt)
    assert len(changes) == 2
    assert receipt["corrected_source_revision"] == pst.CORRECTED_SOURCE_REVISION
    assert sum(c["after"]["bytes"] - c["before"]["bytes"] for c in changes.values()) == 582656
    with pytest.raises(ValueError, match="receipt"):
        pst.load_correction(plan_bytes, receipt, corrected_revision=pst.HISTORICAL_SOURCE_REVISION)
    stale = copy.deepcopy(receipt)
    stale["corrected_source_revision"] = pst.HISTORICAL_SOURCE_REVISION
    with pytest.raises(ValueError, match="receipt"):
        pst.load_correction(plan_bytes, stale)


def test_real_inventory_is_plan_baseline_and_overlay_matches_corrected_fingerprint():
    parts = sorted((ROOT / "initiative/source/inventory").glob("*.parquet"))
    inventory = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    plan = json.loads(REAL_PLAN.read_text())
    assert list(inventory.columns) == pst.COLUMNS
    assert inventory_fingerprint(inventory) == plan["baseline_inventory_fingerprint_sha256"]
    changes = pst.load_correction(REAL_PLAN.read_bytes(), json.loads(REAL_RECEIPT.read_text()))
    indexed = inventory.set_index("path", drop=False)
    for path, change in changes.items():
        assert {k: (int(v) if k == "bytes" else v) for k, v in indexed.loc[path].items()} == change[
            "before"
        ]
        indexed.loc[path, ["sha256", "bytes"]] = [
            change["after"]["sha256"],
            change["after"]["bytes"],
        ]
    corrected = indexed.reset_index(drop=True)
    assert inventory_fingerprint(corrected) == plan["corrected_inventory_fingerprint_sha256"]
    assert int(corrected.bytes.sum()) == plan["source_bytes_after"]
