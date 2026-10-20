"""Exact historical correction tests; never decode or download original MRI."""

import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EVIDENCE = REPO / "initiative/audits/source-pixel-representation-v1"
spec = importlib.util.spec_from_file_location("exact_predictive_source_correction",
                                             HERE / "source_correction.py")
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)


@pytest.fixture
def originals():
    plan = (EVIDENCE / "source_correction_plan.json").read_bytes()
    receipt = (EVIDENCE / "hf-source-correction-receipt.json").read_bytes()
    changes = overlay.validate(plan, receipt)
    records = [
        {"uid": change["study_uid"], "series_csv": "original\r\n",
         "labels_untouched": [None, 0, 1], "mask_untouched": [False, True, True],
         "source_files": [{**change["before"], "extra_original_metadata": "retain exactly"}]}
        for change in changes.values()
    ]
    records.insert(1, {
        "uid": "synthetic_unaffected", "series_csv": "unchanged\n",
        "source_files": [{"study_uid": "synthetic_unaffected",
                          "path": "synthetic/unaffected.dcm", "bytes": 17, "sha256": "a" * 64}],
    })
    return records, plan, receipt


def test_exact_two_payloads_only_and_originals_remain_unchanged(originals):
    records, plan, receipt = originals
    before = overlay.sha(overlay.canonical(records))
    result = overlay.apply(records, plan, receipt)
    assert overlay.sha(overlay.canonical(records)) == before
    assert result["original_records"] is records
    assert result["records"] is not records
    assert result["records"][1] is records[1]  # No bulk source-table copy.
    assert result["lineage"]["byte_delta"] == 582656
    assert result["lineage"]["source_members"] == 3
    assert result["lineage"]["files_skipped"] == 0
    assert len(result["lineage"]["changes"]) == 2
    for i in (0, 2):
        old, new = records[i], result["records"][i]
        assert old is not new and old["source_files"] is not new["source_files"]
        assert list(old) == list(new)
        assert all(old[k] == new[k] for k in old if k != "source_files")
        before_row, after_row = old["source_files"][0], new["source_files"][0]
        assert {k for k in before_row if before_row[k] != after_row[k]} == {"bytes", "sha256"}
    assert [(x["source_ordinal"], x["member_ordinal"]) for x in result["lineage"]["changes"]] == [
        (0, 0), (2, 0),
    ]
    assert all(x["original_revision"] == overlay.ORIGINAL_REVISION
               and x["effective_revision"] == overlay.CORRECTED_REVISION
               for x in result["lineage"]["changes"])


def test_original_seven_field_source_rows_need_no_inventory_storage_column(originals):
    records, plan, receipt = originals
    fields = (*overlay.IDENTITY, "bytes", "sha256")
    for index in (0, 2):
        records[index]["source_files"][0] = {
            key: records[index]["source_files"][0][key] for key in fields
        }
    result = overlay.apply(records, plan, receipt)
    for index in (0, 2):
        assert tuple(result["records"][index]["source_files"][0]) == fields
        assert len(result["lineage"]["changes"][index // 2]["before"]) == 7
    assert result["lineage"]["byte_delta"] == 582656


@pytest.mark.parametrize("which", [1, 2])
def test_even_semantically_similar_plan_or_receipt_bytes_are_rejected(originals, which):
    args = list(originals)
    args[which] += b" "
    with pytest.raises(ValueError, match="evidence changed"):
        overlay.apply(*args)


@pytest.mark.parametrize("field,value", [
    ("bytes", 411051), ("sha256", "0" * 64), ("series_uid", "synthetic_other"),
    ("sop_uid", "synthetic_other"), ("split", "test_series"),
])
def test_wrong_original_member_is_never_overlaid(originals, field, value):
    records, plan, receipt = originals
    records[0]["source_files"][0][field] = value
    with pytest.raises(ValueError, match="before state"):
        overlay.apply(records, plan, receipt)


def test_wrong_study_or_path_never_matches_merely_by_hash(originals):
    records, plan, receipt = originals
    wrong = copy.deepcopy(records)
    wrong[0]["source_files"][0]["path"] = "synthetic/other.dcm"
    with pytest.raises(ValueError, match="different original identity"):
        overlay.apply(wrong, plan, receipt)
    wrong = copy.deepcopy(records)
    wrong[0]["uid"] = "synthetic_other_study"
    with pytest.raises(ValueError, match="study identity"):
        overlay.apply(wrong, plan, receipt)


def test_missing_duplicate_and_already_corrected_sources_fail(originals):
    records, plan, receipt = originals
    with pytest.raises(ValueError, match="Both exact"):
        overlay.apply(records[:-1], plan, receipt)
    duplicated = copy.deepcopy(records)
    duplicated[0]["source_files"].append(copy.deepcopy(duplicated[0]["source_files"][0]))
    with pytest.raises(ValueError, match="membership"):
        overlay.apply(duplicated, plan, receipt)
    corrected = overlay.apply(records, plan, receipt)["records"]
    with pytest.raises(ValueError, match="before state"):
        overlay.apply(corrected, plan, receipt)


def test_within_study_member_order_and_unknown_metadata_are_preserved(originals):
    records, plan, receipt = originals
    extra = {**records[0]["source_files"][0], "path": "synthetic/other.dcm", "sha256": "d" * 64}
    records[0]["source_files"].insert(0, extra)
    result = overlay.apply(records, plan, receipt)
    assert result["records"][0]["source_files"][0] is extra
    assert result["lineage"]["changes"][0]["member_ordinal"] == 1
    assert result["records"][0]["source_files"][1]["extra_original_metadata"] == "retain exactly"
