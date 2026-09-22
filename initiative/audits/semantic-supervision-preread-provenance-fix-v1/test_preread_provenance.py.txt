"""Synthetic failures for original-provenance binding; no real opinions changed."""

import copy
import json

import pytest
from test_semantic_supervision import Fixture

from rsna import semantic_supervision as ss


@pytest.fixture
def f(tmp_path):
    return Fixture(tmp_path)


def inspect_context(f):
    b = ss.Bindings(f.root)
    result = ss._context(f.reports, f.reviews, f.rubric, f.root, f.plan, f.inputs, b, set(), [])
    b.unchanged()
    return result[0]


def meta(f):
    return json.loads((f.root / f.inputs["review_provenance"]["path"]).read_text())


def save(f, obj):
    f.inputs["review_provenance"] = f.write(f.inputs["review_provenance"]["path"], obj)


@pytest.mark.parametrize("alias", ["independent_source_read", "preread", "independent_preread"])
def test_heterogeneous_recorded_paths_are_bound_exactly(f, alias):
    d = meta(f)
    m = d["batch-000"]["reviewed"]
    original = f.root / m.pop("preread_path")
    expected = m.pop("preread_sha256")
    path = f.root / "original-evidence/custom-reader-name.jsonl"
    path.parent.mkdir()
    original.rename(path)
    m[alias + "_path"] = str(path.relative_to(f.root))
    m[alias + "_sha256"] = expected
    save(f, d)
    c = inspect_context(f)
    assert c["original_reading_files"]["prereads"] == {str(path.relative_to(f.root)): expected}
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed bound file"):
        inspect_context(f)


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "missing_hash",
        "conflicting_path",
        "conflicting_hash",
        "escape",
        "wrong_batch",
        "wrong_source",
        "wrong_rubric",
        "wrong_parent",
        "same_reader",
        "boolean_count",
    ],
)
def test_bad_provenance_never_falls_back_to_canonical_folder(f, bad):
    d = meta(f)
    m = d["batch-000"]["reviewed"]
    if bad == "missing":
        m.pop("preread_path")
        m.pop("preread_sha256")
    elif bad == "missing_hash":
        m.pop("preread_sha256")
    elif bad == "conflicting_path":
        m["independent_preread_path"] = "other.jsonl"
        m["independent_preread_sha256"] = m["preread_sha256"]
    elif bad == "conflicting_hash":
        m["independent_preread_sha256"] = "0" * 64
    elif bad == "escape":
        m["preread_path"] = "../outside.jsonl"
    elif bad == "wrong_batch":
        d["batch-001"] = d.pop("batch-000")
    elif bad == "wrong_source":
        m["source_sha256"] = "0" * 64
    elif bad == "wrong_rubric":
        m["rubric_sha256"] = "0" * 64
    elif bad == "wrong_parent":
        m["parent_artifact_sha256"] = "0" * 64
    elif bad == "same_reader":
        m["agent_id"] = d["batch-000"]["draft"]["agent_id"]
    elif bad == "boolean_count":
        m["n_studies"] = True
    save(f, d)
    with pytest.raises(ValueError):
        inspect_context(f)


@pytest.mark.parametrize("bad", ["duplicate_uid", "missing_uid", "wrong_order", "target_count"])
def test_bound_bytes_still_require_exact_batch_membership_and_shape(f, bad):
    d = meta(f)
    m = d["batch-000"]["reviewed"]
    p = f.root / m["preread_path"]
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    if bad == "duplicate_uid":
        rows[1] = copy.deepcopy(rows[0])
    elif bad == "missing_uid":
        rows.pop()
    elif bad == "wrong_order":
        rows.reverse()
    else:
        rows[0]["states"].pop()
    p.write_text("".join(json.dumps(row) + "\n" for row in rows))
    m["preread_sha256"] = ss.file_digest(p)
    save(f, d)
    with pytest.raises(ValueError, match="original reading"):
        inspect_context(f)


def test_preserved_unused_invalid_quote_is_not_a_new_clinical_gate(f):
    d = meta(f)
    m = d["batch-000"]["reviewed"]
    p = f.root / m["preread_path"]
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    rows[0]["evidence"][0] = "Historical nonprojectable quote; do not silently repair."
    p.write_text("".join(json.dumps(row) + "\n" for row in rows))
    m["preread_sha256"] = ss.file_digest(p)
    save(f, d)
    before = p.read_bytes()
    assert (
        inspect_context(f)["original_reading_files"]["prereads"][m["preread_path"]]
        == m["preread_sha256"]
    )
    assert p.read_bytes() == before
    with pytest.raises(ValueError, match="nonverbatim"):
        ss.annotation(
            {"state": rows[0]["states"][0], "evidence": rows[0]["evidence"][0]},
            f.reports[rows[0]["study_uid"]],
        )


def test_missing_draft_still_fails(f):
    (f.root / "initiative/reviews-v2/drafts/batch-000-draft.jsonl").unlink()
    with pytest.raises(ValueError, match="missing original draft"):
        inspect_context(f)
