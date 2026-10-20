import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from rsna.schema import TARGETS

SPEC = importlib.util.spec_from_file_location("fixed_rank_fusion", Path(__file__).with_name("fusion.py"))
fusion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fusion)


def csv_bytes(rows, columns=fusion.COLUMNS):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for uid, scores in rows:
        writer.writerow([uid, *scores])
    return stream.getvalue().encode()


def constant_rows(uids, values):
    return [(uid, [value] * len(TARGETS)) for uid, value in zip(uids, values, strict=True)]


def decoded(data):
    return list(csv.reader(io.StringIO(data.decode())))


def test_schema_matches_canonical_targets():
    assert fusion.COLUMNS == ("StudyInstanceUID", *TARGETS)


def test_permuted_inputs_align_to_official_order_and_are_unchanged():
    roster = ("case-C", "case-A", "case-B")
    reference = csv_bytes(constant_rows(("case-A", "case-B", "case-C"), (0.1, 0.2, 0.9)))
    candidate = csv_bytes(constant_rows(("case-B", "case-C", "case-A"), (0.9, 0.1, 0.5)))
    before = (bytes(reference), bytes(candidate))
    output, receipt = fusion.fuse_csv(reference, candidate, roster)
    rows = decoded(output)
    assert rows[0] == list(fusion.COLUMNS)
    assert [row[0] for row in rows[1:]] == list(roster)
    expected = (2 / 3, 0.25, 7 / 12)
    for row, score in zip(rows[1:], expected, strict=True):
        assert [float(cell) for cell in row[1:]] == pytest.approx([score] * len(TARGETS))
    assert (reference, candidate) == before
    assert receipt["reference_csv_sha256"] == hashlib.sha256(reference).hexdigest()
    assert receipt["candidate_csv_sha256"] == hashlib.sha256(candidate).hexdigest()
    assert receipt["output_csv_sha256"] == hashlib.sha256(output).hexdigest()
    assert receipt["study_count"] == 3
    assert receipt["target_count"] == 12
    assert receipt["reference_weight"] == 0.75
    assert receipt["candidate_weight"] == 0.25
    assert receipt["accuracy_claim"] is None
    assert not receipt["model_provenance_verified_by_helper"]
    assert not any(uid in json.dumps(receipt) for uid in roster)


def test_strictly_increasing_transform_is_invariant_independently_per_target():
    roster = ("a", "b", "c", "d")
    reference = [(uid, [((index + column) % 4 + 1) / 5 for column in range(12)])
                 for index, uid in enumerate(roster)]
    candidate = [(uid, [((3 - index + column) % 4 + 1) / 5 for column in range(12)])
                 for index, uid in enumerate(roster)]
    transformed = [
        [(uid, [value ** 2 for value in scores]) for uid, scores in reference],
        [(uid, [value ** 3 for value in scores]) for uid, scores in candidate],
    ]
    original_output, _ = fusion.fuse_csv(csv_bytes(reference), csv_bytes(candidate), roster)
    transformed_output, _ = fusion.fuse_csv(*(csv_bytes(rows) for rows in transformed), roster)
    assert original_output == transformed_output
    values = [[float(cell) for cell in row[1:]] for row in decoded(original_output)[1:]]
    assert len(set(values[0])) > 1


def test_average_ties_and_all_tied_target():
    roster = ("a", "b", "c", "d")
    reference = constant_rows(roster, (0, 0, 0.5, 1))
    candidate = constant_rows(roster, (1, 0.5, 0.5, 0))
    for _, scores in reference + candidate:
        scores[1] = 0.2
    output, _ = fusion.fuse_csv(csv_bytes(reference), csv_bytes(candidate), roster)
    rows = decoded(output)[1:]
    assert [float(row[1]) for row in rows] == pytest.approx(
        (0.40625, 0.3125, 0.59375, 0.6875)
    )
    assert [float(row[2]) for row in rows] == [0.5] * 4


def test_single_study_is_neutral_rank_without_fallback():
    reference = csv_bytes(constant_rows(("only",), (0,)))
    candidate = csv_bytes(constant_rows(("only",), (1,)))
    output, receipt = fusion.fuse_csv(reference, candidate, ("only",))
    assert [float(cell) for cell in decoded(output)[1][1:]] == [0.5] * 12
    assert receipt["fallback_rows"] == 0


@pytest.mark.parametrize("role", ("reference", "candidate"))
@pytest.mark.parametrize("identifiers", (("a",), ("a", "a"), ("a", "x"), ("a", "b", "c")))
def test_missing_duplicate_extra_or_wrong_studies_rejected(role, identifiers):
    good = csv_bytes(constant_rows(("a", "b"), (0.1, 0.2)))
    bad = csv_bytes(constant_rows(identifiers, (0.3,) * len(identifiers)))
    inputs = (bad, good) if role == "reference" else (good, bad)
    with pytest.raises(ValueError):
        fusion.fuse_csv(*inputs, ("a", "b"))


@pytest.mark.parametrize("roster", ((), ("a", "a"), ("a", ""), ("a", " b"), "ab", ("a", 2)))
def test_invalid_official_roster_rejected(roster):
    good = csv_bytes(constant_rows(("a", "b"), (0.1, 0.2)))
    with pytest.raises(ValueError):
        fusion.fuse_csv(good, good, roster)


@pytest.mark.parametrize("value", ("nan", "NaN", "inf", "-inf", "-0.1", "1.01", "", "invalid"))
@pytest.mark.parametrize("role", ("reference", "candidate"))
def test_nonfinite_out_of_range_and_nonnumeric_scores_rejected(value, role):
    good = csv_bytes(constant_rows(("a",), (0.5,)))
    scores = [0.2] * 12
    scores[7] = value
    bad = csv_bytes([("a", scores)])
    inputs = (bad, good) if role == "reference" else (good, bad)
    with pytest.raises(ValueError):
        fusion.fuse_csv(*inputs, ("a",))


@pytest.mark.parametrize(
    "columns",
    (
        ("wrong-id", *TARGETS),
        ("StudyInstanceUID", *reversed(TARGETS)),
        (*fusion.COLUMNS[:-1], "ACL"),
        fusion.COLUMNS[:-1],
        (*fusion.COLUMNS, "extra"),
    ),
)
def test_wrong_header_rejected(columns):
    good = csv_bytes(constant_rows(("a",), (0.5,)))
    bad = csv_bytes(constant_rows(("a",), (0.5,)), columns)
    with pytest.raises(ValueError):
        fusion.fuse_csv(good, bad, ("a",))


@pytest.mark.parametrize("width", (11, 13))
def test_wrong_row_width_rejected(width):
    good = csv_bytes(constant_rows(("a",), (0.5,)))
    bad = csv_bytes([("a", [0.5] * width)])
    with pytest.raises(ValueError):
        fusion.fuse_csv(bad, good, ("a",))


def test_errors_do_not_disclose_identifiers_or_scores():
    bad = csv_bytes([("private-study", ["private-score"] * 12)])
    with pytest.raises(ValueError) as caught:
        fusion.fuse_csv(bad, bad, ("private-study",))
    assert "private-study" not in str(caught.value)
    assert "private-score" not in str(caught.value)
