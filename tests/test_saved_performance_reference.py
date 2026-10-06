"""Reference reuse rejects wrong model, bytes, study membership and numerical drift."""

import hashlib

import pandas as pd
import pytest

from rsna.schema import TARGETS
from scripts.kaggle.saved_performance_reference import compare_predictions, load_reference


def fixture(tmp_path):
    frame = pd.DataFrame({"StudyInstanceUID": ["a", "b"]})
    for t in TARGETS:
        frame[t] = [0.2, 0.8]
    path = tmp_path / "reference.csv"
    frame.to_csv(path, index=False)
    proof = {
        "status": "passed",
        "inference_precision": "fp32",
        "checkpoint_sha256": "a" * 64,
        "original_exact_source_SHA_verified": True,
        "original_kernel_version": 1,
        "studies": 2,
        "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return path, proof, frame.set_index("StudyInstanceUID")


def test_reuses_all_original_values_and_compares_in_reference_order(tmp_path):
    path, proof, expected = fixture(tmp_path)
    actual = load_reference(path, proof, ["a", "b"], "a" * 64)
    assert actual.equals(expected)
    candidate = actual.iloc[::-1].copy()
    candidate.iloc[0, 0] += 0.001
    result = compare_predictions(actual, candidate)
    assert result["studies"] == 2 and result["max_absolute_probability_error"] == pytest.approx(
        0.001
    )
    assert result["model_accuracy_or_clinical_equivalence_established"] is False


@pytest.mark.parametrize("bad", ["model", "hash", "roster", "version", "precision"])
def test_wrong_original_reference_is_rejected(tmp_path, bad):
    path, proof, _ = fixture(tmp_path)
    roster = ["a", "b"]
    if bad == "model":
        proof["checkpoint_sha256"] = "b" * 64
    if bad == "hash":
        path.write_text("changed")
    if bad == "roster":
        roster = ["b", "a"]
    if bad == "version":
        proof["original_kernel_version"] = 2
    if bad == "precision":
        proof["inference_precision"] = "fp16"
    with pytest.raises(ValueError):
        load_reference(path, proof, roster, "a" * 64)


@pytest.mark.parametrize("bad", ["large_error", "missing", "nonfinite"])
def test_disagreement_is_never_repaired_or_filtered(tmp_path, bad):
    _, _, reference = fixture(tmp_path)
    candidate = reference.copy()
    if bad == "large_error":
        candidate.iloc[0, 0] += 0.1
    if bad == "missing":
        candidate = candidate.iloc[:1]
    if bad == "nonfinite":
        candidate.iloc[0, 0] = float("nan")
    with pytest.raises(ValueError):
        compare_predictions(reference, candidate)
