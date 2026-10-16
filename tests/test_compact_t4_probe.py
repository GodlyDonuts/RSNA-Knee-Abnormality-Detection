import pytest
import torch

from scripts.kaggle.compact_t4_probe import compare, validate_fixture


def test_probability_precision_policy_is_explicit():
    assert compare([0.5] * 12, [0.5001] * 12, 0.0003) < 0.0003
    with pytest.raises(ValueError, match="tolerance"):
        compare([0.5] * 12, [0.504] * 12, 0.003)
    with pytest.raises(ValueError, match="Finite"):
        compare([float("nan")] * 12, [0.5] * 12, 0.003)
    with pytest.raises(ValueError, match="bounds"):
        compare([1.01] * 12, [0.5] * 12, 0.003)


def test_diagnostic_fixture_contains_no_labels_or_original_identifier():
    f = {
        "windows": torch.zeros(1, 3, 384, 384),
        "series_mask": torch.tensor([True]),
        "slice_mask": torch.ones((1, 3), dtype=torch.bool),
        "study_uid": "engineering-fixture-v1",
    }
    assert validate_fixture(f) == 1
    with pytest.raises(ValueError, match="unlabeled"):
        validate_fixture({**f, "labels": torch.zeros(12)})
    with pytest.raises(ValueError, match="identifiers"):
        validate_fixture({**f, "study_uid": "original-clinical-id"})
    f["windows"][0, 0, 0, 0] = 2
    with pytest.raises(ValueError, match="normalized"):
        validate_fixture(f)


def test_payload_discovery_prunes_competition_tree(tmp_path):
    import hashlib

    from scripts.kaggle.compact_t4_entry import discover_bundle

    payload = tmp_path / "datasets" / "private"
    payload.mkdir(parents=True)
    expected = payload / "bundle-manifest.json"
    expected.write_text("{}")
    # A matching file in the MRI tree must never be discovered or counted.
    hidden = tmp_path / "competitions" / "rsna-knee-abnormality-detection" / "test_series" / "study"
    hidden.mkdir(parents=True)
    (hidden / "bundle-manifest.json").write_text("{}")
    assert discover_bundle(tmp_path, hashlib.sha256(b"{}").hexdigest()) == expected
