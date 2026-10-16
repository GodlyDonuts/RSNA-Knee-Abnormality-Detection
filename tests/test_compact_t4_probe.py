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


def test_native_deadline_calls_the_real_bootstrap_interface(tmp_path):
    import inspect

    from scripts.kaggle import notebook_entry
    from scripts.kaggle.compact_t4_entry import run_bounded_native

    assert list(inspect.signature(notebook_entry.bounded_process).parameters) == [
        "command",
        "env",
        "timeout",
    ]
    helper = tmp_path / "code/scripts/kaggle/notebook_entry.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "def bounded_process(command, *, env, timeout):\n"
        " assert command == ['native-runtime']\n"
        " assert env == {'BOUND': '1'} and timeout == 2100\n"
    )
    run_bounded_native(["native-runtime"], tmp_path, {"BOUND": "1"}, 2100)
