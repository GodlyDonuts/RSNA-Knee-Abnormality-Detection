"""Actual cached model reuse must retain the original inference probabilities."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_competitive_inference import _fixture

from rsna.competitive import infer
from rsna.schema import TARGETS

P = Path(__file__).resolve().parents[1] / "scripts/kaggle/resident_infer.py"
spec = importlib.util.spec_from_file_location("resident_test", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize("checkpoint_only", [False, True])
def test_actual_resident_matches_legacy_and_loads_once_across_requests(
    tmp_path, monkeypatch, checkpoint_only
):
    args, _ = _fixture(tmp_path, monkeypatch, size=3, checkpoint_only=checkpoint_only)
    args.precision = "fp32"
    infer.run(args)
    expected = pd.read_csv(args.output)
    args.output = tmp_path / "resident-1.csv"
    original = m.create_encoder_from_training_checkpoint if checkpoint_only else m.create_encoder
    calls = []

    def load(*a, **k):
        calls.append(1)
        return original(*a, **k)

    monkeypatch.setattr(
        m, "create_encoder_from_training_checkpoint" if checkpoint_only else "create_encoder", load
    )
    runner = m.ResidentInference(args)
    assert runner.run(args)["resident_requests"] == 1
    first = pd.read_csv(args.output)
    args.output = tmp_path / "resident-2.csv"
    assert runner.run(args)["resident_requests"] == 2
    assert len(calls) == 1 and runner.model_load_count == 1
    second = pd.read_csv(args.output)
    assert (
        first.StudyInstanceUID.tolist()
        == expected.StudyInstanceUID.tolist()
        == second.StudyInstanceUID.tolist()
    )
    np.testing.assert_array_equal(first[list(TARGETS)].to_numpy(), second[list(TARGETS)].to_numpy())
    np.testing.assert_allclose(
        first[list(TARGETS)].to_numpy(), expected[list(TARGETS)].to_numpy(), rtol=0, atol=0
    )


def test_changed_model_or_runtime_is_rejected_before_output(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch)
    args.precision = "fp32"
    runner = m.ResidentInference(args)
    changed = copy.copy(args)
    changed.output = tmp_path / "changed.csv"
    changed.precision = "bf16"
    with pytest.raises(ValueError, match="admission cannot change"):
        runner.run(changed)
    assert not changed.output.exists()
    args.checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Immutable resident"):
        runner.run(args)
    assert not args.output.exists()


def test_memory_checkpoint_reconstructs_same_predictions_in_resident_worker(tmp_path, monkeypatch):
    args, _ = _fixture(
        tmp_path,
        monkeypatch,
        plane_memory={"weight": 1, "seed": 20261006, "hidden_dim": 16},
    )
    args.precision = "fp32"
    infer.run(args)
    expected = pd.read_csv(args.output)
    args.output = tmp_path / "resident-memory.csv"
    runner = m.ResidentInference(args)
    assert runner.model.plane_memory is not None
    runner.run(args)
    np.testing.assert_array_equal(
        pd.read_csv(args.output)[list(TARGETS)].to_numpy(), expected[list(TARGETS)].to_numpy()
    )


@pytest.mark.parametrize("field,value", [("precision", "bf16"), ("batch_size", 2), ("workers", 1)])
def test_unaccepted_compute_recipe_rejected_before_loading(tmp_path, monkeypatch, field, value):
    args, _ = _fixture(tmp_path, monkeypatch)
    args.precision = "fp32"
    setattr(args, field, value)
    with pytest.raises(ValueError, match="Initial resident admission"):
        m.ResidentInference(args)
