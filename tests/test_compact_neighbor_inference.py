import copy
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_compact_inference import Encoder, checkpoint
from test_compact_runtime import Pool, Preparation, args
from test_compact_source_neighbors import Fixture

from rsna.competitive import compact_infer as original
from rsna.competitive import compact_neighbor_infer as neighbor
from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_source_neighbors import ENGINE, POLICY
from rsna.competitive.compact_train import predict
from rsna.schema import TARGETS
from scripts.kaggle import compact_neighbor_runtime as runtime


def new_checkpoint(tmp_path, diagnostic=False):
    p, value = checkpoint(tmp_path, diagnostic)
    value["binding"].update(engine=ENGINE, input_policy=dict(POLICY))
    torch.save(value, p)
    return p, value


def test_new_and_old_loaders_cannot_silently_swap_input_contracts(tmp_path, monkeypatch):
    p, value = new_checkpoint(tmp_path)
    monkeypatch.setattr(neighbor, "_blueprint", lambda: Encoder(1024))
    model, policy = neighbor.load_model(p, file_sha256(p))
    assert policy["engine"] == ENGINE and policy["input_policy"] == POLICY
    assert policy["competition_eligible"] and not policy["actual_target_runtime_qualified"]
    assert all(torch.equal(model.state_dict()[k], v) for k, v in value["model"].items())
    with pytest.raises(ValueError, match="provenance"):
        original.load_model(p, file_sha256(p))
    value["binding"]["engine"] = "native_coatnet_bag12_N3_full_v1"
    torch.save(value, p)
    with pytest.raises(ValueError, match="provenance"):
        neighbor.load_model(p, file_sha256(p))
    value["binding"]["engine"] = ENGINE
    value["binding"]["input_policy"]["unusable_edge"] = "silently_drop"
    torch.save(value, p)
    with pytest.raises(ValueError, match="provenance"):
        neighbor.load_model(p, file_sha256(p))


def test_diagnostic_checkpoint_requires_explicit_probe_and_stays_ineligible(tmp_path, monkeypatch):
    p, _ = new_checkpoint(tmp_path, diagnostic=True)
    monkeypatch.setattr(neighbor, "_blueprint", lambda: Encoder(1024))
    with pytest.raises(ValueError, match="not competition eligible"):
        neighbor.load_model(p, file_sha256(p))
    _, policy = neighbor.load_model(p, file_sha256(p), allow_diagnostic=True)
    assert not policy["competition_eligible"] and policy["input_policy"] == POLICY


def test_full_inference_equals_actual_training_evaluator_and_never_uses_targets():
    data = Fixture(True)
    for row in data.rows:
        row["legacy_edge_fallback_windows"] = 2
    model = original.DeploymentBagClassifier(Encoder(), precision="float32", dimension=4)
    expected = predict(model, data, torch.device("cpu"))
    # Prediction must not touch report/target/mask contents.
    for row in data.rows:
        row["labels"], row["label_mask"] = object(), object()
    result = neighbor.prediction_rows(
        model, data, list(reversed(expected.study_uid)), device=torch.device("cpu")
    )
    assert result["encoded_windows"] == 84 and result["legacy_edge_fallback_windows"] == 6
    for row in expected.to_dict("records"):
        assert result["rows"][row["study_uid"]] == [row[t] for t in TARGETS]
    with pytest.raises(ValueError, match="Incomplete"):
        neighbor.prediction_rows(
            model, data, [*expected.study_uid, "missing"], device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="Explicit"):
        neighbor.prediction_rows(
            model, data.rows, expected.study_uid.tolist(), device=torch.device("cpu")
        )


def test_unusable_study_fallback_does_not_hide_model_failures():
    data = Fixture(True)
    data.rows = data.rows[:1]
    sample = data[0]
    sample["slice_mask"][:] = False
    sample["slice_mask"][:, :2] = True

    # No model forward is valid for a two-slice-only study.
    class Model(torch.nn.Module):
        def forward(self, *a, **k):
            raise RuntimeError("should not run")

    result = neighbor.prediction_rows(
        Model(), data, [sample["study_uid"]], device=torch.device("cpu")
    )
    assert (
        result["rows"][sample["study_uid"]] == [0.5] * 12
        and result["insufficient_window_studies"] == 1
    )
    sample["slice_mask"][:] = True
    with pytest.raises(RuntimeError, match="should not run"):
        neighbor.prediction_rows(Model(), data, [sample["study_uid"]], device=torch.device("cpu"))


def test_image_only_dataset_constructs_unobserved_placeholders(monkeypatch):
    captured = {}

    def fake_base(rows, series, **options):
        captured.update(rows=rows, options=options)
        return "base"

    monkeypatch.setattr(neighbor, "CompetitiveStudyDataset", fake_base)
    monkeypatch.setattr(neighbor, "MatchedSourceNeighborDataset", lambda base: ("new-view", base))
    result = neighbor.inference_dataset(["b", "a"], pd.DataFrame(), Path("/not-read"))
    assert result == ("new-view", "base")
    assert captured["rows"].study_uid.tolist() == ["b", "a"]
    assert captured["rows"][list(TARGETS)].eq(0).all().all()
    assert captured["rows"][[t + "__mask" for t in TARGETS]].eq(0).all().all()
    assert captured["options"]["raw_split"] == "test_series"
    assert captured["options"]["side"] == 384 and captured["options"]["slices"] == 16


class MatchedPool(Pool):
    legacy_edge_fallback_windows = 3


def test_endpoint_preserves_bounded_raw_roster_and_distinct_final_policy(tmp_path):
    a = args(tmp_path)
    result = runtime.run(
        a,
        pool_factory=MatchedPool,
        preparation_factory=Preparation,
        memory_reader=lambda p, q: {"summed_lifetime_RSS_bytes": 42},
    )
    assert result["engine"] == ENGINE and result["input_policy"] == POLICY
    assert result["status"] == "compact_matched_source_neighbor_test_roster_complete"
    assert result["legacy_edge_fallback_windows"] == 3
    assert pd.read_csv(a.output).StudyInstanceUID.tolist() == ["u5", "u1", "u3", "u0", "u2"]
    a.aggregation = "mc128"
    with pytest.raises(ValueError, match="fixed full-bag"):
        runtime.run(a)


def test_pool_refuses_legacy_policy_and_invalid_geometry_fallback_counts(monkeypatch):
    pool = runtime.NeighborPool.__new__(runtime.NeighborPool)
    pool.legacy_edge_fallback_windows = 0
    reply = {
        "status": "passed",
        "result": {
            "engine": ENGINE,
            "input_policy": dict(POLICY),
            "legacy_edge_fallback_windows": 2,
            "encoded_windows": 14,
        },
    }
    monkeypatch.setattr(runtime.ResidentPool, "exchange", lambda self, commands: [reply])
    assert pool.exchange([{}]) == [reply] and pool.legacy_edge_fallback_windows == 2
    wrong = copy.deepcopy(reply)
    wrong["result"]["input_policy"]["version"] = "legacy"
    monkeypatch.setattr(runtime.ResidentPool, "exchange", lambda self, commands: [wrong])
    with pytest.raises(ValueError, match="policy"):
        pool.exchange([{}])
    reply["result"]["legacy_edge_fallback_windows"] = 15
    monkeypatch.setattr(runtime.ResidentPool, "exchange", lambda self, commands: [reply])
    with pytest.raises(ValueError, match="fallback"):
        pool.exchange([{}])
