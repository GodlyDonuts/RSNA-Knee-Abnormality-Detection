"""Tests only the new export/CSV wrapper; synthetic rows are not MRI acceptance."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "connected-witness-paths-20261017-v1"))
spec = importlib.util.spec_from_file_location("new_path_serving", HERE / "serving.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def exported(tmp_path):
    model = m.PathReader()
    binding = {
        "engine": m.VERSION,
        "diagnostic_only": False,
        "arm": "aligned",
        "source_git": "engineering-only",
        "source_manifest_sha256": "source",
        "GPU_acceptance_sha256": "synthetic-not-GPU",
        "execution_wrapper_source_git": "wrapper",
        "targets": list(m.TARGETS),
        "data": {
            "original": {"parent_sha256": m.PARENT},
            "sampling": "dense",
            "encoder_precision": "float32",
        },
    }
    state = {
        "binding": binding,
        "epoch": 3,
        "completed": 843,
        "cursor": 0,
        "optimizer_boundary": True,
        "model": model.state_dict(),
    }
    p = tmp_path / "checkpoint.pt"
    torch.save(state, p)
    terminal = {
        "status": "actual_fixed_final_dense_coherent_path3epoch_complete",
        "epochs": 3,
        "optimizer_updates": 843,
        "encoder_trained": False,
        "base_checkpoint_sha256": m.PARENT,
        "checkpoint_sha256": m.digest(p),
        "arm": "aligned",
        "source_git": "engineering-only",
        "source_manifest_sha256": "source",
        "GPU_acceptance_sha256": "synthetic-not-GPU",
    }
    t = tmp_path / "result.json"
    t.write_text(json.dumps(terminal))
    out = tmp_path / "head.pt"
    receipt = m.export_final(p, t, out)
    return p, t, out, receipt


class Provider:
    binding = {
        "original": {"parent_sha256": m.PARENT},
        "sampling": "dense",
        "encoder_precision": "float32",
        "encoder_chunk_size": 12,
    }

    def __init__(self, ids=("first", "second")):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {
            "study_uid": self.ids[i],
            "maps": torch.zeros(2, 1),
            "base_logits": torch.arange(12, dtype=torch.float32) - 6,
            "plan": None,
        }


class Formula(torch.nn.Module):
    def forward(self, maps, base, plan, *, arm):
        return base, torch.zeros_like(base)


def runner(exported):
    _, _, out, r = exported
    model = m.PathInference(out, r["sha256"], parent_sha256=m.PARENT, device="cpu")
    model.reader = Formula()
    return model


def test_export_contains_only_new_head_and_provenance(exported):
    _, _, p, r = exported
    v = torch.load(p, weights_only=True)
    assert set(v) == {"policy", "model"} and len(v["model"]) == 9
    assert r["bytes"] < 1024 * 1024 and r["policy"]["new_head_parameters"] == 68560
    assert v["policy"]["T4_runtime_qualified"] is False


def test_csv_order_columns_sigmoid_without_rounding(exported, tmp_path):
    import csv

    out = tmp_path / "submission.csv"
    r = runner(exported).predict_csv(Provider(), out)
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["StudyInstanceUID", *m.TARGETS]
    assert [row[0] for row in rows[1:]] == ["first", "second"]
    assert [float(v) for v in rows[1][1:]] == torch.sigmoid(
        torch.arange(12, dtype=torch.float32) - 6
    ).tolist()
    assert r["rows"] == 2 and r["columns"] == 13 and r["windows"] == 4
    assert not out.with_name("submission.csv.partial").exists()


def test_wrong_parent_refused(exported):
    _, _, out, r = exported
    with pytest.raises(ValueError):
        m.PathInference(out, r["sha256"], parent_sha256="other", device="cpu")


def test_duplicate_never_publishes_final_csv(exported, tmp_path):
    out = tmp_path / "duplicate.csv"
    with pytest.raises(ValueError):
        runner(exported).predict_csv(Provider(("same", "same")), out)
    assert not out.exists() and out.with_name("duplicate.csv.partial").exists()


def test_deadline_never_publishes_incomplete_csv(exported, tmp_path):
    out = tmp_path / "late.csv"
    with pytest.raises(TimeoutError):
        runner(exported).predict_csv(Provider(), out, deadline_seconds=1e-300)
    assert not out.exists()


def test_output_is_immutable(exported, tmp_path):
    out = tmp_path / "existing.csv"
    out.write_text("keep")
    with pytest.raises(FileExistsError):
        runner(exported).predict_csv(Provider(), out)
    assert out.read_text() == "keep"


def test_unfinished_snapshot_not_exported(exported, tmp_path):
    p, t, _, _ = exported
    d = json.loads(t.read_text())
    d["optimizer_updates"] = 842
    t.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        m.export_final(p, t, tmp_path / "invalid.pt")


def test_changed_head_refused(exported):
    _, _, out, r = exported
    with out.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError):
        m.PathInference(out, r["sha256"], parent_sha256=m.PARENT, device="cpu")


def test_two_lanes_equal_original_single_lane_csv(exported, tmp_path):
    provider = Provider(("one", "two", "three", "four", "five"))
    model = runner(exported)
    single = tmp_path / "single.csv"
    model.predict_csv(provider, single)
    lanes = []
    for rank in (0, 1):
        p = tmp_path / f"lane-{rank}.csv"
        model.predict_csv(m.ShardedProvider(provider, rank=rank), p)
        lanes.append(p)
    merged = tmp_path / "submission.csv"
    r = m.merge_shards(lanes, list(provider.ids), merged)
    assert merged.read_bytes() == single.read_bytes() and r["lanes"] == 2


def test_incomplete_lanes_cannot_be_published(exported, tmp_path):
    model = runner(exported)
    p = tmp_path / "lane.csv"
    model.predict_csv(Provider(("one",)), p)
    out = tmp_path / "missing.csv"
    with pytest.raises(ValueError):
        m.merge_shards([p], ["one", "two"], out)
    assert not out.exists()


def test_overlapping_lanes_cannot_be_published(exported, tmp_path):
    paths = []
    for i in (0, 1):
        p = tmp_path / f"duplicate-lane-{i}.csv"
        runner(exported).predict_csv(Provider(("one",)), p)
        paths.append(p)
    out = tmp_path / "overlapping.csv"
    with pytest.raises(ValueError):
        m.merge_shards(paths, ["one"], out)
    assert not out.exists()


def test_explicit_larger_encoder_batch_reaches_prediction_boundary(exported, tmp_path):
    _, _, head, receipt = exported
    model = m.PathInference(
        head, receipt["sha256"], parent_sha256=m.PARENT, device="cpu", encoder_chunk_size=32
    )
    model.reader = Formula()
    provider = Provider()
    provider.binding = {**provider.binding, "encoder_chunk_size": 32}
    output = tmp_path / "batch32.csv"
    result = model.predict_csv(provider, output)
    reference = tmp_path / "batch12.csv"
    runner(exported).predict_csv(Provider(), reference)
    assert output.read_bytes() == reference.read_bytes()
    assert result["encoder_chunk_size"] == 32
    assert result["changed_batching_requires_target_device_comparison"] is True
    # This checks wrapper plumbing on synthetic precomputed maps, not actual
    # encoder equivalence or speed when image batch size changes on T4.


def test_encoder_execution_batch_mismatch_refused_before_study_reads(exported, tmp_path):
    class Unreadable(Provider):
        binding = {**Provider.binding, "encoder_chunk_size": 32}

        def __getitem__(self, i):
            raise AssertionError("Mismatched execution must fail before MRI reads")

    output = tmp_path / "mismatch.csv"
    with pytest.raises(ValueError):
        runner(exported).predict_csv(Unreadable(), output)
    assert not output.exists() and not output.with_name("mismatch.csv.partial").exists()


@pytest.mark.parametrize("size", [0, 97, True, 12.0, "32"])
def test_invalid_encoder_execution_batch_rejected_before_model_reads(size, tmp_path):
    with pytest.raises(ValueError):
        m.PathInference(
            tmp_path / "not-read.pt",
            "unused",
            parent_sha256=m.PARENT,
            device="cpu",
            encoder_chunk_size=size,
        )
