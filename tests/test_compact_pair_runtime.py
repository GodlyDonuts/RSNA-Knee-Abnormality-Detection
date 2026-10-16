
import pandas as pd
import pytest
import torch
from test_compact_runtime import Preparation
from test_compact_runtime import args as base_args

from rsna.schema import TARGETS
from scripts.kaggle import compact_pair_runtime as pair


class Pool:
    last = None
    corrupt_missing = False

    def __init__(self, *, pair, timeout_seconds):
        self.secondary = {}
        self.qualifications = {0: {"accepted": True}, 1: {"accepted": True}}
        self.closed = False
        Pool.last = self

    def exchange(self, commands):
        answers = []
        for c in commands:
            rows = {uid: [0.2] * 12 for uid in c["identifiers"]}
            self.secondary.update({uid: [0.3] * 12 for uid in rows})
            answers.append(
                {
                    "status": "passed",
                    "result": {
                        "rows": rows,
                        "model_load_count": 1,
                        "device_index": 0,
                        "encoded_windows": 14 * len(rows),
                        "insufficient_window_studies": 0,
                        "peak_GPU_bytes": 100,
                        "checkpoint_sha256": c["checkpoint_sha256"],
                    },
                }
            )
        if self.corrupt_missing:
            self.secondary.pop(next(iter(self.secondary)))
        return answers

    def close(self):
        self.closed = True


def arguments(tmp_path):
    a = base_args(tmp_path)
    a.output = tmp_path / "average.csv"
    a.epoch_output = tmp_path / "epoch.csv"
    a.source_shas = ["b" * 64, "c" * 64, "d" * 64]
    a.epoch_sha256 = a.source_shas[-1]
    for name in ["epoch_checkpoint", "fixture", "average_reference", "epoch_reference"]:
        setattr(a, name, tmp_path / name)
    for name in ["fixture_sha256", "average_reference_sha256", "epoch_reference_sha256"]:
        setattr(a, name, "e" * 64)
    return a


def execute(a):
    return pair.run(
        a,
        pool_factory=Pool,
        preparation_factory=Preparation,
        memory_reader=lambda p, q: {"summed_lifetime_RSS_bytes": 42},
    )


def test_same_single_preparation_roster_neutral_fallback_and_separate_predictions(tmp_path):
    a = arguments(tmp_path)
    result = execute(a)
    average, final = pd.read_csv(a.output), pd.read_csv(a.epoch_output)
    assert average.StudyInstanceUID.tolist() == final.StudyInstanceUID.tolist()
    assert Preparation.last.calls == [["u5", "u1", "u3", "u0"], ["u2"]]
    assert average.loc[average.StudyInstanceUID.ne("u3"), list(TARGETS)].eq(0.2).all().all()
    assert final.loc[final.StudyInstanceUID.ne("u3"), list(TARGETS)].eq(0.3).all().all()
    assert final.loc[final.StudyInstanceUID.eq("u3"), list(TARGETS)].eq(0.5).all().all()
    assert result["one_image_decode_for_both_models"] and not result["competition_submission"]
    assert Pool.last.closed and Preparation.last.closed


@pytest.mark.parametrize("change", ["submission", "same", "wrong_source", "diagnostic"])
def test_reject_invalid_research_or_average_scope_before_decoding(tmp_path, change):
    a = arguments(tmp_path)
    if change == "submission":
        a.epoch_output = tmp_path / "submission.csv"
    elif change == "same":
        a.epoch_output = a.output
    elif change == "wrong_source":
        a.source_shas[-1] = "f" * 64
    else:
        a.allow_diagnostic = True
    with pytest.raises(ValueError):
        execute(a)
    assert not a.scratch.exists()


def test_missing_secondary_cannot_be_silently_filled(tmp_path, monkeypatch):
    a = arguments(tmp_path)
    monkeypatch.setattr(Pool, "corrupt_missing", True)
    with pytest.raises(ValueError, match="missingness"):
        execute(a)
    assert not a.epoch_output.exists()


def test_worker_uses_one_exact_window_tensor_for_both_models(tmp_path, monkeypatch):
    a = tmp_path / "series.csv"
    pd.DataFrame({"StudyInstanceUID": ["u"]}).to_csv(a, index=False)
    image = torch.zeros(3, 3, 4, 4)
    sample = {
        "study_uid": "u",
        "images": image,
        "series_mask": torch.tensor([True]),
        "slice_mask": torch.tensor([[True] * 3]),
    }
    monkeypatch.setattr(pair, "CompetitiveStudyDataset", lambda *a, **k: [sample])
    monkeypatch.setattr(pair, "candidate_windows", lambda *a: [1, 2, 3])
    monkeypatch.setattr(pair, "window_tensor", lambda *a: image)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    seen = []

    def model(windows, *, chunk_size):
        seen.append(windows.data_ptr())
        return torch.zeros(12)

    w = object.__new__(pair.PairWorker)
    w.identity = {"checkpoint_sha256": "a", "epoch_sha256": "b"}
    w.device, w.index, w.qualification = torch.device("cpu"), 0, {}
    w.average = w.epoch = model
    request = {**w.identity, "identifiers": ["u"], "series": str(a), "cache": str(tmp_path)}
    result = w.run(request)
    assert seen == [image.data_ptr()] * 2
    assert result["rows"] == result["secondary_rows"] == {"u": [0.5] * 12}
    assert result["encoded_windows"] == 3
    with pytest.raises(ValueError, match="immutable"):
        w.run({**request, "epoch_sha256": "different"})
