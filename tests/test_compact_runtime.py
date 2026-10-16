from argparse import Namespace

import pandas as pd
import pytest

from rsna.schema import TARGETS
from scripts.kaggle import compact_runtime as runtime


class Preparation:
    last = None

    def __init__(self, series, ids, spool, chunk_size):
        self.ids = ids
        self.closed = False
        self.calls = []
        assert chunk_size == 4
        Preparation.last = self

    def prepare(self, frame, ids, **options):
        assert options["cache_side"] == 384 and options["max_series"] == 6
        assert (
            options["recover_slices"] and options["single_decode"] and options["validate_geometry"]
        )
        options["cache"].mkdir()
        self.calls.append(ids)
        bad = [uid for uid in ids if uid == "u3"]
        return frame.loc[~frame.StudyInstanceUID.isin(bad)], {
            "unusable_study_ids": bad,
            "selected_series": len(ids) - len(bad),
        }

    def close(self):
        self.closed = True


class Pool:
    last = None
    failure = False

    def __init__(self, timeout_seconds):
        self.commands = []
        self.closed = False
        Pool.last = self

    def exchange(self, commands):
        self.commands.extend(commands)
        if self.failure:
            return [{"status": "failed", "stderr": "model failure"}]
        return [
            {
                "status": "passed",
                "result": {
                    "rows": {uid: [int(uid[1:]) / 10] * 12 for uid in c["identifiers"]},
                    "model_load_count": 1,
                    "device_index": 0,
                    "encoded_windows": len(c["identifiers"]) * 14,
                    "insufficient_window_studies": 0,
                    "peak_GPU_bytes": 100,
                    "checkpoint_sha256": c["checkpoint_sha256"],
                },
            }
            for c in commands
        ]

    def close(self):
        self.closed = True


def args(tmp_path):
    ids = ["u5", "u1", "u3", "u0", "u2"]
    test = tmp_path / "test.csv"
    series = tmp_path / "series.csv"
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(test, index=False)
    pd.DataFrame({"StudyInstanceUID": ids, "SeriesInstanceUID": ["s" + x for x in ids]}).to_csv(
        series, index=False
    )
    return Namespace(
        test_studies=test,
        series=series,
        checkpoint=tmp_path / "model.pt",
        checkpoint_sha256="a" * 64,
        scratch=tmp_path / "scratch",
        output=tmp_path / "submission.csv",
        dicom_root=tmp_path / "dicoms",
        allow_diagnostic=False,
        aggregation="full",
        precision="float16",
        max_seconds=120,
    )


def execute(a):
    return runtime.run(
        a,
        pool_factory=Pool,
        preparation_factory=Preparation,
        memory_reader=lambda p, q: {"summed_lifetime_RSS_bytes": 42},
    )


def test_exact_roster_tail_chunk_and_only_image_unavailability_fallback(tmp_path):
    a = args(tmp_path)
    result = execute(a)
    frame = pd.read_csv(a.output)
    assert frame.StudyInstanceUID.tolist() == ["u5", "u1", "u3", "u0", "u2"]
    assert list(frame.columns) == ["StudyInstanceUID", *TARGETS]
    assert frame.loc[frame.StudyInstanceUID == "u3", list(TARGETS)].eq(0.5).all().all()
    assert frame.loc[frame.StudyInstanceUID == "u2", list(TARGETS)].eq(0.2).all().all()
    assert Preparation.last.calls == [["u5", "u1", "u3", "u0"], ["u2"]]
    assert [c["identifiers"] for c in Pool.last.commands] == [["u5", "u0"], ["u1"], ["u2"]]
    assert Pool.last.closed and Preparation.last.closed
    assert result["studies"] == 5 and result["chunks"][0]["unusable_MRI_studies"] == 1
    assert not list(a.scratch.glob("chunk-*"))
    import hashlib

    assert result["CSV_sha256"] == hashlib.sha256(a.output.read_bytes()).hexdigest()
    assert len(result["test_roster_sha256"]) == 64
    assert len(result["chunks"][0]["workers"]) == 2
    assert all(c["preparation_wait_seconds"] >= 0 for c in result["chunks"])
    assert all(c["inference_wait_seconds"] >= 0 for c in result["chunks"])


def test_model_failure_is_fatal_and_never_converted_to_neutral(tmp_path, monkeypatch):
    a = args(tmp_path)
    monkeypatch.setattr(Pool, "failure", True)
    with pytest.raises(RuntimeError, match="model failure"):
        execute(a)
    assert not a.output.exists()
    assert Pool.last.closed and Preparation.last.closed


def test_diagnostics_cannot_emit_submission_csv(tmp_path):
    a = args(tmp_path)
    a.allow_diagnostic = True
    with pytest.raises(ValueError, match="Diagnostic"):
        execute(a)
    assert not a.scratch.exists()


def test_roster_and_probability_integrity():
    with pytest.raises(ValueError, match="roster"):
        runtime.frame_from_rows({"u": [0.2] * 12}, ["u", "u"])
    with pytest.raises(ValueError, match="probabilities"):
        runtime.frame_from_rows({"u": [float("nan")] * 12}, ["u"])
