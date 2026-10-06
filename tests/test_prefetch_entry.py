"""Production routing uses the actual test roster and closes all workers on failures."""

from argparse import Namespace

import pandas as pd
import pytest

from scripts.kaggle import prefetch_entry


def options(tmp_path):
    series = tmp_path / "series.csv"
    roster = tmp_path / "test.csv"
    pd.DataFrame({"StudyInstanceUID": ["1.2.2", "1.2.1"], "SeriesInstanceUID": ["b", "a"]}).to_csv(
        series, index=False
    )
    pd.DataFrame({"StudyInstanceUID": ["1.2.1", "1.2.2"]}).to_csv(roster, index=False)
    return Namespace(
        precision="fp32",
        workers=0,
        device="cuda",
        chunk_size=8,
        baseline_epoch_one=False,
        allow_diagnostic=False,
        resilient_series=True,
        recover_slices=True,
        isolate_study_errors=True,
        test_studies=roster,
        series=series,
        cache_root=tmp_path / "cache",
        output=tmp_path / "submission.csv",
    )


def factories(events, *, fail_close=False):
    class Pool:
        def __init__(self, **kwargs):
            assert kwargs == {"workers": 2, "timeout_seconds": 120}

        def invoke(self, *args, **kwargs):
            pass

        def close(self):
            events.append("pool_closed")

    class Preparation:
        completed = 1

        def __init__(self, frame, identifiers, spool, **kwargs):
            pd.testing.assert_frame_equal(frame, pd.read_csv(spool.parent / "series.csv"))
            assert identifiers == ["1.2.1", "1.2.2"]
            assert kwargs == {"chunk_size": 8}

        def prepare(self, *args, **kwargs):
            pass

        def close(self):
            events.append("preparation_closed")
            if fail_close:
                raise RuntimeError("future_failed")

    return Pool, Preparation


def test_original_consumer_receives_both_bound_callbacks(tmp_path, monkeypatch):
    args = options(tmp_path)
    events = []
    pool, preparation = factories(events)

    def consume(current, *, invoke, prepare_invoke):
        assert current.single_decode
        assert isinstance(invoke.__self__, pool)
        assert isinstance(prepare_invoke.__self__, preparation)
        return {"status": "passed", "submission_sha256": "original"}

    monkeypatch.setattr(prefetch_entry.chunked_infer, "run", consume)
    result = prefetch_entry.run(
        args,
        remaining_seconds=120,
        pool_factory=pool,
        preparation_factory=preparation,
        memory_reader=lambda *a: {"process_count": 7, "summed_lifetime_RSS_bytes": 1000},
    )
    assert result["submission_sha256"] == "original"
    assert events == ["preparation_closed", "pool_closed"]
    assert result["completed_prefetch_chunks"] == 1


def test_failed_future_still_closes_resident_processes(tmp_path, monkeypatch):
    args = options(tmp_path)
    events = []
    pool, preparation = factories(events, fail_close=True)
    monkeypatch.setattr(prefetch_entry.chunked_infer, "run", lambda *a, **k: {"status": "passed"})
    with pytest.raises(RuntimeError, match="future_failed"):
        prefetch_entry.run(
            args,
            remaining_seconds=120,
            pool_factory=pool,
            preparation_factory=preparation,
            memory_reader=lambda *a: {},
        )
    assert events == ["preparation_closed", "pool_closed"]
    assert not args.output.with_suffix(".metrics.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [("precision", "fp16"), ("chunk_size", 32), ("test_studies", None), ("allow_diagnostic", True)],
)
def test_unadmitted_production_policy_cannot_spawn(tmp_path, field, value):
    args = options(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="production"):
        prefetch_entry.run(args, pool_factory=lambda **kw: pytest.fail("spawned"))
