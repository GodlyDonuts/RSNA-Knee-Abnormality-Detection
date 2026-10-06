"""Overlap stage ownership without changing cache bytes, order or failure behavior."""

import hashlib
import threading
from pathlib import Path

import pandas as pd
import pytest
from test_parallel_preparation import fixture

from scripts.kaggle.parallel_preparation import ParallelPreparation
from scripts.kaggle.prefetch_preparation import PrefetchPreparation


def hashes(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_actual_spawned_prefetch_two_chunks_keeps_original_cache_bytes_and_order(tmp_path):
    frame, ids, options = fixture(tmp_path, True)
    expected = []
    with ParallelPreparation() as serial:
        for index in range(2):
            group = ids[index * 2 : index * 2 + 2]
            part = frame.loc[frame.StudyInstanceUID.isin(group)]
            expected.append(
                serial.prepare(part, group, cache=tmp_path / f"reference{index}", **options)
            )
    with PrefetchPreparation(frame, ids, tmp_path / "spool", chunk_size=2) as prefetch:
        for index in range(2):
            group = ids[index * 2 : index * 2 + 2]
            part = frame.loc[frame.StudyInstanceUID.isin(group)]
            destination = tmp_path / f"actual{index}"
            selected, receipt = prefetch.prepare(part, group, cache=destination, **options)
            pd.testing.assert_frame_equal(selected, expected[index][0])
            assert receipt == expected[index][1]
            assert hashes(destination) == hashes(tmp_path / f"reference{index}")
        assert prefetch.completed == 1


def test_future_begins_while_caller_can_execute_original_model(tmp_path):
    started, release = threading.Event(), threading.Event()

    class Stub:
        def prepare(self, frame, ids, **options):
            if ids == ["b"]:
                started.set()
                assert release.wait(5)
            cache = Path(options["cache"])
            cache.mkdir(parents=True)
            (cache / "bytes").write_bytes(b"original")
            return frame, {"selected_series": len(frame)}

        def close(self):
            pass

    frame = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "SeriesInstanceUID": ["sa", "sb"]})
    prefetch = PrefetchPreparation(
        frame, ["a", "b"], tmp_path / "spool", chunk_size=1, preparation_factory=Stub
    )
    try:
        options = {"dicom_root": tmp_path, "cache_side": 16}
        prefetch.prepare(frame.iloc[:1], ["a"], cache=tmp_path / "one", **options)
        assert started.wait(5)  # The caller is free to invoke GPU inference at this point.
        release.set()
        prefetch.prepare(frame.iloc[1:], ["b"], cache=tmp_path / "two", **options)
        assert (tmp_path / "two/bytes").read_bytes() == b"original"
    finally:
        release.set()
        prefetch.close()


def test_wrong_chunk_order_or_reused_cache_is_not_silently_repaired(tmp_path):
    frame = pd.DataFrame({"StudyInstanceUID": ["a"], "SeriesInstanceUID": ["sa"]})
    with PrefetchPreparation(frame, ["a"], tmp_path / "spool") as prefetch:
        with pytest.raises(ValueError, match="chunk order"):
            prefetch.prepare(
                frame, ["b"], cache=tmp_path / "cache", dicom_root=tmp_path, cache_side=16
            )
