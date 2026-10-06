"""Actual spawned CPU workers must retain serial MRI recovery/cache bytes and ordering."""

import hashlib

import pandas as pd
import pytest
from test_image_integrity import modified_dicom

from scripts.kaggle.chunked_infer import prepare_usable_series
from scripts.kaggle.parallel_preparation import ParallelPreparation


def fixture(tmp_path, bad):
    root = tmp_path / "source"
    ids = ["1.2.30", "1.2.10", "1.2.40", "1.2.20"]
    rows = []
    for index, uid in enumerate(ids):
        sid = uid + ".1"
        folder = root / "test_series" / uid / sid
        folder.mkdir(parents=True)
        for n in range(3):
            (folder / f"{sid}.{n + 1}.dcm").write_bytes(
                b"invalid"
                if bad and index == 1 and n == 1
                else modified_dicom(
                    StudyInstanceUID=uid,
                    SeriesInstanceUID=sid,
                    SOPInstanceUID=f"{sid}.{n + 1}",
                    InstanceNumber=n + 1,
                )
            )
        rows.append(
            dict(
                StudyInstanceUID=uid,
                SeriesInstanceUID=sid,
                Anatomical_Plane=" sagittal ",
                Fluid_Sensitive=1,
                Fat_Suppression=0,
            )
        )
    frame = pd.DataFrame(rows, index=[8, 2, 7, 1])
    options = dict(
        dicom_root=root,
        cache_side=16,
        max_series=3,
        source_revision="same-source",
        allow_unusable_studies=True,
        recover_slices=True,
        validate_geometry=True,
        single_decode=True,
    )
    return frame, ids, options


@pytest.mark.parametrize("bad", [False, True])
def test_spawned_parallel_cache_and_receipts_equal_serial_in_original_order(tmp_path, bad):
    frame, ids, options = fixture(tmp_path, bad)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    expected, receipt = prepare_usable_series(frame, ids, cache=serial, **options)
    with ParallelPreparation() as runner:
        actual, observed = runner.prepare(frame, ids, cache=parallel, **options)
        assert len(set(runner.worker_pids())) == 2
    pd.testing.assert_frame_equal(actual, expected)
    assert observed == receipt

    def digest(root):
        return {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        }

    assert digest(parallel) == digest(serial)


def test_non_data_cache_integrity_failure_remains_fatal(tmp_path):
    frame, ids, options = fixture(tmp_path, False)
    cache = tmp_path / "cache"
    prepare_usable_series(frame, ids, cache=cache, **options)
    path = next(cache.rglob("*.npz"))
    path.write_bytes(b"corrupted")
    with ParallelPreparation() as runner:
        with pytest.raises(ValueError):
            runner.prepare(frame, ids, cache=cache, **options)


def test_minimum_cpu_budget_is_explicit():
    with pytest.raises(ValueError):
        ParallelPreparation(workers=4)
