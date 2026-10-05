"""The speed path must retain exactly the old MRI pixels and slice recovery."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from test_image_integrity import modified_dicom

import rsna.dicom as dicom
import rsna.preprocess as preprocess
from rsna.preprocess import Source

P = Path(__file__).resolve().parents[1] / "scripts/kaggle/chunked_infer.py"
spec = importlib.util.spec_from_file_location("single_decode_test", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.mark.parametrize("bad", [False, True])
def test_single_decode_cache_equals_legacy_retained_pixels(tmp_path, monkeypatch, bad):
    uid, sid = "1.2.3", "1.2.3.4"
    root = tmp_path / "source"
    folder = root / "test_series" / uid / sid
    folder.mkdir(parents=True)
    for i in range(3):
        name = f"{sid}.{i + 1}.dcm"
        (folder / name).write_bytes(
            b"not a DICOM"
            if bad and i == 1
            else modified_dicom(
                StudyInstanceUID=uid,
                SeriesInstanceUID=sid,
                SOPInstanceUID=f"{sid}.{i + 1}",
                InstanceNumber=i + 1,
            )
        )
    paths = sorted(str(p.relative_to(root)) for p in folder.glob("*.dcm"))
    source = Source(root)
    calls = []
    original = dicom.decode_slice

    def decode(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(dicom, "decode_slice", decode)
    monkeypatch.setattr(preprocess, "decode_slice", decode)
    retained, frozen, rejected = m.healthy_slices(source, paths)
    before = len(calls)
    legacy = m.cache_series(frozen, retained, tmp_path / "legacy", 16, "same-source")
    legacy_total = len(calls)
    calls.clear()
    actual, kept, failures = m.cache_with_lazy_recovery(
        source, paths, tmp_path / "fast", 16, "same-source"
    )
    assert kept == retained and failures == rejected
    assert actual == legacy
    lp = next((tmp_path / "legacy").rglob("*.npz"))
    fp = next((tmp_path / "fast").rglob("*.npz"))
    with np.load(lp, allow_pickle=False) as a, np.load(fp, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    if not bad:
        assert len(calls) == before and legacy_total == 2 * len(calls)


def test_source_read_failure_is_fatal_and_never_recovered(tmp_path):
    class Broken:
        def read(self, path):
            raise OSError("transport failed")

    with pytest.raises(OSError, match="transport failed"):
        m.cache_with_lazy_recovery(Broken(), ["source.dcm"], tmp_path / "cache", 16, "test")


def test_non_data_failure_still_raises_original_error(tmp_path, monkeypatch):
    class Bytes:
        def read(self, path):
            return b"bytes"

    monkeypatch.setattr(
        m, "cache_series", lambda *a: (_ for _ in ()).throw(RuntimeError("CUDA failure"))
    )
    monkeypatch.setattr(
        m,
        "healthy_slices",
        lambda *a: pytest.fail("Infrastructure failures cannot invoke recovery"),
    )
    with pytest.raises(RuntimeError, match="CUDA failure"):
        m.cache_with_lazy_recovery(Bytes(), ["source.dcm"], tmp_path / "cache", 16, "test")
