import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_diagnostic_reproduces_strict_original_metadata_failures(tmp_path):
    b = load("headers", HERE / "benchmark.py")
    f = load("fixture_dicom", REPO / "tests/test_predictive_dicom.py")
    root = tmp_path / f.STUDY
    for path in f.fixture(root):
        assert b.inspect_header(path) == "accepted"
    path = f.write_dicom(
        root, f.SERIES_A, 8, np.zeros((8, 8), dtype=np.uint16), InstanceNumber=None
    )
    assert b.inspect_header(path) == "required_integer_metadata"
    path = f.write_dicom(
        root, f.SERIES_A, 9, np.zeros((8, 8), dtype=np.uint16), ImagePositionPatient=None
    )
    assert b.inspect_header(path) == "geometry"


def test_compilation_and_request_remain_private_CPU_only():
    c = load("header_compile", HERE / "compile.py")
    a = load("header_account", HERE / "account.py")
    source, meta = c.build(
        (REPO / "artifacts/predictive-descriptor-batch-20261021-v1/candidate.py").read_text()
    )
    request = a.request_for(source, meta)
    assert (
        request["enableGpu"] is False
        and request["enableInternet"] is False
        and request["isPrivate"] is True
    )
    assert request["datasetDataSources"] == request["kernelDataSources"] == []
    assert request["slug"].startswith("godlydonuts/rsna-predictive-headercpu-")
    assert "predictive-hidden-input-audit-v1" in source
