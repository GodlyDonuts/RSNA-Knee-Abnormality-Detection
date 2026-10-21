import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_size_only_selection_and_complete_real_pixel_equivalence(tmp_path, monkeypatch):
    b = load("large_study_engineering", HERE / "benchmark.py")
    fixture = load("DICOM_fixture", ROOT / "tests/test_predictive_dicom.py")
    study = tmp_path / fixture.STUDY
    for i in range(4):
        fixture.write_dicom(
            study,
            fixture.SERIES_A,
            i + 1,
            ((np.arange(80 * 80).reshape(80, 80) + i) % 4096).astype(np.uint16),
            position=i,
        )
    rows, count = b.source_inventory(tmp_path, time.monotonic() + 30)
    assert count == 1 and rows[0]["frames"] == 4
    assert rows[0]["estimated_FP32_pixel_bytes"] == 4 * 80 * 80 * 4
    assert b.select(rows) == rows
    modules = {
        "_predictive_descriptor_cache": "predictive-final-serving-20261020-v1/descriptor_cache.py",
        "selected_frames": "predictive-selected-frame-preparation-20261021-v1/selected_frames.py",
        "descriptor_batch": "predictive-descriptor-batch-20261021-v1/descriptor_batch.py",
        "resilient_decoder": "predictive-inference-resilience-20261021-v1/decoder.py",
    }
    for name, path in modules.items():
        monkeypatch.setitem(sys.modules, name, load(name, HERE.parent / path))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *_: None)
    results = [b.child({"variant": v, "path": str(study)}) for v in ("original", "successor")]
    assert all(r["status"] == "complete" for r in results), results
    assert results[0]["context_digest_private"] == results[1]["context_digest_private"]
    assert results[1]["stats"]["decoder_route"] == "original_exact"


def test_digest_checks_metadata_and_tensor_bytes():
    b = load("large_study_digest", HERE / "benchmark.py")
    x = {"images": torch.tensor([1.0, 2.0]), "metadata": torch.tensor([0, 1])}
    a = b.context_digest(x)
    x["metadata"][0] = 1
    assert a != b.context_digest(x)
    x["metadata"][0] = 0
    x["images"][1] = 3
    assert a != b.context_digest(x)


def test_exact_CPU_only_capsule_and_request():
    c = load("stress_compiler", HERE / "compile.py")
    a = load("stress_account", HERE / "account.py")
    source, meta = c.build(
        (
            ROOT / "artifacts/predictive-selected-frame-preparation-20261021-v1/candidate.py"
        ).read_text()
    )
    request = a.request_for(source, meta)
    assert request["isPrivate"] and not request["enableGpu"] and not request["enableInternet"]
    assert request["sessionTimeoutSeconds"] == 2700
    assert request["kernelDataSources"] == request["datasetDataSources"] == []
    assert request["slug"].startswith("godlydonuts/rsna-predictive-largecpu-")
    assert "context_digest_private" in source


def test_actual_child_transport_and_timeout_preserve_failure(tmp_path, monkeypatch):
    b = load("bounded_child_engineering", HERE / "benchmark.py")
    child = tmp_path / "benchmark.py"
    child.write_text(
        "def child(task):\n import time\n time.sleep(task['delay'])\n return {'status':'complete'}\n"
    )
    monkeypatch.setattr(b, "__file__", str(child))
    assert b.bounded_child({"delay": 0}, 5)["status"] == "complete"
    timed = b.bounded_child({"delay": 10}, 0.2)
    assert timed["status"] == "bounded_child_time_limit" and timed["exit_code"] != 0
