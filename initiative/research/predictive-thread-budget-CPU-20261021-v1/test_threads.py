import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_only_declared_image_rounding_is_tolerated():
    b = load("thread_compare", HERE / "benchmark.py")
    a = [
        {
            "images": np.full((2, 3, 8, 8), 0.5, np.float32),
            "targets": np.zeros((2, 64), np.float32),
            "mask": np.array([True, False]),
        }
    ]
    c = copy.deepcopy(a)
    c[0]["images"][0, 0, 0, 0] = np.nextafter(np.float32(0.5), np.float32(1))
    actual = b.compare(a, c)
    assert not actual["bitwise_equal"] and actual["all_other_values_exact"]
    for kind in ("large_image", "nan", "target", "mask", "shape", "dtype", "field"):
        c = copy.deepcopy(a)
        if kind == "large_image":
            c[0]["images"][0, 0, 0, 0] += 0.001
        elif kind == "nan":
            c[0]["images"][0, 0, 0, 0] = np.nan
        elif kind == "target":
            c[0]["targets"][0, 0] = -0.0
        elif kind == "mask":
            c[0]["mask"][0] = False
        elif kind == "shape":
            c[0]["images"] = c[0]["images"][:1]
        elif kind == "dtype":
            c[0]["images"] = c[0]["images"].astype(np.float64)
        else:
            c[0]["extra"] = 1
        with pytest.raises(AssertionError):
            b.compare(a, c)


def test_real_threaded_selected_preparation_bound_on_synthetic_sources(monkeypatch):
    b = load("thread_actual", HERE / "benchmark.py")
    reference = load(
        "reference_context",
        HERE.parent / "predictive-selected-frame-preparation-20261021-v1/benchmark.py",
    )
    f = load("thread_fixture", REPO / "tests/test_predictive_descriptor_cache.py")
    batch = load(
        "descriptor_batch",
        HERE.parent / "predictive-descriptor-batch-20261021-v1/descriptor_batch.py",
    )
    selected = load(
        "selected_frames",
        HERE.parent / "predictive-selected-frame-preparation-20261021-v1/selected_frames.py",
    )
    monkeypatch.setitem(sys.modules, "context_reference", reference)
    monkeypatch.setitem(sys.modules, "descriptor_batch", batch)
    monkeypatch.setitem(sys.modules, "selected_frames", selected)
    import rsna.competitive.predictive.dicom as dicom

    groups = f.synthetic_groups(frames_per_group=9)
    monkeypatch.setattr(dicom, "load_study", lambda *_: (groups, {"synthetic": True}))
    original_threads = torch.get_num_threads()
    try:
        tasks = [(i, "/synthetic", "study") for i in range(3)]
        one, original = b.trial(1, tasks, None, lambda: 30)
        two, _ = b.trial(2, tasks, original, lambda: 30)
        assert one["bitwise_equal"] and two["all_other_values_exact"]
        assert two["max_image_absolute_difference"] <= 1e-6
    finally:
        torch.set_num_threads(original_threads)


def test_actual_generated_capsule_and_CPU_request():
    c = load("thread_compile", HERE / "compile.py")
    a = load("thread_account", HERE / "account.py")
    source, compiled = c.build(
        (
            REPO / "artifacts/predictive-selected-frame-preparation-20261021-v1/candidate.py"
        ).read_text()
    )
    request = a.request_for(source, compiled)
    assert request["isPrivate"] and not request["enableGpu"] and not request["enableInternet"]
    assert request["sessionTimeoutSeconds"] == 900
    assert request["datasetDataSources"] == request["kernelDataSources"] == []
    assert request["slug"].startswith("godlydonuts/rsna-predictive-threadcpu-")
    assert compiled["binding"]["image_max_abs_tolerance"] == 1e-6
    assert compiled["binding"]["all_other_context_values_exact_required"]
