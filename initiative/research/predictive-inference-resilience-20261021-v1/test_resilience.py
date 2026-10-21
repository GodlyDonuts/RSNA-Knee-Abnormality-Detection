import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


D = load("resilient_decoder", HERE / "decoder.py")
F = load("original_DICOM_fixture", ROOT / "tests/test_predictive_dicom.py")
S = load("resilient_serving", HERE / "serving.py")


def same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        for k in ("frames", "aliases", "acquisition", "ordinal"):
            assert torch.equal(getattr(x, k), getattr(y, k))


def test_unchanged_usable_MRI_tensors_and_aliases(tmp_path):
    root = tmp_path / F.STUDY
    for i in range(4):
        F.write_dicom(
            root,
            F.SERIES_A,
            i + 1,
            (np.arange(80 * 80).reshape(80, 80) % 4096).astype(np.uint16),
            position=i,
        )
    original, _ = D.original_load_study(root, F.STUDY)
    actual, stats = D.load_study(root, F.STUDY)
    same(original, actual)
    assert stats["decoder_route"] == "original_exact"
    recovered, _ = D.recover(root, F.STUDY)
    same(original, recovered)


def test_irregular_metadata_and_sidecars_do_not_destroy_usable_pixels(tmp_path):
    root = tmp_path / F.STUDY
    pixels = (np.arange(80 * 80).reshape(80, 80) % 4096).astype(np.uint16)
    F.write_dicom(
        root,
        F.SERIES_A,
        1,
        pixels,
        InstanceNumber=None,
        ImagePositionPatient=None,
        StudyInstanceUID="2.25.5",
    )
    (root / F.SERIES_A / "notes.json").write_text("{}")
    F.write_dicom(root, F.SERIES_A, 2, np.zeros((64, 64), dtype=np.uint16))
    with pytest.raises(ValueError):
        D.original_load_study(root, F.STUDY)
    result, stats = D.load_study(root, F.STUDY)
    assert stats["dicoms"] == 1 and sum(stats["rejected_frames"].values()) == 1
    assert stats["metadata_repairs"] == {
        "path_identity_used": 1,
        "missing_instance_source_order": 1,
        "incomplete_geometry_instance_order": 1,
    }
    expected = D.fixed_metadata_scale(torch.from_numpy(pixels.astype(np.int32)), bits_stored=12)
    assert torch.equal(result[0].frames[0], expected)


def test_zero_usable_MRI_still_fails_explicitly(tmp_path):
    root = tmp_path / F.STUDY
    root.mkdir()
    with pytest.raises(ValueError, match="No usable"):
        D.load_study(root, F.STUDY)


def test_numerical_recovery_recomputes_same_model_without_constant_predictions():
    ns = {}
    exec(S.NUMERICAL_HELPER, ns)
    contexts = object()
    model = object()
    calls = []

    def predict(m, c, *, precision):
        assert m is model and c is contexts
        calls.append(precision)
        if precision == "fp16":
            raise FloatingPointError("overflow")
        return torch.arange(12, dtype=torch.float32) / 12

    result, mode = ns["predict_with_precision_recovery"](predict, model, contexts)
    assert calls == ["fp16", "fp32"] and mode == "fp32_recovery"
    assert torch.equal(result, torch.arange(12, dtype=torch.float32) / 12)

    def broken(*args, **kwargs):
        raise ValueError("invalid")

    with pytest.raises(ValueError):
        ns["predict_with_precision_recovery"](broken, model, contexts)


def test_prospective_worker_keeps_original_science_and_qualification_guards():
    prep = load(
        "base_prep",
        HERE.parent / "predictive-selected-frame-preparation-20261021-v1/serving_upgrade.py",
    )
    original = {n: (prep.PARENT / n).read_text() for n in ("worker.py.txt", "descriptor_cache.py")}
    files = {"serving/preparation_upgrade.py": (prep.HERE / "serving_upgrade.py").read_text()}
    modified = S.capsule(files)["serving/preparation_upgrade.py"]
    ns = {
        "__name__": "prospective_resilient_capsule",
        "__file__": str(prep.HERE / "serving_upgrade.py"),
    }
    exec(modified, ns)
    upgraded = ns["upgrade"](original)
    assert "torch.cuda.device_count() == 2" in upgraded["worker.py.txt"]
    assert "contract.validate_export(export, chosen, base)" in upgraded["worker.py.txt"]
    assert (
        "if prepared is not None:\n            assert memory_ok and precision_ok"
        in upgraded["worker.py.txt"]
    )
    assert original["worker.py.txt"] == (prep.PARENT / "worker.py.txt").read_text()
    with pytest.raises(ValueError):
        ns["require_qualification"]({})
