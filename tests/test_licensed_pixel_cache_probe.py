from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.kaggle.licensed_pixel_cache_probe import CONSTANTS, FUNCTIONS, _build, build

ROOT = Path(__file__).resolve().parents[1]


def fake_parent():
    assignments = [f"{name}=0" for name in sorted(CONSTANTS)]
    functions = [f"def {name}(*args,**kwargs):return None" for name in sorted(FUNCTIONS)]
    return (
        "raise RuntimeError('unselected GPU boot must never execute')\n"
        + "\n".join(assignments + functions)
        + "\nclass RaptorClassifier:\n    pass\n"
        + "\ndef main():raise RuntimeError('unselected competition path')\n"
    )


def codec():
    return (ROOT / "rsna/competitive/licensed_pixel_cache.py").read_text()


def test_only_exact_preprocessing_and_lossless_codec_are_compiled_no_encoder_or_cuda_boot():
    source, binding = _build(fake_parent(), codec())
    namespace = {"__name__": "synthetic_CPU_pixel_cache"}
    exec(compile(source, "synthetic_CPU_pixel_cache", "exec"), namespace)
    assert "RaptorClassifier" not in namespace and "main" not in namespace
    assert "unselected GPU boot" not in source
    assert binding["CPU_only"] and binding["fitting_studies"] == 32
    assert binding["maximum_output_bytes"] == 512 * 2**20
    assert namespace["VOLUME_SHAPE"] == (96, 384, 384)
    volume = np.zeros((96, 384, 384), np.uint8)
    volume[0, 10:20, 20:30] = 255
    mask = (volume.reshape(96, -1).sum(1) > 0).astype(np.uint8)
    raw = namespace["encode"](volume, mask, remaining_bytes=16 * 2**20)
    h = namespace["hashlib"].sha256(raw).hexdigest()
    restored, present = namespace["decode"](raw, h)
    assert np.array_equal(restored, volume) and np.array_equal(present, mask)
    value = torch.zeros(94, 3, 320, 320)
    assert len(namespace["_pixel_tensor_sha"](value)) == 64


@pytest.mark.parametrize("change", ["parent", "codec", "missing", "duplicate"])
def test_changed_sources_or_incomplete_cpu_extraction_are_rejected(change):
    parent = fake_parent()
    if change == "parent":
        with pytest.raises(ValueError, match="licensed scored"):
            build(parent, codec())
    elif change == "codec":
        with pytest.raises(ValueError, match="codec"):
            _build(parent, codec() + "\n# changed\n")
    elif change == "missing":
        with pytest.raises(ValueError, match="Complete original"):
            _build(parent.replace("IMG=0", "OTHER=0"), codec())
    else:
        with pytest.raises(ValueError, match="Unique"):
            _build(parent + "\nIMG=2\n", codec())
