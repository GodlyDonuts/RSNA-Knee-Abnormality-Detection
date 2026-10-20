import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("contrast_collect", Path(__file__).with_name("collect.py"))
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def test_gray_fill_does_not_create_false_visible_contrast():
    images = np.full((1, 3, 320, 320), .5, dtype=np.float32)
    images[:, :, :160] = .02
    result = collect.image_statistics(images)
    assert result["exact_half_fill_fraction"] == .5
    assert result["nonfill_q99_minus_q01"] == 0
    assert result["nonfill_median"] == pytest.approx(.02)
    assert result["nonfill_std"] < 1e-15


def test_visible_dynamic_range_and_all_fill_failure():
    images = np.broadcast_to(np.linspace(0, 1, 320, dtype=np.float32), (1, 3, 320, 320)).copy()
    result = collect.image_statistics(images)
    assert result["exact_half_fill_fraction"] == 0
    assert result["nonfill_q99_minus_q01"] == pytest.approx(.981191, abs=1e-5)
    with pytest.raises(AssertionError, match="All-fill"):
        collect.image_statistics(np.full_like(images, .5))
