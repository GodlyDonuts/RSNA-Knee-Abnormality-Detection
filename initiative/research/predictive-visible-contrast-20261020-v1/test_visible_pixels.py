import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location("visible_pixels", Path(__file__).with_name("visible_pixels.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_interpolated_fill_boundaries_cannot_inflate_visible_contrast():
    hidden = torch.zeros((1, 3, 4, 4), dtype=torch.bool)
    hidden[..., :2] = True
    source = torch.full((1, 3, 4, 4), .02).masked_fill(hidden, .5)
    rendered = F.interpolate(source, size=(3, 3), mode="bilinear", align_corners=False)
    assert rendered[..., 1].mean() > .02  # Interpolated gray edge is contaminated.
    values, count, total = module.pure_visible_values(rendered, hidden)
    assert np.allclose(values, .02) and count == 9 and total == 27
