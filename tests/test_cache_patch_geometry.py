"""Coordinate convention, half-pixel mapping and artificial padding must be correct."""

import numpy as np
import pytest

from scripts.research.cache_patch_geometry import patch_geometry


def meta():
    return {
        "side": 32,
        "original_shape": [2, 32, 32],
        "resized_shape": [32, 32],
        "orientation": [1, 0, 0, 0, 1, 0],
        "positions": [[10, 20, 30], [10, 20, 35]],
        "pixel_spacing": [2, 3],
        "ordering": "patient_geometry",
    }


def test_dicom_row_column_and_exact_patch_center_mapping():
    out = patch_geometry(meta(), [0, 1])
    assert out["valid_content_patch"].all()
    # DICOM first IOP triplet advances column, second advances row.
    np.testing.assert_allclose(out["patient_centers_mm"][0, 0, 0], [10 + 7.5 * 3, 20 + 7.5 * 2, 30])
    np.testing.assert_allclose(
        out["patient_centers_mm"][1, 1, 1], [10 + 23.5 * 3, 20 + 23.5 * 2, 35]
    )
    assert out["cross_series_registration_verified"] is False


def test_letterbox_zero_pixels_do_not_become_content_patches():
    m = meta()
    m.update(side=64, original_shape=[2, 16, 64], resized_shape=[16, 64])
    out = patch_geometry(m, [0])
    assert out["content_fraction"].sum() == 4
    assert not out["valid_content_patch"][0].any() and not out["valid_content_patch"][-1].any()
    assert out["valid_content_patch"].sum() == 4


def test_unknown_patient_geometry_preserves_only_content_not_fake_coordinates():
    m = meta()
    m["ordering"] = "instance_number"
    m["positions"] = []
    out = patch_geometry(m, [0])
    assert out["patient_centers_mm"] is None and out["valid_content_patch"].all()


@pytest.mark.parametrize(
    "key,value",
    [
        ("pixel_spacing", [0, 1]),
        ("orientation", [2, 0, 0, 0, 1, 0]),
        ("positions", [[0, 0, float("nan")], [0, 0, 1]]),
    ],
)
def test_invalid_known_geometry_rejected(key, value):
    m = meta()
    m[key] = value
    with pytest.raises(ValueError):
        patch_geometry(m, [0])


def test_non_source_slice_index_rejected():
    with pytest.raises(ValueError):
        patch_geometry(meta(), [2])
