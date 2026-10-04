"""Engineering invariants of the unintegrated neighbor-channel prototype."""

from __future__ import annotations

import numpy as np
import pytest

from rsna.competitive.neighbor_channels import neighbor_delta_channels


@pytest.mark.parametrize("strength", [0.0, 0.01, 0.25, 0.5])
def test_bounds_and_center_recovery_on_random_and_adversarial_pixels(strength):
    rng = np.random.default_rng(18302)
    slices = [rng.random((48, 64), dtype=np.float32) for _ in range(3)]
    values = np.array(
        [0, -0.0, np.nextafter(np.float32(0), np.float32(1)), 0.5, 1], dtype=np.float32
    )
    for image in slices:
        image[0, :5] = values
    slices[1][0, :5] = 1
    slices[2][0, :5] = 0
    originals = [image.copy() for image in slices]
    output = neighbor_delta_channels(*slices, strength=strength)
    assert output.shape == (3, 48, 64) and output.dtype == np.float32
    assert np.isfinite(output).all() and (output >= 0).all() and (output <= 1).all()
    np.testing.assert_allclose(
        output.mean(axis=0, dtype=np.float64), slices[0], rtol=0, atol=np.finfo(np.float32).eps
    )
    for original, image in zip(originals, slices, strict=True):
        assert original.tobytes() == image.tobytes()


def test_zero_strength_is_exact_gray_bytes_and_no_output_alias():
    center = np.array([[0, -0.0, 0.37, 1]], dtype=np.float32)
    output = neighbor_delta_channels(center, 1 - center, center, strength=0)
    assert output.tobytes() == np.stack((center, center, center)).tobytes()
    output[0, 0, 0] = 0.4
    assert center[0, 0] == 0


def test_identical_neighbors_return_gray_and_missing_context_can_replicate_center():
    center = np.arange(20, dtype=np.float32).reshape(4, 5) / 20
    output = neighbor_delta_channels(center, center, center, strength=0.5)
    assert output.tobytes() == np.stack((center, center, center)).tobytes()


def test_neighbor_signal_stays_local_and_direction_is_explicit():
    center = np.full((7, 9), 0.5, dtype=np.float32)
    previous, following = center.copy(), center.copy()
    previous[3, 4] = 1
    following[3, 4] = 0
    output = neighbor_delta_channels(center, previous, following, strength=0.25)
    changed = np.any(output != center[None], axis=0)
    assert changed.sum() == 1 and changed[3, 4]
    assert output[0, 3, 4] > center[3, 4] > output[2, 3, 4]
    swapped = neighbor_delta_channels(center, following, previous, strength=0.25)
    np.testing.assert_array_equal(output[[2, 1, 0]], swapped)


def test_no_global_random_state_consumption():
    image = np.full((3, 4), 0.3, dtype=np.float32)
    state = np.random.get_state()
    neighbor_delta_channels(image, image / 2, image * 2, strength=0.25)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


@pytest.mark.parametrize("strength", [-0.01, 0.500001, float("nan"), float("inf"), True])
def test_invalid_strength_rejected(strength):
    image = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="strength"):
        neighbor_delta_channels(image, image, image, strength=strength)


@pytest.mark.parametrize("slot", [0, 1, 2])
@pytest.mark.parametrize("bad", ["nan", "range", "dtype", "shape", "empty"])
def test_every_untrusted_input_checked(slot, bad):
    images = [np.full((2, 3), 0.4, dtype=np.float32) for _ in range(3)]
    if bad == "nan":
        images[slot][0, 0] = np.nan
    elif bad == "range":
        images[slot][0, 0] = 1.01
    elif bad == "dtype":
        images[slot] = images[slot].astype(np.float64)
    elif bad == "shape":
        images[slot] = np.ones((1, 2), dtype=np.float32)
    else:
        images[slot] = np.zeros((0, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        neighbor_delta_channels(*images, strength=0.25)
