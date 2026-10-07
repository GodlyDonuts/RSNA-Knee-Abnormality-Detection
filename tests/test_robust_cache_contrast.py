import numpy as np
import pytest
import torch

from scripts.research.robust_cache_contrast import normalize_selected_cache


def test_control_exact_original_conversion_and_input_unchanged():
    source = np.arange(2 * 8 * 8, dtype=np.uint16).reshape(2, 8, 8) * 400
    original = source.copy()
    actual = normalize_selected_cache(source, [8, 8], policy="minmax")
    expected = torch.from_numpy(source).float().div(65535).numpy()
    assert np.array_equal(actual, expected)
    assert np.array_equal(source, original)


def test_padding_excluded_from_percentiles_and_preserved():
    source = np.zeros((2, 10, 10), np.uint16)
    content = np.arange(8, dtype=np.uint16).reshape(2, 2, 2) * 1000 + 10000
    source[:, 4:6, 4:6] = content
    actual = normalize_selected_cache(source, [2, 2], policy="p01_p99")
    low, high = np.percentile(content, [1, 99])
    expected = np.clip((content.astype(np.float32) - low) / (high - low), 0, 1)
    assert np.allclose(actual[:, 4:6, 4:6], expected)
    outside = actual.copy()
    outside[:, 4:6, 4:6] = 0
    assert not outside.any()


def test_shared_stack_range_preserves_inter_slice_difference():
    source = np.stack([np.arange(16).reshape(4, 4) + 100, np.arange(16).reshape(4, 4) + 200])
    actual = normalize_selected_cache(source.astype(np.uint16), [4, 4], policy="p01_p99")
    assert actual[0].max() < actual[1].min()
    assert actual.dtype == np.float32
    assert actual.min() == 0 and actual.max() == 1


def test_no_robust_range_is_explicit_not_invented():
    source = np.full((2, 4, 4), 100, np.uint16)
    with pytest.raises(ValueError, match="no robust"):
        normalize_selected_cache(source, [4, 4], policy="p01_p99")
    assert np.array_equal(
        normalize_selected_cache(source, [4, 4], policy="minmax"), source.astype(np.float32) / 65535
    )


@pytest.mark.parametrize(
    "source,shape,policy",
    [
        (np.ones((2, 4, 4), np.float32), [4, 4], "minmax"),
        (np.ones((0, 4, 4), np.uint16), [4, 4], "minmax"),
        (np.ones((2, 4, 5), np.uint16), [4, 4], "minmax"),
        (np.ones((2, 4, 4), np.uint16), [5, 4], "minmax"),
        (np.ones((2, 4, 4), np.uint16), [4, 4], "unknown"),
        (np.ones((2, 4, 4), np.uint16), [2, 2], "p01_p99"),
    ],
)
def test_invalid_content_and_policy_refused(source, shape, policy):
    with pytest.raises(ValueError):
        normalize_selected_cache(source, shape, policy=policy)
