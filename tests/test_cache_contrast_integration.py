import json

import numpy as np
import pytest
import torch
from test_competitive_data import fixture

from rsna.competitive.cache_contrast import normalize_selected_cache, validate_cache_contrast
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.train import TrainConfig
from scripts.research.robust_cache_contrast import normalize_selected_cache as gated_normalizer


def test_canonical_arithmetic_equals_original_complete_cohort_gate():
    volume = np.arange(3 * 16 * 16, dtype=np.uint16).reshape(3, 16, 16) * 50
    for mode in ("minmax", "p01_p99"):
        assert np.array_equal(
            normalize_selected_cache(volume, [16, 16], policy=mode),
            gated_normalizer(volume, [16, 16], policy=mode),
        )


@pytest.mark.parametrize(
    "policy", [{}, {"mode": []}, {"mode": True}, {"mode": "bad"}, {"mode": "minmax", "extra": 1}]
)
def test_invalid_policy_refused(policy):
    with pytest.raises(ValueError):
        validate_cache_contrast(policy)


def test_actual_loader_default_exact_minmax_and_candidate_nonimage_unchanged(tmp_path):
    studies, series, paths = fixture(tmp_path, counts=(3, 4))
    for path in paths:
        p = path.with_suffix(".json")
        meta = json.loads(p.read_text())
        meta["resized_shape"] = [16, 16]
        p.write_text(json.dumps(meta))
    kwargs = dict(slices=3, side=16)
    original = CompetitiveStudyDataset(studies, series, tmp_path, **kwargs)[0]
    control = CompetitiveStudyDataset(
        studies, series, tmp_path, cache_contrast={"mode": "minmax"}, **kwargs
    )[0]
    candidate = CompetitiveStudyDataset(
        studies, series, tmp_path, cache_contrast={"mode": "p01_p99"}, **kwargs
    )[0]
    for name in original:
        if isinstance(original[name], torch.Tensor):
            torch.testing.assert_close(
                original[name], control[name], atol=0, rtol=0, equal_nan=True
            )
            if name != "images":
                torch.testing.assert_close(
                    original[name], candidate[name], atol=0, rtol=0, equal_nan=True
                )
        else:
            assert original[name] == control[name] == candidate[name]
    assert not torch.equal(original["images"], candidate["images"])
    assert (
        torch.isfinite(candidate["images"]).all()
        and candidate["images"].min() >= 0
        and candidate["images"].max() <= 1
    )
    assert not candidate["images"][2].any()


@pytest.mark.parametrize(
    "change",
    [
        {"augment": True},
        {"slice_sampling": "all"},
        {"neighbor_channels": {"mode": "physical", "strength": 0.25}},
        {"patch_summary": {"weight": 0}},
        {"fit_scope": "all_development"},
    ],
)
def test_recipe_must_isolate_input_change(change):
    with pytest.raises(ValueError, match="Cache contrast"):
        TrainConfig(cache_contrast={"mode": "p01_p99"}, **change).validate()
