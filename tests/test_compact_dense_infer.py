import copy

import numpy as np
import pytest
import torch
from test_compact_source_neighbors import geometry

from rsna.competitive.compact_dense_infer import dense_plan, render_series
from rsna.competitive.compact_source_neighbors import channel_plan, source_window_tensor


@pytest.mark.parametrize("count", [1, 2, 3, 11, 16, 20, 40, 97, 300])
def test_dense_coverage_retains_every_original_window_and_has_strict_cap(count):
    meta = geometry(count)
    original, _ = channel_plan(meta)
    dense, indices, fallback = dense_plan(meta)
    assert len(dense) <= 30 and fallback == 0
    np.testing.assert_array_equal(dense[indices], original)
    assert len(set(dense[:, 1].tolist())) == len(dense)
    if count > 32:
        assert len(dense) == 30
    volume = np.random.default_rng(2).integers(0, 65536, (count, 4, 4), dtype=np.uint16)
    old, _ = source_window_tensor(volume, meta, 4)
    new, idx, _ = render_series(volume, meta, 4)
    assert torch.equal(new[idx], old)


def test_original_cap_is_exact_control_and_selection_deterministic():
    meta = geometry(100)
    a, _, _ = dense_plan(meta)
    b, _, _ = dense_plan(copy.deepcopy(meta))
    assert np.array_equal(a, b)
    original, _ = channel_plan(meta)
    control, _, _ = dense_plan(meta, maximum=14)
    np.testing.assert_array_equal(control, original)
    assert a[:, 1].min() >= 1 and a[:, 1].max() <= 98


@pytest.mark.parametrize("bad", [None, True, 0, 13, 31, 16.0])
def test_invalid_or_unbounded_budgets_cannot_drop_originals(bad):
    with pytest.raises(ValueError):
        dense_plan(geometry(40), bad)


def test_missing_geometry_retains_original_fallbacks_without_inventing_dense_neighbors():
    meta = geometry(50)
    meta["ordering"] = "instance_number"
    old, old_fallback = channel_plan(meta)
    new, indices, fallback = dense_plan(meta)
    np.testing.assert_array_equal(new, old)
    assert fallback == old_fallback.sum() == 14
    np.testing.assert_array_equal(indices, np.arange(14))


def test_geometry_gap_and_shift_never_admit_extra_invalid_edges():
    meta = geometry(80)
    for i in range(40, 80):
        meta["positions"][i][2] += 30
    meta["positions"][20][0] += 0.1
    old, _ = channel_plan(meta)
    old_centers = set(old[:, 1])
    dense, indices, _ = dense_plan(meta)
    np.testing.assert_array_equal(dense[indices], old)
    extra = set(dense[:, 1]) - old_centers
    assert not extra & {19, 20, 21, 39, 40}
    for triple in dense:
        if triple[1] not in old_centers:
            np.testing.assert_array_equal(np.diff(triple), [1, 1])


@pytest.mark.parametrize("bad", ["nan", "missing_positions", "orientation"])
def test_claimed_but_malformed_geometry_is_rejected(bad):
    meta = geometry(40)
    if bad == "nan":
        meta["positions"][3][0] = float("nan")
    elif bad == "missing_positions":
        meta.pop("positions")
    else:
        meta["orientation"] = [1, 0, 0, 1, 0, 0]
    with pytest.raises(ValueError):
        dense_plan(meta)


def test_wrong_pixel_shape_or_dtype_cannot_enter_renderer():
    meta = geometry(40)
    for vol in [np.zeros((40, 4, 4), dtype=np.float32), np.zeros((39, 4, 4), dtype=np.uint16)]:
        with pytest.raises(ValueError):
            render_series(vol, meta, 4)


def native_fixture(tmp_path):
    import json

    from test_competitive_data import fixture

    from rsna.competitive.data import CompetitiveStudyDataset

    studies, series, paths = fixture(tmp_path, counts=(40,))
    from rsna.schema import TARGETS

    for target in TARGETS:
        if target not in studies:
            studies[target] = 0.0
            studies[target + "__mask"] = 0
    meta = json.loads(paths[0].with_suffix(".json").read_text())
    meta["pixel_spacing"] = [0.5, 0.5]
    paths[0].with_suffix(".json").write_text(json.dumps(meta))
    base = CompetitiveStudyDataset(
        studies,
        series,
        tmp_path,
        slices=16,
        side=384,
        max_series=6,
        augment=False,
        slice_sampling="uniform",
        series_sampling="selected",
    )
    return base, paths


def test_actual_dataset_view_preserves_original_subset_and_does_not_read_labels(tmp_path):
    from rsna.competitive.compact_bag import candidate_windows
    from rsna.competitive.compact_dense_infer import DenseSourceNeighborDataset
    from rsna.competitive.compact_source_neighbors import (
        MatchedSourceNeighborDataset,
        select_window_tensor,
    )
    from rsna.schema import TARGETS

    base, _ = native_fixture(tmp_path)
    original = MatchedSourceNeighborDataset(base)[0]
    old = select_window_tensor(
        original, candidate_windows(original["series_mask"], original["slice_mask"])
    )
    for target in TARGETS:
        base.studies[target] = "must-not-be-read"
        base.studies[target + "__mask"] = "must-not-be-read"
    sample = DenseSourceNeighborDataset(base)[0]
    assert sample["dense_windows"].shape == (30, 3, 384, 384)
    assert torch.equal(sample["dense_windows"][sample["original_window_indices"]], old)
    assert "labels" not in sample and "label_mask" not in sample


@pytest.mark.parametrize("mutation", ["metadata", "payload"])
def test_integrity_checked_again_on_each_actual_dense_read(tmp_path, mutation):
    from rsna.competitive.compact_dense_infer import DenseSourceNeighborDataset

    base, paths = native_fixture(tmp_path)
    data = DenseSourceNeighborDataset(base)
    path = paths[0] if mutation == "payload" else paths[0].with_suffix(".json")
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        data[0]


def test_dense_prediction_is_eval_only_and_does_not_hide_model_errors():
    from rsna.competitive.compact_dense_infer import (
        POLICY,
        DenseSourceNeighborDataset,
        prediction_rows,
    )

    class Data(DenseSourceNeighborDataset):
        def __init__(self):
            pass

        def __iter__(self):
            yield {
                "study_uid": "synthetic",
                "dense_windows": torch.ones(30, 3, 4, 4),
                "inference_policy": dict(POLICY),
            }

    class Model(torch.nn.Module):
        def forward(self, x, chunk_size):
            assert chunk_size <= 12
            return torch.arange(12, dtype=torch.float32, device=x.device)

    model = Model()
    with pytest.raises(ValueError, match="evaluation mode"):
        prediction_rows(model, Data(), ["synthetic"], device=torch.device("cpu"))
    result = prediction_rows(model.eval(), Data(), ["synthetic"], device=torch.device("cpu"))
    assert result["encoded_windows"] == 30 and not result["target_runtime_qualified"]
    with pytest.raises(ValueError, match="Incomplete"):
        prediction_rows(model, Data(), ["synthetic", "missing"], device=torch.device("cpu"))
    with pytest.raises(ValueError, match="bounded"):
        prediction_rows(model, Data(), ["synthetic"], device=torch.device("cpu"), chunk_size=13)

    class Broken(Model):
        def forward(self, *args, **kwargs):
            raise RuntimeError("actual model failure")

    with pytest.raises(RuntimeError, match="actual model failure"):
        prediction_rows(Broken().eval(), Data(), ["synthetic"], device=torch.device("cpu"))
