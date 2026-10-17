"""Spatial signal, control invariants and source identity before clinical fitting."""

import copy
import hashlib

import numpy as np
import pytest
import torch

from rsna.competitive.spatial_correspondence import randomized_correspondence_null
from rsna.competitive.spatial_data import CachedSpatialData
from rsna.competitive.spatial_reader import SpatialReader, build_plan


def graph_fixture():
    return {
        "slot": 0,
        "series_uid": "synthetic-series",
        "selected_source_indices": list(range(6)),
        "supports": [list(range(i, i + 3)) for i in range(4)],
        "native_cache_side": 48,
        "native_original_shape": [6, 48, 48],
        "native_resized_shape": [48, 48],
        "input_augmentation_or_mirror": False,
        "full_frame_letterbox_align_corners_false": True,
        "geometry_known": True,
        "pixel_spacing": [0.5, 0.5],
        "image_orientation_patient": [1, 0, 0, 0, 1, 0],
        "selected_image_positions_patient": [[0, 0, i * 2] for i in range(6)],
        "repeat_edges": [[0, 3]],
    }


def plan_fixture():
    return build_plan([graph_fixture()], np.array([[0, i] for i in range(4)]), (8, 6, 6), seed=11)


def cached_fixture(root, count=4):
    root.mkdir()
    records = []
    generator = np.random.default_rng(71)
    for i in range(count):
        path = root / f"case-{i}.npz"
        np.savez_compressed(
            path,
            spatial_maps=generator.normal(size=(4, 8, 6, 6)).astype(np.float32),
            base_logits=generator.normal(size=12).astype(np.float32),
            windows=np.array([[0, j] for j in range(4)]),
        )
        records.append(
            {
                "study_uid": f"case-{i}",
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_graphs": [graph_fixture()],
            }
        )
    labels = generator.integers(0, 2, (count, 12)).astype(np.float32)
    masks = generator.integers(0, 2, (count, 12)).astype(bool)
    masks[:, 0] = True
    return CachedSpatialData(root, records, labels, masks, {"synthetic_input": "source-A"}, seed=11)


def test_randomized_null_preserves_degrees_and_coverage_but_changes_geometric_edges():
    pi = torch.tensor([[0.3, 0.1, 0.0], [0.1, 0.1, 0.1], [0.0, 0.1, 0.1]])
    pi /= pi.sum()
    allowed = torch.ones_like(pi, dtype=torch.bool)
    control, receipt = randomized_correspondence_null(pi, allowed, seed=81)
    assert receipt["testable"] and receipt["same_both_weighted_marginals_and_observed_tokens"]
    assert not torch.equal(control > 0, pi > 0)
    torch.testing.assert_close(control.sum(0), pi.sum(0))
    torch.testing.assert_close(control.sum(1), pi.sum(1))
    assert torch.equal(control, randomized_correspondence_null(pi, allowed, seed=81)[0])


def test_observation_exclusions_apply_to_both_arms():
    graph = graph_fixture()
    graph["native_resized_shape"] = [24, 48]
    plan = build_plan([graph], np.array([[0, i] for i in range(4)]), (8, 6, 6), seed=11)
    assert all(len(x) == 18 for x in plan.valid_tokens)
    local = [x for x in plan.groups if x.kind == "local"]
    assert local
    for group in local:
        assert not group.aligned.diag().any() and not group.control.diag().any()
        assert group.aligned.shape == (18, 18)


def test_full_source_disjoint_and_local_relations_are_available():
    plan = plan_fixture()
    assert {g.kind for g in plan.groups} == {"local", "repeat"}
    group = next(g for g in plan.groups if g.kind == "repeat")
    assert group.first.tolist() == [0] and group.second.tolist() == [3]


def test_all_image_moments_and_nuisance_inputs_stay_identical_between_arms():
    plan = plan_fixture()
    maps = torch.randn(4, 8, 6, 6)
    model = SpatialReader(8, 8)
    first = model.statistics(maps, plan, arm="aligned")
    second = model.statistics(maps, plan, arm="null")
    assert torch.equal(first[:, :4], second[:, :4]) and torch.equal(first[:, 8:], second[:, 8:])
    assert not torch.equal(first[:, 4:8], second[:, 4:8])
    assert first[:, 4:8].abs().max() < 1.001


def test_zero_start_exact_base_unbounded_correction_and_actual_projection_gradient():
    plan = plan_fixture()
    maps = torch.randn(4, 8, 6, 6)
    base = torch.randn(12)
    model = SpatialReader(8, 8)
    for arm in ["aligned", "null"]:
        result, correction = model(maps, base, plan, arm=arm)
        assert torch.equal(result, base) and not correction.any()
    with torch.no_grad():
        model.output[:, 4] = 1
        model.bias.fill_(10)
    result, _ = model(maps, base, plan, arm="aligned")
    assert (result - base).min() > 8
    result.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.grad.abs().sum() for p in model.project.parameters()) > 0


def test_unknown_physical_geometry_keeps_single_acquisition_path():
    graph = graph_fixture()
    graph["geometry_known"] = False
    plan = build_plan([graph], np.array([[0, i] for i in range(4)]), (8, 6, 6), seed=11)
    assert not plan.groups
    values = SpatialReader(8, 8).statistics(torch.randn(4, 8, 6, 6), plan, arm="aligned")
    assert torch.equal(values[:, 4:8], torch.zeros(12, 4)) and torch.isfinite(values).all()


def test_known_slice_centers_with_missing_spacing_do_not_create_physical_links():
    graph = graph_fixture()
    graph["pixel_spacing"] = []
    plan = build_plan([graph], np.array([[0, i] for i in range(4)]), (8, 6, 6), seed=11)
    assert not plan.groups and all(len(x) == 36 for x in plan.valid_tokens)


@pytest.mark.parametrize("change", ["overlap", "selection", "orientation", "mirror", "spacing"])
def test_source_geometry_corruption_is_rejected(change):
    graph = copy.deepcopy(graph_fixture())
    if change == "overlap":
        graph["repeat_edges"] = [[0, 1]]
    if change == "selection":
        graph["selected_source_indices"][1] = 0
    if change == "orientation":
        graph["image_orientation_patient"] = [2, 0, 0, 0, 1, 0]
    if change == "mirror":
        graph["input_augmentation_or_mirror"] = True
    if change == "spacing":
        graph["pixel_spacing"] = [0, 0.5]
    with pytest.raises(ValueError):
        build_plan([graph], np.array([[0, i] for i in range(4)]), (8, 6, 6), seed=11)


def test_cached_member_first_read_hash_and_subsequent_stat_identity(tmp_path):
    data = cached_fixture(tmp_path / "cache")
    sample = data[0]
    assert sample["maps"].shape == (4, 8, 6, 6)
    assert data.verified == {"case-0.npz"}
    path = data.root / "case-0.npz"
    path.touch()
    with pytest.raises(ValueError, match="changed"):
        data[0]


def test_cached_member_with_replaced_original_hash_fails(tmp_path):
    data = cached_fixture(tmp_path / "cache")
    data.records[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes"):
        data[0]


def online_fixture(count=2):
    """Actual frozen convolution/BN and the production online provider boundary."""
    from rsna.competitive.spatial_data import OnlineSpatialData

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 1, stride=64)
            self.bn = torch.nn.BatchNorm2d(8)
            self.head = torch.nn.Module()
            self.head.global_pool = torch.nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            return self.head.global_pool(self.bn(self.conv(x))).flatten(1)

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.classifier = torch.nn.Linear(8, 12)

        def forward(self, x, *, chunk_size):
            return self.classifier(
                torch.cat([self.encoder(chunk) for chunk in x.split(chunk_size)])
            ).mean(0)

    class Data:
        augment = False
        side = 384
        slices = 16
        max_series = 6
        slice_sampling = "uniform"
        series_sampling = "selected"
        neighbor_channels = None
        cache_contrast = None
        input_channels = 1

        def __init__(self):
            self.series = {}
            self._metadata = {}
            for i in range(count):
                uid, sid = f"case-{i}", f"series-{i}"
                self.series[uid] = [
                    {
                        "SeriesInstanceUID": sid,
                        "Anatomical_Plane": "Sagittal",
                        "Fluid_Sensitive": 1,
                        "Fat_Suppression": 1,
                    }
                ]
                self._metadata[(uid, sid)] = (
                    "a" * 64,
                    {
                        "n_slices": 6,
                        "sop_uids": [f"sop-{j}" for j in range(6)],
                        "ordering": "patient_geometry",
                        "orientation": [1, 0, 0, 0, 1, 0],
                        "positions": [[0, 0, j * 2] for j in range(6)],
                        "side": 384,
                        "original_shape": [6, 384, 384],
                        "resized_shape": [384, 384],
                        "pixel_spacing": [0.5, 0.5],
                    },
                )

        def __len__(self):
            return count

        def __getitem__(self, index):
            images = torch.zeros(6, 16, 1, 384, 384)
            images[0, :6] = torch.rand(
                6, 1, 384, 384, generator=torch.Generator().manual_seed(90 + index)
            )
            sm = torch.tensor([True] + [False] * 5)
            km = torch.zeros(6, 16, dtype=torch.bool)
            km[0, :6] = True
            return {
                "study_uid": f"case-{index}",
                "images": images,
                "series_mask": sm,
                "slice_mask": km,
                "labels": torch.tensor([index % 2] * 12, dtype=torch.float32),
                "label_mask": torch.ones(12),
            }

    torch.manual_seed(313)
    parent = Parent().eval().requires_grad_(False)
    return OnlineSpatialData(Data(), parent, {"synthetic_input": "online-v1"}, seed=11), parent


def test_online_maps_are_ordinary_constants_that_support_real_head_backward():
    online, parent = online_fixture()
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    sample = online[0]
    model = SpatialReader(8, 8)
    logits, _ = model(sample["maps"], sample["base_logits"], sample["plan"], arm="aligned")
    logits.square().sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert all(p.grad is None for p in parent.parameters())
    assert all(torch.equal(v, parent.state_dict()[k]) for k, v in before.items())
    assert not any(m.training for m in parent.modules())


def test_realistic_12_by_12_grid_null_converges_with_irregular_degrees():
    yy, xx = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")
    xy = np.stack((xx.ravel(), yy.ravel()), axis=1)
    distance = ((xy[:, None] - xy[None]) ** 2).sum(-1)
    pi = torch.tensor(np.where(distance <= 2.5**2, np.exp(-distance / 2), 0), dtype=torch.float32)
    pi.fill_diagonal_(0)
    pi /= pi.sum()
    allowed = ~torch.eye(len(pi), dtype=torch.bool)
    for seed in range(8):
        result, proof = randomized_correspondence_null(pi, allowed, seed=seed)
        assert proof["testable"] and proof["converged"]
        assert (result - pi).abs().sum() > 1
        assert torch.allclose(result.sum(0), pi.sum(0), atol=1e-6, rtol=0)
        assert torch.allclose(result.sum(1), pi.sum(1), atol=1e-6, rtol=0)
