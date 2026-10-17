import copy

import numpy as np
import pytest
import torch

from rsna.competitive.criterion_geometry import within_series_relations
from rsna.competitive.criterion_residual import CriterionResidual


def fixture():
    metadata = {
        "n_slices": 12,
        "sop_uids": [str(i) for i in range(12)],
        "ordering": "patient_geometry",
        "orientation": [1, 0, 0, 0, 1, 0],
        "positions": [[0, 0, 2 * i] for i in range(12)],
    }
    graph = within_series_relations(metadata, np.arange(12))
    x = torch.randn((10, 16), generator=torch.Generator().manual_seed(17))
    return x, graph


def test_zero_initial_residual_preserves_base_in_both_arms():
    x, graph = fixture()
    head = CriterionResidual(16, 8)
    sham = copy.deepcopy(head)
    base = torch.linspace(-6, 6, 12)
    assert torch.equal(base + head([x], [graph]), base)
    assert torch.equal(base + sham([x], [graph], [np.roll(np.arange(10), 4)]), base)
    assert all(torch.equal(v, sham.state_dict()[k]) for k, v in head.state_dict().items())


def test_sham_changes_relations_but_preserves_image_moments_and_nuisance_exactly():
    x, graph = fixture()
    head = CriterionResidual(16, 8)
    real = head.statistics([x], [graph])
    sham = head.statistics([x], [graph], [np.array([4, 1, 8, 2, 9, 0, 6, 3, 7, 5])])
    assert real.shape == (12, 11)
    assert torch.equal(real[:, :4], sham[:, :4])
    assert torch.equal(real[:, 7:], sham[:, 7:])
    assert not torch.equal(real[:, 4:7], sham[:, 4:7])


def test_real_optimizer_updates_reach_local_evidence_after_zero_start():
    x, graph = fixture()
    head = CriterionResidual(16, 8)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)
    before = head.evidence.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head([x], [graph]), torch.ones(12)
        )
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(before, head.evidence.weight)
    assert torch.isfinite(head([x], [graph])).all()


def test_fake_repeat_edge_sharing_source_slice_is_rejected():
    x, graph = fixture()
    object.__setattr__(graph, "repeat_edges", np.array([[0, 1]]))
    object.__setattr__(graph, "repeat_gaps_mm", np.array([2.0]))
    with pytest.raises(ValueError, match="disjoint"):
        CriterionResidual(16, 8)([x], [graph])


def test_unknown_geometry_has_only_image_and_nuisance_evidence():
    x, _ = fixture()
    graph = within_series_relations(
        {"n_slices": 12, "sop_uids": [str(i) for i in range(12)], "ordering": "unknown"},
        np.arange(12),
    )
    values = CriterionResidual(16, 8).statistics([x], [graph])
    assert torch.equal(values[:, 4:7], torch.zeros(12, 3))
    assert torch.equal(values[:, -1], torch.zeros(12))


@pytest.mark.parametrize("order", [np.arange(9), np.zeros(10, int), np.arange(10, dtype=float)])
def test_incomplete_or_repeated_sham_assignment_rejected(order):
    x, graph = fixture()
    with pytest.raises(ValueError, match="permutation"):
        CriterionResidual(16, 8)([x], [graph], [order])
