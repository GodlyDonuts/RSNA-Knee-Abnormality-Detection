import numpy as np
import pytest

from rsna.competitive.criterion_geometry import within_series_relations


def metadata(n=12):
    return {
        "n_slices": n,
        "sop_uids": [str(i) for i in range(n)],
        "ordering": "patient_geometry",
        "orientation": [1, 0, 0, 0, 1, 0],
        "positions": [[0, 0, 2 * i] for i in range(n)],
    }


def test_repeat_edges_require_three_disjoint_source_observations():
    graph = within_series_relations(metadata(), np.arange(12))
    assert graph.supports.shape == (10, 3)
    assert len(graph.continuity_edges) == 9
    assert len(graph.repeat_edges) == 7
    assert np.all(graph.continuity_gaps_mm == 2)
    assert np.all(graph.repeat_gaps_mm == 6)
    for first, second in graph.repeat_edges:
        assert not set(graph.supports[first]) & set(graph.supports[second])


def test_one_source_slice_cannot_supply_both_sides_of_a_repeat_edge():
    graph = within_series_relations(metadata(), np.arange(12))
    for abnormal_source in range(12):
        evidence = (graph.supports == abnormal_source).any(axis=1)
        assert not any(evidence[i] and evidence[j] for i, j in graph.repeat_edges)
    # The naive overlapping-neighbor rule WOULD double-count source slice 2.
    evidence = (graph.supports == 2).any(axis=1)
    assert any(evidence[i] and evidence[j] for i, j in graph.continuity_edges)


def test_selected_order_is_not_misrepresented_as_acquired_adjacency():
    graph = within_series_relations(metadata(30), np.arange(0, 30, 2))
    assert np.all(graph.continuity_gaps_mm == 4)
    assert np.all(graph.repeat_gaps_mm == 12)
    np.testing.assert_array_equal(graph.supports[0], [0, 2, 4])


def test_large_native_acquisition_gap_breaks_edges():
    value = metadata()
    for row in value["positions"][6:]:
        row[2] += 100
    graph = within_series_relations(value, np.arange(12))
    for edges in (graph.continuity_edges, graph.repeat_edges):
        assert all(not (graph.supports[i, 1] < 6 <= graph.supports[j, 1]) for i, j in edges)


def test_unknown_geometry_keeps_identity_but_invents_no_physical_relation():
    value = metadata()
    value["ordering"] = "instance_number"
    del value["positions"]
    graph = within_series_relations(value, np.arange(12))
    assert graph.geometry_known is False
    assert len(graph.supports) == 10
    assert len(graph.repeat_edges) == len(graph.continuity_edges) == 0
    assert not graph.centers_mm.any()


@pytest.mark.parametrize("indices", [[0, 1, 1], [2, 1, 0], [-1, 0, 1], [0, 1, 12], []])
def test_duplicated_or_malformed_source_indices_rejected(indices):
    with pytest.raises(ValueError):
        within_series_relations(metadata(), np.array(indices, dtype=np.int64))


def test_repeated_sop_identity_and_nonorthogonal_geometry_rejected():
    value = metadata()
    value["sop_uids"][1] = value["sop_uids"][0]
    with pytest.raises(ValueError):
        within_series_relations(value, np.arange(12))
    value = metadata()
    value["orientation"] = [1, 0, 0, 1, 0, 0]
    with pytest.raises(ValueError):
        within_series_relations(value, np.arange(12))


@pytest.mark.parametrize("count", [1, 2, 3])
def test_short_series_have_no_fabricated_repeat_pairs(count):
    graph = within_series_relations(metadata(count), np.arange(count))
    assert len(graph.supports) == max(0, count - 2)
    assert len(graph.repeat_edges) == 0
