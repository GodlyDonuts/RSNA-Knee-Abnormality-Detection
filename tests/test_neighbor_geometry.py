"""Original ordering, physical gaps and grid shifts matter independently."""

import copy

import numpy as np
import pytest

from rsna.competitive.neighbor_geometry import (
    distant_neighbor_indices,
    physical_neighbor_indices,
)


def geometry(count=5):
    return dict(
        n_slices=count,
        ordering="patient_geometry",
        orientation=[1, 0, 0, 0, 1, 0],
        positions=[[0, 0, 3 * i] for i in range(count)],
        pixel_spacing=[0.5, 0.5],
    )


def test_regular_physical_neighbors_preserve_original_order_and_boundaries():
    p, n = physical_neighbor_indices(geometry())
    np.testing.assert_array_equal(p, [0, 0, 1, 2, 3])
    np.testing.assert_array_equal(n, [1, 2, 3, 4, 4])


def test_oblique_orientation_uses_projected_positions():
    m = geometry()
    angle = np.pi / 4
    row = np.array([np.cos(angle), 0, np.sin(angle)])
    column = np.array([0, 1, 0])
    normal = np.cross(row, column)
    m["orientation"] = [*row, *column]
    m["positions"] = [(normal * 3 * i).tolist() for i in range(5)]
    p, n = physical_neighbor_indices(m)
    np.testing.assert_array_equal(p, [0, 0, 1, 2, 3])
    np.testing.assert_array_equal(n, [1, 2, 3, 4, 4])


@pytest.mark.parametrize("ordering", ["instance_number", None, "unverified"])
def test_missing_geometry_never_invents_adjacency(ordering):
    p, n = physical_neighbor_indices({"n_slices": 4, "ordering": ordering})
    np.testing.assert_array_equal(p, np.arange(4))
    np.testing.assert_array_equal(n, p)


def test_large_gap_rejects_only_that_edge():
    m = geometry()
    m["positions"] = [[0, 0, z] for z in [0, 3, 6, 30, 33]]
    p, n = physical_neighbor_indices(m)
    assert n[2] == 2 and p[3] == 3 and n[3] == 4 and p[2] == 1


def test_in_plane_displacement_rejects_only_that_edge():
    m = geometry()
    m["positions"][2] = [0.01, 0, 6]
    p, n = physical_neighbor_indices(m)
    assert n[1] == 1 and p[2] == 2 and n[2] == 2 and p[3] == 3 and n[3] == 4


@pytest.mark.parametrize("z", [[0, 3, 3, 9, 12], [12, 9, 6, 3, 0], [0, 3, 2, 9, 12]])
def test_nonmonotone_geometry_cannot_admit_original_adjacency(z):
    m = geometry()
    m["positions"] = [[0, 0, x] for x in z]
    p, n = physical_neighbor_indices(m)
    np.testing.assert_array_equal(p, np.arange(5))
    np.testing.assert_array_equal(n, p)


@pytest.mark.parametrize(
    "key,value",
    [
        ("n_slices", True),
        ("n_slices", 0),
        ("positions", []),
        ("positions", [[0, 0, float("nan")]] * 5),
        ("orientation", [1, 0, 0, 1, 0, 0]),
        ("pixel_spacing", [0, 0.5]),
        ("pixel_spacing", []),
    ],
)
def test_malformed_claimed_geometry_rejected(key, value):
    m = geometry()
    m[key] = value
    with pytest.raises(ValueError):
        physical_neighbor_indices(m)


def test_single_slice_and_input_metadata_immutable():
    m = geometry(1)
    before = copy.deepcopy(m)
    p, n = physical_neighbor_indices(m)
    assert p.tolist() == n.tolist() == [0] and m == before


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 16, 51])
def test_distant_control_is_deterministic_distinct_and_nonlocal(count):
    p, n = distant_neighbor_indices(count)
    q, r = distant_neighbor_indices(count)
    np.testing.assert_array_equal(p, q)
    np.testing.assert_array_equal(n, r)
    for i in range(count):
        if p[i] == i:
            assert n[i] == i
        else:
            assert p[i] != n[i] and min(abs(int(p[i]) - i), abs(int(n[i]) - i)) >= max(
                2, (count + 3) // 4
            )
