import importlib.util
import struct
from pathlib import Path

import pytest

p = Path(__file__).with_name("audit.py")
spec = importlib.util.spec_from_file_location("coverage_audit", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_exact_even_cap_uniform_spacing_and_full_small_roster():
    for count in range(1, 10001):
        for cap in (24, 96):
            values = m.centers(count, cap)
            reference = (
                list(range(count))
                if count <= cap
                else [round(i * (count - 1) / (cap - 1)) for i in range(cap)]
            )
            assert values == reference
            assert len(set(values)) == min(count, cap)


def test_neighbors_do_not_cross_acquisition_or_matrix_group():
    groups = [([1, 2, 3, 4], [0, 0, 1, 1], [0, 1, 0, 1]), ([5, 6], [1, 1], [2, 3])]
    r = m.coverage(groups, 24)
    assert r["unique_frames"] == r["triplet_unique_frames"] == 6
    assert r["acquisitions"] == 2 and r["acquisitions_without_center"] == 0


def test_aliases_do_not_inflate_frames_and_short_acquisition_can_be_missed():
    group = (
        list(range(101)) + [0],
        [0] * 50 + [1] + [2] * 50 + [0],
        list(range(50)) + [0] + list(range(50)) + [0],
    )
    sparse = m.coverage([group], 24)
    dense = m.coverage([group], 96)
    assert sparse["unique_frames"] == 101
    assert sparse["acquisitions_without_center"] == 1
    assert dense["triplet_unique_frames"] >= sparse["triplet_unique_frames"]


def test_npy_only_admits_integer_metadata():
    header = repr(dict(descr="<i8", fortran_order=False, shape=(3,))).encode() + b"\n"
    raw = (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header
        + struct.pack("<qqq", 1, 2, 3)
    )
    assert m.integers(raw) == (1, 2, 3)
    with pytest.raises(AssertionError):
        m.integers(raw[:-1])
