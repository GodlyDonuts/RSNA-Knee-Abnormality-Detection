import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location("scale_collect", Path(__file__).with_name("collect.py"))
collect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect)


def write_case(tmp_path, mutation=None):
    target = np.arange(16 * 64, dtype=np.float32).reshape(16, 64) / 1000
    observed = np.arange(16) % 2 == 0
    # If collector touches an image/metadata/support payload, allow_pickle=False
    # will reject this deliberately non-loadable sentinel.
    arrays = {k: np.array([object()], dtype=object) for k in collect.CONTEXT_KEYS}
    arrays["manifest"] = np.frombuffer(json.dumps({
        "version": collect.CONTEXT_VERSION, "source": {"original": "bound"},
    }).encode(), dtype=np.uint8)
    for i in (0, 1):
        arrays[f"c{i}_targets"] = target.copy()
        arrays[f"c{i}_observed"] = observed.copy() if i == 0 else ~observed
        arrays[f"c{i}_region_count"] = np.asarray(16, dtype=np.int64)
    if mutation:
        mutation(arrays)
    path = tmp_path / "context.npz"
    np.savez_compressed(path, **arrays)
    row = {"path": str(path), "provenance": {"original": "bound"},
           "proof": {"stat": collect.signature(path), "version": collect.CONTEXT_VERSION}}
    return row, torch.from_numpy(target)


def test_only_unique_descriptor_pair_and_masks_are_read(tmp_path):
    row, expected = write_case(tmp_path)
    assert torch.equal(collect.read_descriptors(row), expected)


@pytest.mark.parametrize("mutation", [
    lambda a: a["c1_targets"].__setitem__((0, 0), 42),
    lambda a: a["c0_targets"].__setitem__((0, 0), np.nan),
    lambda a: a.__setitem__("c1_observed", a["c0_observed"].copy()),
    lambda a: a.__setitem__("c0_region_count", np.asarray(32)),
    lambda a: a.__setitem__("clinical_labels", np.zeros(12)),
    lambda a: a.__setitem__("c0_targets", a["c0_targets"].astype(np.float64)),
])
def test_incomplete_or_modified_input_is_rejected(tmp_path, mutation):
    row, _ = write_case(tmp_path, mutation)
    with pytest.raises(ValueError):
        collect.read_descriptors(row)


def test_private_provenance_and_owning_stat_must_match(tmp_path):
    row, _ = write_case(tmp_path)
    row["provenance"] = {"original": "different"}
    with pytest.raises(ValueError, match="binding"):
        collect.read_descriptors(row)
    row["proof"]["stat"][1] += 1
    with pytest.raises(ValueError, match="immutable"):
        collect.read_descriptors(row)
