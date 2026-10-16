import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.kaggle.series_boundary_notebook import (
    CPU_INDEX,
    GPU_INDEX,
    GUARD,
    _instrument,
    instrument,
)

PARENT = (
    """
import numpy as np
import torch,threading,time
NORM = 'none'
SLOT_SIZES = [26,22,18,12,18]
INPUT_RES = 320
def _eval_centers(mask,D,k):
    active = np.flatnonzero(mask)
    return [int(active[0])]*k if len(active)==1 else list(range(1,D-1))
def gpu_windows(vol,mask,k,res,dev,norm=NORM,mirror=False):
    D=len(vol)
    cs=np.asarray(_eval_centers(mask,D,k))
    return torch.from_numpy(GPU_VALUE)
def eval_windows(vol,mask,k,res,norm=NORM,mirror=False):
    D=len(vol); rows=[]
    for c in _eval_centers(mask,D,k):
        tri = CPU_VALUE
        rows.append(tri)
    return torch.from_numpy(np.stack(rows))
def infer_probs(model,xwins,device):
    return np.zeros(12)
def infer_pair_probs(model,plain,mirrored,device):
    return np.zeros(12)
def main():
    pass
""".replace("GPU_VALUE", GPU_INDEX).replace("CPU_VALUE", CPU_INDEX)
    + GUARD
)


def namespace(policy):
    helper = Path("rsna/competitive/series_boundary.py").read_text()
    source = _instrument(PARENT, helper, policy)
    # Execute the actual generated transformation functions, stopping before any
    # GPU qualification, main, file/network access or post-run receipts.
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_boundary_devices" for t in node.targets
        ):
            break
        nodes.append(node)
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "test_patch", "exec"), scope)
    return source, scope


@pytest.mark.parametrize("policy,count", [("within-series", 86), ("edge-repeat", 94)])
def test_generated_cpu_and_gpu_paths_have_identical_surgical_values(policy, count):
    source, scope = namespace(policy)
    vol = np.arange(96 * 4, dtype=np.uint16).reshape(96, 2, 2)
    mask = np.ones(96)
    left = scope["eval_windows"](vol, mask, 94, 320)
    right = scope["gpu_windows"](vol, mask, 94, 320, "cpu")
    assert left.shape == right.shape == (count, 3, 2, 2)
    assert torch.equal(left, right)
    assert source.count("main()") == PARENT.count("main()")
    if policy == "edge-repeat":
        assert torch.equal(left[:, 1], torch.from_numpy(vol[1:95]))
        assert torch.equal(left[24, 2], torch.from_numpy(vol[25]))


def test_repeated_join_center_explicitly_falls_back_to_original_bag_for_drop_policy():
    _, scope = namespace("within-series")
    vol = np.arange(96 * 4, dtype=np.uint16).reshape(96, 2, 2)
    mask = np.zeros(96)
    mask[25] = 1
    left = scope["eval_windows"](vol, mask, 94, 320)
    assert len(left) == 94
    assert torch.equal(left[0], torch.from_numpy(vol[[24, 25, 26]]))
    keep, fallback = scope["_boundary_keep"](np.full(94, 25))
    assert fallback and keep.all()


@pytest.mark.parametrize("change", ["duplicate_gpu", "missing_cpu", "suffix", "policy"])
def test_changed_patch_contract_is_rejected(change):
    source, policy = PARENT, "edge-repeat"
    if change == "duplicate_gpu":
        source = source.replace(GPU_INDEX, GPU_INDEX + "+" + GPU_INDEX)
    elif change == "missing_cpu":
        source = source.replace(CPU_INDEX, "different_index")
    elif change == "suffix":
        source += "\nprint('unexpected')"
    else:
        policy = "guess"
    with pytest.raises(ValueError):
        _instrument(source, Path("rsna/competitive/series_boundary.py").read_text(), policy)


def test_public_builder_requires_exact_previously_scored_parent_bytes():
    with pytest.raises(ValueError, match="previously scored"):
        instrument(PARENT, Path("rsna/competitive/series_boundary.py").read_text(), "edge-repeat")
