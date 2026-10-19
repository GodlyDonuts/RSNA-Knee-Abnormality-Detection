"""Reindex the actual fixed engineering subset without changing any source signal."""

import hashlib
import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pytest
import torch

from rsna.competitive.licensed_importance import Recipe, sample


def module():
    p = Path("initiative/research/teacher-pair-earlier26-20261019-v1/inputs32.py.txt")
    loader = SourceFileLoader("actual_teacher_pair_inputs32", str(p))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


def inputs():
    uids = [f"synthetic-{i:04d}" for i in range(3887)]
    rng = np.random.default_rng(7)
    mask = rng.random((3887, 12)) > 0.25
    labels = rng.random((3887, 12))
    labels[~mask] = np.nan
    full = {
        "uids": np.array(uids),
        "labels": labels,
        "masks": mask,
        "parent_logits": rng.normal(size=(3887, 2, 12)).astype(np.float32),
    }
    indices = list(range(50, 82))[::-1]
    chosen = [uids[i] for i in indices]
    engineering = {
        "uids": chosen,
        "eligible_fitting_only": True,
        "masks": mask[indices].tolist(),
        "labels": np.nan_to_num(labels[indices].astype(np.float32)).tolist(),
    }
    q = np.full((3887, 94), 0.25 / 94, dtype=np.float64)
    q[np.arange(3887), np.arange(3887) % 94] += 0.75
    rows = [{"study_uid": uid, "unchanged": i} for i, uid in enumerate(reversed(chosen))]
    return rows, full, q, engineering, indices


def test_actual_small_packet_preserves_full_labels_masks_teacher_proposals_and_order(monkeypatch):
    m = module()
    rows, full, q, e, indices = inputs()
    monkeypatch.setattr(
        m,
        "POPULATION_SHA",
        hashlib.sha256(json.dumps(list(full["uids"]), separators=(",", ":")).encode()).hexdigest(),
    )
    r, t, p, actual = m.select(rows, full, q, e)
    assert (
        actual == indices
        and [x["study_uid"] for x in r] == e["uids"]
        and all(x["holder"] == "probe32" for x in r)
    )
    for name in full:
        assert t[name].tobytes() == full[name][indices].tobytes()
    assert p.tobytes() == q[indices].tobytes()
    # Case-index changes do not affect the UID-keyed sampler or source proposal.
    assert np.array_equal(p[0], q[indices[0]])
    recipe = Recipe(epochs=2, role="influence")
    a = sample(e["uids"][0], recipe, 0, torch.from_numpy(p[0]))
    b = sample(str(full["uids"][indices[0]]), recipe, 0, torch.from_numpy(q[indices[0]]))
    assert a[:2] == b[:2] and torch.equal(a[2], b[2])


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_uid",
        "changed_mask",
        "changed_observed_label",
        "wrong_proposal_dtype",
        "source_roster_changed",
    ],
)
def test_actual_reader_rejects_alignment_or_source_changes(monkeypatch, kind):
    m = module()
    rows, full, q, e, _ = inputs()
    monkeypatch.setattr(
        m,
        "POPULATION_SHA",
        hashlib.sha256(json.dumps(list(full["uids"]), separators=(",", ":")).encode()).hexdigest(),
    )
    if kind == "duplicate_uid":
        e["uids"][0] = e["uids"][1]
    elif kind == "changed_mask":
        e["masks"][0][0] = not e["masks"][0][0]
    elif kind == "changed_observed_label":
        k = e["masks"][0].index(True)
        e["labels"][0][k] = 1 - e["labels"][0][k]
    elif kind == "wrong_proposal_dtype":
        q = q.astype(np.float32)
    else:
        full["uids"][0] = "changed"
    with pytest.raises(AssertionError):
        m.select(rows, full, q, e)


def test_actual_probe_compiler_keeps_entire_three_phase_gradient_protocol_unchanged():
    p = Path("initiative/research/teacher-pair-earlier26-20261019-v1/probe_compiler.py.txt")
    loader = SourceFileLoader("actual_packet_probe_compiler", str(p))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    old = Path(
        "initiative/research/teacher-pair-full-encoder-20261019-v1/gpu_probe.py.txt"
    ).read_text()
    new = m.build(old)
    compile(new, "actual_new_node_gradient_probe", "exec")
    assert new[new.index("versions =") :] == old[old.index("versions =") :]
    assert "len(original) == 3887" not in new and "load32(A,targets" in new
    assert 'torch.device("cpu")' in new and "p.dtype == torch.float32" in new
    with pytest.raises(AssertionError):
        m.build(old + "\n# changed\n")
