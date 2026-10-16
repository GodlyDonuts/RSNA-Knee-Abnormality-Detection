import copy

import numpy as np
import pytest
import torch

from scripts.newton.probe_regional_context_recovery import model_hash, same


def state():
    return {
        "model": {"w": torch.arange(6, dtype=torch.float32)},
        "optimizer": {"moment": torch.ones(6), "step": 4},
        "rng": {"numpy": np.arange(8, dtype=np.uint32), "torch": torch.arange(8)},
        "history": [{"update": 1}, {"update": 2}],
        "counter": 4,
    }


def test_complete_deepcopy_is_exact_and_initial_model_hash_covers_names_dtype_values():
    s = state()
    same(s, copy.deepcopy(s))
    assert model_hash(s["model"]) == model_hash(copy.deepcopy(s["model"]))
    assert model_hash(s["model"]) != model_hash({"other": s["model"]["w"]})
    assert model_hash(s["model"]) != model_hash({"w": s["model"]["w"].double()})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s["model"]["w"].add_(1),
        lambda s: s["optimizer"].update(step=3),
        lambda s: s["optimizer"]["moment"].zero_(),
        lambda s: s["rng"]["torch"].add_(1),
        lambda s: s["rng"].update(numpy=s["rng"]["numpy"].astype(np.uint64)),
        lambda s: s["history"].pop(),
        lambda s: s.update(counter=3),
        lambda s: s.pop("optimizer"),
    ],
)
def test_partial_recovery_or_one_changed_state_member_is_rejected(mutation):
    original = state()
    changed = copy.deepcopy(original)
    mutation(changed)
    with pytest.raises(ValueError):
        same(original, changed)
