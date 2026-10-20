import ast
import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("contrast_worker_test", HERE / "worker_builder.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_worker_preserves_real_engine_and_declares_only_matched_contrast_recipe():
    original = (HERE.parent / "predictive-stable-warmup-20261020-v1/worker.py.txt").read_text()
    output = worker.build(original)
    ast.parse(output)
    assert 'metadata_width=10' in output
    assert 'row = index[index_number]["variants"][role]' in output
    assert 'density_weight=0.1,' in output and 'epochs=100,' in output
    assert 'schedule_updates=2916,' in output and 'head_only_updates=3,' in output
    assert 'from rsna.competitive.predictive.train import Recipe, fit, objective' in output
    assert 'stop_after=1 if phase == "stop1" else None' in output
    assert 'epoch_snapshots=False' in output
    assert 'initial_state_is_not_the_eight_coordinate_primary_graph' in output
    assert 'generative_two_mask_loss' not in output and 'SHARED_INITIAL_SHA' not in output
    with pytest.raises(AssertionError):
        worker.build(original + "\n")


def test_ten_coordinate_roles_have_same_initial_state_and_unchanged_RNG():
    import copy
    import hashlib
    import json

    from rsna.competitive.predictive.model import Predictive

    ns = dict(hashlib=hashlib, json=json)
    exec(worker.DIGEST_FUNCTION, ns)
    backbone = torch.nn.Linear(4, 8)
    original_rng = torch.get_rng_state().clone()
    signatures = []
    for _role in worker.ROLES:
        with torch.random.fork_rng():
            torch.manual_seed(20261020)
            model = Predictive(copy.deepcopy(backbone), 8, hidden=8, metadata_width=10, classes=2)
            signatures.append(ns["model_digests"](model))
    assert signatures[0] == signatures[1]
    assert torch.equal(original_rng, torch.get_rng_state())


def test_new_context_graph_uses_complete_unchanged_training_recovery(tmp_path):
    import copy
    import random

    import numpy as np

    from rsna.competitive.predictive import cases, visible_contrast
    from rsna.competitive.predictive.model import Predictive
    from rsna.competitive.predictive.train import Recipe, fit, objective, restore_rng, rng_state

    ramp = torch.linspace(0, 1, 96 * 96).reshape(96, 96)
    group = cases.RawGroup(torch.stack([.002 + .02 * ramp, .5 + .03 * ramp]),
                           torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 0]))
    contexts, _ = visible_contrast.assemble([group], side=16, context_cap=2, classes=2)

    class Data:
        def __len__(self):
            return 2

        def case(self, index, epoch):
            return {"contexts": contexts, "labels": torch.tensor([[float(index), float(1 - index)]]),
                    "mask": torch.ones(1, 2, dtype=torch.bool)}

    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    base = Predictive(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 8)),
                      8, hidden=8, metadata_width=10, classes=2)
    start = rng_state()
    recipe = Recipe("C", epochs=2, effective_batch=1, head_only_updates=1, seed=91)
    binding = {"candidate": worker.CANDIDATE_SHA, "width": 10, "synthetic": True}
    full_model = copy.deepcopy(base)
    full = fit(full_model, Data(), recipe, binding=binding, output=tmp_path / "full", epoch_snapshots=False)
    restore_rng(start)
    fit(copy.deepcopy(base), Data(), recipe, binding=binding, output=tmp_path / "stop",
        stop_after=1, epoch_snapshots=False)
    resumed_model = copy.deepcopy(base)
    resumed = fit(resumed_model, Data(), recipe, binding=binding, output=tmp_path / "resume",
                  resume=tmp_path / "stop/stopped.pt", epoch_snapshots=False)

    def equal(a, b):
        assert type(a) is type(b)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    equal(full, resumed)
    assert full["step"] == 4 and full["history"][-1]["encoder_updated"]
    with torch.inference_mode():
        a = objective(full_model, Data().case(0, 0), recipe, torch.device("cpu"))[1]["probabilities"]
        b = objective(resumed_model, Data().case(0, 0), recipe, torch.device("cpu"))[1]["probabilities"]
    assert torch.equal(a, b)
