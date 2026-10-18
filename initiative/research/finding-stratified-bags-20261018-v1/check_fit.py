"""Real unchanged optimizer/state machine on synthetic MRI-shaped observations."""

import importlib.util
import random

import numpy as np
import pytest
import torch
from fit import fit

from rsna.competitive.licensed_importance import ProposalCases, Recipe
from rsna.competitive.licensed_importance_train import fit as original_fit
from rsna.competitive.licensed_parent_train import PARENT_SHA

spec = importlib.util.spec_from_file_location(
    "existing_importance_fixture", "tests/test_licensed_importance.py"
)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
torch.set_num_threads(1)


def setup():
    base = f.OriginalCases(9)  # Exercises the final one-study effective batch.
    p = np.random.default_rng(31).uniform(0.1, 1, (9, 94, 12))
    p /= p.sum(1, keepdims=True)
    q = (p.sum(2) + 4 / 94) / 16
    return base, p, ProposalCases(base, torch.from_numpy(q))


def train(data, p, path, role, *, resume=False, stop_after=None):
    random.seed(322)
    np.random.seed(322)
    torch.manual_seed(322)
    model = f.Model()
    kwargs = dict(device=torch.device("cpu"), resume=resume, stop_after=stop_after)
    binding = {"parent_checkpoint_sha256": PARENT_SHA, "diagnostic_only": True}
    recipe = Recipe(epochs=2, role="influence")
    if role == "original":
        original_fit(model, data, path, binding, recipe, **kwargs)
    else:
        fit(model, data, path, binding, recipe, components=p, allocation=role, **kwargs)
    return torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)


def prediction(state, base):
    model = f.Model()
    model.load_state_dict(state["model"])
    model.eval()
    with torch.inference_mode():
        return (
            torch.stack(
                [
                    model(base.case(i, list(range(94)), mirror)["images"].unsqueeze(0)).sigmoid()
                    for i in range(9)
                    for mirror in (False, True)
                ]
            )
            .numpy()
            .tobytes()
        )


def test_original_control_complete_numerical_state_and_predictions(tmp_path):
    base, p, data = setup()
    old = train(data, p, tmp_path / "original", "original")
    new = train(data, p, tmp_path / "iid", "iid")
    f.equal(
        {k: v for k, v in old.items() if k != "binding"},
        {k: v for k, v in new.items() if k != "binding"},
    )
    assert prediction(old, base) == prediction(new, base)
    left = dict(new["binding"])
    del left["finding_allocation"]
    f.equal(left, old["binding"])


@pytest.mark.parametrize("role", ["iid", "finding_strata"])
def test_complete_resume_epoch_snapshots_and_csv_probabilities(tmp_path, role):
    base, p, data = setup()
    full = train(data, p, tmp_path / "full", role)
    stopped = train(data, p, tmp_path / "resume", role, stop_after=1)
    assert stopped["completed"] == 1
    resumed = train(data, p, tmp_path / "resume", role, resume=True)
    f.equal(full, resumed)
    assert prediction(full, base) == prediction(resumed, base)
    for epoch in (1, 2):
        f.equal(
            *[
                torch.load(
                    tmp_path / run / "epoch-checkpoints" / f"epoch-{epoch:03d}.pt",
                    weights_only=True,
                )
                for run in ("full", "resume")
            ]
        )


def test_changed_allocation_and_component_semantics_cannot_resume(tmp_path):
    _, p, data = setup()
    train(data, p, tmp_path / "run", "iid", stop_after=1)
    with pytest.raises(ValueError, match="Exact source"):
        train(data, p, tmp_path / "run", "finding_strata", resume=True)
    with pytest.raises(ValueError, match="Exact source"):
        train(data, p[:, :, ::-1].copy(), tmp_path / "run", "iid", resume=True)


def test_allocation_changes_learning_without_changing_global_original_engine(tmp_path):
    base, p, data = setup()
    left = train(data, p, tmp_path / "iid", "iid")
    right = train(data, p, tmp_path / "strata", "finding_strata")
    assert any(
        not torch.equal(value, right["model"][name]) for name, value in left["model"].items()
    )
    assert prediction(left, base) != prediction(right, base)
    # The original imported engine remains unchanged after both closure roles.
    original = train(data, p, tmp_path / "original", "original")
    f.equal(
        {k: v for k, v in original.items() if k != "binding"},
        {k: v for k, v in left.items() if k != "binding"},
    )
