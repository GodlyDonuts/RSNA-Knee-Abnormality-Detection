"""New corrected-training handoff, dropout/recovery and stale-table boundaries."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_licensed_importance import Model, Recipe, cases, equal

from rsna.competitive import paired_logit_train as candidate
from rsna.competitive.licensed_importance import sample
from rsna.competitive.licensed_importance_train import fit as reference
from rsna.competitive.licensed_parent_train import PARENT_SHA
from rsna.competitive.paired_logit_cases import PairedCases
from rsna.competitive.paired_logit_control import PairedLogitReadout


class Corrected(PairedLogitReadout, Model):
    pass


def table(tmp_path, original, recipe, *, nonzero=False, mutate=None):
    draws = [
        [sample(uid, recipe, epoch, original.proposal(i)) for epoch in range(3)]
        for i, uid in enumerate(original.uids)
    ]
    arrays = {
        "study_keys": np.array(
            [list(hashlib.sha256(uid.encode()).digest()) for uid in original.uids], dtype=np.uint8
        ),
        "positions": np.array([[d[0] for d in row] for row in draws], dtype=np.uint8),
        "mirror": np.array([[d[1] for d in row] for row in draws], dtype=np.bool_),
        "log_proposal": np.array([[d[2].numpy() for d in row] for row in draws], dtype=np.float32),
        "shift": np.full((len(original), 3, 12), 0.25 if nonzero else 0, dtype=np.float32),
    }
    if mutate:
        mutate(arrays)
    path = tmp_path / "correction.npz"
    np.savez_compressed(path, **arrays)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha


def dataset(tmp_path, nonzero=False):
    original = cases(3)
    recipe = Recipe(role="influence", effective_batch=2)
    path, sha = table(tmp_path, original, recipe, nonzero=nonzero)
    return PairedCases(original, path, sha, recipe), recipe, original


def run(model, data, recipe, path, **kwargs):
    return candidate.fit(
        model,
        data,
        path,
        {"parent_checkpoint_sha256": PARENT_SHA},
        recipe,
        device=torch.device("cpu"),
        **kwargs,
    )


def test_zero_shift_has_exact_original_dropout_optimizer_trajectory(tmp_path):
    torch.set_num_threads(1)
    data, recipe, original = dataset(tmp_path)
    torch.manual_seed(31)
    a = Model()
    b = Corrected()
    b.load_state_dict(a.state_dict())
    torch.manual_seed(72)
    reference(
        a,
        original,
        tmp_path / "old",
        {"parent_checkpoint_sha256": PARENT_SHA},
        recipe,
        device=torch.device("cpu"),
    )
    torch.manual_seed(72)
    run(b, data, recipe, tmp_path / "new")
    old = torch.load(tmp_path / "old/checkpoint.pt", weights_only=True)
    new = torch.load(tmp_path / "new/checkpoint.pt", weights_only=True)
    for key in ("model", "optimizer", "rng_by_rank", "scaler_by_rank", "history", "schedule"):
        equal(old[key], new[key])
    assert new["binding"]["parent_correction_sha256"] == data.correction_sha256


def test_nonzero_shift_dropout_full_stop_resume_complete_state_equal(tmp_path):
    data, recipe, _ = dataset(tmp_path, nonzero=True)
    torch.manual_seed(28)
    initial = Corrected().state_dict()
    for name, stop in (("full", None), ("split", 1)):
        model = Corrected()
        model.load_state_dict(initial)
        torch.manual_seed(44)
        run(model, data, recipe, tmp_path / name, stop_after=stop)
    run(Corrected(), data, recipe, tmp_path / "split", resume=True)
    equal(
        torch.load(tmp_path / "full/checkpoint.pt", weights_only=True),
        torch.load(tmp_path / "split/checkpoint.pt", weights_only=True),
    )


@pytest.mark.parametrize("kind", ["uid", "draw", "view", "logq", "nan", "dtype"])
def test_wrong_table_refused_before_image_reads(tmp_path, kind):
    original = cases(3)
    recipe = Recipe(role="influence", effective_batch=2)

    def mutate(a):
        if kind == "uid":
            a["study_keys"][0, 0] ^= 1
        elif kind == "draw":
            a["positions"][0, 0, 0] = 94
        elif kind == "view":
            a["mirror"][0, 0] = not a["mirror"][0, 0]
        elif kind == "logq":
            a["log_proposal"][0, 0, 0] += 0.1
        elif kind == "nan":
            a["shift"][0, 0, 0] = np.nan
        else:
            a["shift"] = a["shift"].astype(np.float64)

    path, sha = table(tmp_path, original, recipe, mutate=mutate)
    with pytest.raises(ValueError):
        PairedCases(original, path, sha, recipe)


def test_recipe_and_mutated_table_cannot_resume(tmp_path):
    data, recipe, original = dataset(tmp_path, nonzero=True)
    run(Corrected(), data, recipe, tmp_path / "split", stop_after=1)
    with pytest.raises(ValueError):
        run(Corrected(), data, replace(recipe, encoder_lr=3e-5), tmp_path / "other")
    path, sha = table(tmp_path, original, recipe)
    new = PairedCases(original, path, sha, recipe)
    with pytest.raises(ValueError, match="Exact source"):
        run(Corrected(), new, recipe, tmp_path / "split", resume=True)
