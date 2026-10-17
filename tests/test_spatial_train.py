"""Actual numerical engine replay, not a mirrored mock optimizer test."""

from dataclasses import replace

import pytest
import torch
from test_competitive_training import _assert_tree_equal
from test_spatial_reader import cached_fixture, online_fixture

from rsna.competitive.spatial_train import SpatialRecipe, SpatialTrainer, learning_rate


@pytest.mark.parametrize("arm", ["aligned", "null"])
def test_actual_engine_full_stop_resume_complete_state_and_csv(tmp_path, arm):
    torch.set_num_threads(1)
    data = cached_fixture(tmp_path / "inputs")
    recipe = SpatialRecipe(epochs=2, effective_batch=2, channels=8, width=8)
    binding = {"diagnostic_only": True, "source": "fixture-v1"}
    full = SpatialTrainer(data, recipe, arm=arm, device=torch.device("cpu"), binding=binding)
    full.fit(tmp_path / "full")
    cut = SpatialTrainer(data, recipe, arm=arm, device=torch.device("cpu"), binding=binding)
    cut.fit(tmp_path / "replay", stop_after=1)
    resumed = SpatialTrainer(data, recipe, arm=arm, device=torch.device("cpu"), binding=binding)
    resumed.fit(tmp_path / "replay", resume=True)
    a = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "replay/checkpoint.pt", weights_only=True)
    _assert_tree_equal(a, b)
    assert full.predictions(data).to_csv(index=False) == resumed.predictions(data).to_csv(
        index=False
    )
    assert full.completed == 4 and full.history[0]["projection_gradient_norm"] == 0
    assert full.history[-1]["projection_gradient_norm"] > 0
    assert [x["studies"] for x in full.history] == [2] * 4


def test_final_partial_batch_and_duplicate_original_epoch_are_safe(tmp_path):
    data = cached_fixture(tmp_path / "inputs", count=3)
    recipe = SpatialRecipe(epochs=2, effective_batch=2, channels=8, width=8)
    engine = SpatialTrainer(
        data, recipe, arm="aligned", device=torch.device("cpu"), binding={"diagnostic_only": True}
    )
    engine.fit(tmp_path / "run")
    assert [r["studies"] for r in engine.history] == [2, 1, 2, 1]
    assert len(list((tmp_path / "run/epoch-checkpoints").glob("*.pt"))) == 2
    with pytest.raises(ValueError, match="existing original|Existing original"):
        engine.fit(tmp_path / "run")


def test_wrong_arm_source_or_schedule_cannot_resume(tmp_path):
    data = cached_fixture(tmp_path / "inputs")
    recipe = SpatialRecipe(epochs=2, effective_batch=2, channels=8, width=8)
    engine = SpatialTrainer(
        data, recipe, arm="aligned", device=torch.device("cpu"), binding={"diagnostic_only": True}
    )
    engine.fit(tmp_path / "run", stop_after=1)
    wrong = SpatialTrainer(
        data, recipe, arm="null", device=torch.device("cpu"), binding={"diagnostic_only": True}
    )
    with pytest.raises(ValueError, match="changed"):
        wrong.fit(tmp_path / "run", resume=True)


def test_small_diagnostic_cannot_be_declared_production(tmp_path):
    data = cached_fixture(tmp_path / "inputs")
    with pytest.raises(ValueError, match="Production"):
        SpatialTrainer(
            data,
            SpatialRecipe(channels=8, width=8),
            arm="aligned",
            device=torch.device("cpu"),
            binding={"diagnostic_only": False},
        )


def test_schedule_is_finite_and_fixed_with_explicit_teacher_fraction():
    recipe = SpatialRecipe()
    rates = [learning_rate(recipe, i, 281) for i in range(843)]
    assert rates[280] == recipe.learning_rate and rates[-1] == 0 and min(rates) >= 0
    assert recipe.teacher_fraction == 0.25
    with pytest.raises(ValueError):
        replace(recipe, teacher_fraction=1)
    with pytest.raises(ValueError):
        learning_rate(recipe, 843, 281)


def test_actual_online_and_cached_provider_have_equal_complete_fitting_states(tmp_path):
    import hashlib

    import numpy as np

    from rsna.competitive.spatial_data import CachedSpatialData, source_graphs

    torch.set_num_threads(1)
    online, parent = online_fixture()
    root = tmp_path / "same-maps"
    root.mkdir()
    records = []
    labels = []
    masks = []
    for i in range(len(online)):
        sample = online[i]
        windows = [(0, j) for j in range(4)]
        path = root / f"{i}.npz"
        np.savez_compressed(
            path,
            spatial_maps=sample["maps"].numpy(),
            base_logits=sample["base_logits"].numpy(),
            windows=np.asarray(windows),
        )
        records.append(
            {
                "study_uid": sample["study_uid"],
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_graphs": source_graphs(online.dataset, sample["study_uid"], windows),
            }
        )
        labels.append(sample["labels"].numpy())
        masks.append(sample["label_mask"].numpy())
    cached = CachedSpatialData(root, records, labels, masks, online.binding, seed=11)
    recipe = SpatialRecipe(epochs=2, effective_batch=1, channels=8, width=8)
    binding = {"diagnostic_only": True, "source": "actual-provider-test"}
    a = SpatialTrainer(cached, recipe, arm="aligned", device=torch.device("cpu"), binding=binding)
    a.fit(tmp_path / "cached")
    before = {k: v.clone() for k, v in parent.state_dict().items()}
    b = SpatialTrainer(online, recipe, arm="aligned", device=torch.device("cpu"), binding=binding)
    b.fit(tmp_path / "online")
    _assert_tree_equal(
        torch.load(tmp_path / "cached/checkpoint.pt", weights_only=True),
        torch.load(tmp_path / "online/checkpoint.pt", weights_only=True),
    )
    assert a.predictions(cached).to_csv(index=False) == b.predictions(online).to_csv(index=False)
    assert all(torch.equal(v, parent.state_dict()[k]) for k, v in before.items())


def test_teacher_fraction_and_missing_cells_cannot_be_misread_as_hard_negatives():
    from rsna.competitive.released_context import anchored_loss

    base = torch.linspace(-2, 2, 12)[None]
    logits = base.clone().requires_grad_(True)
    labels = torch.ones_like(base)
    labels[0, 1:] = float("nan")
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[0, 0] = True
    loss = anchored_loss(logits, base, labels, mask, source_weight=0.75)
    loss.backward()
    assert torch.equal(logits.grad[0, 1:], torch.zeros(11))
    torch.testing.assert_close(logits.grad[0, 0], 0.75 * (base[0, 0].sigmoid() - 1) / 12)
