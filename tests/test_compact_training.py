import random
import shutil
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from rsna.competitive.compact_bag import CompactBagClassifier
from rsna.competitive.compact_train import Settings, fit, make_optimizer, masked_cell_loss, predict


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.projection(x)).mean((2, 3))


class FittingFixture:
    def __init__(self):
        generator = torch.Generator().manual_seed(19)
        self.samples = []
        for i in range(3):
            mask = torch.ones(12)
            mask[i : i + 2] = 0
            self.samples.append(
                {
                    "images": torch.rand(2, 16, 1, 4, 4, generator=generator),
                    "series_mask": torch.ones(2, dtype=torch.bool),
                    "slice_mask": torch.ones(2, 16, dtype=torch.bool),
                    "study_uid": f"synthetic-{i}",
                    "labels": torch.rand(12, generator=generator),
                    "label_mask": mask,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def fresh():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    return CompactBagClassifier(TinyEncoder(), dimension=4)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


def test_active_cell_loss_is_global_not_per_target_or_per_bag():
    logits = torch.linspace(-2, 2, 12)
    labels = torch.arange(12).float() / 12
    mask = torch.tensor([1.0] * 3 + [0.0] * 9)
    denominator = 19
    expected = (
        nn.functional.binary_cross_entropy_with_logits(logits[:3], labels[:3], reduction="sum")
        / denominator
    )
    assert torch.equal(masked_cell_loss(logits, labels, mask, denominator), expected)
    labels[3:] = float("nan")
    assert torch.equal(masked_cell_loss(logits, labels, mask, denominator), expected)
    with pytest.raises(ValueError):
        masked_cell_loss(logits, labels, mask, 0)
    mask[4] = 0.5
    with pytest.raises(ValueError):
        masked_cell_loss(logits, labels, mask, denominator)


@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_full_stop_resume_matches_complete_state_history_rng_and_csv(tmp_path, boundary):
    data = FittingFixture()
    settings = Settings(seed=7, epochs=2, effective_batch=2)
    binding = {"diagnostic_only": True, "fixture": "never-clinical"}
    device = torch.device("cpu")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    m = fresh()
    result = fit(m, data, full, binding, settings, device=device)
    assert result["status"] == "fixed_final_fit_complete"
    expected_csv = predict(m, data, device, chunk_size=3).to_csv(index=False)
    other = fresh()
    assert (
        fit(other, data, resumed, binding, settings, device=device, stop_after_updates=boundary)[
            "status"
        ]
        == "stopped_optimizer_boundary"
    )
    other = fresh()
    assert fit(other, data, resumed, binding, settings, device=device, resume=True) == result
    left = torch.load(full / "checkpoint.pt", weights_only=True)
    right = torch.load(resumed / "checkpoint.pt", weights_only=True)
    equal(left, right)
    assert left["optimizer_steps"] == 4
    assert [row["studies"] for row in left["history"]] == [2, 1, 2, 1]
    assert left["scheduler"]["last_epoch"] == 4
    assert expected_csv == predict(other, data, device, chunk_size=3).to_csv(index=False)
    for epoch in (1, 2):
        equal(
            torch.load(full / f"epoch-checkpoints/epoch-{epoch:03d}.pt", weights_only=True),
            torch.load(resumed / f"epoch-checkpoints/epoch-{epoch:03d}.pt", weights_only=True),
        )


def test_replayed_immutable_epoch_must_match_and_is_not_overwritten(tmp_path):
    data, settings = FittingFixture(), Settings(seed=7, epochs=2, effective_batch=2)
    binding = {"diagnostic_only": True}
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    fit(fresh(), data, full, binding, settings, device=torch.device("cpu"))
    fit(
        fresh(),
        data,
        interrupted,
        binding,
        settings,
        device=torch.device("cpu"),
        stop_after_updates=1,
    )
    snapshots = interrupted / "epoch-checkpoints"
    snapshots.mkdir()
    source = full / "epoch-checkpoints/epoch-001.pt"
    target = snapshots / source.name
    shutil.copyfile(source, target)
    original = target.read_bytes()
    fit(fresh(), data, interrupted, binding, settings, device=torch.device("cpu"), resume=True)
    assert target.read_bytes() == original


@pytest.mark.parametrize("tamper", ["binding", "cursor", "order", "steps", "scheduler"])
def test_resume_refuses_mismatched_source_or_incomplete_counters(tmp_path, tamper):
    data, settings = FittingFixture(), Settings(seed=7, epochs=2, effective_batch=2)
    binding = {"diagnostic_only": True}
    fit(
        fresh(), data, tmp_path, binding, settings, device=torch.device("cpu"), stop_after_updates=1
    )
    path = tmp_path / "checkpoint.pt"
    state = torch.load(path, weights_only=True)
    if tamper == "binding":
        state["binding"]["fixture"] = "changed"
    elif tamper == "cursor":
        state["cursor"] = 1
    elif tamper == "order":
        state["order"] = list(reversed(state["order"]))
    elif tamper == "steps":
        state["optimizer_steps"] = 99
    else:
        state["scheduler"]["last_epoch"] = 0
    torch.save(state, path)
    with pytest.raises(ValueError, match="Resume"):
        fit(fresh(), data, tmp_path, binding, settings, device=torch.device("cpu"), resume=True)


def test_empty_and_existing_runs_refused(tmp_path):
    settings = Settings(seed=7, epochs=2, effective_batch=2)
    with pytest.raises(ValueError, match="Empty"):
        fit(fresh(), [], tmp_path, {}, settings, device=torch.device("cpu"))
    data = FittingFixture()
    fit(fresh(), data, tmp_path, {}, settings, device=torch.device("cpu"), stop_after_updates=1)
    with pytest.raises(ValueError, match="explicit resume"):
        fit(fresh(), data, tmp_path, {}, settings, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="settings"):
        replace(settings, clip_norm=float("nan")).validate()


def test_signal_stops_at_full_batch_without_final_predictions(tmp_path):
    settings = Settings(seed=7, epochs=2, effective_batch=2)
    result = fit(
        fresh(),
        FittingFixture(),
        tmp_path,
        {},
        settings,
        device=torch.device("cpu"),
        should_stop=lambda: True,
    )
    assert result["status"] == "stopped_optimizer_boundary"
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert state["cursor"] == 2 and state["optimizer_boundary"]
    assert not (tmp_path / "outer_predictions.csv").exists()


def test_full_schedule_and_optimizer_groups_are_exact():
    settings = Settings()
    assert (2247 + 7) // 8 == 281
    assert 281 * settings.epochs == 4496
    assert 2247 % settings.effective_batch == 7
    model = fresh()
    optimizer, scheduler = make_optimizer(model, settings, 4496)
    assert scheduler.total_steps == 4496
    encoder, head = optimizer.param_groups
    assert encoder["max_lr"] == 3e-5 and head["max_lr"] == 1e-3
    assert encoder["initial_lr"] == 3e-5 / 25
    assert head["initial_lr"] == 1e-3 / 25
    assert encoder["min_lr"] == encoder["initial_lr"] / 10000
    assert encoder["betas"] == (0.9, 0.999) and head["weight_decay"] == 0.02
    assert {id(p) for p in encoder["params"]} == {id(p) for p in model.encoder.parameters()}
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters()
    }
