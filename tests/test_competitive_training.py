import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset

from rsna.competitive.checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from rsna.competitive.loss import effective_batch_loss, observed_counts
from rsna.competitive.sampling import EffectiveBatchSampler, SeededDataset
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS


def test_accumulation_matches_one_effective_batch_with_uneven_masks():
    torch.manual_seed(5)
    logits = torch.randn(7, 12, requires_grad=True)
    labels = torch.randint(0, 2, logits.shape).float()
    masks = torch.randint(0, 2, logits.shape).float()
    masks[:2] = 0
    masks[:, 11] = 0
    counts = observed_counts([masks[:2], masks[2:6], masks[6:]])
    whole = effective_batch_loss(logits, labels, masks, counts)
    accumulated = sum(
        effective_batch_loss(logits[part], labels[part], masks[part], counts)
        for part in (slice(0, 2), slice(2, 6), slice(6, 7))
    )
    expected = torch.autograd.grad(whole, logits, retain_graph=True)[0]
    actual = torch.autograd.grad(accumulated, logits)[0]
    torch.testing.assert_close(accumulated, whole)
    torch.testing.assert_close(actual, expected)
    assert not actual[masks == 0].any()


def test_hidden_labels_including_nan_never_enter_the_loss():
    logits = torch.zeros(2, 12, requires_grad=True)
    mask = torch.zeros_like(logits)
    mask[0, 0] = 1
    labels = torch.full_like(logits, float("nan"))
    labels[0, 0] = 1
    loss = effective_batch_loss(logits, labels, mask, observed_counts([mask]))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0
    assert not logits.grad[mask == 0].any()


def test_all_masked_loss_is_zero_and_differentiable():
    logits = torch.randn(2, 12, requires_grad=True)
    mask = torch.zeros_like(logits)
    loss = effective_batch_loss(logits, torch.zeros_like(logits), mask, observed_counts([mask]))
    loss.backward()
    assert loss == 0 and logits.grad is not None and not logits.grad.any()


def _ddp_rank(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(8)
    inputs, labels = torch.randn(8, 3), torch.randint(2, (8, 12)).float()
    masks = torch.zeros_like(labels)
    masks[2, :3] = 1
    masks[4:6, :8] = 1
    masks[6, :11] = 1
    reference = nn.Linear(3, 12)
    effective_batch_loss(reference(inputs), labels, masks, observed_counts([masks])).backward()
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = nn.Linear(3, 12)
        model.load_state_dict(reference.state_dict())
        wrapped = DistributedDataParallel(model)
        pieces = [slice(rank * 2, rank * 2 + 2), slice(4 + rank * 2, 6 + rank * 2)]
        counts = observed_counts([masks[piece] for piece in pieces])
        with wrapped.no_sync():
            effective_batch_loss(
                wrapped(inputs[pieces[0]]), labels[pieces[0]], masks[pieces[0]], counts
            ).backward()
        effective_batch_loss(
            wrapped(inputs[pieces[1]]), labels[pieces[1]], masks[pieces[1]], counts
        ).backward()
        torch.testing.assert_close(model.weight.grad, reference.weight.grad)
        torch.testing.assert_close(model.bias.grad, reference.bias.grad)
    finally:
        dist.destroy_process_group()


def test_cpu_two_rank_accumulation_matches_global_gradient(tmp_path: Path):
    mp.spawn(_ddp_rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


@pytest.mark.parametrize("size", [1, 7, 8, 9, 19])
def test_distributed_padding_has_no_extra_study_exposure(size):
    keys = []
    for rank in range(2):
        sampler = EffectiveBatchSampler(size, 2, 2, seed=7, epoch=2, rank=rank, world_size=2)
        keys.extend(key for batch in sampler for key in batch)
    real = [key[0] for key in keys if key[1]]
    assert sorted(real) == list(range(size))
    assert len(keys) % 8 == 0


def test_resume_sampler_jumps_to_exact_next_update():
    whole = list(EffectiveBatchSampler(31, 2, 3, seed=7, epoch=2, rank=1, world_size=2))
    resumed = list(
        EffectiveBatchSampler(31, 2, 3, seed=7, epoch=2, rank=1, world_size=2, start_update=1)
    )
    assert resumed == whole[3:]


class RandomDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "image": torch.rand(2),
            "python": random.random(),
            "numpy": np.random.random(),
            "label_mask": torch.ones(12),
        }


def test_item_randomness_is_replayable_and_does_not_mutate_parent_rng():
    dataset = SeededDataset(RandomDataset())
    before = capture_rng(torch.device("cpu"))
    first, second = dataset[(0, True, 42)], dataset[(0, True, 42)]
    torch.testing.assert_close(first["image"], second["image"], rtol=0, atol=0)
    assert first["python"] == second["python"] and first["numpy"] == second["numpy"]
    after = capture_rng(torch.device("cpu"))
    assert before["python"] == after["python"] and before["numpy"] == after["numpy"]
    torch.testing.assert_close(before["torch"], after["torch"], rtol=0, atol=0)
    assert not dataset[(0, False, 42)]["label_mask"].any()


def test_cpu_augmentation_never_calls_the_cuda_seeding_api(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dataset must not reset model CUDA generators")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    assert SeededDataset(RandomDataset())[(0, True, 42)]["sample_valid"]


def test_checkpoint_restores_all_rng_and_rejects_changed_world(tmp_path):
    device = torch.device("cpu")
    binding = {"world_size": 1, "code": "fixed"}
    rng = capture_rng(device)
    checkpoint = {
        "schema_version": 1,
        "binding": binding,
        "optimizer_boundary": True,
        "rng_by_rank": [rng],
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, checkpoint, min_free_gib=0)
    expected = [torch.rand(2), random.random(), np.random.random()]
    loaded = load_checkpoint(path, binding)
    restore_rng(loaded["rng_by_rank"][0], device)
    actual = [torch.rand(2), random.random(), np.random.random()]
    torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
    assert expected[1:] == actual[1:]
    with pytest.raises(ValueError, match="differs"):
        load_checkpoint(path, {"world_size": 2, "code": "fixed"})


def test_failed_checkpoint_reserve_preserves_previous_file(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous")
    with pytest.raises(OSError, match="reserve"):
        save_checkpoint(path, {"x": torch.ones(2)}, min_free_gib=10**9)
    assert path.read_bytes() == b"previous"


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"diagnostic_study_limit": 2, "batch_size": True},
        {"diagnostic_study_limit": 2, "precision": "fp16"},
        {"diagnostic_study_limit": 2, "head_lr": float("nan")},
    ],
)
def test_invalid_training_config_fails_closed(changes):
    with pytest.raises(ValueError):
        TrainConfig(**changes).validate()


def test_diagnostic_configuration_is_explicit():
    TrainConfig(diagnostic_study_limit=4, slices=2, side=32, max_series=3).validate()


class SyntheticStudies(Dataset):
    """Narrow training-loop fixture; real release acceptance is a separate run."""

    cache_side = 256
    source_revision = "synthetic"

    def __init__(self, studies, series, cache, augment=False, **kwargs):
        self.studies = studies.reset_index(drop=True)

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        row = self.studies.iloc[index]
        return {
            "images": torch.rand(3, 2, 1, 16, 16),
            "features": torch.eye(3, 5),
            "series_mask": torch.ones(3, dtype=torch.bool),
            "slice_mask": torch.ones(3, 2, dtype=torch.bool),
            "positions": torch.zeros(3, 2, 2),
            "labels": torch.tensor([float(row[t]) for t in TARGETS]),
            "label_mask": torch.tensor([float(row[t + "__mask"]) for t in TARGETS]),
            "study_uid": row.study_uid,
        }


def _assert_tree_equal(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_tree_equal(left, right)
    else:
        assert first == second


def test_training_mid_epoch_resume_matches_every_checkpoint_value(tmp_path, monkeypatch):
    from rsna.competitive import train
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.model import CompetitiveStudyNet

    rows = []
    for i in range(14):
        row = {"study_uid": f"study{i:02}", "leakage_group": f"group{i}", "split": "train"}
        for j, target in enumerate(TARGETS):
            row[target], row[target + "__mask"] = i % 2, int(j < 6 or i % 3 == 0)
            row[target + "__provenance"] = "synthetic_weak"
        rows.append(row)
    manifest = pd.DataFrame(rows)
    overlay = manifest[["study_uid", "leakage_group"]].copy()
    overlay["fold"] = [0] * 5 + [1] * 9
    series_path = tmp_path / "series.csv"
    series_path.write_text("SeriesInstanceUID\na\n")
    release_binding = {
        "series_sha256": hashlib.sha256(series_path.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    monkeypatch.setattr(
        train, "load_verified_release", lambda *args: (manifest.copy(), release_binding)
    )
    config = TrainConfig(
        diagnostic_study_limit=20,
        epochs=2,
        batch_size=2,
        accumulation_steps=2,
        slices=2,
        side=16,
        max_series=3,
        hidden_dim=16,
        freeze_encoder=False,
        augment=True,
        min_free_gib=0,
    )
    args = argparse.Namespace(
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=tmp_path / "unused",
        series=series_path,
        cache=tmp_path,
        output=tmp_path / "full",
        resume=None,
        stop_after_updates=None,
    )

    def execute():
        return train._run_locked(
            args,
            config,
            [False],
            0,
            1,
            torch.device("cpu"),
            create_encoder,
            SyntheticStudies,
            CompetitiveStudyNet,
            lambda *args: overlay.copy(),
        )

    assert execute() == 0
    full = torch.load(args.output / "checkpoint.pt", weights_only=True)
    expected_predictions = (args.output / "outer_predictions.csv").read_bytes()
    args.output = tmp_path / "resumed"
    args.stop_after_updates = 1
    assert execute() == 75
    partial = torch.load(args.output / "checkpoint.pt", weights_only=True)
    assert partial["epoch"] == 0 and partial["next_update"] == 1
    assert not (args.output / "outer_predictions.csv").exists()
    args.resume = args.output / "checkpoint.pt"
    args.stop_after_updates = None
    assert execute() == 0
    actual = torch.load(args.output / "checkpoint.pt", weights_only=True)
    _assert_tree_equal(full, actual)
    assert expected_predictions == (args.output / "outer_predictions.csv").read_bytes()
