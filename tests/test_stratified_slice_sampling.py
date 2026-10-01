"""Real cache-loader, trainer recovery and offline inference sampling checks.

All MRI pixels and labels are synthetic. No external data or GPU is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_competitive_data import fixture
from test_competitive_inference import _fixture as inference_fixture
from test_competitive_training import _assert_tree_equal
from torch.utils.data import DataLoader

from rsna.competitive import infer, train
from rsna.competitive.backbones import create_encoder
from rsna.competitive.data import CompetitiveStudyDataset, collate_studies
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.sampling import EffectiveBatchSampler, SeededDataset
from rsna.schema import TARGETS


def cache_fixture(root: Path, size=7):
    """Unique identities, real cache checksums/geometry, sparse study-level masks."""
    frames, entries = [], []
    for index in range(size):
        uid = f"1.2.4.{index}"
        part = root / f"part-{index}"
        frame, series, paths = fixture(part, counts=(33, 25, 7, 19, 41, 5, 17))
        frame.loc[0, "study_uid"] = uid
        for j, target in enumerate(TARGETS):
            frame.loc[0, target] = (index + j) % 2
            frame.loc[0, target + "__mask"] = int(j < 3 or index % 2 == 0)
            frame.loc[0, target + "__provenance"] = "synthetic_weak"
        frame["leakage_group"], frame["split"] = uid, "train"
        for number, path in enumerate(paths):
            meta = json.loads(path.with_suffix(".json").read_text())
            sid = f"{uid}.{number}"
            sops = [f"{sid}.{k}" for k in range(meta["n_slices"])]
            relative = f"train_series/{uid}/{sid}.npz"
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
            meta.update(
                study_uid=uid,
                series_uid=sid,
                cache_path=relative,
                source_revision="synthetic",
                sop_uids=sops,
                path_fingerprint=hashlib.sha256(
                    "\n".join(
                        sorted(f"train_series/{uid}/{sid}/{sop}.dcm" for sop in sops)
                    ).encode()
                ).hexdigest(),
            )
            destination.with_suffix(".json").write_text(json.dumps(meta))
            series.loc[number, "StudyInstanceUID"] = uid
            series.loc[number, "SeriesInstanceUID"] = sid
        frames.append(frame)
        entries.append(series)
    return pd.concat(frames, ignore_index=True), pd.concat(entries, ignore_index=True)


def collect(dataset, *, epoch=0, rank=0, world=1, workers=0, start_update=0):
    sampler = EffectiveBatchSampler(
        len(dataset),
        1,
        2,
        seed=20260922,
        epoch=epoch,
        rank=rank,
        world_size=world,
        start_update=start_update,
    )
    loader = DataLoader(
        SeededDataset(dataset),
        batch_sampler=sampler,
        collate_fn=collate_studies,
        num_workers=workers,
        generator=torch.Generator().manual_seed(777),
    )
    result = {}
    for batch in loader:
        if batch["sample_valid"][0]:
            result[batch["study_uid"][0]] = {
                name: value[0].clone()
                for name, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
    return result


@pytest.mark.parametrize("count", [1, 2, 7, 16, 17, 33, 57])
def test_stratified_actual_pixels_are_ordered_disjoint_and_have_matching_geometry(tmp_path, count):
    studies, series, _ = fixture(tmp_path, counts=(count,))
    dataset = CompetitiveStudyDataset(
        studies, series, tmp_path, slices=16, side=16, slice_sampling="stratified"
    )
    for epoch in range(4):
        row = collect(dataset, epoch=epoch)["1.2.3"]
        valid = row["slice_mask"][0]
        indices = (row["images"][0, valid, 0, 0, 0] * 65535 / 1000).round().long() - 1
        selected = min(count, 16)
        edges = np.linspace(0, count, selected + 1).astype(int)
        assert len(indices) == selected and (indices.diff() > 0).all()
        assert (indices >= torch.tensor(edges[:-1])).all()
        assert (indices < torch.tensor(edges[1:])).all()
        torch.testing.assert_close(
            row["positions"][0, valid, 0], (indices * 2 - (count - 1)).float() / 100
        )
        assert row["positions"][0, valid, 1].eq(1).all()
        assert row["images"][~row["slice_mask"]].count_nonzero() == 0
        assert row["label_mask"].sum() == 1 and row["labels"][0] == 1


def test_training_epochs_expose_more_slices_without_changing_series_or_labels(tmp_path):
    studies, series = cache_fixture(tmp_path, size=1)
    variable = CompetitiveStudyDataset(
        studies,
        series,
        tmp_path,
        slices=4,
        side=16,
        max_series=3,
        slice_sampling="stratified",
        augment=False,
    )
    uniform = CompetitiveStudyDataset(studies, series, tmp_path, slices=4, side=16, max_series=3)
    reference = uniform[0]
    rows = [collect(variable, epoch=epoch)[studies.study_uid.iloc[0]] for epoch in range(10)]
    for row in rows:
        for name in ("features", "series_mask", "slice_mask", "labels", "label_mask"):
            assert torch.equal(row[name], reference[name])
    assert any(not torch.equal(rows[0]["images"], row["images"]) for row in rows[1:])
    for slot in range(3):
        seen = torch.cat([r["images"][slot, :, 0, 0, 0] for r in rows]).unique()
        assert len(seen) > 4
    # Sampling never changes the optimizer/model RNG stream in the main process.
    before = torch.random.get_rng_state().clone()
    collect(variable, epoch=5)
    assert torch.equal(before, torch.random.get_rng_state())
    poisoned = studies.copy()
    for target in TARGETS:
        poisoned[target], poisoned[target + "__mask"] = 1 - poisoned[target], 1
    changed_labels = CompetitiveStudyDataset(
        poisoned, series, tmp_path, slices=4, side=16, max_series=3, slice_sampling="stratified"
    )
    assert torch.equal(
        rows[5]["images"], collect(changed_labels, epoch=5)[studies.study_uid.iloc[0]]["images"]
    )


@pytest.mark.parametrize("workers", [0, 2])
def test_sampling_is_invariant_to_worker_count_rank_partition_and_resume(tmp_path, workers):
    studies, series = cache_fixture(tmp_path)
    dataset = CompetitiveStudyDataset(
        studies, series, tmp_path, slices=4, side=16, max_series=3, slice_sampling="stratified"
    )
    reference = collect(dataset, epoch=3)
    distributed = {}
    for rank in range(2):
        part = collect(dataset, epoch=3, rank=rank, world=2, workers=workers)
        assert not distributed.keys() & part.keys()
        distributed.update(part)
    _assert_tree_equal(reference, distributed)
    resumed = collect(dataset, epoch=3, workers=workers, start_update=1)
    assert resumed and len(resumed) < len(reference)
    for uid, row in resumed.items():
        _assert_tree_equal(reference[uid], row)


@pytest.mark.parametrize("fault", ["series_augmentation", "test_split"])
def test_stratified_policy_cannot_silently_change_series_or_randomize_test_input(tmp_path, fault):
    studies, series, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="fixed series selection"):
        CompetitiveStudyDataset(
            studies,
            series,
            tmp_path,
            slice_sampling="stratified",
            augment=fault == "series_augmentation",
            raw_split="test_series" if fault == "test_split" else "train_series",
        )
    with pytest.raises(ValueError, match="fixed series selection"):
        train.TrainConfig(slice_sampling="stratified", augment=True).validate()


def test_actual_cached_trainer_mid_epoch_and_second_epoch_recovery_is_exact(tmp_path, monkeypatch):
    studies, series = cache_fixture(tmp_path / "cache", size=10)
    series_file = tmp_path / "series.csv"
    series.to_csv(series_file, index=False)
    release = {
        "series_sha256": hashlib.sha256(series_file.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    monkeypatch.setattr(train, "load_verified_release", lambda *args: (studies.copy(), release))
    folds = studies[["study_uid", "leakage_group"]].copy()
    folds["fold"] = [0] * 4 + [1] * 6
    config = train.TrainConfig(
        slice_sampling="stratified",
        augment=False,
        epochs=2,
        accumulation_steps=2,
        slices=4,
        side=16,
        max_series=3,
        hidden_dim=8,
        freeze_encoder=False,
        diagnostic_study_limit=20,
        checkpoint_every_updates=1,
        min_free_gib=0,
    )
    args = argparse.Namespace(
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=tmp_path / "unused",
        series=series_file,
        cache=tmp_path / "cache",
        output=tmp_path / "full",
        resume=None,
        stop_after_updates=None,
    )
    modes = []

    class RecordedDataset(CompetitiveStudyDataset):
        def __init__(self, *a, **kw):
            modes.append(kw["slice_sampling"])
            super().__init__(*a, **kw)

    def execute():
        return train._run_locked(
            args,
            config,
            [False],
            0,
            1,
            torch.device("cpu"),
            create_encoder,
            RecordedDataset,
            CompetitiveStudyNet,
            lambda *args: folds.copy(),
        )

    config.validate()
    assert execute() == 0 and modes == ["stratified", "uniform"]
    full = torch.load(args.output / "checkpoint.pt", weights_only=True)
    predictions = (args.output / "outer_predictions.csv").read_bytes()
    assert full["step"] == 6
    for stop in (1, 4):
        args.output, args.resume, args.stop_after_updates = tmp_path / f"resume-{stop}", None, stop
        assert execute() == 75
        partial = torch.load(args.output / "checkpoint.pt", weights_only=True)
        assert partial["next_update"] == 1 and partial["epoch"] == (0 if stop == 1 else 1)
        args.resume, args.stop_after_updates = args.output / "checkpoint.pt", None
        assert execute() == 0
        _assert_tree_equal(full, torch.load(args.output / "checkpoint.pt", weights_only=True))
        assert predictions == (args.output / "outer_predictions.csv").read_bytes()


def test_stratified_checkpoint_actual_offline_inference_uses_fixed_uniform_inputs(
    tmp_path, monkeypatch
):
    args, checkpoint = inference_fixture(tmp_path, monkeypatch)
    checkpoint["binding"]["config"]["slice_sampling"] = "stratified"
    torch.save(checkpoint, args.checkpoint)
    modes = []

    class RecordedDataset(CompetitiveStudyDataset):
        def __init__(self, *a, **kw):
            modes.append(kw["slice_sampling"])
            super().__init__(*a, **kw)

    monkeypatch.setattr(infer, "CompetitiveStudyDataset", RecordedDataset)
    result = infer.run(args)
    first = args.output.read_bytes()
    random.seed(1234)
    np.random.seed(9876)
    torch.manual_seed(6789)
    args.output = tmp_path / "repeat.csv"
    second = infer.run(args)
    assert modes == ["uniform", "uniform"] and first == args.output.read_bytes()
    assert result["training_slice_sampling"] == second["training_slice_sampling"] == "stratified"
    assert result["inference_slice_sampling"] == second["inference_slice_sampling"] == "uniform"
