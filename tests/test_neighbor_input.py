"""Matched raw information, legacy zero parity and real trainer recovery."""

import argparse
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rsna.competitive.backbones import _GrayEncoder, create_encoder
from rsna.competitive.data import CompetitiveStudyDataset, collate_studies
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.neighbor_input import neighbor_input_channels, sample_neighbor_images
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS


def geometry(count=16):
    return dict(
        n_slices=count,
        ordering="patient_geometry",
        orientation=[1, 0, 0, 0, 1, 0],
        positions=[[0, 0, 3 * i] for i in range(count)],
        pixel_spacing=[0.5, 0.5],
    )


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"mode": "physical", "strength": True},
        {"mode": "gray", "strength": 0.25},
        {"mode": "physical", "strength": float("nan")},
        {"mode": "distant", "strength": 0.6},
    ],
)
def test_invalid_input_policies_rejected(policy):
    with pytest.raises(ValueError):
        neighbor_input_channels(policy)


@pytest.mark.parametrize("mode", ["gray", "physical", "distant"])
def test_zero_input_is_exact_legacy_gray_after_resize(mode):
    volume = np.random.default_rng(2).integers(0, 65536, (16, 19, 19), dtype=np.uint16)
    centers = np.array([0, 7, 15], dtype=np.int64)
    before = torch.get_rng_state().clone()
    legacy = sample_neighbor_images(volume, geometry(), centers, 16, None)
    zero = sample_neighbor_images(volume, geometry(), centers, 16, dict(mode=mode, strength=0))
    assert torch.equal(legacy, zero) and torch.equal(before, torch.get_rng_state())


def test_nonzero_context_is_finite_bounded_and_preserves_resized_center_mean():
    volume = np.random.default_rng(5).integers(0, 65536, (16, 19, 19), dtype=np.uint16)
    centers = np.array([0, 7, 15], dtype=np.int64)
    gray = sample_neighbor_images(volume, geometry(), centers, 16, None)
    for mode in ["physical", "distant"]:
        rgb = sample_neighbor_images(
            volume, geometry(), centers, 16, dict(mode=mode, strength=0.25)
        )
        assert (
            rgb.shape == (3, 3, 16, 16)
            and torch.isfinite(rgb).all()
            and rgb.min() >= 0
            and rgb.max() <= 1
        )
        torch.testing.assert_close(rgb.double().mean(1), gray[:, 0].double(), rtol=0, atol=1e-7)
        assert not torch.equal(rgb[:, 0], rgb[:, 2])


def test_canonical_dataset_keeps_masks_positions_and_exact_zero_input(tmp_path):
    from test_competitive_data import fixture

    studies, series, paths = fixture(tmp_path, counts=(16, 6))
    for path in paths:
        meta = json.loads(path.with_suffix(".json").read_text())
        meta["pixel_spacing"] = [0.5, 0.5]
        path.with_suffix(".json").write_text(json.dumps(meta))
    legacy = CompetitiveStudyDataset(studies, series, tmp_path, slices=4, side=16, max_series=3)[0]
    for mode, strength in [("gray", 0), ("physical", 0), ("physical", 0.25), ("distant", 0.25)]:
        row = CompetitiveStudyDataset(
            studies,
            series,
            tmp_path,
            slices=4,
            side=16,
            max_series=3,
            neighbor_channels=dict(mode=mode, strength=strength),
        )[0]
        for key in ["features", "series_mask", "slice_mask", "positions", "labels", "label_mask"]:
            assert torch.equal(row[key], legacy[key])
        if strength == 0:
            assert torch.equal(row["images"], legacy["images"])
        else:
            assert row["images"].shape[2] == 3
            assert not row["images"][~row["slice_mask"]].any()
            torch.testing.assert_close(
                row["images"].double().mean(2),
                legacy["images"][:, :, 0].double(),
                rtol=0,
                atol=1e-7,
            )
        assert collate_studies([row, row])["images"].shape[0] == 2


def test_official_wrapper_same_state_and_normalization_for_repeated_gray():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)

        def forward(self, pixel_values):
            return SimpleNamespace(pooler_output=self.conv(pixel_values).mean((2, 3)))

    torch.manual_seed(42)
    one = _GrayEncoder(Backbone(), 16)
    torch.manual_seed(42)
    three = _GrayEncoder(Backbone(), 16, input_channels=3)
    assert one.state_dict().keys() == three.state_dict().keys()
    for key, value in one.state_dict().items():
        assert torch.equal(value, three.state_dict()[key])
    gray = torch.rand(2, 1, 16, 16)
    rgb = gray.repeat(1, 3, 1, 1)
    assert torch.equal(one._normalized_rgb(gray), three._normalized_rgb(rgb))
    assert torch.equal(one(gray), three(rgb))
    with pytest.raises(ValueError):
        one(rgb)
    with pytest.raises(ValueError):
        three(gray)


@pytest.mark.parametrize(
    "changes",
    [
        {"augment": True},
        {"slice_sampling": "stratified"},
        {"pooling": "patch_attention"},
        {"fit_scope": "all_development"},
        {"slice_context": "neighbor3"},
    ],
)
def test_neighbor_configuration_isolates_fixed_input_experiment(changes):
    with pytest.raises(ValueError):
        replace(
            TrainConfig(neighbor_channels=dict(mode="physical", strength=0.25)), **changes
        ).validate()


@pytest.mark.parametrize("mode,strength", [("gray", 0), ("physical", 0.25), ("distant", 0.25)])
def test_actual_canonical_trainer_full_stop_resume_state_and_csv_equal(
    tmp_path, monkeypatch, mode, strength
):
    from test_competitive_training import SyntheticStudies, _assert_tree_equal

    from rsna.competitive import train

    rows = []
    for i in range(8):
        row = dict(study_uid=f"s{i}", leakage_group=f"g{i}", split="train")
        for t in TARGETS:
            row[t], row[t + "__mask"], row[t + "__provenance"] = i % 2, 1, "synthetic_weak"
        rows.append(row)
    manifest = pd.DataFrame(rows)
    folds = manifest[["study_uid", "leakage_group"]].copy()
    folds["fold"] = [0] * 4 + [1] * 4
    series_path = tmp_path / "series.csv"
    series_path.write_text("SeriesInstanceUID\na\n")
    release = {
        "series_sha256": hashlib.sha256(series_path.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    monkeypatch.setattr(train, "load_verified_release", lambda *args: (manifest.copy(), release))

    class NeighborStudies(SyntheticStudies):
        def __init__(self, *args, neighbor_channels=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.policy = neighbor_channels

        def __getitem__(self, index):
            row = super().__getitem__(index)
            volume = np.arange(16 * 16 * 16, dtype=np.uint16).reshape(16, 16, 16) + index
            image = sample_neighbor_images(
                volume, geometry(), np.array([0, 15], dtype=np.int64), 16, self.policy
            )
            row["images"] = image[None].repeat(3, 1, 1, 1, 1)
            return row

    config = TrainConfig(
        epochs=2,
        batch_size=1,
        accumulation_steps=2,
        slices=2,
        side=16,
        max_series=3,
        hidden_dim=12,
        freeze_encoder=False,
        min_free_gib=0,
        diagnostic_study_limit=16,
        neighbor_channels=dict(mode=mode, strength=strength),
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
            NeighborStudies,
            CompetitiveStudyNet,
            lambda *args: folds.copy(),
        )

    assert execute() == 0
    full = torch.load(args.output / "checkpoint.pt", weights_only=True)
    csv = (args.output / "outer_predictions.csv").read_bytes()
    assert full["binding"]["config"]["neighbor_channels"] == config.neighbor_channels
    args.output = tmp_path / "resume"
    args.stop_after_updates = 1
    assert execute() == 75
    args.resume = args.output / "checkpoint.pt"
    args.stop_after_updates = None
    assert execute() == 0
    _assert_tree_equal(full, torch.load(args.output / "checkpoint.pt", weights_only=True))
    assert csv == (args.output / "outer_predictions.csv").read_bytes()
