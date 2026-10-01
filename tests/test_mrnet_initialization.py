"""Synthetic transfer boundaries and recovery; no Stanford images or clinical metrics."""

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest
import torch
from torch import nn

from rsna.competitive import infer, train
from rsna.competitive.adapters import ResidualLinear
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.mrnet import MRNetAdapterNet
from rsna.competitive.mrnet_initialization import (
    SOURCE_BINDING,
    initialize_mrnet_checkpoint,
    initialize_private_mrnet,
)
from rsna.competitive.mrnet_training import MRNetRecipe
from rsna.schema import TARGETS

PROVENANCE = {
    "pretrained": True,
    "architecture": "dinov3_l16",
    "weights_sha256": "a" * 64,
    "config_sha256": "b" * 64,
    "adapter": {"rank": 2, "alpha": 2},
}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # The same synthetic frozen base in every independently constructed model.
        with torch.random.fork_rng():
            torch.manual_seed(25)
            self.linear = ResidualLinear(nn.Linear(1, 8), rank=2, alpha=2, dropout=0.1)

    def forward(self, images):
        return self.linear(images.mean((2, 3)))


def source_envelope():
    source = MRNetAdapterNet(Encoder(), 8, hidden_dim=8, chunk=3)
    with torch.no_grad():
        source.encoder.linear.lora_A.fill_(0.2)
        source.encoder.linear.lora_B.fill_(0.3)
    original = {
        **SOURCE_BINDING,
        "config_sha256": "c" * 64,
        "entrypoint_sha256": "d" * 64,
    }
    return {
        **FrozenBaseDelta(source, PROVENANCE).checkpoint_model(source),
        "binding": {
            **original,
            "recipe": asdict(MRNetRecipe()),
            "source_target_order": ["acl", "meniscus", "abnormal"],
            "training_cases": [f"{i:04d}" for i in range(1, 1130)],
            "steps_per_epoch": 142,
        },
        "optimizer_boundary": True,
        "step": 1420,
        "optimizer": {"source_only": True},
        "rng": {"source_only": True},
    }, original


def write_source(tmp_path, saved=None, terminal_change=None):
    default, original = source_envelope()
    saved = default if saved is None else saved
    checkpoint = tmp_path / "source.pt"
    torch.save(saved, checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    terminal = tmp_path / "terminal.json"
    receipt = {
        "status": "private_full_Stanford_fitting_complete",
        "binding": original,
        "source_training_exams": 1129,
        "source_validation_exams": 119,
        "step": 1420,
        "checkpoint_sha256": checkpoint_sha,
        "no_competition_submission_or_weight_publication": True,
        "RSNA_or_Kaggle_accuracy_measured": False,
    }
    if terminal_change:
        receipt.update(terminal_change)
    terminal.write_text(json.dumps(receipt))
    return dict(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        terminal=terminal,
        terminal_sha256=hashlib.sha256(terminal.read_bytes()).hexdigest(),
    )


def destination():
    return CompetitiveStudyNet(Encoder(), 8, hidden_dim=8)


def test_only_adapter_tensors_transfer_no_source_heads_base_optimizer_or_rng(tmp_path):
    model = destination()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    kwargs = write_source(tmp_path)
    # Serialization fixtures may construct random heads; initialization itself must not.
    torch.set_rng_state(rng)
    receipt = initialize_mrnet_checkpoint(model, FrozenBaseDelta(model, PROVENANCE), **kwargs)
    assert receipt["encoder_adapter_tensors"] == 2
    assert receipt["private_research_only"] is True
    assert receipt["source_heads_optimizer_counters_or_rng_transferred"] is False
    for name, value in before.items():
        if name.endswith(".lora_A"):
            assert torch.equal(model.state_dict()[name], torch.full_like(value, 0.2))
        elif name.endswith(".lora_B"):
            assert torch.equal(model.state_dict()[name], torch.full_like(value, 0.3))
        else:
            assert torch.equal(model.state_dict()[name], value)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize(
    "change",
    ["pilot", "incomplete", "source", "roster", "base", "rank", "dtype", "nan", "missing", "extra"],
)
def test_invalid_source_is_rejected_before_any_destination_tensor_changes(tmp_path, change):
    saved, _ = source_envelope()
    name = "encoder.linear.lora_A"
    terminal_change = None
    if change == "pilot":
        terminal_change = {"status": "engineering_trajectory_complete"}
    elif change == "incomplete":
        saved["step"] = 1419
    elif change == "source":
        saved["binding"]["source_input_receipt_sha256"] = "0" * 64
    elif change == "roster":
        saved["binding"]["training_cases"][1] = saved["binding"]["training_cases"][0]
    elif change == "base":
        saved["model_storage"]["frozen_tensors"]["encoder.linear.weight"]["sha256"] = "0" * 64
    elif change == "rank":
        saved["model"][name] = torch.zeros(3, 1)
        saved["model_storage"]["learned_tensors"][name]["shape"] = [3, 1]
    elif change == "dtype":
        saved["model"][name] = saved["model"][name].double()
        saved["model_storage"]["learned_tensors"][name]["dtype"] = "torch.float64"
    elif change == "nan":
        saved["model"][name].fill_(float("nan"))
    elif change == "missing":
        del saved["model"][name]
    elif change == "extra":
        saved["model"]["encoder.linear.weight"] = torch.ones(8, 1)
        saved["model_storage"]["learned_tensors"]["encoder.linear.weight"] = {
            "shape": [8, 1],
            "dtype": "torch.float32",
        }
    kwargs = write_source(tmp_path, saved, terminal_change)
    model = destination()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        initialize_mrnet_checkpoint(model, FrozenBaseDelta(model, PROVENANCE), **kwargs)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_changed_original_bytes_and_actual_mutated_frozen_base_rejected(tmp_path):
    kwargs = write_source(tmp_path)
    model = destination()
    guard = FrozenBaseDelta(model, PROVENANCE)
    with pytest.raises(ValueError, match="bytes differ"):
        initialize_mrnet_checkpoint(model, guard, **{**kwargs, "terminal_sha256": "0" * 64})
    with torch.no_grad():
        model.encoder.linear.weight.add_(1)
    with pytest.raises(ValueError, match="frozen pretrained tensor changed"):
        initialize_mrnet_checkpoint(model, guard, **kwargs)


def test_ordinary_run_unchanged_and_private_args_cannot_partially_apply_or_publish(tmp_path):
    model = destination()
    guard = FrozenBaseDelta(model, PROVENANCE)
    assert initialize_private_mrnet(argparse.Namespace(), model, guard) is None
    with pytest.raises(ValueError, match="All source"):
        initialize_private_mrnet(
            argparse.Namespace(mrnet_checkpoint=tmp_path / "source"), model, guard
        )
    kwargs = write_source(tmp_path)
    args = argparse.Namespace(
        output=Path("/home/shared/run"),
        **{"mrnet_" + key: value for key, value in kwargs.items()},
    )
    with pytest.raises(ValueError, match="private node-local"):
        initialize_private_mrnet(args, model, guard)
    with pytest.raises(ValueError, match="not competition candidates"):
        infer.validate_inference_checkpoint(
            {
                "schema_version": 2,
                "optimizer_boundary": True,
                "binding": {"private_research_only": True},
            },
            allow_diagnostic=True,
            allow_scheduled_epoch=True,
        )


class Studies(torch.utils.data.Dataset):
    cache_side = 256
    source_revision = "synthetic"

    def __init__(self, rows, *args, **kwargs):
        self.rows = rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return {
            "images": torch.rand(1, 2, 1, 16, 16),
            "features": torch.ones(1, 5),
            "series_mask": torch.ones(1, dtype=torch.bool),
            "slice_mask": torch.ones(1, 2, dtype=torch.bool),
            "positions": torch.zeros(1, 2, 2),
            "labels": torch.tensor([float(row[t]) for t in TARGETS]),
            "label_mask": torch.ones(12),
            "study_uid": row.study_uid,
        }


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for name in a:
            equal(a[name], b[name])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


@pytest.fixture
def private_tmp():
    with TemporaryDirectory(dir="/tmp", prefix="rsna-private-transfer-test-") as name:
        yield Path(name)


def test_actual_rsna_trainer_initialized_full_stop_resume_matches_every_state_and_csv(
    private_tmp, monkeypatch
):
    tmp_path = private_tmp
    kwargs = write_source(tmp_path)
    rows = [
        {
            "study_uid": f"study{i}",
            "leakage_group": f"g{i}",
            "split": "train",
            **{t: i % 2 for t in TARGETS},
            **{t + "__mask": 1 for t in TARGETS},
            **{t + "__provenance": "synthetic_weak" for t in TARGETS},
        }
        for i in range(6)
    ]
    manifest = pd.DataFrame(rows)
    overlay = manifest[["study_uid", "leakage_group"]].copy()
    overlay["fold"] = [0, 0, 1, 1, 1, 1]
    series = tmp_path / "series.csv"
    series.write_text("SeriesInstanceUID\na\n")
    release = {
        "series_sha256": hashlib.sha256(series.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    monkeypatch.setattr(train, "load_verified_release", lambda *args: (manifest.copy(), release))
    # Exercise the real private /tmp output gate, not a monkeypatched bypass.
    args = argparse.Namespace(
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=tmp_path / "unused",
        series=series,
        cache=tmp_path,
        output=tmp_path / "full",
        resume=None,
        stop_after_updates=None,
        **{"mrnet_" + key: value for key, value in kwargs.items()},
    )
    config = train.TrainConfig(
        epochs=2,
        batch_size=1,
        accumulation_steps=2,
        slices=2,
        side=16,
        max_series=1,
        hidden_dim=8,
        min_free_gib=0,
        checkpoint_format="frozen_base_delta",
    )

    def execute():
        return train._run_locked(
            args,
            config,
            [False],
            0,
            1,
            torch.device("cpu"),
            lambda spec: (Encoder(), 8, PROVENANCE),
            Studies,
            CompetitiveStudyNet,
            lambda *args: overlay.copy(),
        )

    assert execute() == 0
    full = torch.load(args.output / "checkpoint.pt", weights_only=True)
    predictions = (args.output / "outer_predictions.csv").read_bytes()
    assert (
        full["binding"]["external_initialization"]["source_checkpoint_sha256"]
        == kwargs["checkpoint_sha256"]
    )
    assert full["binding"]["private_research_only"] is True
    assert full["step"] == 4  # Fresh RSNA counters, never the source's 1,420.
    args.output, args.stop_after_updates = tmp_path / "resumed", 1
    assert execute() == 75
    stopped = torch.load(args.output / "checkpoint.pt", weights_only=True)
    assert stopped["step"] == 1
    args.resume, args.stop_after_updates = args.output / "checkpoint.pt", None
    assert execute() == 0
    equal(full, torch.load(args.output / "checkpoint.pt", weights_only=True))
    assert predictions == (args.output / "outer_predictions.csv").read_bytes()
