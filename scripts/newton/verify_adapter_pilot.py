"""Verify real adapter training, frozen official tensors and exact pilot recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from rsna.competitive.backbones import (
    _adapt_pretrained_state,
    create_encoder_from_training_checkpoint,
)
from rsna.competitive.checkpoint import atomic_json, file_sha256
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig, _model_inputs
from rsna.schema import TARGETS


def same(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for name in left:
            same(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            same(a, b)
    else:
        assert left == right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(2)
    config = TrainConfig(**json.loads(args.config.read_text()))
    full = torch.load(args.full / "checkpoint.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(args.resumed / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert full["epoch"] == config.epochs and full["next_update"] == 0
    same(full, resumed)
    assert (args.full / "outer_predictions.csv").read_bytes() == (
        args.resumed / "outer_predictions.csv"
    ).read_bytes()
    del resumed
    original = _adapt_pretrained_state(config.encoder, load_file(config.encoder["weights_path"]))
    for name, tensor in original.items():
        assert torch.equal(tensor, full["model"]["encoder.backbone." + name]), name
    base_count = len(original)
    del original
    residuals = {name: tensor for name, tensor in full["model"].items() if name.endswith(".lora_B")}
    assert residuals and all(torch.isfinite(tensor).all() for tensor in residuals.values())
    changed = sum(bool(tensor.abs().sum() > 0) for tensor in residuals.values())
    assert changed == 120, "Every declared residual projection must have actually learned"
    expected = full["binding"]["config"]["encoder"]
    encoder, dim, provenance = create_encoder_from_training_checkpoint(
        expected,
        Path(config.encoder["config_path"]),
        encoder_state={
            name.removeprefix("encoder."): value
            for name, value in full["model"].items()
            if name.startswith("encoder.")
        },
    )
    assert provenance == expected and expected["adapter"]["trainable_parameters"] == 10_223_616
    model = (
        CompetitiveStudyNet(
            encoder,
            dim,
            pooling=config.pooling,
            hidden_dim=config.hidden_dim,
            encode_chunk_size=config.encode_chunk_size,
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(full["model"], strict=True)
    predictions = pd.read_csv(args.full / "outer_predictions.csv")
    studies = pd.read_parquet(args.studies)
    study = studies[studies.study_uid == predictions.study_uid.iloc[0]].copy()
    dataset = CompetitiveStudyDataset(
        study,
        pd.read_csv(args.series),
        args.cache,
        slices=config.slices,
        side=config.side,
        max_series=config.max_series,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.precision == "bf16"),
    ):
        probability = model(*_model_inputs(batch, torch.device("cuda"))).float().sigmoid().cpu()
    torch.testing.assert_close(
        probability[0],
        torch.tensor(predictions[list(TARGETS)].iloc[0].to_numpy(), dtype=torch.float32),
        rtol=0,
        atol=1e-7,
    )
    atomic_json(
        args.output,
        {
            "status": "adapter_pilot_and_exact_recovery_verified",
            "production_model": False,
            "diagnostic_studies": config.diagnostic_study_limit,
            "adapter_parameters": 10_223_616,
            "changed_residual_modules": changed,
            "unchanged_official_base_tensors": base_count,
            "full_payload_resume_equal": True,
            "outer_predictions_byte_equal": True,
            "checkpoint_only_real_mri_reload_equal": True,
            "precision": config.precision,
            "optimizer_steps": full["step"],
            "full_checkpoint_sha256": file_sha256(args.full / "checkpoint.pt"),
            "resumed_checkpoint_sha256": file_sha256(args.resumed / "checkpoint.pt"),
            "outer_predictions_sha256": file_sha256(args.full / "outer_predictions.csv"),
            "binding": full["binding"],
        },
    )
    print(
        json.dumps(
            {
                "status": "adapter_pilot_and_exact_recovery_verified",
                "changed_residual_modules": changed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
