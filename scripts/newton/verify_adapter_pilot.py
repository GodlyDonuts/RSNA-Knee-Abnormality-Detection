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
from rsna.competitive.data import CompetitiveStudyDataset, collate_studies
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
    parser.add_argument("--compact-output", type=Path)
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
            slice_context=config.slice_context,
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
        slice_sampling=config.slice_sampling,
        series_sampling=config.series_sampling,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_studies)))
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
    if config.slice_context == "neighbor3":
        assert full["model"]["slice_context.output.weight"].abs().sum() > 0
    compact_sha = None
    compact_bytes = None
    if args.compact_output is not None:
        if config.slice_context != "neighbor3" or args.compact_output.exists():
            raise ValueError(
                "Distinct compact preservation is only for the whole-series diagnostic"
            )
        compact = {
            "schema_version": 1,
            "status": "diagnostic_adapter_context_delta_only",
            "competition_eligible": False,
            "binding": full["binding"],
            "base_weights_sha256": expected["weights_sha256"],
            "model_trainable": {
                name: value
                for name, value in full["model"].items()
                if not name.startswith("encoder.") or ".lora_A" in name or ".lora_B" in name
            },
            "optimizer": full["optimizer"],
            "epoch": full["epoch"],
            "optimizer_steps": full["step"],
            "full_payload_verified": True,
        }
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(compact, args.compact_output)
        compact_bytes = args.compact_output.stat().st_size
        assert compact_bytes < 180 * 1024**2
        compact_sha = file_sha256(args.compact_output)
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
            "slice_context": config.slice_context,
            "slice_sampling": config.slice_sampling,
            "series_sampling": config.series_sampling,
            "compact_checkpoint_sha256": compact_sha,
            "compact_checkpoint_bytes": compact_bytes,
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
