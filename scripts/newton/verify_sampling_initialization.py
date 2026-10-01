"""Verify a sampling-only comparison starts from identical official-weight models."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder
from rsna.competitive.checkpoint import atomic_json, file_sha256
from rsna.competitive.data import CompetitiveStudyDataset, collate_studies
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig, _model_inputs


def matched_configs(control: dict, candidate: dict) -> tuple[TrainConfig, TrainConfig]:
    """Refuse concurrent changes to capacity, images, membership or optimization."""
    if (
        control.get("slice_sampling") != "uniform"
        or candidate.get("slice_sampling") != "stratified"
        or {**control, "slice_sampling": "stratified"} != candidate
    ):
        raise ValueError("The only configuration difference must be slice_sampling")
    configs = TrainConfig(**control), TrainConfig(**candidate)
    for config in configs:
        config.validate()
        if config.augment or config.fit_scope != "outer_fold":
            raise ValueError("Fixed series and an untouched outer fold are required")
    return configs


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def initialized(
    config: TrainConfig, batch: dict, device: torch.device
) -> tuple[dict, torch.Tensor]:
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    encoder, dimension, provenance = create_encoder(config.encoder)
    configure_encoder_training(encoder, config.freeze_encoder)
    model = CompetitiveStudyNet(
        encoder,
        dimension,
        pooling=config.pooling,
        hidden_dim=config.hidden_dim,
        encode_chunk_size=config.encode_chunk_size,
        slice_context=config.slice_context,
    ).to(device)
    learned = {
        name: tensor_hash(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    residuals = [p for name, p in model.named_parameters() if name.endswith(".lora_B")]
    if not residuals or any(bool(p.count_nonzero()) for p in residuals):
        raise ValueError("Every residual adapter must start at zero")
    model.eval()
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == "bf16"),
    ):
        logits = model(*_model_inputs(batch, device)).float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError("Initial evaluation logits must be finite")
    return {"learned_tensor_sha256": learned, "encoder_provenance": provenance}, logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "control",
        "candidate",
        "studies",
        "series",
        "cache",
        "outer-predictions",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a distinct immutable initialization receipt")
    control, candidate = matched_configs(
        json.loads(args.control.read_text()), json.loads(args.candidate.read_text())
    )
    torch.set_num_threads(2)
    predictions = pd.read_csv(args.outer_predictions)
    if predictions.empty or predictions.study_uid.duplicated().any():
        raise ValueError("Actual outer prediction membership is required")
    studies = pd.read_parquet(args.studies)
    study = studies[studies.study_uid.eq(predictions.study_uid.iloc[0])].copy()
    if len(study) != 1:
        raise ValueError("One source-bound evaluation study is required")
    dataset = CompetitiveStudyDataset(
        study,
        pd.read_csv(args.series),
        args.cache,
        slices=control.slices,
        side=control.side,
        max_series=control.max_series,
        slice_sampling="uniform",
        series_sampling=control.series_sampling,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_studies)))
    device = torch.device(args.device)
    first, first_logits = initialized(control, batch, device)
    second, second_logits = initialized(candidate, batch, device)
    if first != second or not torch.equal(first_logits, second_logits):
        raise ValueError("Initial learned tensors or zero-adapter evaluation logits differ")
    atomic_json(
        args.output,
        {
            "status": "same_initial_learned_tensors_and_zero_adapter_logits_verified",
            "diagnostic_only": True,
            "accuracy_claim": False,
            "control_sha256": file_sha256(args.control),
            "candidate_sha256": file_sha256(args.candidate),
            "evaluation_policy": "uniform",
            "evaluation_study": study.study_uid.iloc[0],
            "initial_logits_sha256": tensor_hash(first_logits),
            "initial_state": first,
            "device": args.device,
            "torch_version": str(torch.__version__),
        },
    )


if __name__ == "__main__":
    main()
