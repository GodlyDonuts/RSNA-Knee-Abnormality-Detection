"""Bounded real-MRI training and inference probe of the largest training examinations."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder
from rsna.competitive.checkpoint import atomic_json, file_sha256
from rsna.competitive.data import CompetitiveStudyDataset, collate_studies
from rsna.competitive.folds import load_folds
from rsna.competitive.loss import effective_batch_loss, observed_counts
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig, _model_inputs, select_development
from rsna.release import load_verified_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "config",
        "compact",
        "studies",
        "series",
        "cache",
        "folds",
        "artifact-root",
        "workloads",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("A distinct probe receipt is required")
    torch.set_num_threads(2)
    config = TrainConfig(**json.loads(args.config.read_text()))
    config.validate()
    assert config.slice_sampling == config.series_sampling == "all"
    assert config.slice_context == "neighbor3" and config.fit_scope == "outer_fold"
    assert config.diagnostic_study_limit is not None
    compact = torch.load(args.compact, map_location="cpu", weights_only=True)
    assert compact["status"] == "diagnostic_adapter_context_delta_only"
    assert compact["competition_eligible"] is False
    assert compact["base_weights_sha256"] == config.encoder["weights_sha256"]
    manifest, binding = load_verified_release(args.studies, args.artifact_root)
    folds = load_folds(args.folds, manifest, binding)
    train, _ = select_development(manifest, folds, config.fold)
    workloads = json.loads(args.workloads.read_text())["worst_training_inputs"]
    assert 1 <= len(workloads) <= 5
    uids = [record["study_uid"] for record in workloads]
    assert len(set(uids)) == len(uids) and set(uids) <= set(train.study_uid)
    studies = train.set_index("study_uid").loc[uids].reset_index()
    dataset = CompetitiveStudyDataset(
        studies,
        pd.read_csv(args.series),
        args.cache,
        slices=config.slices,
        side=config.side,
        max_series=config.max_series,
        slice_sampling=config.slice_sampling,
        series_sampling=config.series_sampling,
    )
    encoder, dimension, provenance = create_encoder(config.encoder)
    configure_encoder_training(encoder, True)
    assert provenance == compact["binding"]["config"]["encoder"]
    model = CompetitiveStudyNet(
        encoder,
        dimension,
        pooling=config.pooling,
        hidden_dim=config.hidden_dim,
        encode_chunk_size=config.encode_chunk_size,
        slice_context=config.slice_context,
    ).cuda()
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(compact["model_trainable"]) == expected
    state = model.state_dict()
    state.update(compact["model_trainable"])
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.encoder_lr,
        weight_decay=config.weight_decay,
    )
    records = []
    for batch, declared in zip(
        DataLoader(dataset, batch_size=1, collate_fn=collate_studies), workloads, strict=True
    ):
        assert batch["study_uid"] == [declared["study_uid"]]
        assert int(batch["slice_mask"].sum()) == declared["valid_slices"]
        assert int(batch["series_mask"].sum()) == declared["series"]
        assert (
            batch["images"].numel() * batch["images"].element_size()
            == declared["all_padded_image_bytes_fp32"]
        )
        optimizer.zero_grad(set_to_none=True)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.precision == "bf16"):
            logits = model(*_model_inputs(batch, torch.device("cuda")))
            labels, masks = batch["labels"].cuda(), batch["label_mask"].cuda()
            loss = effective_batch_loss(logits, labels, masks, observed_counts([masks]))
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], config.clip_grad_norm
        )
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated()
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.precision == "bf16"),
        ):
            probability = model(*_model_inputs(batch, torch.device("cuda"))).float().sigmoid()
        torch.cuda.synchronize()
        inference = time.monotonic() - started
        assert torch.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        assert math.isfinite(elapsed) and math.isfinite(inference)
        record = {
            **declared,
            "training_seconds": elapsed,
            "loss": float(loss),
            "gradient_norm": float(norm),
            "training_peak_gpu_bytes": peak,
            "inference_seconds": inference,
            "inference_peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    atomic_json(
        args.output,
        {
            "status": "largest_training_examinations_full_coverage_probe_passed",
            "diagnostic_only": True,
            "competition_submission_eligible": False,
            "full_fine_tuning_runtime_validated": False,
            "t4_runtime_validated": False,
            "kaggle_score": None,
            "compact_checkpoint_sha256": file_sha256(args.compact),
            "workloads_sha256": file_sha256(args.workloads),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
