"""Verify final production adapters and frozen base tensors on one CPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from rsna.competitive.backbones import (
    _adapt_pretrained_state,
    create_encoder_from_training_checkpoint,
)
from rsna.competitive.checkpoint import atomic_json, file_sha256
from rsna.competitive.model import CompetitiveStudyNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert os.environ.get("SLURM_JOB_ID") and not os.environ.get("SLURM_JOB_GPUS")
    torch.set_num_threads(1)
    run_record = json.loads((args.run / "run.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    assert run_record["status"] == result["status"] == "completed"
    assert run_record["training_studies"] == 16 and run_record["outer_studies"] == 0
    assert result["metrics"] is None and result["evaluation_source"] is None
    assert not (args.run / "outer_predictions.csv").exists()
    checkpoint = args.run / "checkpoint.pt"
    checkpoint_sha = file_sha256(checkpoint)
    assert checkpoint_sha == result["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["epoch"] == 2 and state["next_update"] == 0 and state["step"] == 4
    assert state["binding"]["config"]["fit_scope"] == "all_development"
    assert state["binding"]["config"]["diagnostic_study_limit"] == 16
    assert state["binding"] == run_record["binding"]
    assert run_record["encoder_trainable_parameters"] == 10223616
    config = json.loads(args.config.read_text())
    encoder_config = state["binding"]["config"]["encoder"]
    assert encoder_config["weights_sha256"] == file_sha256(Path(config["encoder"]["weights_path"]))
    original = _adapt_pretrained_state(
        config["encoder"], load_file(config["encoder"]["weights_path"])
    )
    for name, value in original.items():
        assert torch.equal(value, state["model"]["encoder.backbone." + name]), name
    original_count = len(original)
    del original
    residuals = {k: v for k, v in state["model"].items() if k.endswith(".lora_B")}
    assert len(residuals) == 120
    assert all(torch.isfinite(v).all() and v.abs().sum() > 0 for v in residuals.values())
    assert all(torch.isfinite(v).all() for v in state["model"].values())
    encoder, dim, provenance = create_encoder_from_training_checkpoint(
        encoder_config,
        Path(config["encoder"]["config_path"]),
        encoder_state={
            k.removeprefix("encoder."): v
            for k, v in state["model"].items()
            if k.startswith("encoder.")
        },
    )
    assert provenance == encoder_config
    model = CompetitiveStudyNet(
        encoder,
        dim,
        pooling=config["pooling"],
        hidden_dim=config["hidden_dim"],
        encode_chunk_size=config["encode_chunk_size"],
    )
    model.load_state_dict(state["model"], strict=True)
    atomic_json(
        args.output,
        {
            "status": "all_development_pilot_checkpoint_cpu_verified",
            "optimizer_steps": state["step"],
            "checkpoint_sha256": checkpoint_sha,
            "unchanged_official_base_tensors": original_count,
            "changed_residual_modules": len(residuals),
            "all_model_tensors_finite": True,
            "checkpoint_only_strict_model_reconstruction": True,
            "real_mri_prediction_replayed": False,
            "cuda_used": False,
            "binding": state["binding"],
        },
    )
    print(json.dumps({"status": "all_development_pilot_checkpoint_cpu_verified", "modules": 120}))


if __name__ == "__main__":
    main()
