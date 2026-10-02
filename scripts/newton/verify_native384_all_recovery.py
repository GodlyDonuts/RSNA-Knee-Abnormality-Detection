"""Verify allocated all-development recovery including the masked one-study tail.

This verifies a bounded diagnostic only. It cannot admit a publication, report
accuracy or replace the separate full-population production dispatch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def numerical_updates(records):
    result = []
    for record in records:
        clean = {
            k: v
            for k, v in record.items()
            if k not in {"elapsed_seconds", "peak_gpu_allocated_bytes"}
        }
        if any(not math.isfinite(v) for v in clean.values() if isinstance(v, (int, float))):
            raise ValueError("nonfinite_optimizer_record")
        result.append(clean)
    if [x["optimizer_steps"] for x in result] != list(range(1, 7)):
        raise ValueError("six_contiguous_real_optimizer_updates_required")
    return result


def adapter_parameter_count(encoder, provenance):
    learned = {name: p for name, p in encoder.named_parameters() if p.requires_grad}
    adapter = provenance["adapter"]
    if not learned or any(not n.endswith((".lora_A", ".lora_B")) for n in learned):
        raise ValueError("only_declared_encoder_adapters_may_be_trainable")
    count = sum(p.numel() for p in learned.values())
    if count != adapter["trainable_parameters"] or len(learned) != 2 * len(
        adapter["target_modules"]
    ):
        raise ValueError("actual_adapter_graph_must_equal_original_provenance")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file

    from rsna.competitive.adapters import configure_encoder_training
    from rsna.competitive.backbones import _adapt_pretrained_state, create_encoder
    from rsna.competitive.checkpoint import atomic_json, file_sha256
    from rsna.competitive.delta_checkpoint import FrozenBaseDelta, restore_checkpoint_model
    from rsna.competitive.model import CompetitiveStudyNet
    from rsna.competitive.train import TrainConfig
    from scripts.newton.verify_adapter_pilot import same

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("existing_recovery_receipt_requires_readonly_review")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    config = TrainConfig(**json.loads(args.config.read_text()))
    if (
        config.fit_scope != "all_development"
        or config.side != 384
        or config.diagnostic_study_limit != 17
        or config.epochs != 2
    ):
        raise ValueError("exact_native17_all_development_diagnostic_required")
    full, resumed, reference = [
        torch.load(args.root / name / "checkpoint.pt", map_location="cpu", weights_only=True)
        for name in ("full", "resumed", "reference")
    ]
    same(full, resumed)
    if full["step"] != 6 or full["epoch"] != 2 or full["next_update"] != 0:
        raise ValueError("complete_two_epoch_six_update_payload_required")
    records = {}
    for name in ("full", "resumed", "reference"):
        folder = args.root / name
        run = json.loads((folder / "run.json").read_text())
        result = json.loads((folder / "result.json").read_text())
        if run["training_studies"] != 17 or run["outer_studies"] != 0:
            raise ValueError("bounded_all_development_population_required")
        if result.get("metrics") is not None or (folder / "outer_predictions.csv").exists():
            raise ValueError("no_in_sample_evaluation_or_prediction_file_permitted")
        records[name] = json.loads((folder / "invocation.json").read_text())["updates"]
    stopped = json.loads((args.root / "stopped-invocation.json").read_text())
    if stopped["status"] != "checkpointed_for_resume" or stopped["exit_code"] != 75:
        raise ValueError("actual_intentional_optimizer_boundary_stop_required")
    if [r["optimizer_steps"] for r in stopped["updates"]] != [1]:
        raise ValueError("stop_after_exactly_one_update_required")
    learned_records = numerical_updates(records["full"])
    same(learned_records, numerical_updates(stopped["updates"] + records["resumed"]))
    same(learned_records, numerical_updates(records["reference"]))
    reference_binding = dict(reference["binding"])
    reference_binding["config"] = {
        **reference_binding["config"],
        "checkpoint_format": "frozen_base_delta",
    }
    same(full["binding"], reference_binding)
    for key in ("epoch", "next_update", "step", "rng_by_rank", "optimizer", "scheduled_epoch"):
        same(full[key], reference[key])
    encoder, dim, provenance = create_encoder(config.encoder)
    configure_encoder_training(encoder, config.freeze_encoder)
    same(provenance, full["binding"]["config"]["encoder"])
    count = adapter_parameter_count(encoder, provenance)
    model = (
        CompetitiveStudyNet(
            encoder,
            dim,
            pooling=config.pooling,
            hidden_dim=config.hidden_dim,
            encode_chunk_size=config.encode_chunk_size,
            slice_context=config.slice_context,
        )
        .to(args.device)
        .eval()
    )
    guard = FrozenBaseDelta(model, provenance)
    restore_checkpoint_model(model, full, guard)
    expanded = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    same(expanded, reference["model"])
    original = _adapt_pretrained_state(config.encoder, load_file(config.encoder["weights_path"]))
    for name, tensor in original.items():
        same(tensor, expanded["encoder.backbone." + name])
    residuals = [v for k, v in full["model"].items() if k.endswith(".lora_B")]
    if len(residuals) != 120 or not all(
        bool(torch.isfinite(v).all()) and bool(v.abs().sum() > 0) for v in residuals
    ):
        raise ValueError("all_120_declared_projections_must_learn_finitely")
    atomic_json(
        args.output,
        {
            "status": "native384_all_development_actual_recovery_accepted",
            "verification_device": args.device,
            "diagnostic_only": True,
            "clinical_accuracy_evaluated": False,
            "competition_submission_eligible": False,
            "training_studies": 17,
            "outer_studies": 0,
            "optimizer_updates": 6,
            "epochs": 2,
            "one_real_study_masked_padding_tail_exercised": True,
            "full_stop_resume_decoded_payload_equal": True,
            "full_storage_reference_decoded_model_equal": True,
            "all_numerical_updates_equal": True,
            "all_rng_optimizer_lr_counters_equal": True,
            "unchanged_original_base_tensors": len(original),
            "adapter_parameters": count,
            "changed_residual_modules": 120,
            "full_checkpoint_sha256": file_sha256(args.root / "full/checkpoint.pt"),
            "resumed_checkpoint_sha256": file_sha256(args.root / "resumed/checkpoint.pt"),
            "binding": full["binding"],
            "updates": learned_records,
        },
    )


if __name__ == "__main__":
    main()
