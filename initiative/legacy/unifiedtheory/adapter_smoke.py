#!/usr/bin/env python3
"""Validate the real DINO adapter and end-to-end hierarchy on CUDA."""

from __future__ import annotations

import argparse
import json

import torch

try:
    from .dino_adapter import DinoSliceAdapter, EndToEndPatchKneeModel
    from .patch_model import PatchKneeModelConfig
except ImportError:
    from dino_adapter import DinoSliceAdapter, EndToEndPatchKneeModel
    from patch_model import PatchKneeModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    device = torch.device(args.device)
    encoder = DinoSliceAdapter.from_pretrained(
        args.model_name,
        patch_grid=4,
        adapter_bottleneck=32,
        trainable_blocks=0,
        local_files_only=args.local_files_only,
    )
    config = PatchKneeModelConfig(
        feature_dim=2 * encoder.hidden_dim,
        patch_dim=encoder.hidden_dim,
        hidden_dim=32,
        n_heads=4,
        series_depth=1,
        study_depth=1,
        dropout=0.0,
        num_targets=3,
        series_dropout=0.0,
    )
    model = EndToEndPatchKneeModel(encoder, config, encoder_batch_size=2).to(device).train()
    pixels = torch.randn(1, 1, 2, 3, 224, 224, device=device)
    inputs = {
        "pixel_values": pixels,
        "positions_mm": torch.tensor([[[0.0, 3.0]]], device=device),
        "slice_mask": torch.ones(1, 1, 2, dtype=torch.bool, device=device),
        "series_mask": torch.ones(1, 1, dtype=torch.bool, device=device),
        "plane": torch.ones(1, 1, dtype=torch.long, device=device),
        "fluid": torch.full((1, 1), 2, dtype=torch.long, device=device),
        "fatsat": torch.full((1, 1), 2, dtype=torch.long, device=device),
    }
    logits = model(**inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([[1.0, 0.0, 1.0]], device=device)
    )
    loss.backward()
    frozen_gradients = sum(
        parameter.grad is not None for parameter in model.slice_encoder.backbone.parameters()
    )
    adapter_gradients = sum(
        parameter.grad is not None for parameter in model.slice_encoder.adapter.parameters()
    )
    if frozen_gradients:
        raise AssertionError("frozen DINO parameters received gradients")
    if not adapter_gradients:
        raise AssertionError("residual adapter received no gradients")
    if not torch.isfinite(logits).all():
        raise AssertionError("end-to-end model produced non-finite logits")
    print(
        json.dumps(
            {
                "status": "pass",
                "model": args.model_name,
                "device": str(device),
                "logit_shape": list(logits.shape),
                "loss": float(loss.detach()),
                "adapter_gradient_tensors": adapter_gradients,
                "parameter_report": model.trainable_parameter_report(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
