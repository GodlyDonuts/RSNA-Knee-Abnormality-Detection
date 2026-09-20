#!/usr/bin/env python3
"""Synthetic full-size throughput/memory benchmark for a conventional GPU."""

from __future__ import annotations

import argparse
import json
import time

import torch

try:
    from .model import KneeAlibiModel, KneeModelConfig
except ImportError:
    from model import KneeAlibiModel, KneeModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--series", type=int, default=8)
    parser.add_argument("--slices", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=1536)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def make_batch(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    b, r, s, f = args.batch, args.series, args.slices, args.feature_dim
    features = torch.randn(b, r, s, f, device=device, dtype=torch.float16)
    spacing = 3.0 + 0.3 * torch.rand(b, r, 1, device=device)
    positions = spacing * torch.arange(s, device=device)[None, None, :]
    positions = positions.expand(b, r, s).contiguous()
    mask = torch.ones(b, r, s, device=device, dtype=torch.bool)
    return {
        "features": features,
        "positions_mm": positions,
        "slice_mask": mask,
        "series_mask": mask.any(dim=-1),
        "plane": torch.randint(1, 4, (b, r), device=device),
        "fluid": torch.randint(0, 3, (b, r), device=device),
        "fatsat": torch.randint(0, 3, (b, r), device=device),
    }


def benchmark(
    aggregator: str,
    args: argparse.Namespace,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float | int | str]:
    torch.manual_seed(2026)
    config = KneeModelConfig(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.heads,
        series_depth=args.depth,
        study_depth=args.depth,
        dropout=0.0,
        aggregator=aggregator,
    )
    model = KneeAlibiModel(config).to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        for _ in range(args.warmup):
            logits = model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            logits = model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"{aggregator} produced non-finite logits")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "aggregator": aggregator,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "milliseconds_per_batch": 1000 * elapsed / args.iterations,
        "studies_per_second": args.batch * args.iterations / elapsed,
        "peak_allocated_gib": peak / 2**30,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("high")
    batch = make_batch(args, device)
    results = [
        benchmark(aggregator, args, batch, device)
        for aggregator in ("mean", "index_alibi", "physical_alibi")
    ]
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                "shape": {
                    "batch": args.batch,
                    "series": args.series,
                    "slices": args.slices,
                    "feature_dim": args.feature_dim,
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
