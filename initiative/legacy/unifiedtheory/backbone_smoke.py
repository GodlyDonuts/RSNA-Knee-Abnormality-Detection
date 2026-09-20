#!/usr/bin/env python3
"""Download/load a real DINO backbone and validate the extraction interface."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

try:
    from .extract_features import DinoFeatureExtractor
except ImportError:
    from extract_features import DinoFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    generator = np.random.default_rng(2026)
    images = [generator.integers(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(5)]
    extractor = DinoFeatureExtractor(
        args.model_name,
        args.device,
        batch_size=3,
        local_files_only=args.local_files_only,
    )
    encoded = extractor.encode(images)
    if encoded.shape[0] != len(images) or encoded.ndim != 2:
        raise AssertionError(f"unexpected feature shape {tuple(encoded.shape)}")
    if not torch.isfinite(encoded).all():
        raise AssertionError("DINO produced non-finite features")
    print(
        json.dumps(
            {
                "status": "pass",
                "model": args.model_name,
                "device": args.device,
                "shape": list(encoded.shape),
                "dtype": str(encoded.dtype),
                "mean_l2_norm": float(encoded.norm(dim=-1).mean()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
