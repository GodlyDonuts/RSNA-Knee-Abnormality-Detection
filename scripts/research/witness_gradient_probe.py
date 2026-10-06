"""Training-mechanism audit; no MRI labels or clinical accuracy are evaluated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rsna.competitive.witness import pool_normalized_witness_evidence, pool_witness_evidence

normalized_smooth_evidence = pool_normalized_witness_evidence


def probe():
    torch.set_num_threads(1)
    shape = (1, 6, 16, 24 * 24, 12)
    mask = torch.ones(shape[:3], dtype=torch.bool)
    result = {}
    for name, pooling in [
        ("original_hard_top4_patch_top2_slice", lambda x: pool_witness_evidence(x, mask, 4, 2)),
        ("standalone_normalized_smooth_beta8", lambda x: normalized_smooth_evidence(x, mask, 8)),
    ]:
        scores = torch.zeros(shape, requires_grad=True)
        output = pooling(scores)
        output.sum().backward()
        nonzero = scores.grad.ne(0)
        result[name] = {
            "output_exactly_zero": bool(output.eq(0).all()),
            "finite_gradients": bool(torch.isfinite(scores.grad).all()),
            "genuine_score_cells": scores.numel(),
            "nonzero_gradient_cells": int(nonzero.sum()),
            "score_gradient_coverage_fraction": float(nonzero.float().mean()),
            "gradient_sum_per_target": scores.grad.sum(dim=(0, 1, 2, 3)).tolist(),
        }
    return {
        "status": "actual_zero_score_pooling_gradient_probe",
        "score_shape": list(shape),
        "cases": result,
        "scope": "Score-pooling derivative at the exact zero-initialized branch. No MRI pixels, model fitting, labels, held-out metrics or clinical conclusion.",
        "clinical_accuracy_evidence": False,
        "live_scientific_source_modified": False,
        "alternative_model_route_integrated": True,
        "actual_GPU_or_clinical_acceptance": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = probe()
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))
