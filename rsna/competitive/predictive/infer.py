"""MRI-only readout for the new B/C branch; no clinical labels or reports.

This is a callable research readout, not a Kaggle packaging/runtime acceptance.
The two contexts are averaged as probabilities, never multiplied as likelihoods.
"""

from __future__ import annotations

import torch

from rsna.competitive.predictive.density import Supports
from rsna.competitive.predictive.model import Predictive, ReconstructionControl


@torch.inference_mode()
def predict(model, contexts, *, precision="fp32"):
    if not isinstance(model, (Predictive, ReconstructionControl)) or len(contexts) != 2:
        raise ValueError("One B/C MRI model and two original complementary contexts required")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("Explicit serving precision required")
    device = next(model.parameters()).device
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed-precision serving requires an admitted CUDA runtime")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is unavailable; T4 serving must be separately tested with FP16")
    model.eval()
    outputs = []
    for context in contexts:
        if set(context) != {
            "images",
            "token_metadata",
            "region_metadata",
            "targets",
            "observed",
            "supports",
        }:
            raise ValueError(
                "MRI-only source masks/descriptors/geometry; no labels or teacher inputs"
            )
        args = [
            context[n].to(device)
            for n in ("images", "token_metadata", "region_metadata", "targets", "observed")
        ]
        s = context["supports"]
        if not isinstance(s, Supports):
            raise ValueError("Full source-roster support prior required")
        s = Supports(*(x.to(device) for x in (s.indptr, s.regions, s.log_prior)), s.region_count)
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        with torch.autocast(device.type, dtype=dtype, enabled=precision != "fp32"):
            result = model(*args, s) if isinstance(model, Predictive) else model(*args)
        outputs.append(result["logits"].float().sigmoid())
    p = torch.stack(outputs).mean(0)
    if p.shape != (model.classes,) or not torch.isfinite(p).all():
        raise FloatingPointError("Invalid MRI-only prediction; no constant fallback")
    return p
