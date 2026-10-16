"""Predeclared inference-only marginalization of the admitted twelve-window sampler.

Encode each candidate once; compare ordinary full-bag attention with the mean
probability over 128 deterministic training-shaped bags. No labels or fitting.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rsna.competitive.compact_bag import (
    CompactBagClassifier,
    candidate_windows,
    sample_windows,
)

BAGS = 128
VIEW_OFFSET = 10_000
RUN_SEED = 20_260_922


@torch.inference_mode()
def predict_both(
    model: CompactBagClassifier,
    windows: Tensor,
    series_mask: Tensor,
    slice_mask: Tensor,
    *,
    study_uid: str,
    chunk_size: int = 12,
) -> dict[str, Tensor | int]:
    """Return full and MC probabilities from exactly one shared encoder pass.

    ``windows`` must follow candidate_windows canonical order and use the same
    normalization as the admitted evaluator. This helper preserves its precision
    policy through the model's ordinary forward rather than reimplementing it.
    """
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("Saved-statistic evaluation mode required for every module")
    candidates = candidate_windows(series_mask, slice_mask)
    if len(windows) != len(candidates):
        raise ValueError("Exactly one normalized window per canonical candidate required")
    if not isinstance(study_uid, str) or not study_uid:
        raise ValueError("Nonempty study identity required")
    features: list[Tensor] = []

    def capture(_module, _inputs, output):
        if not isinstance(output, Tensor) or output.ndim != 2:
            raise ValueError("Expected encoder matrix output")
        features.append(output.detach().float())

    handle = model.encoder.register_forward_hook(capture)
    try:
        full = model(windows, chunk_size=chunk_size).sigmoid().float()
    finally:
        handle.remove()
    encoded = torch.cat(features)
    if len(encoded) != len(candidates) or not torch.isfinite(encoded).all():
        raise ValueError("Incomplete or nonfinite shared encoder features")
    if len(candidates) <= 12:
        return {"full": full, "mc128": full.clone(), "encoded_windows": len(encoded)}
    lookup = {window: i for i, window in enumerate(candidates)}
    indices = torch.tensor(
        [
            [
                lookup[window]
                for window in sample_windows(
                    series_mask,
                    slice_mask,
                    run_seed=RUN_SEED,
                    epoch=VIEW_OFFSET + view,
                    study_uid=study_uid,
                )
            ]
            for view in range(BAGS)
        ],
        device=encoded.device,
    )
    with torch.autocast(encoded.device.type, enabled=False):
        h = model.norm(encoded)
        scores = model.attention(h)
        selected_h, selected_scores = h[indices], scores[indices]
        pooled = torch.einsum("bkt,bkd->btd", selected_scores.softmax(dim=1), selected_h)
        logits = torch.cat([head(pooled[:, i]) for i, head in enumerate(model.classifiers)], dim=1)
        mc = logits.sigmoid().mean(dim=0)
    if not torch.isfinite(full).all() or not torch.isfinite(mc).all():
        raise ValueError("Nonfinite inference probabilities")
    return {"full": full, "mc128": mc, "encoded_windows": len(encoded)}
