"""Training-weight diagnostics, not a measure of MRI accuracy or feature drift."""

import math

import torch


def effective_update_norm(a, b, base, scaling):
    if (
        a.ndim != 2
        or b.ndim != 2
        or base.ndim != 2
        or b.shape[1] != a.shape[0]
        or base.shape != (b.shape[0], a.shape[1])
    ):
        raise ValueError("Aligned actual A/B/base projection matrices required")
    if not isinstance(scaling, (int, float)) or not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("Positive declared residual scaling required")
    if any(
        not value.is_floating_point() or not torch.isfinite(value).all() for value in (a, b, base)
    ):
        raise ValueError("Finite floating original projection weights required")
    a, b = a.detach().double(), b.detach().double()
    # ||BA||F^2 = trace((B^T B)(A A^T)); never reconstruct the large dense update.
    squared = float(((b.T @ b) * (a @ a.T)).sum())
    if squared < -1e-12:
        raise ValueError("Unexpected negative squared update norm")
    update = scaling * math.sqrt(max(squared, 0))
    original = float(torch.linalg.vector_norm(base.detach().double()))
    if original <= 0:
        raise ValueError("Original pretrained matrix must have positive norm")
    return {
        "effective_update_Frobenius_norm": update,
        "base_Frobenius_norm": original,
        "relative_effective_update_Frobenius_norm": update / original,
        "rank": a.shape[0],
    }
