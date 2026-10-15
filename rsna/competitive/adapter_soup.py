"""CPU-only effective-update averaging, compressed to the original adapter rank.

For identical policies/scaling, average BA rather than the separate B and A
factors. QR and a small-core SVD avoid materializing dense pretrained-sized
updates. This is inference-only math, not a checkpoint loader or trained model.
Source/base/schema/parent provenance must be checked by a separate admission.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def _core(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """left @ right = q_left @ small_core @ q_right.T."""
    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T, mode="reduced")
    return q_left, r_left @ r_right.T, q_right


@torch.no_grad()
def average_adapter_updates(
    parents: Sequence[tuple[Tensor, Tensor]], *, scaling: float
) -> tuple[Tensor, Tensor, dict[str, float | int]]:
    """Return (A, B, diagnostics) for equal-weight rank-preserving mean updates.

    Parents are (A[r,in], B[out,r]) on CPU, all with the same floating dtype,
    shapes and externally admitted policy. Runtime applies the original common
    scaling once; it is not folded into these factors. No optimizer or RNG is
    created, and inputs are neither mutated nor aliased by the outputs.
    """
    if not 1 <= len(parents) <= 10:
        raise ValueError("One to ten explicitly admitted parents required")
    if (
        isinstance(scaling, bool)
        or not isinstance(scaling, (int, float))
        or not math.isfinite(scaling)
        or scaling <= 0
    ):
        raise ValueError("Common adapter scaling must be finite and positive")
    first_a, first_b = parents[0]
    if first_a.ndim != 2 or first_b.ndim != 2:
        raise ValueError("Adapter factors must be matrices")
    rank, width = first_a.shape
    height, inner = first_b.shape
    dtype = first_a.dtype
    if not 1 <= rank <= min(width, height, 256) or inner != rank:
        raise ValueError("Invalid original rank or factor dimensions")
    for a, b in parents:
        if (
            a.device.type != "cpu"
            or b.device.type != "cpu"
            or a.dtype not in {torch.float32, torch.float64}
            or a.dtype != dtype
            or b.dtype != dtype
            or a.shape != first_a.shape
            or b.shape != first_b.shape
            or not torch.isfinite(a).all()
            or not torch.isfinite(b).all()
        ):
            raise ValueError("Parent shapes, CPU dtypes and finite values must match")
    divisor = math.sqrt(len(parents))
    left = torch.cat([b.detach().double() / divisor for _, b in parents], dim=1)
    right = torch.cat([a.detach().double() / divisor for a, _ in parents], dim=0)
    q_left, core, q_right = _core(left, right)
    if not torch.isfinite(core).all():
        raise ValueError("Finite factors overflowed the mean-update core")
    u, singular, vh = torch.linalg.svd(core, full_matrices=False)
    energy = float(singular.square().sum())
    same = all(torch.equal(a, first_a) and torch.equal(b, first_b) for a, b in parents)
    if same:
        result_a, result_b = first_a.detach().clone(), first_b.detach().clone()
    else:
        root = singular[:rank].sqrt()
        result_b = ((q_left @ u[:, :rank]) * root).to(dtype)
        result_a = ((root[:, None] * vh[:rank]) @ q_right.T).to(dtype)
    # Include post-cast quantization error without forming a dense height x width delta.
    _, error_core, _ = _core(
        torch.cat([left, result_b.double()], dim=1),
        torch.cat([right, -result_a.double()], dim=0),
    )
    norm = math.sqrt(energy) * scaling
    error = float(torch.linalg.vector_norm(error_core)) * scaling
    diagnostics: dict[str, float | int] = {
        "parents": len(parents),
        "rank": rank,
        "scaling": scaling,
        "mean_delta_frobenius_norm": norm,
        "optimal_truncation_frobenius_norm": float(singular[rank:].norm()) * scaling,
        "postcast_residual_frobenius_norm": error,
        "postcast_relative_error": error / norm if norm else 0.0,
        "singular_energy_retained": float(singular[:rank].square().sum()) / energy
        if energy
        else 1.0,
    }
    if not all(math.isfinite(value) for value in diagnostics.values()):
        raise ValueError("Merged update diagnostics overflowed")
    return result_a, result_b, diagnostics
