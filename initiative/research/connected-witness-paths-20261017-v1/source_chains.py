"""Admit source-disjoint chains without pretending unsupported geometry passed.

This is a metadata/representation check, not an MRI predictor or registration.
Every original window remains accounted for, including unqualified chains.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from prototype import _joint
from torch import Tensor


@dataclass(frozen=True)
class SourceChain:
    indices: tuple[int, ...]
    tokens: np.ndarray
    joint: Tensor
    gaps_mm: Tensor


def qualify_source_chains(graphs: list[dict], windows: np.ndarray, spatial_plan):
    """Return only genuinely admitted paths plus complete nonclinical accounting.

    Repeat links partition each series into offset3 chains. All original source
    slices within a chain must be distinct. Changing/nonreversible joints are
    reported as unsupported; they are never symmetrized or silently repaired.
    A singleton has no spatial transition and is explicitly counted as focal.
    """
    if (
        windows.ndim != 2
        or windows.shape[1] != 2
        or not np.issubdtype(windows.dtype, np.integer)
        or len(windows) != spatial_plan.windows
        or len(set(map(tuple, windows.tolist()))) != len(windows)
    ):
        raise ValueError("Complete distinct original window membership required")
    operators = {}
    for group in spatial_plan.groups:
        if group.kind != "repeat":
            continue
        for first, second in zip(group.first, group.second, strict=True):
            edge = (int(first), int(second))
            if edge in operators:
                raise ValueError("An original repeat edge has multiple spatial operators")
            operators[edge] = group
    paths = []
    reasons = Counter()
    accounting = {
        "series": 0,
        "windows": len(windows),
        "chains": 0,
        "qualified_chains": 0,
        "qualified_windows": 0,
        "qualified_multistep_chains": 0,
        "qualified_repeat_edges": 0,
        "focal_singleton_chains": 0,
        "unsupported_windows": 0,
    }
    visited = set()
    for graph in graphs:
        take = np.flatnonzero(windows[:, 0] == graph["slot"])
        supports = np.asarray(graph["supports"], dtype=np.int64).reshape(-1, 3)
        selected = np.asarray(graph["selected_source_indices"], dtype=np.int64)
        expected = np.asarray(
            [selected[j : j + 3] for j in range(max(0, len(selected) - 2))], dtype=np.int64
        ).reshape(-1, 3)
        if (
            selected.ndim != 1
            or not len(selected)
            or (selected < 0).any()
            or not np.array_equal(supports, expected)
            or len(take) != len(supports)
            or not np.array_equal(windows[take, 1], np.arange(len(supports)))
            or len(set(selected.tolist())) != len(selected)
            or (np.diff(selected) <= 0).any()
        ):
            raise ValueError("Original selected-source supports and windows differ")
        if not len(take):
            continue
        accounting["series"] += 1
        edges = np.asarray(graph["repeat_edges"], dtype=np.int64).reshape(-1, 2)
        distances = np.asarray(graph["repeat_gaps_mm"], dtype=float)
        centers = np.asarray(graph["centers_mm"], dtype=float)
        if (
            distances.shape != (len(edges),)
            or centers.shape != (len(take),)
            or not np.isfinite(centers).all()
            or not np.isfinite(distances).all()
            or (distances <= 0).any()
            or (edges < 0).any()
            or (edges >= len(take)).any()
            or any(second != first + 3 for first, second in edges)
            or len(set(map(tuple, edges.tolist()))) != len(edges)
            or not np.allclose(
                centers[edges[:, 1]] - centers[edges[:, 0]], distances, rtol=0, atol=1e-8
            )
        ):
            raise ValueError("Exact positive original offset3 distances required")
        successor = {int(a): int(b) for a, b in edges}
        incoming = set(successor.values())
        gaps = {(int(a), int(b)): float(d) for (a, b), d in zip(edges, distances, strict=True)}
        for first in range(len(take)):
            if first in incoming:
                continue
            chain = [first]
            while chain[-1] in successor:
                chain.append(successor[chain[-1]])
            indices = tuple(int(take[j]) for j in chain)
            source_seen = set()
            for j in chain:
                source = set(supports[j].tolist())
                if len(source) != 3 or source_seen & source:
                    raise ValueError("A path reuses an original MRI slice")
                source_seen.update(source)
            if visited & set(indices):
                raise ValueError("Original source-window partition overlaps")
            visited.update(indices)
            accounting["chains"] += 1
            tokens = spatial_plan.valid_tokens[indices[0]]
            if (
                tokens.ndim != 1
                or not len(tokens)
                or len(set(tokens.tolist())) != len(tokens)
                or (tokens < 0).any()
                or (tokens >= spatial_plan.height * spatial_plan.width).any()
            ):
                raise ValueError("Complete valid nominal image-token membership required")
            if any(not np.array_equal(tokens, spatial_plan.valid_tokens[j]) for j in indices):
                raise ValueError("Changing source-token representation needs a separate model")
            why = None
            if len(chain) == 1:
                # No transition is used: a focal path is not corroboration.
                marginal = torch.full((len(tokens),), 1 / len(tokens), dtype=torch.float32)
                joint = marginal[:, None] * marginal[None, :]
            else:
                groups = [operators.get(edge) for edge in zip(indices[:-1], indices[1:])]
                if not graph["geometry_known"] or any(g is None for g in groups):
                    why = "missing_original_qualified_repeat_operator"
                elif any(not np.array_equal(tokens, g.tokens) for g in groups):
                    why = "changing_spatial_token_membership"
                else:
                    joint = groups[0].aligned
                    if joint.shape != (len(tokens), len(tokens)):
                        raise ValueError("Source operator and nominal tokens differ")
                    if any(not torch.equal(joint, g.aligned) for g in groups[1:]):
                        why = "changing_joint_not_supported_by_fixed_joint_prototype"
                    else:
                        try:
                            normalized, marginal = _joint(joint)
                            if not torch.allclose(normalized.sum(0), marginal, atol=1e-6, rtol=0):
                                why = "nonstationary_weighted_marginal_not_supported"
                        except ValueError:
                            why = "nonreversible_or_invalid_joint_not_supported"
            if why is not None:
                reasons[why] += 1
                accounting["unsupported_windows"] += len(chain)
                continue
            delta = torch.tensor(
                [gaps[a, b] for a, b in zip(chain[:-1], chain[1:])], dtype=joint.dtype
            )
            paths.append(SourceChain(indices, tokens, joint, delta))
            accounting["qualified_chains"] += 1
            accounting["qualified_windows"] += len(chain)
            accounting["focal_singleton_chains"] += int(len(chain) == 1)
            accounting["qualified_multistep_chains"] += int(len(chain) > 1)
            accounting["qualified_repeat_edges"] += max(0, len(chain) - 1)
    if visited != set(range(len(windows))):
        raise ValueError("Every original source window must be accounted for")
    if accounting["qualified_windows"] + accounting["unsupported_windows"] != len(windows):
        raise ValueError("Incomplete original input accounting")
    return tuple(paths), {**accounting, "unsupported_chain_reasons": dict(reasons)}
