"""Keep changing/directed original edge kernels; never repair MRI geometry."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from source_chains import qualify_source_chains
from torch import Tensor
from variable_paths import REFERENCE_GAP_MM, inputs


@dataclass(frozen=True)
class VariableSourceChain:
    indices: tuple[int, ...]
    tokens: np.ndarray
    joints: Tensor
    starts: Tensor
    gaps_mm: Tensor


def qualify_variable_chains(graphs, windows, spatial_plan):
    # Reuse every original source-support/partition/token check. Failure of the
    # OLD fixed-joint restriction is retained in accounting, not an input skip.
    _, old = qualify_source_chains(graphs, windows, spatial_plan)
    operators = {
        (int(a), int(b)): group
        for group in spatial_plan.groups
        if group.kind == "repeat"
        for a, b in zip(group.first, group.second, strict=True)
    }
    paths = []
    counts = {
        "windows": len(windows),
        "chains": 0,
        "qualified_chains": 0,
        "qualified_windows": 0,
        "qualified_multistep_chains": 0,
        "qualified_repeat_edges": 0,
        "focal_singleton_chains": 0,
        "varying_joint_chains": 0,
        "nonreversible_edge_matrices": 0,
        "unsupported_windows": 0,
    }
    reasons = Counter()
    visited = set()
    for graph in graphs:
        take = np.flatnonzero(windows[:, 0] == graph["slot"])
        pairs = np.asarray(graph["repeat_edges"], dtype=np.int64).reshape(-1, 2)
        successor = {int(a): int(b) for a, b in pairs}
        incoming = set(successor.values())
        distances = {
            (int(a), int(b)): float(g)
            for (a, b), g in zip(pairs, graph["repeat_gaps_mm"], strict=True)
        }
        for first in range(len(take)):
            if first in incoming:
                continue
            source_indices = [first]
            while source_indices[-1] in successor:
                source_indices.append(successor[source_indices[-1]])
            indices = tuple(int(take[j]) for j in source_indices)
            if visited & set(indices):
                raise ValueError("Duplicate original path membership")
            visited.update(indices)
            counts["chains"] += 1
            tokens = spatial_plan.valid_tokens[indices[0]]
            starts = torch.full((len(indices), len(tokens)), 1 / len(tokens), dtype=torch.float32)
            delta = torch.tensor(
                [distances[a, b] for a, b in zip(source_indices[:-1], source_indices[1:])],
                dtype=torch.float32,
            )
            if len(indices) == 1:
                joints = torch.empty(0, len(tokens), len(tokens), dtype=torch.float32)
            else:
                groups = [operators.get(edge) for edge in zip(indices[:-1], indices[1:])]
                why = None
                if not graph["geometry_known"] or any(g is None for g in groups):
                    why = "missing_original_qualified_repeat_operator"
                elif any(not np.array_equal(tokens, g.tokens) for g in groups):
                    why = "changing_observed_token_membership_needs_separate_policy"
                elif any(g.aligned.shape != (len(tokens), len(tokens)) for g in groups):
                    raise ValueError("Original kernel shape and tokens differ")
                elif any((g.aligned.sum(1) <= 0).any() for g in groups):
                    why = "zero_outgoing_observation_state_needs_explicit_policy"
                if why is not None:
                    reasons[why] += 1
                    counts["unsupported_windows"] += len(indices)
                    continue
                joints = torch.stack([g.aligned for g in groups])
                counts["varying_joint_chains"] += int(
                    any(not torch.equal(joints[0], j) for j in joints[1:])
                )
                counts["nonreversible_edge_matrices"] += sum(
                    not torch.allclose(j, j.T, atol=1e-6, rtol=0) for j in joints
                )
            # Uniform starts over EVERY observed token. Only transitions use
            # row-normalized original kernels. Both arms share this prior;
            # no stationarity, clinical location or registration is invented.
            inputs(
                torch.zeros(len(indices), len(tokens), 1, dtype=torch.float32),
                joints,
                starts,
                delta,
                torch.zeros(1),
                REFERENCE_GAP_MM,
            )
            paths.append(VariableSourceChain(indices, tokens, joints, starts, delta))
            counts["qualified_chains"] += 1
            counts["qualified_windows"] += len(indices)
            counts["qualified_multistep_chains"] += int(len(indices) > 1)
            counts["qualified_repeat_edges"] += max(0, len(indices) - 1)
            counts["focal_singleton_chains"] += int(len(indices) == 1)
    if visited != set(range(len(windows))):
        raise ValueError("An original window was omitted")
    if counts["qualified_windows"] + counts["unsupported_windows"] != len(windows):
        raise ValueError("Incomplete original input accounting")
    return tuple(paths), {
        **counts,
        "unsupported_chain_reasons": dict(reasons),
        "old_fixed_joint_unsupported_windows": old["unsupported_windows"],
    }
