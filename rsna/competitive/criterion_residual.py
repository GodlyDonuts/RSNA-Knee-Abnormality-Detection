"""Criterion-relational residual on unchanged original N3 window features.

Local evidence is learned model evidence, not a lesion annotation. The branch
adds distinct-source corroboration and ordered variation to image-only moments.
An aligned/sham comparison must share ALL parameters, features and nuisance
inputs; only feature-to-within-series-geometry assignment differs. No cross-plane
registration or lesion area/volume is inferred by these operators.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from rsna.competitive.criterion_geometry import WindowRelations


def _logmeanexp(values: Tensor) -> Tensor:
    return torch.logsumexp(values, dim=0) - math.log(len(values))


class CriterionResidual(nn.Module):
    """Fixed-base residual; no encoder or base-reader tensors belong to this head."""

    def __init__(self, dimension: int = 1024, width: int = 64):
        super().__init__()
        if type(dimension) is not int or dimension < 1 or type(width) is not int or width < 1:
            raise ValueError("Positive representation dimensions required")
        self.dimension, self.width = dimension, width
        self.project = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, width), nn.GELU()
        )
        self.evidence = nn.Linear(width, 12)
        self.variation = nn.Linear(width, 12, bias=False)
        # Seven image/relation and four acquisition nuisance statistics. Zero
        # output preserves the actual base before fitting, in BOTH arms.
        self.output_weight = nn.Parameter(torch.zeros(12, 11))
        self.output_bias = nn.Parameter(torch.zeros(12))

    def statistics(
        self,
        features: Sequence[Tensor],
        graphs: Sequence[WindowRelations],
        permutations: Sequence[np.ndarray] | None = None,
    ) -> Tensor:
        if (
            not features
            or len(features) != len(graphs)
            or (permutations is not None and len(permutations) != len(features))
        ):
            raise ValueError("One source-support graph per nonempty represented series required")
        global_evidence, repeat, continuity, variation = [], [], [], []
        spans, known, total_nodes, total_edges = 0.0, 0, 0, 0
        for index, (x, graph) in enumerate(zip(features, graphs, strict=True)):
            if (
                not isinstance(graph, WindowRelations)
                or x.ndim != 2
                or x.shape != (len(graph.supports), self.dimension)
                or not len(x)
                or x.dtype != torch.float32
                or not torch.isfinite(x).all()
            ):
                raise ValueError(
                    "Finite original FP32 window features and exact support graph required"
                )
            h = self.project(x)
            e = self.evidence(h)
            # Global image moments remain EXACTLY unchanged by the sham. Only
            # relational terms see the counterfactual geometry assignment.
            global_evidence.append(e)
            if permutations is not None:
                order = permutations[index]
                if (
                    not isinstance(order, np.ndarray)
                    or not np.issubdtype(order.dtype, np.integer)
                    or order.shape != (len(x),)
                    or not np.array_equal(np.sort(order), np.arange(len(x)))
                ):
                    raise ValueError("Sham must be a complete within-series permutation")
                take = torch.as_tensor(order, device=x.device, dtype=torch.long)
                h, e = h[take], e[take]
            total_nodes += len(x)
            known += int(graph.geometry_known)
            if graph.geometry_known:
                spans += float(np.ptp(graph.centers_mm))
            for edges, gaps, kind in (
                (graph.repeat_edges, graph.repeat_gaps_mm, "repeat"),
                (graph.continuity_edges, graph.continuity_gaps_mm, "continuity"),
            ):
                if (
                    edges.ndim != 2
                    or edges.shape[1] != 2
                    or gaps.shape != (len(edges),)
                    or (edges < 0).any()
                    or (edges >= len(x)).any()
                    or not np.isfinite(gaps).all()
                    or (gaps <= 0).any()
                    or (len(edges) and not graph.geometry_known)
                ):
                    raise ValueError(
                        "Valid within-series physical edges and positive actual gaps required"
                    )
                if not len(edges):
                    continue
                pair = torch.as_tensor(edges, device=x.device, dtype=torch.long)
                first, second = pair[:, 0], pair[:, 1]
                distance = torch.as_tensor(gaps, device=x.device, dtype=torch.float32)
                weight = torch.exp(-distance / 10)
                if kind == "repeat":
                    if any(set(graph.supports[i]) & set(graph.supports[j]) for i, j in edges):
                        raise ValueError(
                            "Repeat evidence requires disjoint ORIGINAL slice supports"
                        )
                    repeat.append(((e[first] + e[second]) * 0.5, weight))
                else:
                    total_edges += len(edges)
                    continuity.append((e[first].sigmoid() * e[second].sigmoid(), weight))
                    delta = torch.abs(h[first] - h[second]) / math.sqrt(self.width)
                    variation.append((self.variation(delta), weight))
        all_e = torch.cat(global_evidence)
        zero = all_e.sum(0) * 0

        def weighted(rows):
            if not rows:
                return zero
            values = torch.cat([value for value, _ in rows])
            weights = torch.cat([weight for _, weight in rows])
            return (values * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-12)

        def weighted_logmean(rows):
            if not rows:
                return zero
            values = torch.cat([value for value, _ in rows])
            log_weights = torch.cat([weight for _, weight in rows]).clamp_min(1e-30).log()
            return torch.logsumexp(values + log_weights[:, None], 0) - torch.logsumexp(
                log_weights, 0
            )

        learned = [
            all_e.mean(0),
            all_e.std(0, unbiased=False),
            _logmeanexp(all_e),
            all_e.sigmoid().mean(0),
            weighted_logmean(repeat),
            weighted(continuity),
            weighted(variation),
        ]
        nuisance = [
            math.log1p(total_nodes),
            math.log1p(total_edges),
            math.log1p(spans),
            known / len(graphs),
        ]
        return torch.stack(learned + [all_e.new_full((12,), value) for value in nuisance], dim=1)

    def forward(
        self,
        features: Sequence[Tensor],
        graphs: Sequence[WindowRelations],
        permutations: Sequence[np.ndarray] | None = None,
    ) -> Tensor:
        values = self.statistics(features, graphs, permutations)
        correction = (values * self.output_weight).sum(1) + self.output_bias
        return 2 * torch.tanh(correction / 2)
