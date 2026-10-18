"""Fitting-only label-values constructor; no source admission or trainer mutation.

Callers supply independently admitted canonical probabilities and semantic states.
Raw CSV values must not be passed here as probabilities without that admission.
Private arrays stay with their owning data; only aggregate audit() is publishable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MEANINGFUL = ("positive", "negative", "probability")
STATES = (*MEANINGFUL, "uncertain", "unmentioned", "incompatible")


def ancestry(source: str, parents: dict[str, tuple[str, ...]], path=()) -> set[str]:
    if source in path or source not in parents:
        raise ValueError("Cyclic or incompletely documented ancestry")
    result = {source}
    for parent in parents[source]:
        result |= ancestry(parent, parents, (*path, source))
    return result


@dataclass(frozen=True)
class Overlay:
    targets: np.ndarray
    mask: np.ndarray
    eligible: np.ndarray
    changed: np.ndarray
    family_count: np.ndarray
    disagreement_variance: np.ndarray
    effective_weights: np.ndarray

    def audit(self, names: tuple[str, ...]) -> dict:
        if len(names) != self.targets.shape[1] or len(set(names)) != len(names):
            raise ValueError("Exact distinct target names required")
        return {
            "status": "target_construction_only_not_training_or_accuracy",
            "fitting_studies": len(self.targets),
            "mask_unchanged": True,
            "coverage_expansion_cells": 0,
            "per_target": [
                {
                    "target": name,
                    "supervised_cells": int(self.mask[:, c].sum()),
                    "safe_common_cells": int(self.eligible[:, c].sum()),
                    "changed_cells": int(self.changed[:, c].sum()),
                    "available_family_counts": {
                        str(int(k)): int((self.family_count[:, c] == k).sum())
                        for k in np.unique(self.family_count[:, c])
                    },
                    "mean_disagreement_on_safe_common_cells": (
                        float(self.disagreement_variance[self.eligible[:, c], c].mean())
                        if self.eligible[:, c].any()
                        else None
                    ),
                }
                for c, name in enumerate(names)
            ],
        }


def construct(
    baseline: np.ndarray,
    baseline_mask: np.ndarray,
    values: np.ndarray,
    states: np.ndarray,
    *,
    source_ids: tuple[str, ...],
    parents: dict[str, tuple[str, ...]],
    canonical_by_target: tuple[tuple[str, ...], ...],
    baseline_source: str,
    admitted: np.ndarray,
    fitting_ids: tuple[str, ...],
    fitting_components: tuple[str, ...],
    excluded_ids: frozenset[str],
    excluded_components: frozenset[str],
    accepted_override_mask: np.ndarray,
    weights: np.ndarray | None = None,
) -> Overlay:
    """Replace only common-meaningful, currently supervised, nonoverride cells.

    Baseline source must be a selected canonical representative for each target.
    Each selected source has disjoint documented transitive ancestry. Equal family
    weights are the default. The legacy supervised denominator/mask is unchanged.
    Neither missingness expansion nor reliability estimation is performed here.
    """
    if baseline.ndim != 2 or baseline.dtype not in (np.float32, np.float64):
        raise ValueError("Original floating baseline matrix required")
    n, c = baseline.shape
    s = len(source_ids)
    if not n or not c or not s or len(set(source_ids)) != s:
        raise ValueError("Nonempty distinct sources and target population required")
    if (
        baseline_mask.shape != (n, c)
        or baseline_mask.dtype != np.bool_
        or accepted_override_mask.shape != (n, c)
        or accepted_override_mask.dtype != np.bool_
        or values.shape != (n, c, s)
        or values.dtype not in (np.float32, np.float64)
        or states.shape != values.shape
        or admitted.shape != (c, s)
        or admitted.dtype != np.bool_
        or len(canonical_by_target) != c
    ):
        raise ValueError("Exact shapes, original bool masks and source admission required")
    if (
        len(fitting_ids) != n
        or len(fitting_components) != n
        or len(set(fitting_ids)) != n
        or any(not isinstance(x, str) or not x for x in (*fitting_ids, *fitting_components))
        or set(fitting_ids) & excluded_ids
        or set(fitting_components) & excluded_components
    ):
        raise ValueError("Unique fitting identities and disjoint evaluation components required")
    if not np.isin(states, STATES).all():
        raise ValueError("Explicit source-specific semantic states required")
    meaningful = np.isin(states, MEANINGFUL)
    observed = meaningful & admitted[None, :, :]
    if (
        not np.isfinite(values[meaningful]).all()
        or ((values[meaningful] < 0) | (values[meaningful] > 1)).any()
        or (values[states == "positive"] != 1).any()
        or (values[states == "negative"] != 0).any()
        or not np.isfinite(baseline[baseline_mask]).all()
        or ((baseline[baseline_mask] < 0) | (baseline[baseline_mask] > 1)).any()
    ):
        raise ValueError("Canonical probabilities required; no clipping or severity inference")
    selected = np.zeros((c, s), dtype=bool)
    closures = {name: ancestry(name, parents) for name in source_ids}
    for j, group in enumerate(canonical_by_target):
        if not group or len(set(group)) != len(group) or baseline_source not in group:
            raise ValueError("Canonical sources must include the baseline once")
        seen: set[str] = set()
        for name in group:
            if name not in source_ids or not admitted[j, source_ids.index(name)]:
                raise ValueError("Unadmitted canonical source")
            if seen & closures[name]:
                raise ValueError("Mirror, ancestor or overlapping blend counted twice")
            seen |= closures[name]
            selected[j, source_ids.index(name)] = True
    b = source_ids.index(baseline_source)
    comparable = baseline_mask & observed[:, :, b]
    if not np.array_equal(baseline[comparable], values[:, :, b][comparable]):
        raise ValueError("Meaningful baseline source differs from original targets")
    if weights is None:
        weights = np.ones((c, s), dtype=np.float64)
    if (
        weights.shape != (c, s)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or (weights[selected] <= 0).any()
    ):
        raise ValueError("Predeclared positive canonical family weights required")
    active = observed & selected[None, :, :]
    mass = active * weights[None, :, :]
    denominator = mass.sum(axis=2, keepdims=True)
    normalized = np.divide(mass, denominator, out=np.zeros_like(mass), where=denominator > 0)
    # Unavailable raw placeholders can be NaN or arbitrary sentinels: zero before math.
    safe = np.where(active, values, 0)
    q = (safe * normalized).sum(axis=2)
    count = active.sum(axis=2)
    variance = (normalized * (safe - q[:, :, None]) ** 2).sum(axis=2)
    eligible = comparable & (count >= 2) & ~accepted_override_mask
    result = baseline.copy()
    result[eligible] = q[eligible].astype(baseline.dtype)
    changed = eligible & (result != baseline)
    return Overlay(result, baseline_mask.copy(), eligible, changed, count, variance, normalized)
