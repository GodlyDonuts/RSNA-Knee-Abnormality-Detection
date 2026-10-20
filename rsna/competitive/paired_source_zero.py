"""Separate fixed-IID16 paired-view zero-source recipe for one/two GPU ranks.

The original positive-weight recipe and live trainers remain unchanged. This
module changes no labels, masks, teacher values or sampling probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rsna.competitive.licensed_importance import Recipe as ReferenceRecipe

VERSION = "paired_IID16_zero_source_one_or_two_rank_v1"


@dataclass(frozen=True)
class Recipe(ReferenceRecipe):
    role: str = "influence"
    source_weight: float = 0.0

    def __post_init__(self):
        if (
            type(self.source_weight) is not float
            or self.source_weight != 0.0
            or self.role != "influence"
        ):
            raise ValueError("Explicit zero-source, unchanged influence proposal recipe required")
        ReferenceRecipe(**{**asdict(self), "source_weight": 0.25})
