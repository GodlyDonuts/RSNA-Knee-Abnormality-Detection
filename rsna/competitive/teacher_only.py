"""Isolated zero-source-weight ablation over the accepted paired-view trainer.

Original labels/masks remain validated and unchanged, but contribute zero loss.
This is a falsifiable supervision ablation, not a claim that source labels are
wrong or that teacher predictions are clinical truth. No live module is edited.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from rsna.competitive import teacher_pair
from rsna.competitive.teacher_pair import Recipe as ReferenceRecipe

VERSION = "full_encoder_two_view_zero_source_weight_ablation_v1"
ENGINE_SHA = "8384b578724022a7951ee084a45e27ac54dd038c8af84e57dbca7310f56dd3c3"


@dataclass(frozen=True)
class Recipe(ReferenceRecipe):
    source_weight: float = 0.0

    def __post_init__(self):
        if type(self.source_weight) is not float or self.source_weight != 0.0:
            raise ValueError("Explicit zero source contribution required")
        if self.objective != "pointwise":
            raise ValueError("Only the matched pointwise source-weight ablation is admitted")
        # Reuse every original recipe constraint without weakening that class's
        # positive-source-weight contract or changing any other field.
        ReferenceRecipe(**{**asdict(self), "source_weight": 0.25})


def teacher_loss(logits, parent_logits, labels, mask, *, source_weight):
    if (
        type(source_weight) is not float
        or source_weight != 0.0
        or logits.ndim != 2
        or not logits.numel()
        or any(x.shape != logits.shape for x in (parent_logits, labels, mask))
        or mask.dtype != torch.bool
        or any(x.device != logits.device for x in (parent_logits, labels, mask))
        or not torch.isfinite(logits).all()
        or not torch.isfinite(parent_logits).all()
        or not torch.isfinite(labels[mask]).all()
        or ((labels[mask] < 0) | (labels[mask] > 1)).any()
    ):
        raise ValueError("Exact finite original inputs/masks and zero-source ablation required")
    return F.binary_cross_entropy_with_logits(logits, parent_logits.detach().sigmoid())


@contextlib.contextmanager
def _isolated_objective():
    """A new worker process alone may substitute and must restore its objective."""
    original = teacher_pair.anchored_loss
    if original is teacher_loss:
        raise ValueError("Nested or already active objective substitution is not allowed")
    teacher_pair.anchored_loss = teacher_loss
    try:
        yield
    finally:
        teacher_pair.anchored_loss = original


def fit(model, data, output, binding, recipe, *, engine, **kwargs):
    """Call the exact accepted engine in a separate source-bound worker process.

    This does not dispatch a GPU, admit pixels or authorize competition use.
    Actual full/stop/resume acceptance is required before production fitting.
    """
    if not isinstance(recipe, Recipe):
        raise ValueError("Separate ablation recipe required; never coerce a live reference")
    namespace = getattr(engine, "__globals__", {})
    path = Path(namespace.get("__file__", "/missing-original-engine"))
    if (
        not callable(engine)
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != ENGINE_SHA
        or namespace.get("coupled_batch") is not teacher_pair.coupled_batch
        or namespace.get("Recipe") is not ReferenceRecipe
    ):
        raise ValueError("Exact accepted original one-rank numerical engine required")
    extra = {
        "source_supervision_ablation": VERSION,
        "observed_source_loss_weight": 0.0,
        "source_labels_masks_and_population_unchanged": True,
    }
    if any(k in binding for k in extra):
        raise ValueError("Fresh explicit ablation provenance required")
    with _isolated_objective():
        return engine(model, data, output, {**binding, **extra}, recipe, **kwargs)
