"""Use the proven bounded MRI path with a separately typed confidence component.

All original source hashes are enforced by the original compiler. Only its
whitelisted component schema/loader block changes; MRI math and queue limits do
not. No accepted original notebook or running controller is modified.
"""

from __future__ import annotations

import hashlib

from scripts.kaggle.released_context_notebook import _build as original_build
from scripts.kaggle.released_context_notebook import bodies
from scripts.kaggle.released_context_notebook import build as original_public_build

INFER_SHA = "5ec75773bc70a9a1dd8666a87e7c67dba73f97b5362fe3608fe176166a800d48"
NAMES = {
    "FORMAT",
    "SOURCE_GIT",
    "PARENT_SOURCE",
    "HELPER_SHA",
    "POPULATION_SHA",
    "LABEL_SOURCE_SHA",
    "FEATURE_SOURCE_SHA",
    "PROVENANCE_KEYS",
    "sha256",
    "is_sha",
    "model_from_component",
    "load_component",
    "predict_context_pair",
}


def build(parent, context, trainer, old_infer, bounded, infer, component_sha, **kwargs):
    source, binding = original_public_build(
        parent, context, trainer, old_infer, bounded, component_sha, **kwargs
    )
    return _replace(source, binding, old_infer, infer)


def _build(parent, context, trainer, old_infer, bounded, infer, component_sha, **kwargs):
    """Synthetic-parent engineering entry; public build still pins the parent."""
    source, binding = original_build(
        parent, context, trainer, old_infer, bounded, component_sha, **kwargs
    )
    return _replace(source, binding, old_infer, infer)


def _replace(source, binding, old_infer, infer):
    if hashlib.sha256(infer.encode()).hexdigest() != INFER_SHA:
        raise ValueError("Exact separately typed confidence serving source required")
    old = bodies(old_infer, selected=NAMES)
    new = bodies(infer, selected=NAMES | {"VERSION", "ORIGINAL_SOURCE_GIT", "FIXED_RECIPE"})
    if source.count(old) != 1:
        raise ValueError("Unique original component loader block required")
    source = source.replace(old, new)
    compile(source, "licensed_confidence_context_notebook", "exec")
    return source, {
        **binding,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        "original_serving_module_sha256": binding["serving_module_sha256"],
        "serving_module_sha256": INFER_SHA,
        "objective": "source_confidence",
        "separately_typed_confidence_component": True,
        "original_bounded_MRI_path_unchanged": True,
    }
