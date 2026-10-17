"""Compile label-only fixed-final inference through unchanged full94 serving."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import FunctionType

from rsna.competitive import licensed_label_serving as label_scope
from scripts.kaggle import licensed_importance_serving as original

BASE_HELPER_SHA = "e4c7a71ec219bbe645274af0c72dde851fcd2b7990b2d4888cf2b16b93b43f8d"
LABEL_HELPER_SHA = "c82147533e0e59722b953d37dfdd70427425caa1e6cae9883541986417aec609"
BASE_COMPILER_SHA = "9b1798683fe21cbc36cbceb48837ee7a4e7da3e464d238e683636163939d04c2"


def standalone(base, label):
    """Keep base function globals isolated while extending only provenance tags."""
    if original.digest(base) != BASE_HELPER_SHA or original.digest(label) != LABEL_HELPER_SHA:
        raise ValueError("Exact original and fixed label-scope helper sources required")
    parts = []
    removed = 0
    for node in ast.parse(label).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "rsna.competitive.licensed_importance_serving"
        ):
            removed += 1
            continue
        text = ast.get_source_segment(label, node)
        if text is None:
            raise ValueError("Complete label scope source span required")
        parts.append(text)
    if removed != 2:
        raise ValueError("Exact isolated provenance and model validators required")
    header = (
        "_base_serving_namespace={}\n"
        f"exec(compile({base!r},'unchanged_full94_base_validation','exec'),"
        "_base_serving_namespace)\n"
        "base_provenance=_base_serving_namespace['provenance']\n"
        "base_validate=_base_serving_namespace['validate']\n"
    )
    result = header + "\n\n".join(parts) + "\n"
    compile(result, "standalone_fixed_final_label_scope", "exec")
    return result


def build(
    parent,
    bounded,
    base_serving,
    label_serving,
    codec,
    mixin,
    importance,
    training,
    publication,
    *,
    mode,
):
    if hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != BASE_COMPILER_SHA:
        raise ValueError("Exact unchanged original full94 notebook compiler required")
    helper = standalone(base_serving, label_serving)
    # A new function owns its globals. No mutation of the original compiler or
    # current numerical source is needed, including during concurrent builds.
    compiler = FunctionType(
        original.build.__code__,
        {
            **original.build.__globals__,
            "provenance": label_scope.provenance,
            "SERVING_SHA": original.digest(helper),
        },
        "fixed_final_label_full94_build",
    )
    source, binding = compiler(
        parent, bounded, helper, codec, mixin, importance, training, publication, mode=mode
    )
    old = "result.update(role=_ADAPTED_TRAINING['role'],"
    if source.count(old) != 1:
        raise ValueError("Unique original qualification/public result binding required")
    source = source.replace(
        old,
        "result.update(source_label_role='released_S6_source',"
        f"actual_label_recovery_kernel={label_scope.RECOVERY['kernel_id']},"
        "role=_ADAPTED_TRAINING['role'],",
    )
    compile(source, "fixed_final_label_full94_notebook", "exec")
    return source, {
        **binding,
        "source_sha256": original.digest(source),
        "source_label_role": "released_S6_source",
        "source_label_target_sha256": label_scope.TARGET,
        "actual_label_recovery_kernel": label_scope.RECOVERY["kernel_id"],
        "base_compiler_sha256": BASE_COMPILER_SHA,
        "label_scope_helper_sha256": LABEL_HELPER_SHA,
        "fixed_final_only": True,
    }
