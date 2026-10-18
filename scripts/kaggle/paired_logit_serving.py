"""Prepare paired fixed-final inference through unchanged original MRI math."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import FunctionType

from rsna.competitive import paired_logit_serving as scope
from scripts.kaggle import licensed_importance_serving as original

BASE_COMPILER_SHA = "9b1798683fe21cbc36cbceb48837ee7a4e7da3e464d238e683636163939d04c2"
HELPER_SHA = "a5479e03630ea189055fad28e8113f0756b99c159e19e008dd5bdc76f370543c"
PAIRED_SHA = "c95b35b672654cabeac6a8d46432ef878c2e084d2d8ea7dbcb38b0f1fa0ce896"


def literal(source, name, value=None, *, replace=False):
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == name
    ]
    if len(nodes) != 1:
        raise ValueError("One exact serving literal required: " + name)
    n = nodes[0]
    if not replace:
        return ast.literal_eval(n.value)
    assert n.end_lineno is not None and n.end_col_offset is not None
    lines = source.encode().splitlines(keepends=True)
    a = sum(map(len, lines[: n.lineno - 1])) + n.col_offset
    b = sum(map(len, lines[: n.end_lineno - 1])) + n.end_col_offset
    return (
        source.encode()[:a] + (name + "=" + repr(value)).encode() + source.encode()[b:]
    ).decode()


def build(
    parent, bounded, serving, codec, mixin, importance, paired, training, publication, *, mode
):
    if (
        hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != BASE_COMPILER_SHA
        or original.digest(serving) != HELPER_SHA
        or original.digest(paired) != PAIRED_SHA
    ):
        raise ValueError(
            "Exact original full94 compiler and distinct paired source/helpers required"
        )
    compiler = FunctionType(
        original.build.__code__,
        {**original.build.__globals__, "provenance": scope.provenance, "SERVING_SHA": HELPER_SHA},
        "fixed_final_paired_original_full94_build",
    )
    source, meta = compiler(
        parent, bounded, serving, codec, mixin, importance, training, publication, mode=mode
    )
    classes = [
        n
        for n in ast.parse(paired).body
        if isinstance(n, ast.ClassDef) and n.name == "PairedLogitReadout"
    ]
    if len(classes) != 1:
        raise ValueError("Exact actual training correction class required")
    paired_class = ast.get_source_segment(paired, classes[0])
    assert paired_class is not None
    embedded = literal(source, "_ADAPTED_IMPORTANCE") + "\n" + paired_class + "\n"
    source = literal(source, "_ADAPTED_IMPORTANCE", embedded, replace=True)
    old = "class TrainingGraph(importance['ImportanceReadout'],mix['CheckpointedEncode'],RaptorClassifier):pass"
    if source.count(old) != 1:
        raise ValueError("One original GPU qualification reference graph required")
    source = source.replace(
        old,
        "class TrainingGraph(importance['PairedLogitReadout'],importance['ImportanceReadout'],mix['CheckpointedEncode'],RaptorClassifier):pass",
    )
    old = "result.update(role=_ADAPTED_TRAINING['role'],"
    if source.count(old) != 1:
        raise ValueError("One original output receipt binding required")
    source = source.replace(
        old,
        "result.update(paired_parent_logit_training=True,parent_logit_correction_applied_in_inference=False,fixed_final_only=True,role=_ADAPTED_TRAINING['role'],",
    )
    compile(source, "fixed_final_paired_original_full94_notebook", "exec")
    return source, {
        **meta,
        "source_sha256": original.digest(source),
        "helper_sha256": HELPER_SHA,
        "paired_readout_sha256": PAIRED_SHA,
        "original_compiler_sha256": BASE_COMPILER_SHA,
        "fixed_final_only": True,
        "parent_logit_correction_applied_in_inference": False,
        "qualification_reference_includes_actual_PairedLogitReadout": True,
        "actual_new_model_GPU_qualification_claim": False,
        "new_fit_GPU_or_submission_dispatched": False,
    }
