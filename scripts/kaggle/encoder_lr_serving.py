"""Prepare fixed-final LR treatment serving without changing original MRI math."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import FunctionType

from rsna.competitive import encoder_lr_serving as scope
from scripts.kaggle import licensed_importance_serving as original

BASE_COMPILER_SHA = "9b1798683fe21cbc36cbceb48837ee7a4e7da3e464d238e683636163939d04c2"
HELPER_SHA = "d02ee89f9c31bf3927795bbcc2d6b68de0b6c3c8d952d77002ca6b3d460d08fa"


def build(parent, bounded, serving, codec, mixin, importance, training, publication, *, mode):
    if (
        hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() != BASE_COMPILER_SHA
        or original.digest(serving) != HELPER_SHA
    ):
        raise ValueError("Exact unchanged full94 compiler and separate LR helper required")
    compiler = FunctionType(
        original.build.__code__,
        {
            **original.build.__globals__,
            "provenance": scope.provenance,
            "SERVING_SHA": HELPER_SHA,
        },
        "fixed_final_encoder_LR_full94_build",
    )
    source, meta = compiler(
        parent, bounded, serving, codec, mixin, importance, training, publication, mode=mode
    )
    old = "result.update(role=_ADAPTED_TRAINING['role'],"
    if source.count(old) != 1:
        raise ValueError("One original output receipt binding required")
    source = source.replace(
        old,
        "result.update(encoder_peak_lr=3e-5,head_peak_lr=3e-5,"
        "fixed_final_only=True,role=_ADAPTED_TRAINING['role'],",
    )
    compile(source, "fixed_final_encoder_LR_full94_notebook", "exec")
    return source, {
        **meta,
        "source_sha256": original.digest(source),
        "encoder_peak_lr": 3e-5,
        "head_peak_lr": 3e-5,
        "fixed_final_only": True,
        "model_filename": "epoch-003.pt",
        "helper_sha256": HELPER_SHA,
        "original_compiler_sha256": BASE_COMPILER_SHA,
        "actual_new_final_model_GPU_qualification_claim": False,
    }
