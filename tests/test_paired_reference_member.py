"""One member is replaced; all other seven cells and blend rules stay byte-identical."""

import copy
import hashlib
import json

import pytest
from test_reference_member import PARENT, external
from test_reference_member import paired as fake_paired

from scripts.kaggle import paired_reference_member as compiler
from scripts.kaggle.paired_logit_serving import literal


@pytest.fixture
def case(monkeypatch):
    cells = [
        external(),
        "# execute exactly the original stage1\npass\n",
        *["pass\n" for _ in range(6)],
    ]
    nb = json.dumps(
        {
            "nbformat": 4,
            "cells": [
                {"cell_type": "markdown", "source": "Original author credits"},
                *[{"cell_type": "code", "source": s} for s in cells],
            ],
        }
    )
    monkeypatch.setattr(compiler, "NOTEBOOK_SHA", hashlib.sha256(nb.encode()).hexdigest())
    monkeypatch.setattr(
        compiler,
        "CELLS_SHA",
        hashlib.sha256(
            json.dumps(cells, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    model = "1" * 64
    source = "2" * 64
    training = {"kernel_id": 123, "source_sha256": source}

    def build(*args, mode):
        text = fake_paired()
        return text, {
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "model_sha256": model,
        }

    monkeypatch.setattr(compiler.paired, "build", build)
    proof = {
        "status": "COMPLETE",
        "version": 1,
        "kernel_id": 456,
        "source_sha256": hashlib.sha256(fake_paired().encode()).hexdigest(),
        "receipt": {
            "status": "actual_matched_IID16_parent_epoch_bothT4_original32_MRI_and_training_graph_qualified",
            "model_sha256": model,
            "training_source_sha256": source,
            "epoch": 3,
            "qualified_studies": 32,
            "qualification_cannot_submit": True,
            "fixed_final_only": True,
            "paired_parent_logit_training": True,
            "parent_logit_correction_applied_in_inference": False,
            "devices": [
                {
                    "index": i,
                    "GPU": "Tesla T4",
                    "MRI_studies": 32,
                    "both_original94_input_views_bit_equal": True,
                    "complete_learned_graph_tensor_values_equal": True,
                    "maximum_original_class_vs_training_graph_probability_error": 0.0,
                    "peak_allocated_bytes": 1024,
                }
                for i in range(2)
            ],
        },
    }
    args = [
        nb,
        PARENT,
        "bounded",
        "serving",
        "codec",
        "mixin",
        "importance",
        "correction",
        training,
        {"fixture_only": True},
        proof,
    ]
    return args, cells


def test_only_first_cell_changes_and_no_model_or_blend_is_added(case):
    args, original = case
    source, meta = compiler.build(*args)
    actual = literal(source, "_CELLS")
    assert actual[0] != original[0] and actual[1:] == original[1:]
    assert meta["changed_original_cells"] == [0] and meta["unchanged_original_cells"] == list(
        range(1, 8)
    )
    assert (
        meta["ensemble_blend_weights_changed"] is False and meta["new_inference_models_added"] == 0
    )
    assert "R6-paired-member-accepted.json" in source and "raptor_raw.csv" in source
    assert (
        "Original author credits" in source
        and meta["actual_new_full_R6_run_or_hidden_score"] is False
    )
    assert literal(source, "ACTUAL_CELLS_SHA") == meta["actual_all8_cells_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        lambda q: q.update(status="RUNNING"),
        lambda q: q.update(source_sha256="0" * 64),
        lambda q: q["receipt"].update(model_sha256="0" * 64),
        lambda q: q["receipt"].update(qualified_studies=16),
        lambda q: q["receipt"]["devices"][0].update(GPU="H100"),
        lambda q: q["receipt"].update(parent_logit_correction_applied_in_inference=True),
    ],
)
def test_missing_or_unrelated_actual_qualification_rejected(case, change):
    args, _ = copy.deepcopy(case)
    change(args[-1])
    with pytest.raises(ValueError, match="qualification"):
        compiler.build(*args)


def member():
    return {
        "status": "actual_matched_IID16_parent_epoch_MRI_test_roster_CSV_complete",
        "model_sha256": "1" * 64,
        "training_source_sha256": "2" * 64,
        "epoch": 3,
        "fixed_final_only": True,
        "paired_parent_logit_training": True,
        "parent_logit_correction_applied_in_inference": False,
        "studies": 3,
        "columns": 13,
        "CSV_sha256": "3" * 64,
        "CSV_bytes": 123,
        "prediction_counts": {
            "started": 6,
            "completed": 6,
            "failed": 0,
            "decode_errors": 0,
            "empty_views": 0,
        },
        "devices": ["Tesla T4", "Tesla T4"],
    }


def test_member_receipt_binds_CSV_and_exact_source():
    result = compiler.validate_member_receipt(
        member(), "1" * 64, "2" * 64, 3, {"sha256": "3" * 64, "bytes": 123}
    )
    assert result["new_inference_models_added"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("CSV_sha256", "0" * 64),
        ("epoch", 1),
        ("parent_logit_correction_applied_in_inference", True),
    ],
)
def test_failed_or_mixed_member_cannot_become_ensemble_result(key, value):
    m = member()
    m[key] = value
    with pytest.raises(ValueError):
        compiler.validate_member_receipt(
            m, "1" * 64, "2" * 64, 3, {"sha256": "3" * 64, "bytes": 123}
        )
