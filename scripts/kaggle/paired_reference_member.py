"""Replace only R6's 384-pixel member with a qualified fixed-final paired model.

All upstream source bodies and compiled notebooks remain private. This prepares
one candidate; no runtime, resource allowance or new accuracy is implied.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from rsna.competitive.reference_member import export_as_member
from scripts.kaggle import paired_logit_serving as paired
from scripts.kaggle import public_reference_script as reference

BASE_SHA = "79a8b3539ad017057a2b90fb4d2481293dff109be2aa00a63e7b6fc058d4f98e"
NOTEBOOK_SHA = reference.NOTEBOOK_SHA
CELLS_SHA = reference.CELLS_SHA


def qualified(value, training, model_sha, expected_source):
    r = value.get("receipt", {})
    if (
        value.get("status") != "COMPLETE"
        or value.get("version") != 1
        or type(value.get("kernel_id")) is not int
        or value["kernel_id"] <= 0
        or value.get("source_sha256") != expected_source
        or r.get("status")
        != "actual_matched_IID16_parent_epoch_bothT4_original32_MRI_and_training_graph_qualified"
        or r.get("model_sha256") != model_sha
        or r.get("epoch") != 3
        or r.get("training_source_sha256") != training["source_sha256"]
        or r.get("qualified_studies") != 32
        or r.get("qualification_cannot_submit") is not True
        or r.get("fixed_final_only") is not True
        or r.get("paired_parent_logit_training") is not True
        or r.get("parent_logit_correction_applied_in_inference") is not False
        or len(r.get("devices", [])) != 2
        or {d.get("index") for d in r["devices"]} != {0, 1}
        or any(
            "T4" not in d.get("GPU", "")
            or d.get("MRI_studies") != 32
            or d.get("both_original94_input_views_bit_equal") is not True
            or d.get("complete_learned_graph_tensor_values_equal") is not True
            or not 0
            <= d.get("maximum_original_class_vs_training_graph_probability_error", float("inf"))
            <= 1e-6
            or not 0 < d.get("peak_allocated_bytes", 0) < 16 * 2**30
            for d in r["devices"]
        )
    ):
        raise ValueError("Actual exact final paired model two-T4/32-MRI qualification required")


def validate_member_receipt(value, model_sha, training_source, rows, proof):
    if (
        value.get("status") != "actual_matched_IID16_parent_epoch_MRI_test_roster_CSV_complete"
        or value.get("model_sha256") != model_sha
        or value.get("training_source_sha256") != training_source
        or value.get("epoch") != 3
        or value.get("fixed_final_only") is not True
        or value.get("paired_parent_logit_training") is not True
        or value.get("parent_logit_correction_applied_in_inference") is not False
        or value.get("studies") != rows
        or value.get("columns") != 13
        or value.get("CSV_sha256") != proof["sha256"]
        or value.get("CSV_bytes") != proof["bytes"]
        or value.get("prediction_counts")
        != {
            "started": 2 * rows,
            "completed": 2 * rows,
            "failed": 0,
            "decode_errors": 0,
            "empty_views": 0,
        }
        or len(value.get("devices", [])) != 2
        or any("T4" not in x for x in value["devices"])
    ):
        raise ValueError("Actual complete paired member CSV/source/zero-fallback receipt required")
    return {
        "model_sha256": model_sha,
        "training_source_sha256": training_source,
        "epoch": 3,
        "member_CSV_sha256": proof["sha256"],
        "new_inference_models_added": 0,
    }


def unique(source, old, new):
    if source.count(old) != 1:
        raise ValueError("One exact reference launcher replacement required")
    return source.replace(old, new)


def build(
    notebook,
    parent,
    bounded,
    serving,
    codec,
    mixin,
    importance,
    correction,
    training,
    publication,
    qualification,
):
    if hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError("Exact admitted original R6 batch launcher required")
    cells, _ = reference.code_cells(notebook, notebook_sha=NOTEBOOK_SHA, cells_sha=CELLS_SHA)
    inputs = (parent, bounded, serving, codec, mixin, importance, correction, training, publication)
    _, qmeta = paired.build(*inputs, mode="qualification")
    source, meta = paired.build(*inputs, mode="competition")
    qualified(qualification, training, meta["model_sha256"], qmeta["source_sha256"])
    member, closure = export_as_member(parent, cells[0], source)
    changed = [member, *cells[1:]]
    actual_sha = reference.digest(json.dumps(changed, ensure_ascii=False, separators=(",", ":")))
    unchanged = [reference.digest(x) for x in cells[1:]]
    driver, _ = reference.build(notebook, notebook_sha=NOTEBOOK_SHA, cells_sha=CELLS_SHA)
    driver = paired.literal(driver, "_CELLS", changed, replace=True)
    driver = unique(
        driver,
        "# Private batch launcher for the exact public Helios v6 reference.",
        "# Private R6 variant: one qualified own paired member replaces the original384 member.",
    )
    driver = unique(driver, "== ORIGINAL_CELLS_SHA", "== ACTUAL_CELLS_SHA")
    driver = unique(
        driver,
        "    manifest = _original_manifest()",
        "    assert [hashlib.sha256(x.encode()).hexdigest() for x in _CELLS[1:]] == UNCHANGED_CELL_SHAS\n    manifest = _original_manifest()",
    )
    driver = unique(
        driver,
        "'actual_exact8cell_reference_script_MRI_and_final_reader_accepted'",
        "'actual_R6_with_fixed_final_paired_384_member_MRI_and_final_reader_accepted'",
    )
    driver = unique(
        driver,
        "'all8code_cells_sha256': ORIGINAL_CELLS_SHA,",
        "'all8code_cells_sha256': ACTUAL_CELLS_SHA, 'base_all8code_cells_sha256': ORIGINAL_CELLS_SHA, 'changed_original_cell_indices':[0], 'unchanged_original_cell_indices':list(range(1,8)),",
    )
    driver = unique(
        driver,
        "'raptor_sub.csv', 'raptor224_sub.csv'):",
        "'raptor_sub.csv', 'raptor224_sub.csv', 'raptor_raw.csv'):",
    )
    insertion = "    member=json.loads(Path('/kaggle/working/licensed-IID16-epoch-public-accepted.json').read_text())\n    member_proof=validate_member_receipt(member,PAIRED_MODEL_SHA,PAIRED_TRAINING_SOURCE,len(ids),proofs['raptor_sub.csv'])\n"
    driver = unique(
        driver,
        "    out = {'status': 'actual_R6_with_fixed_final_paired_384_member_MRI_and_final_reader_accepted',",
        insertion
        + "    out = {'status': 'actual_R6_with_fixed_final_paired_384_member_MRI_and_final_reader_accepted',",
    )
    driver = unique(
        driver,
        "    Path('/kaggle/working/exact8cell-script-accepted.json').write_text",
        "    out.update(paired_member=member_proof,original_R6_ensemble_weights_unchanged=True,new_inference_models_added=0)\n    Path('/kaggle/working/R6-paired-member-accepted.json').write_text",
    )
    node = next(
        n
        for n in ast.parse(Path(__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_member_receipt"
    )
    function = ast.get_source_segment(Path(__file__).read_text(), node)
    assert function is not None
    header = (
        f"ACTUAL_CELLS_SHA={actual_sha!r}\nUNCHANGED_CELL_SHAS={unchanged!r}\n"
        f"PAIRED_MODEL_SHA={meta['model_sha256']!r}\nPAIRED_TRAINING_SOURCE={training['source_sha256']!r}\n"
    )
    driver = header + function + "\n\n" + driver
    compile(driver, "R6_one_qualified_paired_member", "exec")
    return driver, {
        "source_sha256": reference.digest(driver),
        "source_bytes": len(driver.encode()),
        "upstream_notebook_sha256": NOTEBOOK_SHA,
        "upstream_all8_cells_sha256": CELLS_SHA,
        "actual_all8_cells_sha256": actual_sha,
        "changed_original_cells": [0],
        "unchanged_original_cells": list(range(1, 8)),
        "unchanged_original_cell_sha256": unchanged,
        "actual_original_qualification_kernel": qualification["kernel_id"],
        "actual_original_qualification_source_sha256": qualification["source_sha256"],
        "model_sha256": meta["model_sha256"],
        "training_kernel": training["kernel_id"],
        "training_source_sha256": training["source_sha256"],
        "fixed_final_epoch": 3,
        "parent_logit_correction_applied_in_inference": False,
        "ensemble_blend_weights_changed": False,
        "new_inference_models_added": 0,
        "original_attribution_retained": True,
        "member_closure": closure,
        "requires_original_R6_Dataset_version": 1,
        "actual_new_full_R6_run_or_hidden_score": False,
    }
