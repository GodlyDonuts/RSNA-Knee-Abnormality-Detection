"""Compile a separate public-three delta check; no execution or writes on import."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL_SOURCE_SHA = "dd3b7ea13383c5132a3d02c60e87aa65767a21efe8032de43c2bd524718f455e"
MODEL_SHA = "d478a502e6fd33be2ad9d781acd91d3d2b6b252629262031fba8282a91cf5bca"
WORKER_SHA = "ffc1fbf914b9e1a97476ac40b3d4cc0c58b7948c45061079a497f1135c2df461"
INHERITED_SHA = "13d7e576e15a9036ebf463ea3a74b5ff39c6c836a1f3d2bef9f49274a8c7463a"
PUBLIC_ROSTER_SHA = "e743f6d8dca6eabedec2f05a98cf86ed820ef1fe8a0eb06338af5b377dc8a59b"
STATUS = "actual_teacher_rank_public3_bothT4_delta_qualified"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def inherited_evidence(text):
    if digest(text) != INHERITED_SHA:
        raise ValueError("Exact completed inherited same-code qualification receipt required")
    actual = json.loads(text)
    receipt = actual["receipt"]
    if (
        actual["status"] != "COMPLETE"
        or actual["kernel_id"] != 139818636
        or actual["version"] != 1
        or actual["role"] != "pointwise"
        or receipt["qualified_studies"] != 32
        or receipt["serving_windows"] != 94
        or [d["index"] for d in receipt["devices"]] != [0, 1]
        or any(
            d["MRI_studies"] != 32
            or d["both_original94_input_views_bit_equal"] is not True
            or d["complete_learned_graph_tensor_values_equal"] is not True
            or d["maximum_original_class_vs_training_graph_probability_error"] != 0
            for d in receipt["devices"]
        )
    ):
        raise ValueError("Inherited qualification scope differs")
    return actual


def _literal_bindings(source):
    wanted = {"_ADAPTED_MODE", "_ADAPTED_SHA", "_ADAPTED_TRAINING", "_ADAPTED_PUBLICATION"}
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    if target.id in result:
                        raise ValueError("Duplicate original binding")
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != wanted:
        raise ValueError("Original literal training/model/publication bindings required")
    training, publication = result["_ADAPTED_TRAINING"], result["_ADAPTED_PUBLICATION"]
    if (
        result["_ADAPTED_MODE"] != "competition"
        or result["_ADAPTED_SHA"] != MODEL_SHA
        or training["role"] != "teacher_rank"
        or training["epoch"] != 3
        or training["status"] != "COMPLETE"
        or training["kernel_id"] is not None
        or training["training_Slurm_job"] != "975297"
        or training["source_sha256"] != WORKER_SHA
        or training["receipt"]["completed"] != 1458
        or training["receipt"]["epoch"] != {"bytes": 292867251, "sha256": MODEL_SHA}
        or publication["model_sha256"] != MODEL_SHA
        or publication["role"] != "teacher_rank"
        or publication["epoch"] != 3
        or publication["completed_updates"] != 1458
        or publication["objects"] != 6
        or publication["revision"] != "60a64c2be5d59f60e458caca1b6f475f82a95fa7"
    ):
        raise ValueError("Exact original final H100 ranking state required")
    return result


def _replace_entry(source, helper, prelude, binding):
    tree = ast.parse(source)
    guards = [
        node for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
    ]
    if len(guards) != 1 or guards[0] is not tree.body[-1]:
        raise ValueError("One final original main guard required")
    guard = guards[0]
    calls = {
        node.func.id for node in ast.walk(guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    if not {"_adapted_boot", "_adapted_qualify", "_adapted_competition"} <= calls:
        raise ValueError("Original final serving dispatch required")
    prefix = "".join(source.splitlines(keepends=True)[:guard.lineno - 1])
    result = (
        "_RANK_DELTA_BINDING = " + repr(binding) + "\n" + prelude + "\n"
        + prefix + "\n" + helper
        + "\nif __name__ == '__main__':\n    _rank_delta_main()\n"
    )
    compile(result, "teacher_rank_public3_delta_candidate", "exec")
    return result


def adapt(compiled_source, *, inherited_qualification_text):
    """Return new immutable source/metadata; actual public acceptance remains required."""
    if digest(compiled_source) != ORIGINAL_SOURCE_SHA:
        raise ValueError("Exact original ranking competition source required")
    inherited_evidence(inherited_qualification_text)
    _literal_bindings(compiled_source)
    helper = (HERE / "publiccheck.py").read_text()
    prelude = (HERE / "prelude.py").read_text()
    binding = {
        "original_competition_sha256": ORIGINAL_SOURCE_SHA,
        "model_sha256": MODEL_SHA,
        "training_source_sha256": WORKER_SHA,
        "inherited32_receipt_sha256": INHERITED_SHA,
        "public_roster_sha256": PUBLIC_ROSTER_SHA,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "publiccheck_sha256": digest(helper),
        "prelude_sha256": digest(prelude),
        "public_watchdog_seconds": 180,
        "public_VM_setup_reserve_seconds": 80,
        "qualification_scope": "inherited32_same_code_plus_actual_public3_new_weights",
    }
    result = _replace_entry(compiled_source, helper, prelude, binding)
    return result, {
        **binding,
        "status": "prepared_teacher_rank_single_public_run_delta_source",
        "source_sha256": digest(result),
        "inline_acceptance_status": STATUS,
        "actual_new_weight_studies_per_device": 3,
        "inherited_code_qualification_studies": 32,
        "inherited32_reexecuted": False,
        "total_reserved_VM_seconds": 260,
        "hidden_session_timeout_seconds": 32400,
        "competition_inference_functions_unchanged": True,
        "zero_fallback_postcondition_for_all_rosters": True,
        "actual_runtime_or_entry_accepted": False,
    }


def validate_receipt(receipt, metadata):
    """Check actual public delta evidence; caller must still bind exact kernel/version/source."""
    if (
        metadata["status"] != "prepared_teacher_rank_single_public_run_delta_source"
        or metadata["original_competition_sha256"] != ORIGINAL_SOURCE_SHA
        or metadata["model_sha256"] != MODEL_SHA
        or metadata["inherited32_receipt_sha256"] != INHERITED_SHA
        or metadata["public_watchdog_seconds"] != 180
        or metadata["public_VM_setup_reserve_seconds"] != 80
        or metadata["total_reserved_VM_seconds"] != 260
        or metadata["hidden_session_timeout_seconds"] != 32400
    ):
        raise ValueError("Exact prepared delta candidate metadata required")
    inline = receipt["inline_new_weight_acceptance"]
    keys = (
        "original_competition_sha256", "model_sha256", "training_source_sha256",
        "inherited32_receipt_sha256", "public_roster_sha256", "adapter_sha256",
        "publiccheck_sha256", "prelude_sha256", "public_watchdog_seconds",
        "public_VM_setup_reserve_seconds", "qualification_scope",
    )
    if any(inline[key] != metadata[key] for key in keys):
        raise ValueError("Actual delta source/weight/inherited evidence differs")
    expected_counts = {"started": 6, "completed": 6, "failed": 0, "decode_errors": 0, "empty_views": 0}
    if (
        receipt["status"] != "actual_teacher_pair_final_epoch_MRI_test_roster_CSV_complete"
        or receipt["role"] != "teacher_rank"
        or receipt["epoch"] != 3
        or receipt["model_sha256"] != MODEL_SHA
        or receipt["training_source_sha256"] != WORKER_SHA
        or receipt["test_roster_sha256"] != PUBLIC_ROSTER_SHA
        or receipt["studies"] != 3
        or receipt["columns"] != 13
        or receipt["prediction_counts"] != expected_counts
        or inline["status"] != STATUS
        or inline["views_per_study"] != 2
        or inline["windows_per_view"] != 94
        or inline["actual_new_weight_studies_per_device"] != 3
        or inline["inherited_code_qualification_studies"] != 32
        or inline["inherited32_reexecuted"] is not False
        or inline["qualification_scope"] != "inherited32_same_code_plus_actual_public3_new_weights"
    ):
        raise ValueError("Actual public-three delta/CSV scope differs")
    timings = [receipt["public_total_elapsed_seconds"], inline["elapsed_seconds"]]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 180 for v in timings):
        raise ValueError("Finite actual public execution within watchdog required")
    devices = inline["devices"]
    if len(devices) != 2 or [d["index"] for d in devices] != [0, 1]:
        raise ValueError("Both original T4 devices required")
    for device in devices:
        error = device["maximum_original_class_vs_training_graph_probability_error"]
        memory = device["peak_allocated_bytes"]
        if (
            "T4" not in device["GPU"]
            or device["MRI_studies"] != 3
            or device["both_original94_input_views_bit_equal"] is not True
            or device["complete_learned_graph_tensor_values_equal"] is not True
            or type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= 1e-6
            or type(memory) is not int or not 0 < memory < 16 * 2**30
        ):
            raise ValueError("Actual learned state/FP16 graph/input/memory check failed")
    if devices[0]["input_views_aggregate_sha256"] != devices[1]["input_views_aggregate_sha256"]:
        raise ValueError("Device input views differ")
    for value in (
        receipt["CSV_sha256"], devices[0]["input_views_aggregate_sha256"], metadata["source_sha256"],
    ):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Exact output/source/input hashes required")
    if type(receipt["CSV_bytes"]) is not int or not 0 < receipt["CSV_bytes"] < 16384:
        raise ValueError("Bounded public CSV required")
    return {
        "status": "actual_teacher_rank_public3_delta_and_CSV_accepted",
        "source_sha256": metadata["source_sha256"],
        "model_sha256": MODEL_SHA,
        "CSV_sha256": receipt["CSV_sha256"],
        "public_roster_sha256": PUBLIC_ROSTER_SHA,
        "inherited32_receipt_sha256": INHERITED_SHA,
        "public_total_elapsed_seconds": receipt["public_total_elapsed_seconds"],
        "new_weight_studies_per_device": 3,
        "same_exact_kernel_version_still_required": True,
        "competition_entry_authority": False,
    }
