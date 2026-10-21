"""Prospective fixed96 competition source, admitted by the actual same-model test.

This pure compiler neither pushes a notebook nor submits. Original final model,
training provenance and exact successful decoder stay unchanged.
"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("context96_original_diagnostic", HERE / "compile.py")
D = importlib.util.module_from_spec(s)
s.loader.exec_module(D)
PARENT_COMPETITION = "055a8f8ed7e0d64f47537fb25480a0f08c7ac947ec8f0472e22f7dcdc117d6e9"
DIAGNOSTIC_SOURCE = "12b46428d718e95c2b872b3e9a19fb6050d4fda08d1e8b757d2219f5d4d1e4a3"
DIAGNOSTIC_KERNEL = 140066402
RUNTIME_SHA = "49308cd82b88503ab54c869868c7d0b1e42e11c4712fd1b95237e56a2f4a376e"


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def qualification_function(runtime, actual, chosen):
    """Extend only the original cap/status gate, retaining all numerical checks."""
    assert D.sha(runtime) == RUNTIME_SHA
    original = None
    for node in ast.parse(runtime).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_adapter"
        ):
            original = ast.literal_eval(node.value.args[0])
    assert (
        original
        and D.sha(original) == "ee830d60e578163524c9534064e07badca6e05fbd488df0029c2b6d53c5b1075"
    )
    function = next(
        n
        for n in ast.parse(original).body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_qualification"
    )
    source = ast.get_source_segment(original, function)
    source = D.replace(
        source, "def validate_qualification(", "def _validate_fixed96_qualification("
    )
    source = D.replace(
        source,
        'receipt["status"] == "actual_trained_predictive_public_T4_qualification_accepted"',
        'receipt["status"] == "actual_fixed96_public_T4_diagnostic_complete"',
    )
    source = D.replace(source, '0 < m["windows"] <= 48', '0 < m["windows"] <= 192')
    receipt_sha = hashlib.sha256(canonical(actual)).hexdigest()
    source = (
        "import math\n"
        + source
        + f"""

def validate_qualification(receipt, chosen):
    require(digest(canonical(receipt)) == {receipt_sha!r}, "Exact actual fixed96 diagnostic required")
    require(chosen.get("branch") == "C100" and chosen["architecture"]["context_cap"] == 24,
            "Preserve original training cap24 provenance")
    require(receipt["inference_context_cap"] == 96 and receipt["training_context_cap"] == 24
            and receipt["context_coverage_version"] == {D.VERSION!r}
            and receipt["competition_admitted"] is False, "Diagnostic provenance differs")
    row = _validate_fixed96_qualification(receipt, chosen)
    require(row["reference24_prediction_sha256"] == {D.REFERENCE_PREDICTIONS!r},
            "Original24 actual predictions changed")
    require(all(m["complete_descriptor_support_fields_exact"] is True
                and m["original_context_bitwise_equal"] is True
                and m["decoder_route"] == "original_exact"
                and m["prediction_precision"] == "fp16"
                and m["inference_context_cap"] == 96 for m in row["measurements"]),
            "Original context, descriptors, precision and decoder proof required")
    return row
"""
    )
    namespace = dict(
        digest=lambda b: hashlib.sha256(b).hexdigest(),
        canonical=canonical,
        require=lambda ok, msg: None if ok else (_ for _ in ()).throw(ValueError(msg)),
        DOCKER=actual["pinned_Docker"],
    )
    exec(compile(source, "fixed96_receipt_validator", "exec"), namespace)
    namespace["validate_qualification"](actual, chosen)
    return source


def build(original, diagnostic):
    assert D.sha(original) == PARENT_COMPETITION
    assert diagnostic["kernel_id"] == DIAGNOSTIC_KERNEL and diagnostic["version"] == 1
    assert diagnostic["status"] == "COMPLETE" and diagnostic["source_sha256"] == DIAGNOSTIC_SOURCE
    assert diagnostic["competition_entry"] is False
    values, worker = D.split_source(original)
    files, binding = values["FILES"], values["BINDING"]
    assert binding["mode"] == "competition" and len(binding["models"]) == 1
    chosen, actual = binding["models"][0], diagnostic["receipt"]
    assert chosen["model"]["sha256"] == D.MODEL
    assert all(D.sha(body) == binding["module_sha256"][name] for name, body in files.items())
    helper = files["_predictive_descriptor_cache.py"]
    assert D.sha(helper) == D.HELPER and helper.count("context_cap=24") == 2
    helper = helper.replace("context_cap=24", "context_cap=96")
    helper = D.replace(
        helper,
        '        "preparation_version": "exact_selected_frames_and_batched_descriptors_v1",',
        f'        "preparation_version": "exact_selected_frames_and_batched_descriptors_v1",\n        "context_coverage_version": {D.VERSION!r},\n        "inference_context_cap": 96,\n        "training_context_cap": 24,',
    )
    files["_predictive_descriptor_cache.py"] = helper
    runtime = files["_predictive_serving_contract.py"]
    runtime += "\n" + qualification_function(runtime, actual, chosen)
    files["_predictive_serving_contract.py"] = runtime
    worker = D.replace(
        worker,
        'sum(len(c["images"]) for c in contexts) <= 48',
        'sum(len(c["images"]) for c in contexts) <= 192',
    )
    worker = D.replace(
        worker,
        '        "clinical_score": None,',
        f'        "clinical_score": None,\n        "context_coverage_version": {D.VERSION!r},\n        "inference_context_cap": 96,\n        "training_context_cap": 24,',
    )
    binding.update(
        qualification=actual,
        parent_competition_source_sha256=PARENT_COMPETITION,
        actual_context96_diagnostic_kernel=DIAGNOSTIC_KERNEL,
        actual_context96_diagnostic_source_sha256=DIAGNOSTIC_SOURCE,
        context_coverage_version=D.VERSION,
        inference_context_cap=96,
        training_context_cap=24,
        maximum_encoder_windows_per_study=192,
    )
    binding["module_sha256"]["_predictive_descriptor_cache.py"] = D.sha(helper)
    binding["module_sha256"]["_predictive_serving_contract.py"] = D.sha(runtime)
    binding["serving_helpers_sha256"]["_predictive_descriptor_cache.py"] = D.sha(helper)
    binding["serving_contract_sha256"] = D.sha(runtime)
    binding["serving_worker_sha256"] = D.sha(worker)
    for name, value in values.items():
        worker = D.replace(worker, "__" + name + "_VALUE__", repr(value))
    ast.parse(worker)
    ast.parse(helper)
    return worker, dict(
        status="fixed96_competition_candidate_compiled_not_submitted",
        source_sha256=D.sha(worker),
        source_bytes=len(worker.encode()),
        actual_diagnostic_kernel=DIAGNOSTIC_KERNEL,
        actual_diagnostic_source_sha256=DIAGNOSTIC_SOURCE,
        diagnostic_receipt_sha256=hashlib.sha256(canonical(actual)).hexdigest(),
        expected_public_prediction_sha256=actual["models"][0]["prediction_sha256"],
        binding=binding,
        competition_entry=False,
        new_Dataset=False,
    )
