"""Compile the actual independent Colab C100 final using the unchanged two-T4 serving worker.

This is a CPU-only source compiler. It performs no upload, allocation, model
transfer or submission, and it cannot accept a pilot as a final model.
"""

import ast
import base64
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "predictive-final-serving-20261020-v1"
PINS = {
    "contract.py": "ee830d60e578163524c9534064e07badca6e05fbd488df0029c2b6d53c5b1075",
    "worker.py.txt": "0b1a218482cddb9fd0ec16ac9ae18c0bf7cd7fa9e03b1e0a5c4e998f4c447525",
    "mounts.py": "f2a302499720bb3a439f40f174d334588d69aa1333e750ea672aff3f6516462d",
    "descriptor_cache.py": "a40a3e313922d2577706eb430429675bac8cfe0c52df768fcd8eac0435078560",
}
OPTIONS = {"module_sha256"}

spec = importlib.util.spec_from_file_location("Colab_C100_compile_contract", HERE / "contract.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def originals():
    result = {name: (ORIGINAL / name).read_text() for name in PINS}
    C.require(
        all(C.digest(result[n]) == h for n, h in PINS.items()),
        "Exact accepted original serving source required",
    )
    return result


def runtime_contract(original, adapter_source, options):
    """Expose only the C100 adapter's validated interface to the original worker."""
    C.require(set(options) == OPTIONS, "Complete explicit C100 evidence options required")
    # JSON round-trip disallows executable/custom objects in embedded evidence.
    options = json.loads(C.canonical(options))
    text = (
        "import types as _types\n"
        "_implementation = _types.ModuleType('Colab_C100_evidence_contract')\n"
        f"exec(compile({adapter_source!r}, 'Colab_C100_evidence_contract', 'exec'), "
        "_implementation.__dict__)\n"
        f"_adapter = _implementation.Contract({original!r}, **{options!r})\n"
        "digest = _implementation.digest\n"
        "canonical = _implementation.canonical\n"
        "require = _implementation.require\n"
        "validate_export = _adapter.validate_export\n"
        "validate_qualification = _adapter.validate_qualification\n"
        "check_predictions = _adapter.check_predictions\n"
        "COLUMNS = _adapter._base.COLUMNS\n"
        "LICENSE = _adapter._base.LICENSE\n"
        "PARENT_SOURCE = _adapter._base.PARENT_SOURCE\n"
        "PARENT = _adapter._base.PARENT\n"
        "DOCKER = _adapter._base.DOCKER\n"
    )
    ast.parse(text)
    return text


def build(parent_source, export, *, contract_options, mode, model_dataset_ref, qualification=None):
    original = originals()
    C.require(mode in ("qualification", "competition"), "Explicit serving mode required")
    C.require(set(contract_options) == OPTIONS, "Complete explicit C100 evidence required")
    adapter = C.Contract(original["contract.py"], **contract_options)
    base = adapter._base
    C.require(
        C.digest(parent_source) == base.PARENT_SOURCE, "Exact inherited MRI encoder source required"
    )
    mounts = {}
    exec(compile(original["mounts.py"], "original_serving_mounts", "exec"), mounts)
    mounts["dataset_parts"](model_dataset_ref)
    chosen = adapter.candidate(**export)
    C.require(
        chosen["branch"] == C.BRANCH and chosen["arm"] == "C",
        "Only the distinct fixed-final C100 model is compiled",
    )
    C.require(
        model_dataset_ref == chosen["model_dataset_ref"],
        "Use the actual existing private final epoch Dataset",
    )
    if mode == "competition":
        C.require(qualification is not None, "Actual trained C100 qualification required")
        adapter.validate_qualification(qualification, chosen)
    else:
        C.require(qualification is None, "Fresh qualification cannot inherit acceptance")

    own = json.loads(export["own_source_text"])
    sources = {name: own[name] for name in base.MODULES}
    adapter_source = (HERE / "contract.py").read_text()
    runtime = runtime_contract(original["contract.py"], adapter_source, contract_options)
    helpers = {
        "_predictive_serving_mounts.py": original["mounts.py"],
        "_predictive_descriptor_cache.py": original["descriptor_cache.py"],
    }
    helper_hashes = {n: C.digest(s) for n, s in helpers.items()}
    worker_sha, contract_sha = C.digest(original["worker.py.txt"]), C.digest(runtime)
    if mode == "competition":
        C.require(
            qualification["serving_worker_sha256"] == worker_sha
            and qualification["serving_contract_sha256"] == contract_sha
            and qualification.get("serving_helpers_sha256") == helper_hashes
            and qualification.get("model_dataset_ref") == model_dataset_ref,
            "Actual C100 qualification used a different source or Dataset",
        )
    sources["_predictive_serving_contract.py"] = runtime
    sources.update(helpers)
    binding = {
        "schema": "predictive_full3887_final_serving_v1",
        "branch": C.BRANCH,
        "mode": mode,
        "models": [chosen],
        "qualification": qualification,
        "module_sha256": {name: C.digest(text) for name, text in sources.items()},
        "serving_worker_sha256": worker_sha,
        "serving_contract_sha256": contract_sha,
        "serving_helpers_sha256": helper_hashes,
        "model_dataset_ref": model_dataset_ref,
        "public_roster_sha256": "e743f6d8dca6eabedec2f05a98cf86ed820ef1fe8a0eb06338af5b377dc8a59b",
        "public_watchdog_seconds": 600,
        "pinned_Docker": base.DOCKER,
        "internet": False,
        "GPU_count": 2,
        "CPU_study_workers": 2,
        "maximum_encoder_windows_per_study": 48,
    }
    result = original["worker.py.txt"]
    for marker, value in {
        "__FILES_VALUE__": repr(sources),
        "__PARENT_SOURCE_VALUE__": repr(base64.b64encode(parent_source.encode()).decode()),
        "__PERMISSION_VALUE__": repr(base64.b64encode(export["license_bytes"]).decode()),
        "__BINDING_VALUE__": repr(binding),
    }.items():
        C.require(result.count(marker) == 1, "One original worker placeholder required")
        result = result.replace(marker, value)
    ast.parse(result)
    return result, {
        "status": "Colab_C100_final_offline_source_compiled_CPU_only",
        "branch": C.BRANCH,
        "mode": mode,
        "source_sha256": C.digest(result),
        "source_bytes": len(result.encode()),
        "worker_sha256": worker_sha,
        "contract_sha256": contract_sha,
        "Colab_C100_adapter_sha256": C.digest(adapter_source),
        "original_serving_pins": dict(PINS),
        "serving_helpers_sha256": helper_hashes,
        "model_dataset_ref": model_dataset_ref,
        "model_files": {chosen["model_filename"]: chosen["model"]},
        "candidate_sha256": C.digest(C.canonical(chosen)),
        "public_watchdog_seconds": 600,
        "pinned_Docker": base.DOCKER,
        "actual_trained_GPU_execution": False,
        "new_Dataset_notebook_or_entry_mutations": 0,
        "qualification_may_create_submission_csv": False,
        "clinical_or_hidden_runtime_claim": False,
        "account_budget_admitted": False,
        "primary_predictive_VM_reservation_seconds": 3360,
    }
