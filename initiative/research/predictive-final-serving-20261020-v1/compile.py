"""CPU-only compiler for actual preserved predictive finals; no remote operations.

Each export input is a mapping with publication, provenance_text, own_source_text,
license_bytes. Actual model bytes stay private on the owning node and must be
present under model_filename when the compiled offline notebook runs.
"""

import ast
import base64
import importlib.util
from pathlib import Path

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("predictive_serving_contract", D / "contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def build(parent_source, exports, *, mode, qualification=None):
    contract.require(mode in ("qualification", "competition"), "Explicit serving mode required")
    contract.require(
        exports and len(exports) <= 3, "One to three actual final trained exports required"
    )
    contract.require(
        contract.digest(parent_source) == contract.PARENT_SOURCE,
        "Exact private inherited encoder source required",
    )
    candidates = []
    sources = None
    license_bytes = None
    for entry in exports:
        chosen = contract.candidate(**entry)
        import json

        own = json.loads(entry["own_source_text"])
        selected = {name: own[name] for name in contract.MODULES}
        contract.require(
            sources is None or sources == selected,
            "Matched exact serving modules required across arms",
        )
        contract.require(
            license_bytes is None or license_bytes == entry["license_bytes"],
            "One exact inherited license required",
        )
        sources, license_bytes = selected, entry["license_bytes"]
        candidates.append(chosen)
    contract.require(
        len({c["arm"] for c in candidates}) == len(candidates), "No duplicated arm/export"
    )
    contract.require(
        len({c["provenance"]["source_git"] for c in candidates}) == 1
        and len({c["provenance"]["initial_shared_tensor_sha256"] for c in candidates}) == 1,
        "Matched production source and shared initialization required",
    )
    if mode == "competition":
        contract.require(
            len(candidates) == 1 and qualification is not None,
            "One actually qualified model per code candidate",
        )
        contract.validate_qualification(qualification, candidates[0])
    else:
        contract.require(
            qualification is None, "Fresh actual qualification has no inherited acceptance"
        )
    template = (D / "worker.py.txt").read_text()
    contract_source = (D / "contract.py").read_text()
    worker_sha, contract_sha = contract.digest(template), contract.digest(contract_source)
    if mode == "competition":
        contract.require(
            qualification["serving_worker_sha256"] == worker_sha
            and qualification["serving_contract_sha256"] == contract_sha,
            "Actual qualification used a different serving worker or contract",
        )
    sources["_predictive_serving_contract.py"] = contract_source
    binding = {
        "schema": "predictive_full3887_final_serving_v1",
        "mode": mode,
        "models": candidates,
        "qualification": qualification,
        "module_sha256": {n: contract.digest(text) for n, text in sources.items()},
        "serving_worker_sha256": worker_sha,
        "serving_contract_sha256": contract_sha,
        "pinned_Docker": contract.DOCKER,
        "internet": False,
        "GPU_count": 2,
        "CPU_study_workers": 2,
        "maximum_encoder_windows_per_study": 48,
    }
    replacements = {
        "__FILES_VALUE__": repr(sources),
        "__PARENT_SOURCE_VALUE__": repr(base64.b64encode(parent_source.encode()).decode()),
        "__PERMISSION_VALUE__": repr(base64.b64encode(license_bytes).decode()),
        "__BINDING_VALUE__": repr(binding),
    }
    for marker, value in replacements.items():
        contract.require(template.count(marker) == 1, "One immutable worker placeholder required")
        template = template.replace(marker, value)
    ast.parse(template)
    return template, {
        "status": "predictive_final_offline_source_compiled_CPU_only",
        "mode": mode,
        "arms": [c["arm"] for c in candidates],
        "source_sha256": contract.digest(template),
        "source_bytes": len(template.encode()),
        "worker_sha256": worker_sha,
        "contract_sha256": contract_sha,
        "model_files": {c["model_filename"]: c["model"] for c in candidates},
        "pinned_Docker": contract.DOCKER,
        "actual_trained_GPU_execution": False,
        "new_Dataset_notebook_or_entry_mutations": 0,
        "qualification_may_create_submission_csv": False,
        "clinical_or_hidden_runtime_claim": False,
    }
