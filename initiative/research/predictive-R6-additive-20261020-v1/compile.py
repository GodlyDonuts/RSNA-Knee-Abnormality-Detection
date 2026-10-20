"""Conditional CPU-only combined compiler; generated external source stays private."""

import ast
import base64
import importlib.util
import json
import math
from pathlib import Path

D = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


supervisor = _load("_additive_supervisor", D / "supervisor.py")
predictive = _load(
    "_additive_predictive_compiler", D.parent / "predictive-final-serving-20261020-v1/compile.py",
)


def build(
    r6_driver_bytes, parent_source, c_export, *,
    qualification, model_dataset_ref, standalone_c_source_sha256,
    standalone_c_acceptance, mode="candidate",
):
    """Return private source only after the exact standalone C has actual acceptance.

    This calls the canonical final-export compiler; arbitrary precompiled C bodies
    cannot be supplied. One candidate run creates the full fused CSV and actual
    public receipt. Compilation grants no entry authority: accept_entry must
    validate that same run before the canonical coordinator may submit it.
    """
    require, digest = supervisor.require, supervisor.digest
    require(mode == "candidate", "Only one complete combined candidate mode is supported")
    require(
        isinstance(r6_driver_bytes, bytes) and digest(r6_driver_bytes) in supervisor.R6_DRIVER_SHAS,
        "One of the two exact admitted original R6 driver byte sequences required",
    )
    chosen = predictive.contract.candidate(**c_export)
    require(chosen["arm"] == "C", "Only the complete fixed-final two-mask C candidate is admitted")
    c_source, c_compilation = predictive.build(
        parent_source, [c_export], mode="competition",
        model_dataset_ref=model_dataset_ref, qualification=qualification,
    )
    c_bytes = c_source.encode()
    require(
        digest(c_bytes) == standalone_c_source_sha256 == c_compilation["source_sha256"],
        "Generated C source differs from the original standalone accepted source",
    )
    runtime = {key: qualification[key] for key in ("torch", "timm", "pydicom", "pinned_Docker")}
    serving = {
        key: qualification[key] for key in (
            "serving_worker_sha256", "serving_contract_sha256",
            "serving_helpers_sha256", "model_dataset_ref",
        )
    }
    # Check the retained standalone public acceptance without inventing its CSV bytes.
    proof = standalone_c_acceptance
    require(
        proof["status"] == "actual_predictive_exact_test_roster_submission_created"
        and proof["mode"] == "competition" and proof["actual_test_studies"] == 3
        and proof["test_roster_sha256"] == supervisor.PUBLIC_ROSTER
        and proof["submission_csv_absent"] is False
        and {key: proof[key] for key in runtime} == runtime
        and {key: proof[key] for key in serving} == serving
        and len(proof["models"]) == 1 and proof["models"][0]["arm"] == "C"
        and proof["models"][0]["trained_export_sha256"] == chosen["model"]["sha256"]
        and proof["models"][0]["candidate_sha256"] == digest(predictive.contract.canonical(chosen))
        and proof["models"][0]["precision_accepted"] is True
        and proof["models"][0]["memory_accepted"] is True
        and type(proof["CSV_bytes"]) is int and proof["CSV_bytes"] > 0
        and isinstance(proof["CSV_sha256"], str) and len(proof["CSV_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in proof["CSV_sha256"]),
        "Actual exact standalone C public acceptance and runtime are required",
    )
    fusion_bytes = (D / "fusion.py").read_bytes()
    supervisor_bytes = (D / "supervisor.py").read_bytes()
    binding = {
        "schema": "conditional_predictive_C_R6_fixed_rank_fusion_v1", "mode": mode,
        "source_sha256": {
            "C": digest(c_bytes), "R6": digest(r6_driver_bytes),
            "fusion": digest(fusion_bytes), "supervisor": digest(supervisor_bytes),
        },
        "C_model_sha256": chosen["model"]["sha256"],
        "C_candidate_sha256": digest(predictive.contract.canonical(chosen)),
        "C_runtime": runtime, "C_serving": serving,
        "trained_qualification_sha256": digest(supervisor.canonical(qualification)),
        "standalone_C_acceptance_sha256": digest(supervisor.canonical(proof)),
        "standalone_C_public_CSV_sha256": proof["CSV_sha256"],
        "total_watchdog_seconds": 32400, "fixed_reference_weight": 0.75,
        "fixed_candidate_weight": 0.25, "internet": False,
    }
    payload = {
        name: base64.b64encode(data).decode()
        for name, data in (
            ("C", c_bytes), ("R6", r6_driver_bytes),
            ("fusion", fusion_bytes), ("supervisor", supervisor_bytes),
        )
    }
    source = (
        "# PRIVATE: exact external R6 body embedded; never commit this generated file.\n"
        "import base64,hashlib,json,os\n"
        "os.umask(0o077)\n"
        f"_binding=json.loads({json.dumps(binding, sort_keys=True)!r})\n"
        f"_payload={payload!r}\n"
        "_source={k:base64.b64decode(v,validate=True) for k,v in _payload.items()}\n"
        "assert all(hashlib.sha256(v).hexdigest()==_binding['source_sha256'][k] "
        "for k,v in _source.items())\n"
        "_namespace={'__name__':'_private_combined_supervisor'}\n"
        "exec(compile(_source['supervisor'],'bound_combined_supervisor','exec'),_namespace)\n"
        "_namespace['run'](_source['C'],_source['R6'],_source['fusion'],_binding)\n"
    )
    ast.parse(source)
    return source, {
        "status": "conditional_C_R6_source_prepared_CPU_only",
        "mode": mode, "source_sha256": digest(source.encode()), "source_bytes": len(source.encode()),
        "binding": binding, "C_compilation_source_sha256": c_compilation["source_sha256"],
        "actual_combined_GPU_execution": False, "combined_runtime_accepted": False,
        "actual_public_receipt_and_runtime_entry_gate_required": True,
        "compilation_grants_entry_authority": False,
        "clinical_or_hidden_score_claim": False,
        "successful_candidate_creates_complete_submission_csv": True, "new_remote_mutations": 0,
    }


def accept_entry(binding, actual_public_receipt, runtime_decision):
    """Pure post-run gate for the already executed candidate; no recompilation or submission.

    The decision must bind the actual combined receipt, source/model hashes,
    component projections, startup, tail margin, study count and evidence hashes.
    It is supplied by the caller after evidence review, never inferred here from
    the old R6 public time, model scores or apparent spare GPU time.

    The canonical coordinator must additionally bind the returned gate to the
    authenticated same kernel/version, complete wrapper source and mounted
    Dataset versions. Service metadata is not invented by this CPU helper.
    """
    receipt, decision = actual_public_receipt, runtime_decision
    require, digest = supervisor.require, supervisor.digest
    require(isinstance(receipt, dict) and isinstance(decision, dict),
            "Actual candidate public receipt and explicit hidden runtime decision required")
    require(
        binding["mode"] == "candidate"
        and receipt["status"] == "actual_sequential_C_then_exact_R6_fixed_rank_fusion_completed"
        and receipt["mode"] == "candidate" and receipt["submission_csv_created"] is True
        and receipt["binding_sha256"] == digest(supervisor.canonical(binding))
        and receipt["source_sha256"] == binding["source_sha256"]
        and receipt["C_model_sha256"] == binding["C_model_sha256"]
        and receipt["C_runtime"] == binding["C_runtime"]
        and receipt["C_serving"] == binding["C_serving"]
        and receipt["test_roster_sha256"] == supervisor.PUBLIC_ROSTER and receipt["studies"] == 3
        and receipt["total_watchdog_seconds"] == 32400
        and math.isfinite(receipt["elapsed_seconds"]) and 0 < receipt["elapsed_seconds"] < 32400
        and receipt["stage_order"] == ["C", "R6"] and len(receipt["stages"]) == 2,
        "Actual matching complete candidate public execution required",
    )
    for stage, arm, csv_hash in zip(
        receipt["stages"], ("C", "R6"),
        (binding["standalone_C_public_CSV_sha256"], supervisor.R6_PUBLIC_CSV), strict=True,
    ):
        require(
            stage["arm"] == arm and stage["exit_code"] == 0
            and stage["source_sha256"] == binding["source_sha256"][arm]
            and stage["CSV_sha256"] == csv_hash
            and math.isfinite(stage["elapsed_seconds"]) and stage["elapsed_seconds"] > 0,
            "Combined child source/public CSV proof differs",
        )
    fusion = receipt["fusion"]
    require(
        fusion["status"] == "fixed_exploratory_rank_fusion_computed"
        and fusion["reference_csv_sha256"] == supervisor.R6_PUBLIC_CSV
        and fusion["candidate_csv_sha256"] == binding["standalone_C_public_CSV_sha256"]
        and fusion["official_ordered_roster_sha256"] == supervisor.PUBLIC_ROSTER
        and fusion["study_count"] == 3 and fusion["target_count"] == 12
        and fusion["columns"] == predictive.contract.COLUMNS
        and fusion["reference_weight"] == 0.75 and fusion["candidate_weight"] == 0.25
        and fusion["fallback_rows"] == 0
        and receipt["submission_csv_sha256"] == fusion["output_csv_sha256"]
        and isinstance(receipt["submission_csv_sha256"], str)
        and len(receipt["submission_csv_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in receipt["submission_csv_sha256"])
        and type(receipt["submission_csv_bytes"]) is int and receipt["submission_csv_bytes"] > 0,
        "Complete exact fixed-rule public fusion and final CSV proof required",
    )
    require(
        decision["status"] == "accepted_combined_C_R6_hidden_runtime_scenario"
        and decision["combined_public_receipt_sha256"] == digest(supervisor.canonical(receipt))
        and decision["source_sha256"] == binding["source_sha256"]
        and decision["C_model_sha256"] == binding["C_model_sha256"]
        and type(decision["planned_hidden_studies"]) is int
        and decision["planned_hidden_studies"] >= 1300
        and isinstance(decision["scenario_basis"], str) and bool(decision["scenario_basis"].strip())
        and decision["hidden_runtime_is_guaranteed"] is False,
        "Explicit evidence-bound conditional hidden runtime scenario required",
    )
    terms = ("C_projected_seconds", "R6_projected_seconds", "startup_seconds", "tail_margin_seconds")
    require(
        all(
            type(decision[key]) in (int, float) and math.isfinite(decision[key])
            and decision[key] > 0 for key in terms
        )
        and decision["startup_seconds"] >= 600
        and decision["tail_margin_seconds"] >= 1800
        and sum(decision[key] for key in terms) <= 32400
        and decision["projected_total_seconds"] == sum(decision[key] for key in terms),
        "Both model projections plus explicit startup/tail margin must fit nine hours",
    )
    for key in ("C_projection_evidence_sha256", "R6_projection_evidence_sha256"):
        value = decision[key]
        require(
            isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            "Both component projections require exact evidence bindings",
        )
    return {
        "status": "accepted_same_candidate_public_run_and_combined_runtime_for_entry",
        "binding_sha256": digest(supervisor.canonical(binding)),
        "actual_public_receipt_sha256": digest(supervisor.canonical(receipt)),
        "runtime_decision_sha256": digest(supervisor.canonical(decision)),
        "source_sha256": binding["source_sha256"],
        "C_model_sha256": binding["C_model_sha256"],
        "public_fusion_csv_sha256": receipt["submission_csv_sha256"],
        "planned_hidden_studies": decision["planned_hidden_studies"],
        "projected_total_seconds": decision["projected_total_seconds"],
        "same_kernel_version_source_Datasets_require_canonical_coordinator_binding": True,
        "recompile_or_second_public_run_required": False,
        "hidden_runtime_guaranteed": False, "clinical_or_hidden_score_claim": False,
        "competition_entry_submitted": False, "new_remote_mutations": 0,
    }
