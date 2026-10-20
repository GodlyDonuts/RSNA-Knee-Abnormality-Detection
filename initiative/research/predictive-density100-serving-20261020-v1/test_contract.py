"""Synthetic C100 admission failures and unchanged inference/precision contracts."""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("C100_serving_adapter_test", HERE / "contract.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
OLD = REPO / "initiative/research/predictive-final-serving-20261020-v1/contract.py"
SOURCE = OLD.read_text()


def encode(value):
    return json.dumps(value, sort_keys=True)


def scheduler(job):
    return dict(job_id=job, state="COMPLETED", exit_code="0:0", queue_empty=True, node="evc27")


def pilot():
    return {
        "status": "actual_C100_full_stop_resume_engineering_complete",
        "branch": "C100", "source_git": C.PILOT_SOURCE, "GPU_job": C.PILOT_JOB,
        "binding_sha256": C.PILOT_BINDING, "original_numerical_source_git": C.PILOT_NUMERICAL_SOURCE,
        "original_engineering_receipt_sha256": C.PILOT_ORIGINAL_RECEIPT,
        "objective_measurement_sha256": C.OBJECTIVE_RECEIPT,
        "full_stop_resume_complete_state_and_prediction_bytes_equal": True,
        "clinical_or_held_out_accuracy_claim": False, "automatic_full_training_promotion": False,
        "primary_B_C_C0_pipeline_unchanged": True,
        "real_optimizer_history": [{"step": i} for i in range(1, 25)],
        "terminal": {
            "arm": "C", "phase": "full", "steps": 24, "density_weight": 100,
            "source_studies": 12, "disjoint_engineering_prediction_studies": 4,
            "production_LR_horizon_updates": 2916, "clinical_accuracy_or_submission_candidate": False,
        },
    }


def options():
    text = encode(pilot())
    return dict(full_source_git="b" * 40, publisher_sha256="b" * 64,
                pilot_receipt_text=text, pilot_receipt_sha256=C.digest(text),
                pilot_scheduler=scheduler(C.PILOT_JOB))


def original_sources():
    values = {}
    for name in C.MODULE_SHA256:
        values[name] = (
            "" if name in ("rsna/__init__.py", "rsna/competitive/__init__.py")
            else subprocess.check_output(
                ["git", "show", C.ORIGINAL_FULL_SOURCE + ":" + name], cwd=REPO,
            ).decode()
        )
    assert {n: C.digest(v) for n, v in values.items()} == C.MODULE_SHA256
    return values


@pytest.fixture
def case():
    adapter = C.Contract(SOURCE, **options())
    base = adapter._base
    license_bytes = b"synthetic fixture permission; no model or clinical data"
    base.LICENSE = C.digest(license_bytes)
    own = encode(original_sources())
    p = {
        "arm": "C", "branch": "C100", "source_git": adapter.full_source_git,
        "base_science_git": C.ORIGINAL_FULL_SOURCE,
        "density100_pilot_sha256": adapter.pilot_receipt_sha256,
        "density100_pilot_source_git": adapter.pilot["source_git"], "density100_pilot_GPU_job": adapter.pilot["GPU_job"],
        "original_portable_admission_sha256": "7" * 64,
        "original_C0_binding_sha256": "8" * 64,
        "publisher_source_git": adapter.full_source_git,
        "GPU_job": "900002", "epoch": 6, "completed_updates": 2916, "cohort_size": 3887,
        "recipe": base.recipe("C"), "parent_weight_sha256": base.PARENT,
        "parent_source_sha256": base.PARENT_SOURCE, "source_revision": base.SOURCE_REVISION,
        "targets_sha256": base.TARGETS, "fitting_order_sha256": base.ROSTER,
        "license_sha256": base.LICENSE, "parameter_count": base.PARAMETERS["C"],
        "model_schema_sha256": base.SCHEMAS["C"], "publisher_sha256": adapter.publisher_sha256,
        "original_publisher_core_sha256": base.PUBLISHER_CORE,
        "original_numerical_source_git": base.ORIGINAL_SCIENCE,
        "alias_regression_receipt_sha256": base.ALIAS_REGRESSION,
        "accepted_renderer_sha256": [base.ORIGINAL_RENDERERS, base.CORRECTED_RENDERERS],
        "renderer_sha256": base.CORRECTED_RENDERERS,
        "renderer_provenance_counts": {"original": 500, "corrected": 3387},
        "model_source": "public_licensed_MRI_parent_and_unchanged_RSNA_fitting_only",
        "additional_Stanford_or_KneeCoT_or_teacher_probabilities_used": False,
        "optimizer_RNG_private_paths_labels_IDs_exported": False,
        "code_sha256": dict(C.MODULE_SHA256),
        "initial_shared_tensor_sha256": C.INITIAL,
        "runtime_versions": {"torch": "2.14.0+cu130", "timm": "1.0.30", "cuda": "13.0"},
    }
    for name in (
        "binding", "context_index", "source_context_index", "runtime_receipt",
        "engineering_decision", "input_admission", "funding", "checkpoint", "history", "sealed_sidecar",
    ):
        p[name + "_sha256"] = "d" * 64
    text = encode(p)
    files = {
        name: {"bytes": len(body), "sha256": C.digest(body)}
        for name, body in (
            ("provenance.json", text.encode()), ("own-source.json", own.encode()),
            ("source-permission.txt", license_bytes), ("manifest.json", b"synthetic"),
        )
    }
    files["model.pt"] = {"bytes": 293300000, "sha256": "e" * 64}
    pubs = []
    for epoch in range(1, 7):
        pubs.append({
            "status": C.PUBLICATION_STATUS, "objects": 5,
            "canonical_shared_publication_lock_used": True, "files": copy.deepcopy(files),
            "revision": format(epoch, "040x"), "epoch": epoch,
            "arm": "C", "branch": "C100", "source_git": adapter.full_source_git,
            "base_science_git": C.ORIGINAL_FULL_SOURCE,
            "density100_pilot_sha256": adapter.pilot_receipt_sha256,
            "density100_pilot_source_git": adapter.pilot["source_git"], "density100_pilot_GPU_job": adapter.pilot["GPU_job"],
            "original_portable_admission_sha256": p["original_portable_admission_sha256"],
            "original_C0_binding_sha256": p["original_C0_binding_sha256"],
            "GPU_job": p["GPU_job"],
            "prefix": f"checkpoints/predictive-C100-full3887-{adapter.full_source_git[:12]}/C/epoch-{epoch:03d}",
        })
    terminal = {name: p[name] for name in C.FINAL_MATCH_FIELDS}
    terminal.update(
        status="actual_fixed_final_full3887_predictive_arm_complete", epochs=6, updates=2916,
        study_exposures=3887 * 6, retained_full_checkpoints=7, immutable_epoch_sidecars=6,
        all_original_fitting_labels_masks_unchanged=True, protected_or_gold_cases_read=False,
        final_checkpoint_sha256="f" * 64, final_checkpoint_bytes=900000000,
    )
    args = dict(
        publication=pubs[-1], provenance_text=text, own_source_text=own, license_bytes=license_bytes,
        epoch_publications=pubs, full_terminal=terminal, full_scheduler=scheduler("900002"),
    )
    return adapter, args


def rewrite(args, change):
    p = json.loads(args["provenance_text"])
    change(p)
    text = encode(p)
    args["provenance_text"] = text
    args["publication"]["files"]["provenance.json"] = {
        "bytes": len(text.encode()), "sha256": C.digest(text),
    }


def test_complete_candidate_distinct_identity_and_same_original_architecture(case):
    adapter, args = case
    original_bytes = OLD.read_bytes()
    chosen = adapter.candidate(**args)
    assert chosen["arm"] == "C" and chosen["branch"] == "C100"
    assert chosen["model_filename"] == "predictive-C100-epoch-006.pt"
    assert chosen["architecture"] == adapter._base.architecture("C")
    assert chosen["provenance"]["recipe"]["density_weight"] == 100
    assert chosen["provenance"]["initial_shared_tensor_sha256"] == C.INITIAL
    assert chosen["full_terminal_sha256"] == C.digest(C.canonical(args["full_terminal"]))
    assert chosen["six_publications_sha256"] == C.digest(C.canonical(args["epoch_publications"]))
    assert chosen["pilot_scheduler_sha256"] == C.digest(C.canonical(options()["pilot_scheduler"]))
    assert OLD.read_bytes() == original_bytes
    ns = {}
    exec(compile(SOURCE, "untouched_original_contract", "exec"), ns)
    assert ns["recipe"]("C")["density_weight"] == 0.1
    assert ns["PUBLISHER"] != adapter._base.PUBLISHER


@pytest.mark.parametrize("field", [
    "full_source_git", "publisher_sha256", "pilot_receipt_text",
    "pilot_receipt_sha256", "pilot_scheduler",
])
def test_missing_finalized_binding_fails_closed(field):
    value = options()
    value.pop(field)
    with pytest.raises(ValueError):
        C.Contract(SOURCE, **value)


@pytest.mark.parametrize("problem", [
    "pending", "failed", "wrong_job", "wrong_node", "in_queue", "partial",
    "wrong_coefficient", "wrong_horizon", "no_recovery", "wrong_original", "changed_bytes",
    "substitute_job", "substitute_binding",
])
def test_pending_or_incomplete_pilot_cannot_authorize_full_candidate(problem):
    value = options()
    p = json.loads(value["pilot_receipt_text"])
    if problem in ("pending", "failed"):
        value["pilot_scheduler"]["state"] = problem.upper()
    elif problem == "wrong_job":
        value["pilot_scheduler"]["job_id"] = "999999"
    elif problem == "substitute_job":
        p["GPU_job"] = value["pilot_scheduler"]["job_id"] = "999999"
    elif problem == "substitute_binding":
        p["binding_sha256"] = "a" * 64
    elif problem == "wrong_node":
        value["pilot_scheduler"]["node"] = "evc33"
    elif problem == "in_queue":
        value["pilot_scheduler"]["queue_empty"] = False
    elif problem == "partial":
        p["terminal"]["steps"] = 23
    elif problem == "wrong_coefficient":
        p["terminal"]["density_weight"] = 0.1
    elif problem == "wrong_horizon":
        p["terminal"]["production_LR_horizon_updates"] = 24
    elif problem == "no_recovery":
        p["full_stop_resume_complete_state_and_prediction_bytes_equal"] = False
    elif problem == "wrong_original":
        p["original_numerical_source_git"] = "0" * 40
    value["pilot_receipt_text"] = encode(p)
    if problem != "changed_bytes":
        value["pilot_receipt_sha256"] = C.digest(value["pilot_receipt_text"])
    else:
        value["pilot_receipt_text"] += "\n"
    with pytest.raises(ValueError):
        C.Contract(SOURCE, **value)


@pytest.mark.parametrize("problem", [
    "branch", "full_source", "pilot_pin", "publisher", "coefficient", "initial",
    "short_fit", "no_final", "scheduler", "missing_epoch", "duplicate_epoch", "unverified",
    "wrong_epoch_job", "wrong_epoch_branch", "old_prefix", "wrong_epoch_source",
    "wrong_epoch_model", "wrong_epoch_license", "module",
    "missing_portable", "epoch_portable", "terminal_C0_binding",
])
def test_other_arm_source_recipe_partial_fit_or_incomplete_preservation_rejected(case, problem):
    adapter, args = case
    if problem == "branch":
        rewrite(args, lambda p: p.update(branch="C"))
    elif problem == "full_source":
        rewrite(args, lambda p: p.update(source_git=C.ORIGINAL_FULL_SOURCE))
    elif problem == "pilot_pin":
        rewrite(args, lambda p: p.update(density100_pilot_sha256="0" * 64))
    elif problem == "publisher":
        rewrite(args, lambda p: p.update(publisher_sha256="0" * 64))
    elif problem == "coefficient":
        rewrite(args, lambda p: p["recipe"].update(density_weight=0.1))
    elif problem == "initial":
        rewrite(args, lambda p: p.update(initial_shared_tensor_sha256="0" * 64))
    elif problem == "short_fit":
        args["full_terminal"]["updates"] = 24
    elif problem == "no_final":
        args["full_terminal"].pop("final_checkpoint_sha256")
    elif problem == "scheduler":
        args["full_scheduler"]["state"] = "RUNNING"
    elif problem == "missing_epoch":
        args["epoch_publications"] = args["epoch_publications"][1:]
    elif problem == "duplicate_epoch":
        args["epoch_publications"][1] = args["epoch_publications"][0]
    elif problem == "unverified":
        args["epoch_publications"][2]["canonical_shared_publication_lock_used"] = False
    elif problem == "wrong_epoch_job":
        args["epoch_publications"][1]["GPU_job"] = "900003"
    elif problem == "wrong_epoch_branch":
        args["epoch_publications"][1]["branch"] = "C0"
    elif problem == "old_prefix":
        args["publication"]["prefix"] = args["publication"]["prefix"].replace("C100-", "")
    elif problem == "wrong_epoch_source":
        args["epoch_publications"][1]["source_git"] = "e" * 40
    elif problem == "wrong_epoch_model":
        args["epoch_publications"][1]["files"]["model.pt"]["bytes"] = 12
    elif problem == "wrong_epoch_license":
        args["epoch_publications"][1]["files"]["source-permission.txt"]["sha256"] = "0" * 64
    elif problem == "missing_portable":
        rewrite(args, lambda p: p.pop("original_portable_admission_sha256"))
    elif problem == "epoch_portable":
        args["epoch_publications"][1]["original_portable_admission_sha256"] = "0" * 64
    elif problem == "terminal_C0_binding":
        args["full_terminal"]["original_C0_binding_sha256"] = "0" * 64
    else:
        own = json.loads(args["own_source_text"])
        name = "rsna/competitive/predictive/model.py"
        own[name] += "\n# a graph-changing source must not be self-admitted\n"
        args["own_source_text"] = encode(own)
        args["publication"]["files"]["own-source.json"] = {
            "bytes": len(args["own_source_text"].encode()),
            "sha256": C.digest(args["own_source_text"]),
        }
        rewrite(args, lambda p: p["code_sha256"].update({name: C.digest(own[name])}))
    with pytest.raises(ValueError):
        adapter.candidate(**args)


def qualification(chosen, base):
    return {
        "status": "actual_trained_predictive_public_T4_qualification_accepted", "public_studies": 3,
        "GPU_names": ["Tesla T4", "Tesla T4"], "pinned_Docker": base.DOCKER,
        "submission_csv_absent": True, "hidden_runtime_acceptance": False,
        "models": [{
            "arm": "C", "candidate_sha256": C.digest(C.canonical(chosen)),
            "trained_export_sha256": chosen["model"]["sha256"],
            "precision_accepted": True, "memory_accepted": True,
            "FP16_FP32_tolerance": 0.005, "cross_device_FP16_tolerance": 0.0001,
            "cross_device_FP16_gap": 0.00001, "device_peak_bytes": [10000, 10000],
            "host_peak_bytes": 100000,
            "measurements": [
                dict(device=d, study_index=i, maximum_probability_gap=0.001,
                     FP16_seconds=1.0, FP32_seconds=2.0, windows=48, regions=100)
                for d, i in ((0, 0), (0, 2), (1, 1), (1, 0))
            ],
        }],
    }


def test_existing_T4_export_and_prediction_checks_still_apply(case):
    adapter, args = case
    chosen = adapter.candidate(**args)
    q = qualification(chosen, adapter._base)
    assert adapter.validate_qualification(q, chosen)["precision_accepted"]
    q["models"][0]["measurements"][2]["maximum_probability_gap"] = 0.006
    with pytest.raises(ValueError):
        adapter.validate_qualification(q, chosen)
    q = qualification(chosen, adapter._base)
    q["models"][0]["candidate_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        adapter.validate_qualification(q, chosen)
    adapter.check_predictions(["1.2.3"], [[0.5] * 12])
    with pytest.raises(ValueError):
        adapter.check_predictions(["1.2.3", "1.2.3"], [[0.5] * 12] * 2)
    with pytest.raises(ValueError):
        adapter.check_predictions(["1.2.3"], [[float("nan")] * 12])
    initialized = torch.nn.BatchNorm1d(2)
    state = {n: t.clone() for n, t in initialized.state_dict().items()}
    state["running_mean"].fill_(3)
    expected = copy.deepcopy(chosen)
    expected["provenance"]["parameter_count"] = sum(p.numel() for p in initialized.parameters())
    expected["provenance"]["model_schema_sha256"] = C.digest(C.canonical({
        n: {"shape": list(t.shape), "dtype": str(t.dtype)} for n, t in state.items()
    }))
    export = dict(model=state, architecture=expected["architecture"], provenance=expected["provenance"])
    adapter.validate_export(export, expected, initialized)
    assert torch.equal(initialized.running_mean, torch.tensor([3.0, 3.0]))
    export["model"].pop("running_mean")
    with pytest.raises(ValueError):
        adapter.validate_export(export, expected, initialized)


def test_source_pin_and_validation_function_bodies_remain_original():
    with pytest.raises(ValueError):
        C.Contract(SOURCE + "\n", **options())
    adapter = C.Contract(SOURCE, **options())
    expected = {}
    exec(compile(SOURCE, "isolated_original", "exec"), expected)
    for name in ("validate_export", "validate_qualification", "check_predictions"):
        # The functions execute their exact pinned original bytecode; only recipe/prefix globals differ.
        actual = getattr(adapter._base, name)
        assert actual.__code__.co_code == expected[name].__code__.co_code
        assert actual.__code__.co_consts == expected[name].__code__.co_consts
