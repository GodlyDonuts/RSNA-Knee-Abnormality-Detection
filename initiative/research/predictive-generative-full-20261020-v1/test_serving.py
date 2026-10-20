"""G100 uses the original inference graph but cannot borrow another run's proof."""

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load("G100_serving_tests", HERE / "serving.py")
FULL = load("G100_serving_full_fixtures", HERE / "test_full_worker.py")
OLD = load("G100_serving_original_fixtures", HERE.parent / "predictive-density100-serving-20261020-v1/test_contract.py")


def options():
    value = OLD.options()
    value["pilot_receipt_text"] = json.dumps(FULL.fixture({})[1], sort_keys=True)
    value["pilot_receipt_sha256"] = OLD.C.digest(value["pilot_receipt_text"])
    value["pilot_scheduler"] = OLD.scheduler("986011")
    return value


@pytest.fixture
def prepared():
    compiled = S.compiler()
    adapter = compiled.C.Contract(OLD.SOURCE, **options())
    _, entry = OLD.case.__wrapped__()
    adapter._base.LICENSE = compiled.C.digest(entry["license_bytes"])

    def convert(row):
        result = {k.replace("density100_pilot", "generative_pilot"): v for k, v in row.items()}
        result.update(
            branch="G100", generative_pilot_sha256=adapter.pilot_receipt_sha256,
            generative_pilot_source_git=adapter.pilot["source_git"],
            generative_pilot_GPU_job="986011", **compiled.C.ROUTING_PROVENANCE,
        )
        return result

    p = convert(json.loads(entry["provenance_text"]))
    p["code_sha256"]["rsna/competitive/predictive/generative_routing.py"] = compiled.C.ROUTING_SHA
    text = json.dumps(p, sort_keys=True)
    entry["provenance_text"] = text
    publications = []
    for pub in entry["epoch_publications"]:
        pub = convert(pub)
        pub["prefix"] = pub["prefix"].replace("C100", "G100")
        pub["files"]["provenance.json"] = {"bytes": len(text), "sha256": compiled.C.digest(text)}
        publications.append(pub)
    entry["epoch_publications"] = publications
    entry["publication"] = publications[-1]
    entry["full_terminal"] = convert(entry["full_terminal"])
    return compiled, adapter, entry


def embedded(text):
    return {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(text).body
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in {"FILES", "BINDING", "PERMISSION", "PARENT_SOURCE"}
    }


def test_g100_inference_same_architecture_and_unchanged_worker(prepared, monkeypatch):
    compiled, adapter, entry = prepared
    chosen = adapter.candidate(**entry)
    assert chosen["branch"] == "G100" and chosen["arm"] == "C"
    assert chosen["architecture"] == adapter._base.architecture("C")
    assert chosen["model_filename"] == "predictive-G100-epoch-006.pt"
    assert chosen["provenance"]["exam_gradients_through_Bayes_factor"] is False
    assert chosen["provenance"]["proper_conditional_density_gradients_attached"] is True

    parent = "synthetic test parent; no MRI, labels or weights"
    adapter._base.PARENT_SOURCE = compiled.C.digest(parent)
    OLD.rewrite(entry, lambda p: p.update(parent_source_sha256=compiled.C.digest(parent)))
    entry["full_terminal"]["parent_source_sha256"] = compiled.C.digest(parent)
    monkeypatch.setattr(compiled.C, "Contract", lambda *args, **kwargs: adapter)
    arguments = dict(contract_options=options(), model_dataset_ref="godlydonuts/synthetic-g100")
    script, receipt = compiled.build(parent, entry, mode="qualification", **arguments)
    values = embedded(script)
    assert values["BINDING"]["branch"] == "G100"
    assert values["BINDING"]["GPU_count"] == 2
    assert values["BINDING"]["public_watchdog_seconds"] == 600
    assert receipt["account_budget_admitted"] is False
    assert receipt["actual_trained_GPU_execution"] is False
    for name, sha in compiled.C.MODULE_SHA256.items():
        assert compiled.C.digest(values["FILES"][name]) == sha
    original = compiled.originals()["worker.py.txt"]
    for key, value in values.items():
        original = original.replace("__" + key + "_VALUE__", repr(value))
    assert original == script

    q = OLD.qualification(adapter.candidate(**entry), adapter._base)
    q.update(serving_worker_sha256=receipt["worker_sha256"],
             serving_contract_sha256=receipt["contract_sha256"],
             serving_helpers_sha256=receipt["serving_helpers_sha256"],
             model_dataset_ref=receipt["model_dataset_ref"])
    compiled.build(parent, entry, mode="competition", qualification=q, **arguments)
    for key in ("serving_contract_sha256", "model_dataset_ref"):
        invalid = {**q, key: "different"}
        with pytest.raises(ValueError):
            compiled.build(parent, entry, mode="competition", qualification=invalid, **arguments)
    with pytest.raises(ValueError):
        compiled.build(parent, entry, mode="competition", **arguments)


@pytest.mark.parametrize("problem", ["C100_pilot", "pending", "partial", "bad_LR", "no_recovery", "different_bytes"])
def test_no_old_or_partial_engineering_result_can_admit_g100(problem):
    compiled = S.compiler()
    opt = options()
    pilot = json.loads(opt["pilot_receipt_text"])
    if problem == "C100_pilot":
        opt = OLD.options()
    elif problem == "pending":
        opt["pilot_scheduler"]["state"] = "PENDING"
    else:
        if problem == "partial":
            pilot["roles"][1]["real_optimizer_history"].pop()
        elif problem == "bad_LR":
            pilot["roles"][0]["real_optimizer_history"][145]["LR"] = [1e-4, 1e-4]
        elif problem == "no_recovery":
            pilot["G100_full_stop_resume_complete_state_and_prediction_bytes_equal"] = False
        opt["pilot_receipt_text"] = json.dumps(pilot, sort_keys=True)
        opt["pilot_receipt_sha256"] = compiled.C.digest(opt["pilot_receipt_text"])
        if problem == "different_bytes":
            opt["pilot_receipt_text"] += "\n"
    with pytest.raises(ValueError):
        compiled.C.Contract(OLD.SOURCE, **opt)


@pytest.mark.parametrize("problem", ["wrong_routing", "wrong_source", "old_branch", "missing_epoch", "partial_fit", "terminal_mismatch"])
def test_other_model_or_incomplete_fit_rejected(prepared, problem):
    _, adapter, entry = prepared
    if problem == "wrong_routing":
        OLD.rewrite(entry, lambda p: p.update(exam_gradients_through_Bayes_factor=True))
    elif problem == "wrong_source":
        OLD.rewrite(entry, lambda p: p["code_sha256"].update({
            "rsna/competitive/predictive/generative_routing.py": "0" * 64,
        }))
    elif problem == "old_branch":
        OLD.rewrite(entry, lambda p: p.update(branch="C100"))
    elif problem == "missing_epoch":
        entry["epoch_publications"].pop(0)
    elif problem == "partial_fit":
        entry["full_terminal"]["updates"] = 200
    else:
        entry["full_terminal"]["proper_conditional_density_gradients_attached"] = False
    with pytest.raises(ValueError):
        adapter.candidate(**entry)


def test_embedded_standalone_contract_executes_same_export_and_csv_validation(prepared):
    compiled, adapter, entry = prepared
    runtime = compiled.runtime_contract(OLD.SOURCE, compiled.ADAPTER_SOURCE, options())
    ns = {}
    exec(compile(runtime, "standalone_G100_runtime", "exec"), ns)
    chosen = adapter.candidate(**entry)
    model = torch.nn.BatchNorm1d(2)
    state = copy.deepcopy(model.state_dict())
    state["running_mean"].fill_(5)
    chosen["provenance"]["parameter_count"] = sum(p.numel() for p in model.parameters())
    chosen["provenance"]["model_schema_sha256"] = compiled.C.digest(compiled.C.canonical({
        n: {"shape": list(v.shape), "dtype": str(v.dtype)} for n, v in state.items()
    }))
    export = dict(model=state, architecture=chosen["architecture"], provenance=chosen["provenance"])
    ns["validate_export"](export, chosen, model)
    assert torch.equal(model.running_mean, torch.tensor([5., 5.]))
    ns["check_predictions"](["1.2.3"], [[.5] * 12])
    with pytest.raises(ValueError):
        ns["check_predictions"](["1.2.3"], [[float("nan")] * 12])
    chosen["branch"] = "C100"
    with pytest.raises(ValueError):
        ns["validate_export"](export, chosen, model)
