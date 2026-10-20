"""Exercise C100 compilation and rejection of unqualified or substituted models."""

import ast
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = load("C100_source_compiler_tests", HERE / "compile.py")
FIX = load("C100_contract_fixtures_for_compile", HERE / "test_contract.py")


@pytest.fixture
def synthetic(monkeypatch):
    adapter, entry = FIX.case.__wrapped__()
    parent = "synthetic parent code; no clinical data or model weights"
    adapter._base.PARENT_SOURCE = FIX.C.digest(parent)
    FIX.rewrite(entry, lambda p: p.update(parent_source_sha256=FIX.C.digest(parent)))
    entry["full_terminal"]["parent_source_sha256"] = FIX.C.digest(parent)
    monkeypatch.setattr(BUILD.C, "Contract", lambda source, **options: adapter)
    return adapter, entry, parent


def embedded(source):
    names = {"FILES", "BINDING", "PERMISSION", "PARENT_SOURCE"}
    return {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names
    }


def build(entry, parent, **overrides):
    return BUILD.build(
        parent, entry, contract_options=FIX.options(),
        **{"mode": "qualification", "model_dataset_ref": "godlydonuts/synthetic-c100",
           **overrides},
    )


def test_qualification_source_keeps_original_worker_and_exact_numerical_modules(synthetic):
    adapter, entry, parent = synthetic
    source, receipt = build(entry, parent)
    values = embedded(source)
    chosen = adapter.candidate(**entry)
    binding = values["BINDING"]
    assert binding["models"] == [chosen]
    assert chosen["model_filename"] == "predictive-C100-epoch-006.pt"
    assert binding["branch"] == "C100"
    assert binding["mode"] == "qualification"
    assert binding["internet"] is False
    assert binding["GPU_count"] == binding["CPU_study_workers"] == 2
    assert binding["public_watchdog_seconds"] == 600
    assert binding["qualification"] is None
    assert receipt["account_budget_admitted"] is False
    assert receipt["actual_trained_GPU_execution"] is False
    assert receipt["new_Dataset_notebook_or_entry_mutations"] == 0
    assert receipt["primary_predictive_VM_reservation_seconds"] == 3360
    assert receipt["source_sha256"] == FIX.C.digest(source)
    for name, digest in FIX.C.MODULE_SHA256.items():
        assert FIX.C.digest(values["FILES"][name]) == digest
    # Only the four original placeholders are replaced. No inference branch,
    # decoder, output writer, precision tolerance or time limit is rewritten.
    expected = (BUILD.ORIGINAL / "worker.py.txt").read_text()
    for name in ("FILES", "BINDING", "PERMISSION", "PARENT_SOURCE"):
        marker = "__" + name + "_VALUE__"
        expected = expected.replace(marker, repr(values[name]))
    assert source == expected


def test_generated_runtime_contract_runs_original_validation_and_preserves_buffers():
    options = FIX.options()
    runtime = BUILD.runtime_contract(
        FIX.SOURCE, (HERE / "contract.py").read_text(), options,
    )
    namespace = {}
    exec(compile(runtime, "synthetic_C100_runtime_contract", "exec"), namespace)
    assert namespace["_adapter"]._base.recipe("C")["density_weight"] == 100
    namespace["check_predictions"](["1.2.3"], [[0.25] * 12])
    with pytest.raises(ValueError):
        namespace["check_predictions"](["1.2.3"], [[float("nan")] * 12])
    with pytest.raises(ValueError):
        namespace["check_predictions"](["1.2.3", "1.2.3"], [[0.25] * 12] * 2)
    adapter, entry = FIX.case.__wrapped__()
    chosen = adapter.candidate(**entry)
    model = torch.nn.BatchNorm1d(2)
    tensors = {n: t.clone() for n, t in model.state_dict().items()}
    tensors["running_mean"].fill_(4)
    expected = copy.deepcopy(chosen)
    expected["provenance"]["parameter_count"] = sum(p.numel() for p in model.parameters())
    expected["provenance"]["model_schema_sha256"] = FIX.C.digest(FIX.C.canonical({
        n: {"shape": list(t.shape), "dtype": str(t.dtype)} for n, t in tensors.items()
    }))
    export = dict(model=tensors, architecture=expected["architecture"],
                  provenance=expected["provenance"])
    namespace["validate_export"](export, expected, model)
    assert torch.equal(model.running_mean, torch.tensor([4.0, 4.0]))
    expected["branch"] = "C"
    with pytest.raises(ValueError):
        namespace["validate_export"](export, expected, model)


@pytest.mark.parametrize("failure", [
    "no_qualification", "different_model", "different_dataset",
    "different_worker", "different_contract", "different_helpers", "precision_failure",
])
def test_competition_requires_its_own_actual_trained_qualification(synthetic, failure):
    adapter, entry, parent = synthetic
    _, receipt = build(entry, parent)
    q = FIX.qualification(adapter.candidate(**entry), adapter._base)
    q.update(
        serving_worker_sha256=receipt["worker_sha256"],
        serving_contract_sha256=receipt["contract_sha256"],
        serving_helpers_sha256=receipt["serving_helpers_sha256"],
        model_dataset_ref=receipt["model_dataset_ref"],
    )
    source, _ = build(entry, parent, mode="competition", qualification=q)
    assert embedded(source)["BINDING"]["mode"] == "competition"
    if failure == "no_qualification":
        q = None
    elif failure == "different_model":
        q["models"][0]["candidate_sha256"] = "0" * 64
    elif failure == "different_dataset":
        q["model_dataset_ref"] = "godlydonuts/different"
    elif failure == "different_worker":
        q["serving_worker_sha256"] = "0" * 64
    elif failure == "different_contract":
        q["serving_contract_sha256"] = "0" * 64
    elif failure == "different_helpers":
        q["serving_helpers_sha256"] = {}
    else:
        q["models"][0]["precision_accepted"] = False
    with pytest.raises(ValueError):
        build(entry, parent, mode="competition", qualification=q)


@pytest.mark.parametrize("failure", ["parent", "incomplete_fit", "missing_epoch", "wrong_input"])
def test_compiler_cannot_promote_invalid_final_or_reinterpret_inputs(synthetic, failure):
    _, entry, parent = synthetic
    if failure == "parent":
        parent += "\nchanged"
    elif failure == "incomplete_fit":
        entry["full_terminal"]["updates"] = 24
    elif failure == "missing_epoch":
        entry["epoch_publications"].pop(0)
    else:
        entry["full_terminal"]["input_admission_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        build(entry, parent)


def test_original_worker_edit_or_extra_runtime_option_is_rejected(monkeypatch):
    old = Path.read_text

    def changed(path, *args, **kwargs):
        text = old(path, *args, **kwargs)
        return text + "\n" if path == BUILD.ORIGINAL / "worker.py.txt" else text

    monkeypatch.setattr(Path, "read_text", changed)
    with pytest.raises(ValueError):
        BUILD.originals()
    with pytest.raises(ValueError):
        BUILD.runtime_contract(FIX.SOURCE, "", {**FIX.options(), "skip_validation": True})
