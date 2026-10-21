"""Execute the isolated serving capsule and reject unsafe account admission."""

import ast
import copy
import importlib.util
import json
import sqlite3
import subprocess
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CTL = load("C100_deployment_tests", HERE / "controls.py")
FIX = load("C100_deployment_fixtures", CTL.SERVING / "test_contract.py")


@pytest.fixture
def capsule(monkeypatch, tmp_path):
    _, entry = FIX.case.__wrapped__()
    options = FIX.options()
    monkeypatch.setattr(CTL, "FULL_SOURCE", options["full_source_git"])
    monkeypatch.setattr(CTL, "FULL_JOB", entry["full_terminal"]["GPU_job"])
    config = dict(
        owning_node="evc27",
        original_runs={"C": {"run_root": "/tmp/rsna-predictive-full-C100-20261020-v1"}},
        root="/tmp/rsna-predictive-final-deployment-20261020-v1-C100-synthetic",
        C100_contract_options=options,
        C100_final_evidence={k: entry[k] for k in CTL.EXTRA_KEYS},
    )
    files = CTL.build(config)
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    facade = load("staged_C100_facade_test", tmp_path / "serving/compile.py")
    facade.adapter._base.LICENSE = FIX.C.digest(entry["license_bytes"])
    policy = load("staged_C100_policy_test", tmp_path / "policy.py")
    return config, files, entry, facade, policy


def test_staged_facade_admits_only_its_original_final(capsule):
    _, _, entry, facade, _ = capsule
    export = {k: v for k, v in entry.items() if k not in CTL.EXTRA_KEYS}
    chosen = facade.contract.candidate(**export)
    assert chosen["branch"] == "C100"
    assert chosen["model_filename"] == "predictive-C100-epoch-006.pt"
    assert chosen["provenance"]["recipe"]["density_weight"] == 100
    bad = copy.deepcopy(export)
    p = json.loads(bad["provenance_text"])
    p["branch"] = "C"
    bad["provenance_text"] = json.dumps(p)
    with pytest.raises(ValueError):
        facade.contract.candidate(**bad)
    facade.EVIDENCE["epoch_publications"].pop()
    with pytest.raises(ValueError, match="All six"):
        facade.contract.candidate(**export)


def test_staged_compiler_executes_and_binds_exact_qualification(capsule, monkeypatch):
    _, files, entry, facade, _ = capsule
    parent = "synthetic parent; no pixels or clinical weights"
    facade.adapter._base.PARENT_SOURCE = FIX.C.digest(parent)
    FIX.rewrite(entry, lambda p: p.update(parent_source_sha256=FIX.C.digest(parent)))
    entry["full_terminal"]["parent_source_sha256"] = FIX.C.digest(parent)
    facade.EVIDENCE = {k: entry[k] for k in CTL.EXTRA_KEYS}
    monkeypatch.setattr(facade.implementation.C, "Contract", lambda *_a, **_kw: facade.adapter)
    export = {k: v for k, v in entry.items() if k not in CTL.EXTRA_KEYS}
    source, receipt = facade.build(
        parent, [export], mode="qualification", model_dataset_ref="godlydonuts/synthetic-c100"
    )
    embedded = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("FILES", "BINDING", "PERMISSION", "PARENT_SOURCE")
    }
    assert embedded["BINDING"]["branch"] == "C100"
    assert embedded["BINDING"]["models"] == [facade.contract.candidate(**export)]
    original = files["serving/worker.py.txt"]
    for key, value in embedded.items():
        original = original.replace("__" + key + "_VALUE__", repr(value))
    assert original == source  # Decoder, graph, precision and CSV handling unchanged.
    assert receipt["actual_trained_GPU_execution"] is False
    assert receipt["account_budget_admitted"] is False
    q = FIX.qualification(facade.contract.candidate(**export), facade.adapter._base)
    q.update(
        serving_worker_sha256=receipt["worker_sha256"],
        serving_contract_sha256=receipt["contract_sha256"],
        serving_helpers_sha256=receipt["serving_helpers_sha256"],
        model_dataset_ref=receipt["model_dataset_ref"],
    )
    facade.build(
        parent,
        [export],
        mode="competition",
        model_dataset_ref=receipt["model_dataset_ref"],
        qualification=q,
    )
    q["models"][0]["candidate_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        facade.build(
            parent,
            [export],
            mode="competition",
            model_dataset_ref=receipt["model_dataset_ref"],
            qualification=q,
        )
    with pytest.raises(ValueError, match="One original"):
        facade.build(
            parent,
            [export, export],
            mode="qualification",
            model_dataset_ref=receipt["model_dataset_ref"],
        )


@pytest.mark.parametrize("fault", ["node", "run", "incomplete", "pilot", "source", "pending"])
def test_preparation_does_not_accept_pilot_or_incomplete_fit(capsule, fault):
    config = copy.deepcopy(capsule[0])
    evidence = config["C100_final_evidence"]
    if fault == "node":
        config["owning_node"] = "evc33"
    elif fault == "run":
        config["original_runs"]["C"]["run_root"] = "/tmp/rsna-predictive-full-C-20261020-v1"
    elif fault == "incomplete":
        evidence["full_terminal"]["updates"] = 100
    elif fault == "pilot":
        evidence["full_terminal"]["GPU_job"] = "981620"
    elif fault == "source":
        config["C100_contract_options"]["full_source_git"] = "c" * 40
    else:
        evidence["full_scheduler"]["state"] = "PENDING"
    with pytest.raises(ValueError):
        CTL.build(config)


def test_canonical_account_code_and_pins_are_preserved(capsule):
    _, files, _, _, policy = capsule
    assert policy.ARMS == ("C",)
    assert policy.FAMILY != policy.PRIMARY_FAMILY
    assert policy.sha(files["common/workflow.py"]) == policy.WORKFLOW_SHA
    assert policy.sha(files["common/gpu_capacity.py"]) == policy.CAPACITY_SHA
    original = CTL.git(CTL.ORIGINAL + "account.py")
    expected = original.replace(
        'kind="predictive_final_deployment_v1"', 'kind="predictive_C100_final_deployment_v1"'
    )
    expected = expected.replace(
        '"RSNA predictive final B C C0"', '"RSNA predictive C100 fixed final"'
    )
    expected = expected.replace(
        "                actual_budget = p.budget(\n",
        "                plan = p.protect_primary_budget(plan, db)\n"
        "                actual_budget = p.budget(\n",
    )
    assert files["account.py"] == expected
    assert "coordinator-claim" in files["account.py"]
    assert '"fileName": "submission.csv"' in files["account.py"]
    assert '"godlydonuts/rsna-predictive-c100-final-"' in files["cpu.py"]


def budget_policy():
    ns = types.ModuleType("primary_policy_test")
    exec(CTL.git(CTL.ORIGINAL + "policy.py"), ns.__dict__)
    exec((HERE / "primary_budget.py").read_text(), ns.__dict__)
    return ns


def primary_rows(p):
    return {
        a: dict(
            bundle_identity="d" * 64,
            mode=a,
            state="predictive_final_submitted",
            submission_id=1000 + i,
            version=1,
            model_files={
                "predictive-" + a + "-epoch-006.pt": {"sha256": p.PRIMARY_MODELS.get(a, "e" * 64)}
            },
        )
        for i, a in enumerate(("B", "C", "C0"))
    }


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "absent",
        "incomplete",
        "uncertain",
        "different_bundle",
        "wrong_model",
        "duplicate_entry",
        "zero_entry",
        "unrelated",
    ],
)
def test_primary_allowance_can_only_be_retired_by_canonical_submissions(fault):
    p = budget_policy()
    rows = primary_rows(p)
    if fault == "absent":
        rows = {}
    elif fault == "incomplete":
        del rows["C0"]
    elif fault == "uncertain":
        rows["C0"]["state"] = "predictive_final_push_intent"
    elif fault == "different_bundle":
        rows["C0"]["bundle_identity"] = "f" * 64
    elif fault == "wrong_model":
        rows["C"]["model_files"]["predictive-C-epoch-006.pt"]["sha256"] = "f" * 64
    elif fault == "duplicate_entry":
        rows["C0"]["submission_id"] = rows["B"]["submission_id"]
    elif fault == "zero_entry":
        rows["C0"]["submission_id"] = 0
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        for mode, row in rows.items():
            key = (
                (p.PRIMARY_FAMILY if fault != "unrelated" else "other:")
                + row["bundle_identity"]
                + ":"
                + mode
            )
            db.execute("INSERT INTO meta VALUES (?,?)", (key, json.dumps(row)))
        protected = p.protect_primary_budget({"other_reserved_VM_seconds": 0}, db)
        assert protected["other_reserved_VM_seconds"] == (0 if fault is None else 3360)
        assert (
            p.protect_primary_budget({"other_reserved_VM_seconds": 5000}, db)[
                "other_reserved_VM_seconds"
            ]
            == 5000
        )
        plan = {**protected, "qualification_cap_seconds": 600, "public_watchdog_seconds": 600}
        result = p.budget(plan, {"totalTimeAllowed": "3587.5076641s"}, "qualification", 1)
        assert result["available"] == (fault is None)


def test_one_phase_entrypoint_has_no_implicit_remote_action():
    runner = load("C100_single_phase_test", HERE / "run_once.py")
    ns = runner.namespace()
    assert ns["ACCOUNT_PHASES"] == (
        "dataset",
        "dataset-reconcile",
        "push",
        "status",
        "submit",
        "outcome",
    )
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        ns["controls"](CTL.PIN, {})
