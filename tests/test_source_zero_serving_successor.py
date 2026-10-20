"""Versioned serving controls preserve the actual compiled final and original protocol."""

import ast
import hashlib
import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

D = Path("initiative/research/source-zero-final-serving-20261020-v2")
OLD = Path("initiative/research/source-zero-final-serving-20261020-v1")
BASE = Path("initiative/research/S6-Kaggle-production-route-20261019-v1")


def load(path, name):
    loader = SourceFileLoader(name, str(path))
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(m)
    return m


adapter = load(D / "account.py.txt", "zero_successor_adapter")
old = load(OLD / "account.py.txt", "zero_original_adapter")
original = old.core((BASE / "serving_cloud.py.txt").read_text())
source = adapter.build(original)


def test_only_control_location_and_pre_stage_reconciliation_change():
    assert hashlib.sha256(original.encode()).hexdigest() == adapter.ORIGINAL_PROGRAM_SHA
    ns = {"__name__": "synthetic_no_remote"}
    exec(compile(source, "successor", "exec"), ns)
    assert str(ns["F"]).endswith("/source-zero-full-v1/serving-account-control-v2")
    assert 'key = "licensed-parent-full-epoch-serving:source-zero-final:" + model_sha' in source
    assert 'target = S / "account-phase-v1"' in source
    # All actual account operations remain precisely the old AST after the stage branch.
    old_fn = next(
        n
        for n in ast.parse(original).body
        if isinstance(n, ast.FunctionDef) and n.name == "execute"
    )
    new_fn = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "execute"
    )
    start = next(
        i
        for i, node in enumerate(old_fn.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "stage"
    )
    new_start = next(
        i
        for i, node in enumerate(new_fn.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "stage"
    )
    assert [ast.dump(n) for n in old_fn.body[start:]] == [
        ast.dump(n) for n in new_fn.body[new_start:]
    ]
    with pytest.raises(ValueError):
        adapter.build(original + "# changed")


def setup_reconciliation(tmp_path, monkeypatch):
    ns = {"__name__": "synthetic_no_remote"}
    exec(compile(source, "successor", "exec"), ns)
    root = tmp_path / "source-zero-full-v1"
    ns["D"] = root
    original_stage = {"source_git": adapter.ORIGINAL_STAGE_GIT, "source_sha256": {}}
    old_root = root / "serving-account-control-v1"
    (old_root / "code").mkdir(parents=True)
    for name, text in {
        "program.py": "old program",
        "gpu_capacity.py": "old capacity",
        "policy.py": "same policy",
    }.items():
        (old_root / "code" / name).write_text(text)
        original_stage["source_sha256"][name] = hashlib.sha256(text.encode()).hexdigest()
    (old_root / "staged.json").write_text(json.dumps(original_stage))
    compiled = {
        "status": "actual_source_zero_fixed_final_serving_sources_compiled",
        "CPU_job": "978933",
        "epoch": 3,
        "model_sha256": adapter.MODEL_SHA,
        "outputs": {},
    }
    serving = root / "epoch-3/serving-v1"
    serving.mkdir(parents=True)
    for mode in ("qualification", "competition"):
        path = serving / (mode + ".private.py")
        path.write_text(mode)
        compiled["outputs"][mode] = {"bytes": path.stat().st_size, "sha256": ns["sha"](path)}
    (serving / "compiled.json").write_text(json.dumps(compiled))
    import sqlite3

    dbpath = tmp_path / "account-ledger.sqlite3"
    with sqlite3.connect(dbpath) as db:
        db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)")
    claim = tmp_path / "coordinator-claim"
    original_path = ns["Path"]

    def mapped(path):
        if str(path) == "/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3":
            return dbpath
        if str(path) == "/home/sa305415/rsna-knee/kaggle/coordinator-claim":
            return claim
        return original_path(path)

    ns["Path"] = mapped
    payload = {
        "original_stage": original_stage,
        "original_compiled": compiled,
        "source_sha256": {
            **original_stage["source_sha256"],
            "gpu_capacity.py": adapter.CAPACITY_SHA,
            "program.py": hashlib.sha256(source.encode()).hexdigest(),
        },
    }
    return ns, payload, root, dbpath, claim


@pytest.mark.parametrize(
    "problem", [None, "claim", "old_code", "compiled", "capacity", "policy", "phase", "ledger"]
)
def test_readonly_reconciliation_blocks_conflicts_without_mutating_originals(
    tmp_path, monkeypatch, problem
):
    ns, value, root, dbpath, claim = setup_reconciliation(tmp_path, monkeypatch)
    if problem == "claim":
        claim.mkdir()
    elif problem == "old_code":
        (root / "serving-account-control-v1/code/program.py").write_text("altered")
    elif problem == "compiled":
        (root / "epoch-3/serving-v1/qualification.private.py").write_text("altered")
    elif problem in ("capacity", "policy"):
        value["source_sha256"]["gpu_capacity.py" if problem == "capacity" else "policy.py"] = (
            "0" * 64
        )
    elif problem == "phase":
        (root / "epoch-3/serving-v1/account-phase-v1").mkdir()
    elif problem == "ledger":
        import sqlite3

        with sqlite3.connect(dbpath) as db:
            db.execute(
                "INSERT INTO meta VALUES (?,?)",
                ("licensed-parent-full-epoch-serving:source-zero-final:" + adapter.MODEL_SHA, "{}"),
            )
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if problem not in (None, "claim"):
        with pytest.raises(AssertionError):
            ns["reconcile_original"](value)
    else:
        out = ns["reconcile_original"](value)
        assert out["status"] == (
            "waiting_existing_canonical_claim_no_mutation"
            if problem == "claim"
            else "original_compiled_final_and_absent_serving_reconciled_read_only"
        )
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not (root / "serving-account-control-v2").exists()


def test_successor_owner_does_not_call_any_training_publication_or_compiler():
    code = (D / "follow_once.py.txt").read_text()
    tree = ast.parse(code)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert "prepare.run" not in calls and "account.run" in calls
    assert 'action("prepare")' in code and 'action("stage")' in code
    assert 'action("submit", "competition")' in code
    assert "prepare_once" not in code and "record_publication" not in code
    assert 'maximum_code_entries": 1' in code
    compile((D / "run_once.py.txt").read_text(), "run_once", "exec")


def test_frozen_source_assembly_changes_only_capacity_and_account_wrapper(monkeypatch):
    runner = load(D / "run_once.py.txt", "zero_successor_runner")
    original_git_bytes = runner.git_bytes

    def git_bytes(pin, path):
        if pin == "synthetic-uncommitted-local-controls":
            return path.read_bytes()
        return original_git_bytes(pin, path)

    monkeypatch.setattr(runner, "git_bytes", git_bytes)
    files, stage, compiled = runner.sources("synthetic-uncommitted-local-controls")
    for name, digest in stage["source_sha256"].items():
        actual = hashlib.sha256(files[name].encode()).hexdigest()
        if name == "gpu_capacity.py":
            assert actual == adapter.CAPACITY_SHA != digest
        elif name == "program.py":
            assert files[name] == source and actual != digest
        else:
            assert actual == digest
    assert compiled["CPU_job"] == "978933" and compiled["model_sha256"] == adapter.MODEL_SHA
