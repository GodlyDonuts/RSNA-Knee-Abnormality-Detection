"""Operational provenance/handoff tests; no remote calls or scientific training."""
import ast
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("successor_contract_tests", HERE / "contract.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def original(name):
    return subprocess.check_output(["git", "show", c.SCIENCE + ":" + c.FIT + "/" + name],
                                   cwd=ROOT).decode()


def stopped(role):
    return dict(status="actual_idle_v3_cache_dependent_owner_stopped_before_phase", role=role,
                pid={"fitting": 80995, "deployment": 80996}[role], source_git=c.PARENT,
                phase_intents=[], children=[], owner_process_absent_after_stop=True,
                remote_admission_or_dispatch_exists=False, remote_jobs_cancelled=[],
                remote_read_only_snapshot_sha256="a" * 64)


@pytest.mark.parametrize("role", ["fitting", "deployment"])
@pytest.mark.parametrize("key,bad", [("phase_intents", ["admission"]), ("children", [123]),
    ("owner_process_absent_after_stop", False), ("remote_admission_or_dispatch_exists", True),
    ("source_git", c.SCIENCE), ("remote_jobs_cancelled", ["979664"])])
def test_reject_unproved_or_advanced_handoff(role, key, bad):
    value = stopped(role)
    c.handoff(value, role)
    value[key] = bad
    with pytest.raises(ValueError):
        c.handoff(value, role)


def lineage():
    return {role: dict(job=job, root=c.OLD_ROOTS[role], intent_sha256=c.INTENTS[role],
        source_git=c.OLD_SOURCE, actual_terminal=dict(job_id=job, state="FAILED", exit_code="1:0",
        live_queue_empty=True, sha256="f" * 64)) for role, job in c.PREDECESSORS.items()}


@pytest.mark.parametrize("key,bad", [("state", "RUNNING"), ("state", "TIMEOUT"),
                                    ("exit_code", "0:0"), ("live_queue_empty", False)])
def test_exact_failure_chain_required(key, bad):
    value = lineage()
    assert len(c.chain(value)) == 64
    value["context"]["actual_terminal"][key] = bad
    with pytest.raises(ValueError):
        c.chain(value)


def test_original_preparation_initialization_and_all_other_functions_unchanged():
    source = original("admit_cpu.py.txt")
    changed = c.derived_admission(source)
    old = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
           if isinstance(n, ast.FunctionDef)}
    new = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(changed).body
           if isinstance(n, ast.FunctionDef)}
    assert old.keys() == new.keys()
    assert {name for name in old if old[name] != new[name]} == {"admit"}
    assert "cache_successor_control" in changed
    assert "validate_successor_chain(terminal, intent, raw, terminal_job)" in changed
    for role in c.ROOTS:
        assert c.ROOTS[role] in changed and c.OLD_ROOTS[role] not in changed
    with pytest.raises(ValueError):
        c.derived_admission(source + "\n")


def test_remote_changes_only_roots_and_extra_two_gib_scratch():
    source = original("owner_remote.py.txt")
    changed = c.derived_remote(source)
    expected = source
    for role in c.ROOTS:
        expected = expected.replace(c.OLD_ROOTS[role], c.ROOTS[role])
    expected = expected.replace("return 30 * GIB + 594 * MIB + remaining_cache(v)",
                                "return 32 * GIB + 594 * MIB + remaining_cache(v)")
    assert changed == expected


def test_local_handoff_rejects_phase_files_and_live_owner(tmp_path):
    journal = tmp_path / "artifacts" / Path(c.OLD).name / "finite-owner"
    journal.mkdir(parents=True)
    (journal / "owner-intent.json").write_text("{}")
    c.local_handoff(stopped("fitting"), "fitting", tmp_path, lambda _: "")
    with pytest.raises(ValueError):
        c.local_handoff(stopped("fitting"), "fitting", tmp_path, lambda _: "still alive")
    (journal / "admission-BC-intent.json").write_text("{}")
    with pytest.raises(ValueError):
        c.local_handoff(stopped("fitting"), "fitting", tmp_path, lambda _: "")


def load_launcher(monkeypatch):
    monkeypatch.setitem(sys.modules, "contract", c)
    spec = importlib.util.spec_from_file_location("successor_launch_tests", HERE / "launch.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    return launch


def test_frozen_originals_and_new_program_integrate_without_remote_or_reprepare(tmp_path, monkeypatch):
    launch = load_launcher(monkeypatch)
    real_git = launch.git_file
    pin = "f" * 40
    monkeypatch.setattr(launch, "git_file", lambda commit, path:
        (ROOT / path).read_bytes() if commit == pin else real_git(commit, path))
    # Simulate only the future local handoff check; original CPU979667 evidence is real and read-only.
    monkeypatch.setattr(c, "local_handoff", lambda value, role, *_: c.handoff(value, role))
    files = {}
    for role in ("fitting", "deployment"):
        files[role] = tmp_path / (role + ".json")
        files[role].write_text(json.dumps(stopped(role)))
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps(dict(status="accepted_v4_cache_owner_handoff",
        placement_source_git=pin, science_git=c.SCIENCE,
        shared_future_bytes=123 * 2**20, shared_future_files=376,
        retained_total_scratch_bytes=6 * 2**30, no_paid_credits=True)))
    args = SimpleNamespace(source_git=pin, decision=decision, handoff=files["fitting"],
                           deployment_handoff=files["deployment"])
    own, originals, fit, p, old, transport, program, payload = launch.freeze(args)
    assert old.COLLECT == ROOT / c.CACHE
    assert fit.HERE == HERE and p.SCIENCE == c.SCIENCE
    assert payload["prepared"]["CPU_job"] == "979667"
    assert c.sha(payload["original_remote"].encode()) == c.REMOTE_SHA
    assert payload["handoff"]["pid"] == 80995
    compile(program, "stdin-program", "exec")
    # Exercise remote namespace construction using a stub original namespace, no I/O or Slurm.
    ns = {}
    monkeypatch.setitem(sys.modules, "successor_contract", c)
    exec(compile(own["remote_adapter.py"], "adapter", "exec"), ns)
    captured = []
    remote = {"namespace": lambda value, node: captured.append((value, node)) or {}}
    ns["install"](remote, payload)
    adapted = remote["namespace"](payload, "evc33")
    assert "runpy" in adapted
    assert c.ROOTS["context"] in captured[0][0]["original_remote"]
    assert captured[0][0]["prepared"] == payload["prepared"]
    assert payload["prepared"]["code_sha256"]["admit_cpu.py"] == c.ADMIT_SHA
    # Byte-identical worker and publisher remain bound to actual preparation.
    for name in ("worker.py", "publish_epoch.py"):
        assert c.sha(original(name + ".txt").encode()) == payload["prepared"]["code_sha256"][name]


def test_remote_adapter_rejects_mismatched_chain_before_inputs_are_accepted(monkeypatch, tmp_path):
    # Extract the pure validator nested in the actual adapter and provide controlled I/O.
    tree = ast.parse((HERE / "remote_adapter.py").read_text())
    validator = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "validate")
    scope = {"c": c, "json": json, "Path": Path, "control": {}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[validator], type_ignores=[])), "validator", "exec"), scope)
    context = {"continuation_chain": lineage()}
    wrong = copy.deepcopy(context)
    wrong["continuation_chain"]["raw"]["job"] = "123"
    with pytest.raises(ValueError):
        scope["validate"](context, context, wrong, lambda *a: pytest.fail("No scheduler read yet"))


def test_actual_portable_namespace_uses_adapter_without_touching_staged_helpers(monkeypatch):
    launch = load_launcher(monkeypatch)
    p_source = subprocess.check_output(["git", "show", c.PARENT + ":" + c.OLD + "/placement.py"], cwd=ROOT).decode()
    remote_source = subprocess.check_output(["git", "show", c.PARENT + ":" + c.OLD + "/remote.py"], cwd=ROOT).decode()
    p = launch.module(c.derived_placement(p_source), "namespace_placement", HERE / "virtual.py",
                      {"successor_handoff": c.handoff})
    monkeypatch.setitem(sys.modules, "placement", p)
    monkeypatch.setitem(sys.modules, "successor_contract", c)
    remote, adapter = {}, {}
    exec(compile(remote_source, "original_remote", "exec"), remote)
    exec(compile((HERE / "remote_adapter.py").read_text(), "adapter", "exec"), adapter)
    payload = dict(original_remote=original("owner_remote.py.txt"), shared_bytes=123 * 2**20,
                   shared_files=376, placement_source_git="f" * 40,
                   successor_code_sha256={"contract.py": "a" * 64, "remote_adapter.py": "b" * 64})
    adapter["install"](remote, payload)
    ns = remote["namespace"](payload, "evc33")
    assert ns["ARMS"] == ("B", "C")
    assert ns["admission"].__globals__ is ns
    assert ns["cache_roots"]({"original_actual": {"status":
        "actual_missing_only3887_exact_FP32_context_alias_recovery_accepted"}}) == Path(c.ROOTS["context"])
    staged = Path("/tmp/rsna-predictive-full-fitting-20261020-v2/code/admit_cpu.py")
    old_read = Path.read_text
    admission_source = original("admit_cpu.py.txt")
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw:
                        admission_source if path == staged else old_read(path, *a, **kw))
    derived = ns["runpy"].run_path(str(staged), run_name="admission_library")
    assert "validate_successor_chain" in derived
    assert derived["initialize"].__globals__ is derived
    assert ns["ROOT"] == Path("/tmp/rsna-predictive-full-fitting-20261020-v2")
    ns27 = remote["namespace"](payload, "evc27")
    assert ns27["ARMS"] == ("C0",)
    assert ns27["ROOT"] == p.ROOT27


def test_deployment_reads_successor_paid_cache_receipts_and_retains_six_gib_scratch(
    tmp_path, monkeypatch
):
    launch = load_launcher(monkeypatch)
    pin = "8fd43b9818c85cbbff9e608ae5ba7d3ae2c6a26f"
    rel = "initiative/research/predictive-final-deployment-owner-20261020-v1"
    source = launch.git_file(pin, rel + "/owner.py").decode()
    changed = c.derived_deployment_owner(source)
    old = {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(source))
           if isinstance(n, ast.FunctionDef)}
    new = {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(changed))
           if isinstance(n, ast.FunctionDef)}
    assert old.keys() == new.keys()
    assert {name for name in old if old[name] != new[name]} == {"config"}
    canonical = launch.module(changed, "deployment_receipt_scope", ROOT / rel / "owner.py")
    canonical.ROOT = tmp_path
    monkeypatch.setattr(canonical, "runtime_plan", lambda *args: {})
    cache = tmp_path / c.CACHE
    cache.mkdir(parents=True)
    paid = {"raw": 60 * 2**30, "context": 20 * 2**30}
    for role, size in paid.items():
        (cache / f"actual-owner-{role}_collect.json").write_text(json.dumps({
            "status": "actual_original_full_cache_terminal_collected", "studies": 3887,
            "target_terminal": "COMPLETED", "exit_code": "0:0", "existing_cache_bytes": size,
        }))
    funding = {
        "other_node_base_bytes": 68 * 2**30 + 594 * 2**20,
        "other_shared_unpaid_bytes": 122 * 2**20,
        "other_shared_unpaid_files": 368,
    }
    c.deployment_funding({"funding": funding})
    owner = canonical.Owner.__new__(canonical.Owner)
    owner.decision = {"funding": funding, "root": "/tmp/test-private-deployment"}
    owner.qualification, owner.policy, owner.remaining_public = None, None, 3
    observed = {
        "observed_at": canonical.time.time(),
        "paid_full_run_bytes": dict(B=2**30, C=2**30, C0=2**29),
    }
    result = owner.config(observed, "qualification")
    assert result["funding"]["other_node_unpaid_bytes"] == (
        funding["other_node_base_bytes"] + 416 * 2**30
        - sum(paid.values()) - sum(observed["paid_full_run_bytes"].values())
    )
    with pytest.raises(ValueError):
        c.deployment_funding({"funding": {**funding,
            "other_node_base_bytes": 66 * 2**30 + 594 * 2**20}})
