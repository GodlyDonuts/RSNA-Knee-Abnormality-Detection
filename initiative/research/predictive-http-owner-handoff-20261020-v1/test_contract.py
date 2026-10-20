"""Exercise actual frozen operational sources without remote work or model loading."""
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


def load(name, file, monkeypatch=None):
    spec = importlib.util.spec_from_file_location(name, file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def env(monkeypatch):
    c = load("http_contract_tests", HERE / "contract.py")
    monkeypatch.setitem(sys.modules, "contract", c)
    launch = load("http_launch_tests", HERE / "launch.py")
    launch.base()
    return c, launch


def source(c, name):
    return subprocess.check_output(["git", "show", c.SCIENCE + ":" + c.FIT + "/" + name], cwd=ROOT).decode()


def ancestor(c):
    p = c.prior
    return {role: dict(job=job, root=p.OLD_ROOTS[role], source_git=p.OLD_SOURCE,
        intent_sha256=p.INTENTS[role], actual_terminal=dict(job_id=job, state="FAILED",
        exit_code="1:0", live_queue_empty=True, sha256="e" * 64)) for role, job in p.PREDECESSORS.items()}


def lineage(c):
    return {role: dict(job=job, root=c.OLD_ROOTS[role], source_git=c.OLD_SOURCE,
        intent_sha256=c.INTENTS[role], actual_terminal=dict(job_id=job, state="FAILED",
        exit_code="1:0", live_queue_empty=True, sha256="f" * 64), ancestor_chain=ancestor(c))
        for role, job in c.PREDECESSORS.items()}


@pytest.mark.parametrize("role", ("raw", "context"))
@pytest.mark.parametrize("level", ("immediate", "ancestor"))
def test_nested_lineage_rejects_wrong_job_state_or_intent(env, role, level):
    c, _ = env
    value = lineage(c)
    assert len(c.chain(value)) == 64
    target = value[role] if level == "immediate" else value[role]["ancestor_chain"][role]
    target["intent_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        c.chain(value)
    value = lineage(c)
    target = value[role] if level == "immediate" else value[role]["ancestor_chain"][role]
    target["actual_terminal"]["state"] = "RUNNING"
    with pytest.raises(ValueError):
        c.chain(value)


def test_real_immediate_hashes_match_actual_retained_buffer_receipt(env):
    c, _ = env
    actual = json.loads((ROOT / "initiative/research/predictive-source-http-failure-20261020-v1/actual-retained-buffers.json").read_text())
    for role in c.PREDECESSORS:
        assert c.INTENTS[role] == actual["roles"][role]["intent_sha256"]
        assert c.PREDECESSORS[role] == actual["roles"][role]["job"]
        assert c.OLD_SOURCE == actual["roles"][role]["source_git"]


def test_only_operational_admission_changes_all_scientific_helpers_unchanged(env):
    c, _ = env
    raw = source(c, "admit_cpu.py.txt")
    changed = c.derived_admission(raw)
    old = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    new = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(changed).body if isinstance(n, ast.FunctionDef)}
    assert {n for n in old if old[n] != new[n]} == {"admit"}
    for role in c.ROOTS:
        assert c.ROOTS[role] in changed and c.OLD_ROOTS[role] not in changed
    prior = c.prior.derived_remote(source(c, "owner_remote.py.txt"))
    expected = prior
    for role in c.ROOTS:
        expected = expected.replace(c.OLD_ROOTS[role], c.ROOTS[role])
    assert c.derived_remote(source(c, "owner_remote.py.txt")) == expected
    assert 'return 32 * GIB + 594 * MIB + remaining_cache(v)' in expected


@pytest.mark.parametrize("role", ("fitting", "deployment"))
def test_actual_handoffs_accept_and_advanced_owner_rejected(env, role, monkeypatch):
    c, _ = env
    pid = {"fitting": 87673, "deployment": 87674}[role]
    value = json.loads((HERE / f"actual-{role}-owner{pid}-stopped.json").read_text())
    c.local_handoff(value, role, ROOT, lambda _: "")
    with pytest.raises(ValueError):
        c.local_handoff(value, role, ROOT, lambda _: "still running")
    value["phase_intents"] = ["admit"]
    with pytest.raises(ValueError):
        c.handoff(value, role)


def test_actual_observer_handoff_guard(env):
    c, _ = env
    value = json.loads((HERE / "actual-observer87804-stopped.json").read_text())
    c.observer_handoff(value, ROOT, lambda _: "")
    value["actual_training_updates_observed"] = 1
    with pytest.raises(ValueError):
        c.observer_handoff(value, ROOT, lambda _: "")


def test_full_local_freeze_reuses_actual979667_and_builds_readonly_capsule(env, tmp_path, monkeypatch):
    c, launch = env
    base = launch.base()
    pin = "f" * 40
    real_git = base.git_file
    monkeypatch.setattr(base, "git_file", lambda commit, path:
                        (ROOT / path).read_bytes() if commit == pin else real_git(commit, path))
    monkeypatch.setattr(base, "process_command", lambda _: "")
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps(dict(status="accepted_http_cache_owner_handoff",
        placement_source_git=pin, science_git=c.SCIENCE, shared_future_bytes=123 * 2**20,
        shared_future_files=376, retained_total_scratch_bytes=6 * 2**30, no_paid_credits=True)))
    args = SimpleNamespace(source_git=pin, decision=decision,
        handoff=HERE / "actual-fitting-owner87673-stopped.json",
        deployment_handoff=HERE / "actual-deployment-owner87674-stopped.json",
        observer_handoff=HERE / "actual-observer87804-stopped.json")
    own, originals, fit, p, old, transport, program, payload = launch.freeze(args)
    assert old.COLLECT == ROOT / c.CACHE and fit.HERE == HERE
    assert payload["prepared"]["CPU_job"] == "979667"
    assert payload["prior_operational_sources_sha256"] == c.PRIOR_SHA
    assert payload["handoff"]["pid"] == 87673
    for name in ("worker.py", "publish_epoch.py"):
        assert c.sha(source(c, name + ".txt").encode()) == payload["prepared"]["code_sha256"][name]
    compile(program, "HTTP_capsule", "exec")
    guarded = launch.build_program(own, originals, readonly=True)
    assert guarded.index("assert v['mode']=='progress'") < guarded.index("exec(compile")
    import io
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"mode":"dispatch"}'))
    with pytest.raises(AssertionError):
        exec(compile(guarded, "readonly", "exec"), {})


def test_proofs_require_all_four_objects_and_each_new_hash(env):
    c, _ = env
    values = [{key: str(i + 1) * 64 for i, key in enumerate(c.PROOFS)} for _ in range(4)]
    assert c.proofs(*values) == values[0]
    values[-1]["source_http_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        c.proofs(*values)


def test_deployment_config_reads_new_cache_and_retains_prior_budget(env, monkeypatch, tmp_path):
    c, launch = env
    rel = "initiative/research/predictive-final-deployment-owner-20261020-v1"
    text = launch.git_file("8fd43b9818c85cbbff9e608ae5ba7d3ae2c6a26f", rel + "/owner.py").decode()
    changed = c.derived_deployment_owner(text)
    assert c.CACHE in changed and c.prior.CACHE not in changed
    module = launch.base().module(changed, "http_canonical", ROOT / rel / "owner.py")
    module.ROOT = tmp_path
    monkeypatch.setattr(module, "runtime_plan", lambda *a: {})
    cache = tmp_path / c.CACHE
    cache.mkdir(parents=True)
    for role in ("raw", "context"):
        (cache / f"actual-owner-{role}_collect.json").write_text(json.dumps(dict(
            status="actual_original_full_cache_terminal_collected", studies=3887,
            target_terminal="COMPLETED", exit_code="0:0", existing_cache_bytes=2**30)))
    funding = dict(other_node_base_bytes=68 * 2**30 + 594 * 2**20,
                   other_shared_unpaid_bytes=122 * 2**20, other_shared_unpaid_files=368)
    c.deployment_funding(dict(funding=funding))
    instance = module.Owner.__new__(module.Owner)
    instance.decision = dict(funding=funding, root="/tmp/private-test")
    instance.qualification, instance.policy, instance.remaining_public = None, None, 3
    result = instance.config(dict(observed_at=module.time.time(), paid_full_run_bytes=dict(B=0, C=0, C0=0)), "qualification")
    assert result["funding"]["other_node_unpaid_bytes"] == funding["other_node_base_bytes"] + 414 * 2**30
    with pytest.raises(ValueError):
        c.deployment_funding(dict(funding=dict(funding, other_node_base_bytes=66 * 2**30)))


def test_observer_source_selects_new_owner_journal_without_mutation(env, monkeypatch):
    c, launch = env
    monkeypatch.setitem(sys.modules, "launch", launch)
    obs = load("http_observer_tests", HERE / "observe.py").observer()
    assert obs.OWNER_JOURNAL == ROOT / "artifacts" / HERE.name / "finite-owner"
    assert obs.SUCCESSOR == HERE.relative_to(ROOT).as_posix()
    assert obs.HERE == HERE
    assert obs.DEADLINE == "2026-10-22T09:00:00+00:00"


def test_validator_rejects_changed_raw_ancestry_before_scheduler_reads(env, monkeypatch):
    c, _ = env
    monkeypatch.setitem(sys.modules, "successor_contract", c)
    adapter = load("http_remote_tests", HERE / "remote_adapter.py")
    good = {"continuation_chain": lineage(c)}
    bad = copy.deepcopy(good)
    bad["continuation_chain"]["raw"]["ancestor_chain"]["raw"]["job"] = "1"
    with pytest.raises(ValueError):
        adapter.validate(good, good, bad, lambda *a: pytest.fail("No read before common chain"), {})


def test_complete_remote_chain_rechecks_retained_intents_and_all_final_proofs(env, monkeypatch, tmp_path):
    c, _ = env
    monkeypatch.setitem(sys.modules, "successor_contract", c)
    adapter = load("http_remote_positive", HERE / "remote_adapter.py")
    p = c.prior
    old_roots = {role: str(tmp_path / ("v3-" + role)) for role in p.PREDECESSORS}
    current_roots = {role: str(tmp_path / ("v4-" + role)) for role in c.PREDECESSORS}
    new_roots = {role: str(tmp_path / ("http-" + role)) for role in c.PREDECESSORS}
    monkeypatch.setattr(p, "OLD_ROOTS", old_roots)
    monkeypatch.setattr(c, "OLD_ROOTS", current_roots)
    monkeypatch.setattr(c, "ROOTS", new_roots)
    old_hashes = {}
    for role, root in old_roots.items():
        path = Path(root)
        path.mkdir()
        body = c.canonical(dict(original_role=role))
        (path / "intent.json").write_bytes(body)
        old_hashes[role] = c.sha(body)
    monkeypatch.setattr(p, "INTENTS", old_hashes)
    old_chain = ancestor(c)
    hashes = {}
    for role, root in current_roots.items():
        path = Path(root)
        path.mkdir()
        body = c.canonical(dict(continuation_chain=old_chain, original_role=role))
        (path / "intent.json").write_bytes(body)
        hashes[role] = c.sha(body)
    monkeypatch.setattr(c, "INTENTS", hashes)
    common = dict(continuation_chain=lineage(c), scratch_cap_bytes=6 * 2**30,
                  **{key: str(i + 1) * 64 for i, key in enumerate(c.PROOFS)})
    terminal = dict(common, successor_raw_job="1234")
    final = dict(common, CPU_job="1234")
    raw_root = Path(new_roots["raw"])
    raw_root.mkdir()
    raw_final = raw_root / "actual.json"
    raw_final.write_text(json.dumps(final))
    queried = []
    def scheduler(job, accepted):
        queried.append((job, accepted))
        return dict(exit_code="1:0")
    result = adapter.validate(terminal, common, common, scheduler, {"science": c.SCIENCE})
    assert {job for job, _ in queried} == {"979664", "979665", "980743", "980744"}
    assert result["source_http_sha256"] == common["source_http_sha256"]
    assert result["science"] == c.SCIENCE
    final["source_http_sha256"] = "0" * 64
    raw_final.write_text(json.dumps(final))
    with pytest.raises(ValueError, match="HTTP admission proof"):
        adapter.validate(terminal, common, common, scheduler, {})
    (Path(current_roots["raw"]) / "intent.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Retained ancestor intent"):
        adapter.validate(terminal, common, common, scheduler, {})


def test_actual_observer_frozen_path_builds_new_readonly_capsule(env, monkeypatch):
    c, launch = env
    monkeypatch.setitem(sys.modules, "launch", launch)
    obs = load("http_observer_frozen", HERE / "observe.py").observer()
    pin = "f" * 40
    own = {name: (HERE / name).read_text() for name in ("launch.py", "contract.py", "remote_adapter.py")}
    real_git, real_read = obs.git_file, obs.read
    monkeypatch.setattr(obs, "git_file", lambda commit, path:
                        (ROOT / path).read_bytes() if commit == pin else real_git(commit, path))
    handoff_path = HERE / "actual-fitting-owner87673-stopped.json"
    handoff, handoff_sha = real_read(handoff_path)
    decision = dict(status="accepted_http_cache_owner_handoff", placement_source_git=pin,
                    shared_future_bytes=123 * 2**20, shared_future_files=376)
    decision_sha = c.sha(c.canonical(decision))
    decision_path = HERE / "test-not-created-decision.json"
    owner = dict(source_git=pin, source_sha256={k: c.sha(v.encode()) for k, v in own.items()},
        science_git=c.SCIENCE, original_owner_source_git=c.PRIOR, original_owner_pid=87673,
        decision_path=str(decision_path.relative_to(ROOT)), decision_sha256=decision_sha,
        handoff_path=str(handoff_path.relative_to(ROOT)), handoff_sha256=handoff_sha)
    def read(path):
        if path == obs.OWNER_JOURNAL / "owner-intent.json":
            return owner, c.sha(c.canonical(owner))
        if path == decision_path:
            return decision, decision_sha
        return real_read(path)
    monkeypatch.setattr(obs, "read", read)
    transport, program, payload, proof = obs.frozen()
    assert proof["placement_source_git"] == pin
    assert payload["handoff"]["pid"] == 87673
    assert payload["prepared"]["CPU_job"] == "979667"
    assert "token" not in payload and payload["mode"] == "progress"
    assert "http-recovery-20261020-v1" in program
    compile(program, "read_only_HTTP_owner_capsule", "exec")
