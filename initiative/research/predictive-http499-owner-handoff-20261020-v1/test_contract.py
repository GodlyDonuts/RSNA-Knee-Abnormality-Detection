"""Run the actual frozen preparation/capsule and reject ancestry or ownership drift."""

import ast
import copy
import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def env(monkeypatch):
    c = load("http499_test_contract", HERE / "contract.py")
    monkeypatch.setitem(sys.modules, "contract", c)
    launch = load("http499_test_launch", HERE / "launch.py")
    launch.base()
    previous_tests = load("old_corrected_test_helpers", ROOT / c.LEGACY_DIR / "test_contract.py")
    return c, launch, previous_tests


def failure(c, old):
    previous = old.levels(c.legacy)
    return {role: dict(job=job, root=c.OLD_ROOTS[role], source_git=c.LEGACY,
                       intent_sha256=c.INTENTS[role], ancestor_chain=previous,
                       actual_terminal=dict(job_id=job, state="FAILED", exit_code="1:0",
                                            live_queue_empty=True, sha256="a" * 64))
            for role, job in c.PREDECESSORS.items()}


def stopped(c, role):
    return dict(status="actual_idle_http499_owner_stopped_before_phase", role=role,
                pid=11013 if role == "fitting" else 11014, source_git=c.LEGACY,
                phase_intents=[], children=[], owner_process_absent_after_stop=True,
                remote_admission_or_dispatch_exists=False, remote_jobs_cancelled=[],
                remote_read_only_snapshot_sha256="a" * 64, original_intent_sha256="b" * 64,
                observed_at=datetime.now(timezone.utc).isoformat())


def http(c):
    return dict(http499_recovery_decision_sha256="1" * 64,
                previous_source_http_sha256=c.PREVIOUS_HTTP, source_http_sha256="2" * 64)


@pytest.mark.parametrize("depth", range(4))
def test_exact_all_four_ancestry_levels(env, depth):
    c, _, old = env
    value = failure(c, old)
    assert len(c.chain(value)) == 64
    layer = value
    for _ in range(depth):
        layer = layer["raw"]["ancestor_chain"]
    layer["raw"]["intent_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        c.chain(value)


@pytest.mark.parametrize("role", ["fitting", "deployment"])
def test_actual_original_journals_and_started_phase_rejection(env, role, tmp_path):
    c, _, _ = env
    value = stopped(c, role)
    old = ROOT / "artifacts" / Path(c.LEGACY_DIR).name
    journal = old / "finite-owner" if role == "fitting" else old.with_name(old.name + "-deployment")
    value["original_intent_sha256"] = c.sha((journal / "owner-intent.json").read_bytes())
    c.local_handoff(value, role, ROOT, lambda _: "")
    with pytest.raises(ValueError, match="still exists"):
        c.local_handoff(value, role, ROOT, lambda _: "running")
    value["phase_intents"] = ["dispatch"]
    with pytest.raises(ValueError, match="handoff"):
        c.local_handoff(value, role, ROOT, lambda _: "")


def test_actual_observer_continuation_and_both_intent_hashes(env):
    c, _, _ = env
    journal = ROOT / "artifacts" / Path(c.OBSERVER_DIR).name
    value = dict(status="actual_idle_training_observer_stopped_before_dispatch", pid=11927,
                 source_git=c.OBSERVER_SOURCE, children=[], process_absent_after_stop=True,
                 remote_allocations=0, remote_mutations=0, actual_training_updates_observed=0,
                 observed_at=datetime.now(timezone.utc).isoformat(),
                 original_intent_sha256=c.sha((journal / "observer-intent.json").read_bytes()),
                 original_continuation_intent_sha256=c.sha((journal / "continuation-intent.json").read_bytes()))
    c.observer_handoff(value, ROOT, lambda _: "")
    value["original_continuation_intent_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="intents changed"):
        c.observer_handoff(value, ROOT, lambda _: "")


@pytest.mark.parametrize("key", ["source_http_sha256", "http499_recovery_decision_sha256",
                                 "source_correction_regression_sha256", "effective_source_index_sha256"])
def test_four_original_records_share_transport_and_corrected_sources(env, key):
    c, _, old = env
    value = {**old.correction(c), **{k: "f" * 64 for k in c.legacy.prior.PROOFS}, **http(c)}
    values = [copy.deepcopy(value) for _ in range(4)]
    expected = {**http(c), "source_correction_regression_sha256": "c" * 64}
    assert c.proofs(*values, expected=expected)["source_http_sha256"] == "2" * 64
    values[-1][key] = "0" * 64
    with pytest.raises(ValueError):
        c.proofs(*values, expected=expected)


def freeze(env, tmp_path, monkeypatch):
    c, launch, _ = env
    base, pin = launch.base(), "f" * 40
    original_git = base.git_file
    monkeypatch.setattr(base, "git_file", lambda commit, path:
                        (ROOT / path).read_bytes() if commit == pin else original_git(commit, path))
    monkeypatch.setattr(c, "local_handoff", lambda *args: None)
    monkeypatch.setattr(c, "observer_handoff", lambda *args: None)
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps(dict(
        status="accepted_http499_owner_handoff", placement_source_git=pin,
        science_git=c.SCIENCE, shared_future_bytes=123 * 2**20, shared_future_files=376,
        retained_total_scratch_bytes=6 * 2**30, no_paid_credits=True,
        source_correction_policy=c.correction_policy(), preserve_original_scientific_sources=True,
        source_correction_regression_sha256="e" * 64, **http(c))))
    paths = {}
    for role in ("fitting", "deployment", "observer"):
        paths[role] = tmp_path / (role + ".json")
        paths[role].write_text(json.dumps(stopped(c, role) if role != "observer" else {}))
    args = SimpleNamespace(source_git=pin, decision=decision, handoff=paths["fitting"],
                           deployment_handoff=paths["deployment"], observer_handoff=paths["observer"])
    return args, launch.freeze(args)


def test_actual979667_freeze_and_readonly_capsule(env, tmp_path, monkeypatch):
    c, launch, _ = env
    _, frozen = freeze(env, tmp_path, monkeypatch)
    own, originals, fit, _, original, _, program, payload = frozen
    assert original.COLLECT == ROOT / c.CACHE and fit.HERE == HERE
    assert payload["prepared"]["CPU_job"] == "979667"
    assert payload["accepted_http499_proofs"] == http(c)
    for name in ("worker.py", "publish_epoch.py"):
        raw = launch.git_file(c.SCIENCE, c.FIT + "/" + name + ".txt")
        assert c.sha(raw) == payload["prepared"]["code_sha256"][name]
    compile(program, "actual_full_capsule", "exec")
    guard = launch.build_program(own, originals, readonly=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"mode":"dispatch"}'))
    with pytest.raises(AssertionError):
        exec(compile(guard, "readonly_capsule", "exec"), {})
    original_source = launch.git_file(c.SCIENCE, c.FIT + "/admit_cpu.py.txt").decode()
    derived = c.derived_admission(original_source)
    before = {n.name: ast.dump(n) for n in ast.parse(original_source).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n) for n in ast.parse(derived).body if isinstance(n, ast.FunctionDef)}
    assert {name for name in before if before[name] != after[name]} == {"admit"}
    assert all(root in derived for root in c.ROOTS.values())
    assert all(root not in derived for root in c.OLD_ROOTS.values())


def test_observer_frozen_capsule_matches_new_owner_and_actual_preparation(env, monkeypatch):
    c, launch, _ = env
    monkeypatch.setitem(sys.modules, "launch", launch)
    observer = load("http499_observer", HERE / "observe.py").observer()
    pin = "f" * 40
    owner = dict(source_git=pin, science_git=c.SCIENCE, original_owner_source_git=c.LEGACY,
                 original_owner_pid=11013, decision_sha256="a" * 64, handoff_sha256="b" * 64,
                 source_sha256={name: c.sha((HERE / name).read_bytes())
                                for name in ("launch.py", "contract.py", "remote_adapter.py")},
                 decision_path="test-decision.json", handoff_path="test-handoff.json")
    decision = dict(status="accepted_http499_owner_handoff", placement_source_git=pin,
                    shared_future_bytes=123 * 2**20, shared_future_files=376)
    handoff = stopped(c, "fitting")
    original_read, original_git = observer.read, observer.git_file
    def read(path):
        if path == observer.OWNER_JOURNAL / "owner-intent.json":
            return owner, "d" * 64
        if path == ROOT / "test-decision.json":
            return decision, "a" * 64
        if path == ROOT / "test-handoff.json":
            return handoff, "b" * 64
        return original_read(path)
    monkeypatch.setattr(observer, "read", read)
    monkeypatch.setattr(observer, "git_file", lambda commit, path:
                        (ROOT / path).read_bytes() if commit == pin else original_git(commit, path))
    original_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw:
                        json.dumps(handoff) if path == ROOT / "test-handoff.json" else original_text(path, *a, **kw))
    _, program, payload, provenance = observer.frozen()
    assert observer.OWNER_JOURNAL == ROOT / "artifacts" / HERE.name / "finite-owner"
    assert payload["mode"] == "progress" and payload["prepared"]["CPU_job"] == "979667"
    assert provenance["placement_source_git"] == pin
    assert provenance["progress_program_sha256"] == c.sha(program.encode())


def test_final_source_gate_keeps_correction_and_new_http_lineage(env, tmp_path):
    c, _, old = env
    control = dict(old.correction(c), **http(c), source_correction_policy=c.correction_policy(),
                   accepted_source_correction_proofs={"source_correction_regression_sha256": "c" * 64},
                   accepted_http499_proofs=http(c))
    result = {"finals": {}, "publications": {}}
    for name, arms, status in (
        ("admission-BC", ("B", "C"), "actual_full3887_predictive_CPU_admission_complete_no_GPU_dispatched"),
        ("contexts-C0", ("C0",), "actual_C0_context_transfer_and_CPU_admission_accepted")):
        inputs = {arm: str(i + 1) * 64 for i, arm in enumerate(arms)}
        (tmp_path / ("actual-owner-" + name + ".json")).write_bytes(c.canonical(dict(
            status=status, input_admission_sha256=inputs, cache_successor_control=control)))
        for arm in arms:
            result["finals"][arm] = dict(GPU_job=arm, source_revision=c.BASELINE, input_admission_sha256=inputs[arm])
            result["publications"][arm] = dict(GPU_job=arm)
    assert set(c.final_source_lineage(result, tmp_path)["effective_source_lineage"]["per_arm"]) == {"B", "C", "C0"}
    path = tmp_path / "actual-owner-contexts-C0.json"
    changed = json.loads(path.read_bytes())
    changed["cache_successor_control"]["source_http_sha256"] = "0" * 64
    path.write_bytes(c.canonical(changed))
    with pytest.raises(ValueError, match="HTTP499"):
        c.final_source_lineage(result, tmp_path)


def test_canonical_deployment_reads_only_new_cache_collection(env):
    c, launch, _ = env
    old = "initiative/research/predictive-final-deployment-owner-20261020-v1/owner.py"
    source = launch.git_file("8fd43b9818c85cbbff9e608ae5ba7d3ae2c6a26f", old).decode()
    derived = c.derived_deployment_owner(source)
    compile(derived, "unchanged_canonical_owner_with_successor_cache", "exec")
    assert c.CACHE in derived
    assert c.legacy.CACHE not in derived
    c.deployment_funding({"funding": {"other_node_base_bytes": 73637298176}})
    with pytest.raises(ValueError, match="scratch"):
        c.deployment_funding({"funding": {"other_node_base_bytes": 73637298175}})


def test_actual_stop_receipts_bind_all_original_journals(env, monkeypatch):
    c, launch, _ = env
    # Historical immutable proofs remain testable after the launch freshness window.
    monkeypatch.setattr(c, "fresh", lambda value: None)
    for role in ("fitting", "deployment"):
        value = json.loads((HERE / f"actual-{role}-owner-stopped.json").read_bytes())
        c.local_handoff(value, role, ROOT, lambda _: "")
    c.observer_handoff(json.loads((HERE / "actual-observer-owner-stopped.json").read_bytes()),
                        ROOT, lambda _: "")
    actual = json.loads((HERE / "actual-exclusive-prephase-readonly.json").read_bytes())
    expected = c.sha((HERE / "actual-exclusive-prephase-readonly.json").read_bytes())
    assert actual and all(json.loads((HERE / f"actual-{role}-owner-stopped.json").read_bytes())[
        "remote_read_only_snapshot_sha256"] == expected for role in ("fitting", "deployment", "observer"))
