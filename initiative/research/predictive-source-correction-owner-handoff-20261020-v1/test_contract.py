"""Exercise exact prior sources and future correction/handoff evidence locally."""

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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(monkeypatch):
    c = load("source_corrected_test_contract", HERE / "contract.py")
    monkeypatch.setitem(sys.modules, "contract", c)
    launch = load("source_corrected_test_launch", HERE / "launch.py")
    launch.base()
    return c, launch


def levels(c):
    older = {r: dict(job=j, root=c.prior.prior.OLD_ROOTS[r],
                    source_git=c.prior.prior.OLD_SOURCE, intent_sha256=c.prior.prior.INTENTS[r],
                    actual_terminal=dict(job_id=j, state="FAILED", exit_code="1:0",
                                         live_queue_empty=True, sha256="a" * 64))
             for r, j in c.prior.prior.PREDECESSORS.items()}
    def next_level(contract, ancestor):
        return {r: dict(job=j, root=contract.OLD_ROOTS[r], source_git=contract.OLD_SOURCE,
                        intent_sha256=contract.INTENTS[r], ancestor_chain=ancestor,
                        actual_terminal=dict(job_id=j, state="FAILED", exit_code="1:0",
                                             live_queue_empty=True, sha256="b" * 64))
                for r, j in contract.PREDECESSORS.items()}
    return next_level(c, next_level(c.prior, older))


def handoff(c, role):
    return dict(
        status="actual_idle_source_correction_owner_stopped_before_phase", role=role,
        pid=96074 if role == "fitting" else 91866,
        source_git=c.RESILIENT if role == "fitting" else c.PRIOR,
        phase_intents=[], children=[], owner_process_absent_after_stop=True,
        remote_admission_or_dispatch_exists=False, remote_jobs_cancelled=[],
        remote_read_only_snapshot_sha256="a" * 64, original_intent_sha256="b" * 64,
        observed_at=datetime.now(timezone.utc).isoformat(),
    )


def test_exact_prior_failure_and_no_scientific_helper_changes(env):
    c, launch = env
    failure = json.loads((ROOT / "initiative/research/predictive-source-decode-failure-20261020-v1/actual-readonly-failure.json").read_text())
    for role in c.PREDECESSORS:
        assert c.INTENTS[role] == failure["roles"][role]["intent_sha256"]
        assert c.PREDECESSORS[role] == failure["roles"][role]["job"]
    source = launch.git_file(c.SCIENCE, c.FIT + "/admit_cpu.py.txt").decode()
    changed = c.derived_admission(source)
    before = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n) for n in ast.parse(changed).body if isinstance(n, ast.FunctionDef)}
    assert {name for name in before if before[name] != after[name]} == {"admit"}
    for root in c.ROOTS.values():
        assert root in changed
    assert 'result["input_admission_sha256"]' in changed
    remote = launch.git_file(c.PARENT, c.OLD + "/remote.py").decode()
    portable = c.derived_portable_remote(remote)
    before = {n.name: ast.dump(n) for n in ast.parse(remote).body if isinstance(n, ast.FunctionDef)}
    after = {n.name: ast.dump(n) for n in ast.parse(portable).body if isinstance(n, ast.FunctionDef)}
    assert {name for name in before if before[name] != after[name]} == {"portable_admit"}


@pytest.mark.parametrize("level", range(3))
@pytest.mark.parametrize("key,value", [("job", "1"), ("intent_sha256", "0" * 64)])
def test_all_three_levels_require_exact_originals(env, level, key, value):
    c, _ = env
    lineage = levels(c)
    assert len(c.chain(lineage)) == 64
    item = lineage
    for _ in range(level):
        item = item["raw"]["ancestor_chain"]
    item["raw"][key] = value
    with pytest.raises(ValueError):
        c.chain(lineage)


@pytest.mark.parametrize("role", ["fitting", "deployment"])
def test_idle_handoff_rejects_started_work(env, role):
    c, _ = env
    value = handoff(c, role)
    assert c.handoff(value, role) == value
    value["phase_intents"] = ["dispatch"]
    with pytest.raises(ValueError):
        c.handoff(value, role)
    with pytest.raises(ValueError):
        c.fresh(dict(observed_at="2026-01-01T00:00:00+00:00"))
    with pytest.raises(ValueError):
        c.fresh(dict(observed_at="2026-10-20T00:00:00"))


def test_real_local_journal_hashes_and_live_process_refusal(env, monkeypatch):
    c, _ = env
    value = handoff(c, "fitting")
    raw = (ROOT / "artifacts" / Path(c.RESILIENT_DIR).name / "controller-intent.json").read_bytes()
    value["original_intent_sha256"] = c.sha(raw)
    value["original_phase_owner_intent_sha256"] = json.loads(raw)["original_owner_intent_sha256"]
    c.local_handoff(value, "fitting", ROOT, lambda _: "")
    with pytest.raises(ValueError, match="still exists"):
        c.local_handoff(value, "fitting", ROOT, lambda _: "original controller active")
    value["original_intent_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="controller changed"):
        c.local_handoff(value, "fitting", ROOT, lambda _: "")


def test_complete_freeze_builds_original_preparation_without_launch(env, tmp_path, monkeypatch):
    c, launch = env
    base = launch.base()
    pin = "f" * 40
    original_git = base.git_file
    monkeypatch.setattr(base, "git_file", lambda commit, path:
                        (ROOT / path).read_bytes() if commit == pin else original_git(commit, path))
    monkeypatch.setattr(c, "local_handoff", lambda *args: None)
    monkeypatch.setattr(c, "observer_handoff", lambda *args: None)
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps(dict(
        status="accepted_source_correction_owner_handoff", placement_source_git=pin,
        science_git=c.SCIENCE, shared_future_bytes=123 * 2**20, shared_future_files=376,
        retained_total_scratch_bytes=6 * 2**30, no_paid_credits=True,
        source_correction_policy=c.correction_policy(), preserve_original_scientific_sources=True,
        source_correction_regression_sha256="e" * 64)))
    paths = {}
    for role in ("fitting", "deployment", "observer"):
        paths[role] = tmp_path / (role + ".json")
        paths[role].write_text(json.dumps(handoff(c, role) if role != "observer" else {}))
    args = SimpleNamespace(
        source_git=pin, decision=decision, handoff=paths["fitting"],
        deployment_handoff=paths["deployment"], observer_handoff=paths["observer"])
    own, originals, fit, p, original, transport, program, payload = launch.freeze(args)
    assert original.COLLECT == ROOT / c.CACHE and fit.HERE == HERE
    assert payload["prepared"]["CPU_job"] == "979667"
    assert payload["source_correction_policy"] == c.correction_policy()
    for name in ("worker.py", "publish_epoch.py"):
        raw = launch.git_file(c.SCIENCE, c.FIT + "/" + name + ".txt")
        assert c.sha(raw) == payload["prepared"]["code_sha256"][name]
    compile(program, "corrected_source_capsule", "exec")
    guarded = launch.build_program(own, originals, readonly=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"mode":"dispatch"}'))
    with pytest.raises(AssertionError):
        exec(compile(guarded, "allocation_free", "exec"), {})
    monkeypatch.setitem(sys.modules, "launch", launch)
    observer = load("source_corrected_observer_test", HERE / "observe.py").observer()
    assert observer.OWNER_JOURNAL == ROOT / "artifacts" / HERE.name / "finite-owner"
    assert observer.DEADLINE == "2026-10-22T12:00:00+00:00"


def test_policy_names_baseline_plus_two_objects_not_whole_revision(env):
    c, _ = env
    value = c.correction_policy()
    assert value["baseline_source_revision"] == c.BASELINE
    assert value["corrected_object_revision"] == c.CORRECTED_OBJECT_REVISION
    assert len(value["changed_source_files"]) == 2
    assert sum(r["after_bytes"] - r["before_bytes"] for r in value["changed_source_files"]) == 582656
    assert value["unchanged_source_members_use_baseline"] is True


def correction(c):
    return dict(
        effective_source_policy="baseline_plus_exact_authoritative_two_file_patch",
        source_correction_plan_sha256=c.PLAN, source_correction_receipt_sha256=c.RECEIPT,
        source_correction_helper_sha256=c.HELPER, source_correction_regression_sha256="c" * 64,
        effective_source_index_sha256="d" * 64, original_source_index_sha256="e" * 64,
        source_correction_revision=c.CORRECTED_OBJECT_REVISION,
    )


@pytest.mark.parametrize("field", [
    "source_correction_plan_sha256", "source_correction_helper_sha256",
    "source_correction_regression_sha256", "effective_source_index_sha256",
])
def test_all_four_records_and_accepted_regression_agree(env, field):
    c, _ = env
    values = [dict(correction(c), **{key: "f" * 64 for key in c.prior.PROOFS})
              for _ in range(4)]
    expected = {"source_correction_regression_sha256": "c" * 64}
    assert c.proofs(*values, expected=expected)["effective_source_index_sha256"] == "d" * 64
    values[-1][field] = "0" * 64
    with pytest.raises(ValueError):
        c.proofs(*values, expected=expected)


def test_final_exports_require_actual_BC_and_portable_admission_hashes(env, tmp_path):
    c, _ = env
    control = dict(correction(c), source_correction_policy=c.correction_policy(),
                   accepted_source_correction_proofs={"source_correction_regression_sha256": "c" * 64})
    value = {"finals": {}, "publications": {}}
    for name, arms, status in (
        ("admission-BC", ("B", "C"), "actual_full3887_predictive_CPU_admission_complete_no_GPU_dispatched"),
        ("contexts-C0", ("C0",), "actual_C0_context_transfer_and_CPU_admission_accepted"),
    ):
        inputs = {arm: str(i + 1) * 64 for i, arm in enumerate(arms)}
        (tmp_path / ("actual-owner-" + name + ".json")).write_bytes(c.canonical(dict(
            status=status, input_admission_sha256=inputs, cache_successor_control=control)))
        for arm in arms:
            value["finals"][arm] = dict(GPU_job=arm + "-test", source_revision=c.BASELINE,
                                        input_admission_sha256=inputs[arm])
            value["publications"][arm] = dict(GPU_job=arm + "-test")
    admitted = c.final_source_lineage(copy.deepcopy(value), tmp_path)
    assert set(admitted["effective_source_lineage"]["per_arm"]) == {"B", "C", "C0"}
    changed = copy.deepcopy(value)
    changed["finals"]["C0"]["input_admission_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="actual corrected-input"):
        c.final_source_lineage(changed, tmp_path)
    changed = copy.deepcopy(value)
    changed["finals"]["C"]["source_revision"] = c.CORRECTED_OBJECT_REVISION
    with pytest.raises(ValueError):
        c.final_source_lineage(changed, tmp_path)


def test_actual_stop_receipts_match_original_immutable_journals(env, monkeypatch):
    c, _ = env
    # Time freshness itself is covered separately; historical receipts remain
    # useful for deterministic validation after the actual launch window.
    monkeypatch.setattr(c, "fresh", lambda *args: None)
    for role in ("fitting", "deployment"):
        receipt = json.loads((HERE / f"actual-{role}-owner-stopped.json").read_text())
        c.local_handoff(receipt, role, ROOT, lambda _: "")
    receipt = json.loads((HERE / "actual-observer-owner-stopped.json").read_text())
    c.observer_handoff(receipt, ROOT, lambda _: "")


def test_context_roster_requires_both_exact_corrected_members_and_retains_baseline(env):
    c, _ = env
    rows = [{"provenance": {"original_source_revision": c.BASELINE}} for _ in range(3887)]
    for index, (before, (old_bytes, after, new_bytes)) in zip((1740, 1850), c.CORRECTIONS.items()):
        rows[index]["provenance"]["source_correction"] = dict(
            effective_source_policy=c.correction_policy()["policy"],
            original_revision=c.BASELINE, corrected_revision=c.CORRECTED_OBJECT_REVISION,
            plan_sha256=c.PLAN, receipt_sha256=c.RECEIPT, before_sha256=before,
            before_bytes=old_bytes, after_sha256=after, after_bytes=new_bytes)
    control = dict(correction(c), accepted_source_correction_proofs={
        "source_correction_regression_sha256": "c" * 64})
    assert c.validate_corrected_context_rows(rows, control)
    changed = copy.deepcopy(rows)
    changed[1850]["provenance"].pop("source_correction")
    with pytest.raises(ValueError, match="Both corrected"):
        c.validate_corrected_context_rows(changed, control)
    changed = copy.deepcopy(rows)
    changed[1740]["provenance"]["source_correction"]["after_bytes"] -= 1
    with pytest.raises(ValueError, match="authoritative bytes"):
        c.validate_corrected_context_rows(changed, control)


def test_observer_builds_pure_progress_capsule_from_actual_preparation(env, monkeypatch):
    c, launch = env
    monkeypatch.setitem(sys.modules, "launch", launch)
    observer = load("source_corrected_observer_capsule_test", HERE / "observe.py").observer()
    pin = "f" * 40
    owner = dict(
        source_git=pin, science_git=c.SCIENCE, original_owner_source_git=c.RESILIENT,
        original_owner_pid=96074, decision_sha256="a" * 64, handoff_sha256="b" * 64,
        source_sha256={name: c.sha((HERE / name).read_bytes())
                       for name in ("launch.py", "contract.py", "remote_adapter.py")},
        decision_path="test-decision.json", handoff_path="test-handoff.json",
    )
    decision = dict(status="accepted_source_correction_owner_handoff", placement_source_git=pin,
                    shared_future_bytes=123 * 2**20, shared_future_files=376)
    fit_handoff = handoff(c, "fitting")
    original_read, original_git = observer.read, observer.git_file

    def read(path):
        if path == observer.OWNER_JOURNAL / "owner-intent.json":
            return owner, "d" * 64
        if path == ROOT / "test-decision.json":
            return decision, "a" * 64
        if path == ROOT / "test-handoff.json":
            return fit_handoff, "b" * 64
        return original_read(path)

    monkeypatch.setattr(observer, "read", read)
    monkeypatch.setattr(observer, "git_file", lambda commit, path:
                        (ROOT / path).read_bytes() if commit == pin else original_git(commit, path))
    original_read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
                        json.dumps(fit_handoff) if path == ROOT / "test-handoff.json"
                        else original_read_text(path, *args, **kwargs))
    _, program, payload, provenance = observer.frozen()
    assert payload["mode"] == "progress" and payload["prepared"]["CPU_job"] == "979667"
    assert provenance["placement_source_git"] == pin
    assert provenance["progress_program_sha256"] == c.sha(program.encode())
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"mode":"dispatch"}'))
    with pytest.raises(AssertionError):
        exec(compile(program, "observer_rejects_dispatch", "exec"), {})
