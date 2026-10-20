"""Real frozen capsule plus isolated phase/return/funding acceptance tests."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def load(name, **imports):
    before = {k: sys.modules.get(k) for k in imports}
    sys.modules.update(imports)
    try:
        spec = importlib.util.spec_from_file_location("deadline_test_" + name, HERE / (name + ".py"))
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        for key, value in before.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


D = load("contract")
OWNER = load("owner", contract=D)
DEP = load("deployment", contract=D, owner=OWNER)
R = load("remote")


def once(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f)


def test_actual_stopped_journals_and_original_sources_still_bind():
    for role in D.PIDS:
        value = json.loads((HERE / f"actual-{role}-owner-stopped.json").read_bytes())
        D.local_handoff(value, role, OWNER.ROOT, lambda _: "", now=D.stamp(value["observed_at"]) + 1)
        with pytest.raises(ValueError, match="live"):
            D.local_handoff(value, role, OWNER.ROOT, lambda _: "python original owner", now=D.stamp(value["observed_at"]) + 1)


def test_real_HTTP499_capsule_includes_exact_return_source_and_new_handoff():
    parent, c = OWNER.previous()
    originals = {n: OWNER.git_file(c.PARENT, c.OLD + "/" + n).decode()
                 for n in ("placement.py", "owner.py", "remote.py", "deployment.py")}
    own = {n: (HERE / n).read_text() for n in OWNER.FILES}
    program = OWNER.build_program(parent, originals, own)
    compile(program, "all27_capsule", "exec")
    prefix = program[:program.index("v=json.load(sys.stdin)")]
    namespace = {}
    before = {k: sys.modules.get(k) for k in ("successor_contract", "placement")}
    try:
        exec(prefix, namespace)
        assert namespace["n"]["p"].SCIENCE == D.SCIENCE
        assert namespace["n"]["return_sender"].__code__.co_name == "return_sender"
    finally:
        for key, value in before.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    assert "n['__portable_source__']=" + repr(c.derived_portable_remote(originals["remote.py"])) in program
    assert "p.successor_handoff=d.handoff" in program


def test_real_full_freeze_reuses_prepared_models_with_new_control_receipts(tmp_path, monkeypatch):
    parent, lineage = OWNER.previous()
    old_decision = json.loads((OWNER.ROOT / D.PREVIOUS_DIR / "decision.accepted.json").read_bytes())
    pin = "f" * 40
    decision = {**old_decision,
        "status": "accepted_all_arms_evc27_deadline_placement", "placement_source_git": pin,
        "science_git": D.SCIENCE, "ownership": dict.fromkeys(D.ARMS, "evc27"),
        "dispatch_order": ["C", "B", "C0"], "same_six_shared_GPU_logs_reassigned": True,
        "training_wave_seconds": 57600, "stage_cutoffs_UTC": D.CUTOFFS,
        "preservation_end_UTC": D.END, "node": "evc27", "scientific_source_git": D.SCIENCE,
        "original_preparation_CPU_job": "979667", "funding": R.FUNDING}
    path = tmp_path / "accepted.json"
    path.write_text(json.dumps(decision))
    args = SimpleNamespace(source_git=pin, decision=path,
        handoff=HERE / "actual-fitting-owner-stopped.json",
        deployment_handoff=HERE / "actual-deployment-owner-stopped.json",
        observer_handoff=HERE / "actual-observer-owner-stopped.json")
    original_git = OWNER.git_file
    monkeypatch.setattr(OWNER, "git_file", lambda source, name:
        (OWNER.ROOT / name).read_bytes() if source == pin else original_git(source, name))
    original_handoff = D.local_handoff
    monkeypatch.setattr(D, "local_handoff", lambda v, role, repo, process:
        original_handoff(v, role, repo, process, now=D.stamp(v["observed_at"]) + 1))
    frozen = OWNER.freeze(args)
    assert frozen["payload"]["prepared"]["CPU_job"] == "979667"
    assert frozen["payload"]["prepared"]["source_git"] == D.SCIENCE
    assert frozen["payload"]["placement_source_git"] == pin
    assert frozen["payload"]["handoff"]["pid"] == 29171
    assert frozen["payload"]["accepted_http499_proofs"] == lineage.accepted_http(decision)
    assert frozen["original"].cache_receipts() is None
    R.validate_decision(frozen["payload"]["deadline_decision"])
    compile(frozen["program"], "full_new_bootstrap", "exec")


def fake_owner(tmp_path, *, malformed=None, publication_fail=None, partial=False):
    journal, audit = tmp_path / "journal", tmp_path / "audit"
    journal.mkdir()
    audit.mkdir()
    once(journal / "owner-intent.json", {"fixture": True})
    jobs = {a: {"arm": a, "node": "evc27", "source_git": D.SCIENCE, "GPU_job": str(i + 100)}
            for i, a in enumerate(D.ARMS)}
    calls = []

    def ready(progress, published):
        choices = [(e, a) for a, item in progress["arms"].items()
                   for e in range(1, item["sealed_epochs"] + 1) if (a, e) not in published]
        if choices:
            e, a = min(choices)
            return a, e

    def invoke(node, mode, phase=None, extra=None):
        calls.append((node, mode, phase, extra))
        if mode == "admit":
            return {"arms": ["B", "C"]}
        if mode == "dispatch":
            if partial:
                raise ConnectionError("uncertain after original first sbatch")
            return {"status": "actual_predictive_placement_jobs_submitted_once", "source_git": D.SCIENCE, "jobs": jobs}
        if mode == "dispatch_reconcile":
            return {"status": "actual_all27_original_dispatch_reconciled_readonly",
                    "jobs": {"C": jobs["C"]}, "submission_completeness_known": False}
        if mode == "progress":
            selected = extra["dispatched"]["jobs"]
            assert len(selected) == 1
            arm = next(iter(selected))
            if arm == malformed:
                raise ValueError("single bad arm must not block preservation")
            return {"arms": {arm: {"GPU_job": jobs[arm]["GPU_job"],
                "slurm": {"state": "COMPLETED"}, "terminal": {"arm": arm}, "stderr_bytes": 0,
                "sealed_epochs": 6, "first200": {"actual_updates": 200}}}}
        if mode == "publish":
            arm, epoch = extra["arm"], extra["epoch"]
            if arm == publication_fail:
                raise ConnectionError("unknown publication")
            return {"status": OWNER.VERIFIED, "objects": 5, "arm": arm, "epoch": epoch, "source_git": D.SCIENCE}
        raise AssertionError(mode)

    def transfer(arm=None):
        calls.append(("transfer", arm))
        return {"admitted_arms": ["C", "B", "C0"]} if arm is None else {"arm": arm}

    owner = SimpleNamespace(
        journal=journal, payload={}, placement=SimpleNamespace(once=once),
        args=SimpleNamespace(source_git="f" * 40, poll_seconds=120), invoke=invoke, transfer=transfer,
        original=SimpleNamespace(cache_receipts=lambda: {"raw": {}, "context": {}}, ready_epoch=ready,
                                 completed=lambda s: s["state"] == "COMPLETED", terminal=lambda s: True))
    return owner, audit, calls


@pytest.mark.parametrize("partial", [False, True])
def test_single_transfer_one_dispatch_all_preservation_and_exact_returns(tmp_path, partial):
    owner, audit, calls = fake_owner(tmp_path, partial=partial)
    result = OWNER.run(owner, audit=audit, clock=lambda: D.stamp("2026-10-20T21:00:00+00:00"), sleep=lambda _: None)
    expected = {"C"} if partial else set(D.ARMS)
    assert set(result["complete"]) == expected and set(result["returns"]) == expected
    assert len(result["privately_verified_epochs"]) == len(expected) * 6
    assert calls.count(("transfer", None)) == 1
    assert sum(len(c) == 4 and c[1] == "dispatch" for c in calls) == 1
    assert not any(len(c) == 4 and c[:2] == ("evc33", "dispatch") for c in calls)
    assert all((audit / f"actual-first200-{a}.json").exists() for a in expected)


@pytest.mark.parametrize("kind", ["malformed", "publication"])
def test_one_arm_error_preserves_other_arms_without_mutation_retry(tmp_path, kind):
    kwargs = {"malformed": "B"} if kind == "malformed" else {"publication_fail": "B"}
    owner, audit, calls = fake_owner(tmp_path, **kwargs)
    ticks = [D.stamp("2026-10-20T21:00:00+00:00")]
    pauses = []
    def sleep(_):
        pauses.append(True)
        if len(pauses) == 2:
            ticks[0] = D.stamp(D.END)
    result = OWNER.run(owner, audit=audit, clock=lambda: ticks[0], sleep=sleep)
    assert {"C", "C0"} <= set(result["returns"])
    assert sum(len(c) == 4 and c[1] == "publish" and c[3]["arm"] == "B" for c in calls) <= 1
    assert result["remote_jobs_cancelled"] == []


def test_two_wave_cutoff_never_submits_or_transfers(tmp_path):
    owner, audit, calls = fake_owner(tmp_path)
    result = OWNER.run(owner, audit=audit, clock=lambda: D.stamp("2026-10-21T09:16:00+00:00"), sleep=lambda _: None)
    assert not calls and not result["jobs"]


def test_observation_loss_after_terminal_does_not_abandon_remaining_epochs(tmp_path):
    owner, audit, calls = fake_owner(tmp_path)
    original, lost = owner.invoke, set()
    def invoke(node, mode, phase=None, extra=None):
        if mode == "progress" and any(len(c) == 4 and c[1] == "publish" for c in calls):
            arm = next(iter(extra["dispatched"]["jobs"]))
            if arm not in lost:
                lost.add(arm)
                raise ConnectionError("one observation outage after actual terminal")
        return original(node, mode, phase, extra)
    owner.invoke = invoke
    result = OWNER.run(owner, audit=audit, clock=lambda: D.stamp("2026-10-20T21:00:00+00:00"), sleep=lambda _: None)
    assert lost == set(D.ARMS)
    assert len(result["privately_verified_epochs"]) == 18 and len(result["returns"]) == 3
    assert result["reason"] == "all_original_jobs_terminal"


def test_three_return_mounts_and_paid_log_reservation_are_exact():
    value = {"original_runs": {}, "funding": {"other_shared_unpaid_files": 368}}
    paid = {"files": 6, "paths": [str(i) for i in range(6)], "originals_deleted": False}
    result = DEP.rebind_config(value, paid)
    assert set(result["original_runs"]) == set(D.ARMS)
    assert len({v["run_root"] for v in result["original_runs"].values()}) == 3
    assert result["funding"]["other_shared_unpaid_files"] + 8 == 370
    assert value["funding"]["other_shared_unpaid_files"] == 368
    with pytest.raises(ValueError):
        DEP.rebind_config(value, {**paid, "paths": ["same"] * 6})


def test_three_mirror_observer_preserves_account_API_and_original_source():
    parent, c = OWNER.previous()
    source = OWNER.git_file(DEP.CANONICAL, DEP.CANONICAL_DIR + "/observe.py.txt").decode()
    placement = OWNER.git_file(c.PARENT, c.OLD + "/placement.py").decode()
    result = DEP.observer_source(source, {"return_receipt_sha256": dict.fromkeys(D.ARMS, "a"*64)}, {}, placement)
    compile(result, "all_mirrors", "exec")
    assert all(str(D.return_root(a)) in result for a in D.ARMS)
    assert "GetAcceleratorQuotaStatistics" in result and "GetSubmissionLimits" in result
    assert 'job["node"] == "evc27"' in result
