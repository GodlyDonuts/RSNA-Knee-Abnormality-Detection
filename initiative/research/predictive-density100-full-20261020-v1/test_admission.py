"""Synthetic admission/ownership regressions; these are not actual pilot receipts."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
PLACEMENT = HERE.parent / "predictive-c0-portable-placement-20261020-v1/placement.py"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def api(monkeypatch):
    admission = module(HERE / "admission.py", "C100_test_admission")
    monkeypatch.setitem(sys.modules, "admission", admission)
    operations = module(HERE / "operations.py", "C100_test_operations")
    return admission, operations


@pytest.fixture
def coordinator(api, monkeypatch, tmp_path):
    a, _ = api
    derived = module(HERE / "derive.py", "C100_test_derive")
    monkeypatch.setitem(sys.modules, "derive", derived)
    result = module(HERE / "coordinator.py", "C100_test_coordinator")
    monkeypatch.setattr(result, "JOURNAL", tmp_path / "journal")
    return a, result


def private(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)
    return path


def pilot_fixture(a):
    pilot = {
        "status": "actual_C100_full_stop_resume_engineering_complete",
        "GPU_job": "981620",
        "source_git": a.PILOT_SOURCE,
        "binding_sha256": a.PILOT_BINDING,
        "branch": "C100",
        "full_stop_resume_complete_state_and_prediction_bytes_equal": True,
        "automatic_full_training_promotion": False,
        "clinical_or_held_out_accuracy_claim": False,
        "real_optimizer_history": [
            dict(step=i, studies=8 if i % 2 else 4, loss=92.7, exam_loss=0.8,
                 density_loss=0.919, encoder_gradient_norm=4.0,
                 head_gradient_norm=10.0, gradient_norm=11.0)
            for i in range(1, 25)
        ],
        "terminal": {"density_weight": 100, "steps": 24},
        "engineering_warnings": ["synthetic warning requiring explicit review"],
    }
    scheduler = dict(GPU_job="981620", state="COMPLETED", exit_code="0:0", node="evc27")
    return pilot, scheduler


def test_actual_pilot_requires_successful_original_scheduler_and_complete_history(api):
    a, _ = api
    pilot, scheduler = pilot_fixture(a)
    a.validate_pilot(pilot, scheduler)
    for state in ("PENDING", "RUNNING", "FAILED", "TIMEOUT"):
        with pytest.raises(ValueError, match="original C100 recovery"):
            a.validate_pilot(pilot, {**scheduler, "state": state})
    for change in (
        {"GPU_job": "981621"}, {"exit_code": "1:0"}, {"node": "evc33"},
    ):
        with pytest.raises(ValueError):
            a.validate_pilot(pilot, {**scheduler, **change})
    for history in (
        pilot["real_optimizer_history"][:-1],
        list(reversed(pilot["real_optimizer_history"])),
    ):
        with pytest.raises(ValueError, match="history"):
            a.validate_pilot({**pilot, "real_optimizer_history": history}, scheduler)


@pytest.mark.parametrize("field", [
    "loss", "exam_loss", "density_loss", "encoder_gradient_norm",
    "head_gradient_norm", "gradient_norm",
])
def test_pilot_rejects_nonfinite_at_any_loss_or_gradient(api, field):
    a, _ = api
    pilot, scheduler = pilot_fixture(a)
    pilot["real_optimizer_history"][17][field] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        a.validate_pilot(pilot, scheduler)


@pytest.mark.parametrize("change", [
    {"full_stop_resume_complete_state_and_prediction_bytes_equal": False},
    {"automatic_full_training_promotion": True},
    {"clinical_or_held_out_accuracy_claim": True},
    {"source_git": "0" * 40},
    {"binding_sha256": "0" * 64},
    {"terminal": {"density_weight": 0.1, "steps": 24}},
])
def test_pilot_rejects_unaccepted_or_other_science(api, change):
    a, _ = api
    pilot, scheduler = pilot_fixture(a)
    with pytest.raises(ValueError):
        a.validate_pilot({**pilot, **change}, scheduler)


def test_agent_decision_binds_actual_warnings_and_fixed_recipe(api):
    a, _ = api
    pilot, scheduler = pilot_fixture(a)
    source, code = "1" * 40, {"worker.py": "2" * 64}
    pilot_sha = a.sha(a.canonical(pilot))
    decision = dict(
        status="accepted_separate_full_C100_after_actual_pilot_review",
        pilot_scheduler=scheduler, pilot_acceptance_sha256=pilot_sha,
        source_git=source, code_sha256=code, recipe=a.recipe(),
        pilot_warnings_reviewed=pilot["engineering_warnings"],
        agent_reasoning="Synthetic review: the recorded warning was examined without changing "
        "the fixed recipe or selecting on engineering predictions.",
        observed_C_C0_gradient_risk_reviewed=True, primary_B_C_C0_unchanged=True,
        primary_Kaggle_reserve_seconds=3360, engineering_predictions_used_for_selection=False,
        automatic_recipe_retuning=False,
    )
    a.validate_decision(decision, pilot, pilot_sha, source, code)
    for change in (
        {"pilot_warnings_reviewed": []},
        {"pilot_acceptance_sha256": "3" * 64},
        {"engineering_predictions_used_for_selection": True},
        {"automatic_recipe_retuning": True},
        {"primary_B_C_C0_unchanged": False},
        {"primary_Kaggle_reserve_seconds": 3359},
        {"agent_reasoning": "approved"},
        {"recipe": {**a.recipe(), "density_weight": 10.0}},
        {"recipe": {**a.recipe(), "encoder_lr": 1e-5}},
    ):
        with pytest.raises(ValueError):
            a.validate_decision({**decision, **change}, pilot, pilot_sha, source, code)


def reuse_fixture(a, monkeypatch):
    # Generated identifiers are deliberately unrelated to private fitting studies.
    index = [{"uid": f"synthetic-{i:04d}", "proof": {"bytes": 100 + i}}
             for i in range(3887)]
    roster = a.sha(a.canonical([r["uid"] for r in index]))
    monkeypatch.setattr(a, "ROSTER", roster)
    index_sha, binding_sha = a.sha(a.canonical(index)), "b" * 64
    code = {"rsna/competitive/predictive/model.py": "c" * 64}
    old = dict(source_git=a.BASE, expected_initial_shared_tensor_sha256=a.INITIAL,
               arm="C0", node="evc27", code_sha256=code,
               context_index={"sha256": index_sha})
    portable = dict(
        status="actual_C0_context_transfer_and_CPU_admission_accepted",
        science_git=a.BASE, binding_sha256={"C0": binding_sha},
        initial_shared_tensor_sha256=a.INITIAL, destination_context_index_sha256=index_sha,
        context_bytes=sum(row["proof"]["bytes"] for row in index),
    )
    proof = dict(
        status="actual_complete3887_predictive_fitting_inputs_accepted",
        source_git=a.BASE, code_sha256=code, context_index_sha256=index_sha,
        all_context_original_hashes_and_stats_verified=True,
        fitting_order_sha256=roster, original_context_count=3887,
    )
    decision = json.loads((HERE.parent / "predictive-http499-owner-handoff-20261020-v1"
                           / "decision.accepted.json").read_bytes())
    control = dict(
        operational_control_git=a.PRIMARY_SOURCE,
        **{key: decision[key] for key in a.HTTP499_PROOFS},
        accepted_http499_proofs={key: decision[key] for key in a.HTTP499_PROOFS},
        source_correction_policy=decision["source_correction_policy"],
        source_correction_regression_sha256=decision["source_correction_regression_sha256"],
        accepted_source_correction_proofs={
            "source_correction_regression_sha256": decision["source_correction_regression_sha256"],
        },
    )
    for value in (portable, old, proof):
        value["cache_successor_control"] = copy.deepcopy(control)
        value["placement_control_source_git"] = a.PRIMARY_SOURCE
    old["input_admission"] = {"sha256": "d" * 64}
    portable["input_admission_sha256"] = {"C0": old["input_admission"]["sha256"]}
    return portable, old, binding_sha, proof, index


def test_reuse_requires_exact_complete_membership_and_prior_proofs(api, monkeypatch):
    a, _ = api
    args = reuse_fixture(a, monkeypatch)
    a.validate_reuse(*args)
    mutations = [
        (0, "status", "transfer_in_progress"),
        (0, "binding_sha256", {"C0": "a" * 64}),
        (0, "destination_context_index_sha256", "a" * 64),
        (0, "context_bytes", args[0]["context_bytes"] + 1),
        (1, "node", "evc33"), (1, "source_git", "a" * 40),
        (1, "expected_initial_shared_tensor_sha256", "a" * 64),
        (3, "code_sha256", {}), (3, "all_context_original_hashes_and_stats_verified", False),
        (3, "fitting_order_sha256", "a" * 64),
        (0, "placement_control_source_git", "2b834507e19c405971228e1bc405462cd370f560"),
        (1, "placement_control_source_git", "2b834507e19c405971228e1bc405462cd370f560"),
        (0, "input_admission_sha256", {"C0": "0" * 64}),
    ]
    for position, key, value in mutations:
        changed = copy.deepcopy(args)
        changed[position][key] = value
        with pytest.raises(ValueError):
            a.validate_reuse(*changed)
    for kind in ("missing", "duplicate", "changed", "reordered"):
        changed = copy.deepcopy(args)
        rows = changed[-1]
        if kind == "missing":
            rows.pop()
        elif kind == "duplicate":
            rows[-1] = copy.deepcopy(rows[-2])
        elif kind == "changed":
            rows[-1]["uid"] = "synthetic-9999"
        else:
            rows[0], rows[1] = rows[1], rows[0]
        with pytest.raises(ValueError):
            a.validate_reuse(*changed)


def test_reuse_rejects_consistently_stale_or_changed_correction_lineage(api, monkeypatch):
    a, _ = api
    args = reuse_fixture(a, monkeypatch)
    for key, value in (
        ("operational_control_git", "2b834507e19c405971228e1bc405462cd370f560"),
        ("source_correction_regression_sha256", "0" * 64),
        ("source_correction_policy", {}),
        ("accepted_source_correction_proofs", {"source_correction_regression_sha256": "0" * 64}),
    ):
        changed = copy.deepcopy(args)
        for position in (0, 1, 3):
            changed[position]["cache_successor_control"][key] = value
        with pytest.raises(ValueError, match="source-corrected"):
            a.validate_reuse(*changed)


def test_original_reference_rejects_changed_bytes_and_links(api, tmp_path):
    a, _ = api
    path = private(tmp_path / "input.json", b'{"original":true}\r\n')
    ref = a.reference(path)
    assert a.read_ref(ref) == path.read_bytes()
    path.write_bytes(b'{"original":false}\n')
    with pytest.raises(ValueError, match="reference changed"):
        a.read_ref(ref)
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="owner-only"):
        a.reference(link)


def zero_paid():
    return dict(contexts_paid=0, C0_controls_paid=0, C0_fit_paid=0,
                pilot_checkpoints_paid=0, pilot_controls_paid=0)


def test_node_budget_strict_floor_and_paid_bytes_counted_once(api):
    a, _ = api
    paid = zero_paid()
    required = a.node_budget(1000 * a.GIB, **paid)["required_bytes"]
    with pytest.raises(ValueError, match="liability"):
        a.node_budget(required, **paid)
    assert a.node_budget(required + 1, **paid)["remaining_bytes"] == 1
    paid.update(contexts_paid=10 * a.GIB, C0_fit_paid=2 * a.GIB,
                pilot_checkpoints_paid=a.GIB)
    result = a.node_budget(required + 1, **paid)
    assert result["required_bytes"] == required - 13 * a.GIB
    # Actual free space falls as these original bytes become physically paid.
    assert a.node_budget(required + 1 - 13 * a.GIB, **paid)["remaining_bytes"] == 1
    # Deleting an original increases free bytes and reinstates its full liability.
    paid["C0_fit_paid"] = 0
    assert a.node_budget(required + 1 - 11 * a.GIB, **paid)["remaining_bytes"] == 1


@pytest.mark.parametrize("key,value", [
    ("contexts_paid", -1), ("contexts_paid", True), ("contexts_paid", 96 * 2**30 + 1),
    ("C0_controls_paid", 64 * 2**20 + 1), ("C0_fit_paid", 12 * 2**30 + 1),
    ("pilot_checkpoints_paid", 6 * 2**30 + 1), ("pilot_controls_paid", 4 * 2**20 + 1),
    ("own_controls_paid", 16 * 2**20 + 1), ("own_fit_paid", 12 * 2**30 + 1),
])
def test_invalid_paid_scopes_never_create_capacity(api, key, value):
    a, _ = api
    paid = {**zero_paid(), key: value}
    with pytest.raises(ValueError, match="paid bytes"):
        a.node_budget(1000 * a.GIB, **paid)


def test_transient_downloads_never_retire_permanent_obligations(api, tmp_path):
    a, _ = api
    private(tmp_path / "checkpoints/epoch-001.pt", b"retained")
    private(tmp_path / "publication/epoch-001/independent-download-abc/model.pt", b"x" * 101)
    private(tmp_path / "contexts/context-0000.npz", b"x" * 37)
    assert a.used(tmp_path, 1000) == len(b"retained") + 37
    assert a.used(tmp_path, 1000, exclude_contexts=True) == len(b"retained")
    with pytest.raises(ValueError, match="retained bound"):
        a.used(tmp_path, 1)


def test_fresh_node_samples_df_after_all_paid_file_scans(api, monkeypatch, tmp_path):
    a, _ = api
    monkeypatch.setattr(a, "PILOT", tmp_path)
    events = []
    monkeypatch.setattr(a, "used", lambda *args, **kw: events.append("paid") or 0)

    def disk_usage(path):
        assert path == "/tmp"
        events.append("df")
        return SimpleNamespace(free=1000 * a.GIB)

    monkeypatch.setattr(a.shutil, "disk_usage", disk_usage)
    a.fresh_node({"context_bytes": 0})
    assert events[-1] == "df" and events.count("paid") == 6


def quota_text(a, free_files, free_bytes=None):
    free_bytes = 30 * a.GIB if free_bytes is None else free_bytes
    return f"/home/sa305415 0 {free_bytes // 1024} 0 - {1000000 - free_files} 1000000 0 -\n"


def job_fixture(a, root, arm, job):
    name = "rsna-predictive-full2-" + arm
    return dict(arm=arm, GPU_job=job, source_git=a.BASE,
                node="evc27" if arm == "C0" else "evc33", job_name=name,
                run_root=f"/tmp/rsna-predictive-full-{arm}-20261020-v2",
                binding_sha256="b" * 64, script_sha256="c" * 64,
                stdout=str(root / f"{name}-{job}.out"),
                stderr=str(root / f"{name}-{job}.err"))


def primary_dispatch_fixture(a, c, monkeypatch, tmp_path):
    audit, journal = tmp_path / "primary-audit", tmp_path / "primary-journal"
    monkeypatch.setattr(c, "ORIGINAL_OWNER", audit)
    monkeypatch.setattr(c, "ORIGINAL_JOURNAL", journal)
    private(journal / "owner-intent.json", a.canonical({
        "source_git": a.PRIMARY_SOURCE, "science_git": a.BASE, "pid": a.PRIMARY_OWNER_PID,
    }))
    jobs = {arm: job_fixture(a, Path("/home/sa305415/rsna-knee/kaggle"), arm, str(900020 + i))
            for i, arm in enumerate(("B", "C", "C0"))}
    for role, arms in (("BC", ("B", "C")), ("C0", ("C0",))):
        receipt = dict(status="actual_predictive_placement_jobs_submitted_once",
                       source_git=a.BASE, jobs={arm: jobs[arm] for arm in arms})
        private(audit / f"actual-owner-dispatch-{role}.json", a.canonical(receipt))
        private(journal / f"dispatch-{role}-result.json", a.canonical(receipt))
        private(journal / f"dispatch-{role}-intent.json", a.canonical({
            "mode": "dispatch", "node": "evc27" if role == "C0" else "evc33",
            "placement_source_git": a.PRIMARY_SOURCE, "program_sha256": "e" * 64,
        }))
    return jobs


def test_current_owner_retains_all_six_paid_logs_including_four_bc(
    coordinator, monkeypatch, tmp_path,
):
    a, c = coordinator
    expected = primary_dispatch_fixture(a, c, monkeypatch, tmp_path)
    jobs = c.original_jobs()
    assert jobs == expected
    # Exercise actual physical-inode accounting using only synthetic local files.
    logs = tmp_path / "logs"
    local_jobs = copy.deepcopy(jobs)
    for value in local_jobs.values():
        for key in ("stdout", "stderr"):
            value[key] = str(logs / Path(value[key]).name)
            private(Path(value[key]), b"")
    placement = module(PLACEMENT, "C100_current_owner_paid_logs")
    paid = placement.paid_shared_logs(local_jobs, logs)
    assert paid["files"] == 6
    funded = a.shared_budget(quota_text(a, 5373), paid["files"])
    assert funded["free_files"] - funded["unpaid_files"] == 5003
    old_c0_only = placement.paid_shared_logs({"C0": local_jobs["C0"]}, logs)
    with pytest.raises(ValueError, match="shared reserve"):
        a.shared_budget(quota_text(a, 5373), old_c0_only["files"])


@pytest.mark.parametrize("stale_part", ["owner", "phase", "unmatched_result", "job_source"])
def test_coordinator_refuses_stale_or_unmatched_primary_dispatch(
    coordinator, monkeypatch, tmp_path, stale_part,
):
    a, c = coordinator
    primary_dispatch_fixture(a, c, monkeypatch, tmp_path)
    if stale_part == "owner":
        path = c.ORIGINAL_JOURNAL / "owner-intent.json"
        value = json.loads(path.read_bytes())
        value["source_git"] = "2b834507e19c405971228e1bc405462cd370f560"
    elif stale_part == "phase":
        path = c.ORIGINAL_JOURNAL / "dispatch-BC-intent.json"
        value = json.loads(path.read_bytes())
        value["placement_source_git"] = "2b834507e19c405971228e1bc405462cd370f560"
    else:
        path = c.ORIGINAL_OWNER / "actual-owner-dispatch-BC.json"
        value = json.loads(path.read_bytes())
        if stale_part == "unmatched_result":
            value["jobs"]["B"]["binding_sha256"] = "0" * 64
        else:
            value["jobs"]["B"]["source_git"] = "0" * 40
            private(c.ORIGINAL_JOURNAL / "dispatch-BC-result.json", a.canonical(value))
    private(path, a.canonical(value))
    with pytest.raises(ValueError):
        c.original_jobs()


def test_shared_reserve_credits_only_existing_exact_original_logs(api, tmp_path):
    a, _ = api
    placement = module(PLACEMENT, "C100_test_actual_placement")
    jobs = {arm: job_fixture(a, tmp_path, arm, str(900000 + i))
            for i, arm in enumerate(("B", "C", "C0"))}
    assert placement.paid_shared_logs(jobs, tmp_path)["files"] == 0
    a.shared_budget(quota_text(a, 5379), 0)
    for job in jobs.values():
        private(Path(job["stdout"]), b'{"step":1}\n')
        private(Path(job["stderr"]), b"")
    paid = placement.paid_shared_logs(jobs, tmp_path)
    assert paid["files"] == 6 and paid["originals_deleted"] is False
    assert a.shared_budget(quota_text(a, 5373), paid["files"])["unpaid_files"] == 370
    with pytest.raises(ValueError, match="shared reserve"):
        a.shared_budget(quota_text(a, 5373), 2)  # Seeing C0 alone is insufficient.
    spoofed = copy.deepcopy(jobs)
    spoofed["B"]["stdout"] = str(tmp_path / "unrelated.out")
    with pytest.raises(ValueError, match="Log path"):
        placement.paid_shared_logs(spoofed, tmp_path)
    # Hard links cannot produce two physically paid original log credits.
    err = Path(jobs["B"]["stderr"])
    err.unlink()
    os.link(jobs["B"]["stdout"], err)
    with pytest.raises(ValueError, match="Duplicate paid log inode"):
        placement.paid_shared_logs(jobs, tmp_path)


def test_shared_budget_retains_byte_floor_and_rejects_guessed_credit(api):
    a, _ = api
    with pytest.raises(ValueError, match="shared reserve"):
        a.shared_budget(quota_text(a, 6000, 20 * a.GIB + a.SHARED_BYTES), 0)
    for credit in (-1, 7, True, 6.0):
        with pytest.raises(ValueError, match="original six"):
            a.shared_budget(quota_text(a, 6000), credit)


def test_original_jobs_merges_exact_submitted_receipt_and_rejects_conflicts(api, monkeypatch,
                                                                         tmp_path):
    a, ops = api
    monkeypatch.setattr(a, "C0", tmp_path / "C0")
    jobs = {arm: job_fixture(a, Path("/home/sa305415/rsna-knee/kaggle"), arm, str(900010 + i))
            for i, arm in enumerate(("B", "C", "C0"))}
    private(a.C0 / "dispatch/actual.json", a.canonical({"jobs": {"C0": {"wrong": True}}}))
    payload = {"original_dispatched_jobs": {arm: jobs[arm] for arm in ("B", "C")}}
    assert ops.original_jobs(payload) == payload["original_dispatched_jobs"]
    private(a.C0 / "dispatch/submitted.json", a.canonical({
        "status": "actual_predictive_placement_jobs_submitted_once",
        "source_git": a.BASE, "jobs": {"C0": jobs["C0"]},
    }))
    private(a.C0 / "dispatch/intent.json", a.canonical({
        "source_git": a.BASE, "owner_git": a.PRIMARY_SOURCE, "arms": ["C0"],
    }))
    assert ops.original_jobs(payload) == jobs
    assert ops.original_jobs({"original_dispatched_jobs": jobs}) == jobs
    changed = copy.deepcopy(jobs)
    changed["C0"]["binding_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="ownership differs"):
        ops.original_jobs({"original_dispatched_jobs": changed})
    private(a.C0 / "dispatch/intent.json", a.canonical({
        "source_git": a.BASE, "owner_git": "2b834507e19c405971228e1bc405462cd370f560",
        "arms": ["C0"],
    }))
    with pytest.raises(ValueError, match="Stale primary dispatch"):
        ops.original_jobs(payload)


def test_completed_job_can_be_absent_from_live_controller(api, monkeypatch):
    _, ops = api
    calls = []

    def output(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "sacct":
            return "981620|COMPLETED|0:0|evc27|2026-10-20T17:00:00|2026-10-20T17:15:00|\n"
        if cmd[0] == "squeue":
            if "-j" in cmd:
                raise subprocess.CalledProcessError(1, cmd, stderr="Invalid job id specified")
            return "1981620\n9816200\n"  # Unrelated and prefix-like IDs are not the original.
        raise AssertionError("Unexpected external command")

    monkeypatch.setattr(ops.subprocess, "check_output", output)
    state = ops.scheduler("981620")
    assert state["state"] == "COMPLETED" and state["queue_empty"] is True
    assert any(cmd[0] == "squeue" and "-u" in cmd for cmd in calls)


def test_full_c100_cannot_jump_primary_c0_and_uses_node_local_logs(api):
    a, ops = api
    script = ops.script("a" * 64, "900012")
    assert "#SBATCH --dependency=after:900012\n" in script
    assert "#SBATCH --nodelist=evc27\n" in script
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1\n" in script
    assert "#SBATCH --time=08:00:00\n" in script
    assert f"#SBATCH --output={a.ROOT}/GPU-%j.out\n" in script
    assert f"#SBATCH --error={a.ROOT}/GPU-%j.err\n" in script
    assert "/home/sa305415/rsna-knee/kaggle" not in script
    assert "--binding-sha256 " + "a" * 64 in script


def test_dispatch_refuses_before_primary_c0_without_any_sbatch(api, monkeypatch, tmp_path):
    a, ops = api
    monkeypatch.setattr(a, "ROOT", tmp_path / "C100")
    monkeypatch.setattr(a, "RUN", tmp_path / "run-C100")
    monkeypatch.setattr(ops, "load_binding", lambda source: ({}, {}, {"sha256": "a" * 64}))
    monkeypatch.setattr(ops, "original_jobs", lambda v: {})

    def prohibited(*args, **kwargs):
        raise AssertionError("No scheduler allocation or submission is permitted")

    monkeypatch.setattr(ops.subprocess, "run", prohibited)
    monkeypatch.setattr(ops.subprocess, "check_output", prohibited)
    with pytest.raises(ValueError, match="primary C0 dispatch"):
        ops.dispatch({"source_git": "a" * 40})
    assert not a.ROOT.exists() and not a.RUN.exists()


def test_dispatch_applies_utc_before_slurm_parses_deadline(api, monkeypatch, tmp_path):
    a, ops = api
    monkeypatch.setattr(a, "ROOT", tmp_path / "C100")
    monkeypatch.setattr(a, "RUN", tmp_path / "run-C100")
    monkeypatch.setattr(a, "C0", tmp_path / "C0")
    a.ROOT.mkdir(mode=0o700)
    private(a.C0 / "portable-accepted.json", b"{}")
    c0 = job_fixture(a, Path("/home/sa305415/rsna-knee/kaggle"), "C0", "900012")
    monkeypatch.setattr(ops, "load_binding", lambda source: (
        {"density100_pilot_sha256": "b" * 64}, {}, {"sha256": "a" * 64},
    ))
    monkeypatch.setattr(ops, "original_jobs", lambda v: {"C0": c0})
    monkeypatch.setattr(ops, "scheduler", lambda job: {"state": "RUNNING"})
    monkeypatch.setattr(a, "fresh_node", lambda portable: {"synthetic": True})
    monkeypatch.setattr(ops, "shared", lambda v: {"synthetic": True})
    monkeypatch.setattr(ops, "datetime", SimpleNamespace(
        now=lambda tz: a.datetime(2026, 10, 20, 18, tzinfo=a.timezone.utc),
    ))
    monkeypatch.setenv("TZ", "America/New_York")

    def node(command, **kwargs):
        assert command == ["scontrol", "show", "node", "evc27", "-o"]
        return "NodeName=evc27 Gres=gpu:nvidia_h100_pcie:2 CfgTRES=cpu=64,gres/gpu=2"

    calls = []

    def submit(command, **kwargs):
        assert command == ["sbatch", "--parsable"]
        assert kwargs["env"]["TZ"] == "UTC"
        assert "#SBATCH --deadline=2026-10-22T04:00:00\n" in kwargs["input"]
        assert "#SBATCH --dependency=after:900012\n" in kwargs["input"]
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="999001\n", stderr="")

    monkeypatch.setattr(ops.subprocess, "check_output", node)
    monkeypatch.setattr(ops.subprocess, "run", submit)
    result = ops.dispatch({"source_git": "a" * 40})
    assert result["GPU_job"] == "999001" and len(calls) == 1
    assert json.loads((a.ROOT / "GPU-submitted.json").read_bytes()) == result


def test_coordinator_pending_prerequisite_does_not_create_admission_intent(
    coordinator, monkeypatch, tmp_path,
):
    a, c = coordinator
    decision = private(tmp_path / "decision.json", b'{"synthetic":true}')
    monkeypatch.setattr(sys, "argv", [
        "coordinator.py", "--source-git", "a" * 40, "--phase", "admit",
        "--decision", str(decision),
    ])
    monkeypatch.setattr(c, "frozen", lambda args: ({}, "pass", None))
    calls = []

    def call(args, frozen_source, phase, **kwargs):
        calls.append(phase)
        assert phase == "status", "An incomplete prerequisite must not enter mutation code"
        return {"pilot_scheduler": {"GPU_job": "981620", "state": "PENDING"}}

    monkeypatch.setattr(c, "call", call)
    # main changes process umask intentionally; restore it for the test runner.
    old_umask = os.umask(0o077)
    try:
        with pytest.raises(ValueError, match="prerequisites before mutation"):
            c.main()
    finally:
        os.umask(old_umask)
    assert calls == ["status"] and not c.JOURNAL.exists()


def test_freeze_rejects_changed_control_before_derivation_or_transport(
    coordinator, monkeypatch, tmp_path,
):
    _, c = coordinator
    frozen_here = tmp_path / "repo/initiative/research/C100"
    private(frozen_here / "admission.py", b"# altered local bytes\n")
    monkeypatch.setattr(c, "REPO", tmp_path / "repo")
    monkeypatch.setattr(c, "HERE", frozen_here)
    pins = []

    def git_file(pin, path):
        pins.append((pin, path))
        return b"# original committed bytes\n"

    monkeypatch.setattr(c, "git_file", git_file)
    monkeypatch.setattr(c.derive, "build", lambda repo: pytest.fail("Changed source was admitted"))
    with pytest.raises(ValueError, match="exact committed"):
        c.frozen(SimpleNamespace(source_git="a" * 40))
    assert pins == [("a" * 40, "initiative/research/C100/admission.py")]


@pytest.mark.parametrize("outcome", ["timeout", "nonzero", "unparseable_success"])
def test_uncertain_publication_never_retries_and_token_stays_in_stdin(
    coordinator, monkeypatch, outcome,
):
    _, c = coordinator
    args = SimpleNamespace(source_git="a" * 40, socket="/tmp/synthetic-ssh.sock")
    token = "hf_SYNTHETIC_NOT_A_REAL_CREDENTIAL"
    monkeypatch.setattr(c, "token", lambda: token)
    monkeypatch.setattr(c, "original_jobs", dict)
    transport = SimpleNamespace(
        COMMAND="stdin_envelope_only",
        envelope=lambda program, payload: json.dumps({"program": program, "payload": payload}),
    )
    frozen = ({"code_sha256": {"worker.py": "b" * 64}}, "pass", transport)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert token not in " ".join(command)
        assert json.loads(kwargs["input"])["payload"]["token"] == token
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 3000)
        return SimpleNamespace(returncode=1 if outcome == "nonzero" else 0,
                               stdout="not a receipt", stderr="synthetic failure")

    monkeypatch.setattr(c.subprocess, "run", run)
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        c.call(args, frozen, "publish", epoch=1)
    assert (c.JOURNAL / "publish-001-intent.json").is_file()
    assert not (c.JOURNAL / "publish-001-result.json").exists()
    assert all(token.encode() not in path.read_bytes() for path in c.JOURNAL.iterdir())
    with pytest.raises(FileExistsError):
        c.call(args, frozen, "publish", epoch=1)
    assert len(calls) == 1


def test_terminal_preservation_collects_final_epoch_before_finishing(coordinator, monkeypatch):
    a, c = coordinator
    args = SimpleNamespace(source_git="a" * 40)
    verified = "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"
    private(c.JOURNAL / "dispatch-result.json", b'{"synthetic_dispatch":true}')
    before = {
        "dispatch": {"source_git": args.source_git},
        "ready_epochs": list(range(1, 7)),
        "publications": {str(e): {"status": verified} for e in range(1, 6)},
        "full_scheduler": {"state": "COMPLETED", "exit_code": "0:0"},
    }
    after = copy.deepcopy(before)
    after["publications"]["6"] = {"status": verified}
    calls = []

    def call(args, frozen, phase, **kwargs):
        calls.append((phase, kwargs.get("epoch")))
        if phase == "publish":
            assert kwargs["epoch"] == 6
            return {"status": verified}
        return before if len(calls) == 1 else after

    monkeypatch.setattr(c, "call", call)
    monkeypatch.setattr(c.time, "time", lambda: a.datetime(
        2026, 10, 21, 20, tzinfo=a.timezone.utc,
    ).timestamp())
    final = c.preserve(args, None)
    assert calls == [("status", None), ("publish", 6), ("status", None)]
    assert len(final["publications"]) == 6
    assert json.loads((c.JOURNAL / "preservation-owner-result.json").read_text()) == final


@pytest.mark.parametrize("returncode,stderr,retryable", [
    (255, "ssh: connection lost", True),
    (1, "srun: error: Unable to allocate resources: Resource temporarily unavailable\n", True),
    (1, "srun: error: Unable to allocate resources: Requested nodes are busy\n", True),
    (1, '{"error_type":"ValueError","site_line":142}\n', False),
    (1, "srun: error: Invalid account or account/partition combination specified\n", False),
])
def test_only_explicit_readonly_transport_failures_are_retryable(
    coordinator, monkeypatch, returncode, stderr, retryable,
):
    _, c = coordinator
    args = SimpleNamespace(source_git="a" * 40, socket="/tmp/synthetic.sock")
    frozen = (
        {"code_sha256": {}}, "pass",
        SimpleNamespace(COMMAND="stdin_only", envelope=lambda *args: "{}"),
    )
    monkeypatch.setattr(c, "original_jobs", dict)
    monkeypatch.setattr(c.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=returncode, stdout="", stderr=stderr,
    ))
    error = c.ObservationUnavailable if retryable else ValueError
    with pytest.raises(error):
        c.call(args, frozen, "status")
    assert not c.JOURNAL.exists()
    # Exactly the same SSH/srun failure for a mutation must never become retryable.
    with pytest.raises(ValueError):
        c.call(args, frozen, "dispatch")
    assert (c.JOURNAL / "dispatch-intent.json").exists()


def fake_clock(c, monkeypatch):
    now, sleeps = [0.0], []
    monkeypatch.setattr(c.time, "time", lambda: now[0])

    def sleep(seconds):
        assert 0 < seconds <= 120
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(c.time, "sleep", sleep)
    return now, sleeps


def test_readonly_observation_recovers_without_mutation_or_resetting_deadline(
    coordinator, monkeypatch,
):
    _, c = coordinator
    _, sleeps = fake_clock(c, monkeypatch)
    calls = []

    def call(args, frozen, phase):
        assert phase == "status"
        calls.append(phase)
        if len(calls) < 3:
            raise c.ObservationUnavailable("synthetic observation loss")
        return {"observed": True}

    monkeypatch.setattr(c, "call", call)
    assert c.observe(None, None, 1000) == {"observed": True}
    assert calls == ["status"] * 3 and sleeps == [120, 120]
    assert not c.JOURNAL.exists()


def test_readonly_reconnect_is_bounded_by_incident_and_absolute_deadlines(
    coordinator, monkeypatch,
):
    _, c = coordinator
    now, sleeps = fake_clock(c, monkeypatch)
    calls = []

    def unavailable(args, frozen, phase):
        calls.append(phase)
        raise c.ObservationUnavailable("synthetic observation loss")

    monkeypatch.setattr(c, "call", unavailable)
    with pytest.raises(c.ObservationUnavailable, match="Thirty-minute"):
        c.observe(None, None, 3600)
    assert now[0] == 1800 and len(calls) == 15 and sum(sleeps) == 1800
    calls.clear()
    sleeps.clear()
    now[0] = 0
    assert c.observe(None, None, 250) is None
    assert now[0] == 250 and len(calls) == 3 and sleeps == [120, 120, 10]


def test_source_or_receipt_validation_failure_is_never_retried(coordinator, monkeypatch):
    _, c = coordinator
    _, sleeps = fake_clock(c, monkeypatch)
    calls = []

    def invalid(args, frozen, phase):
        calls.append(phase)
        raise ValueError("Original binding differs")

    monkeypatch.setattr(c, "call", invalid)
    with pytest.raises(ValueError, match="binding"):
        c.observe(None, None, 3600)
    assert calls == ["status"] and sleeps == []


@pytest.mark.parametrize("field", ["source_http_sha256", "previous_source_http_sha256",
                                  "http499_recovery_decision_sha256", "accepted_http499_proofs"])
def test_reuse_rejects_consistently_changed_HTTP499_proofs(api, monkeypatch, field):
    a, _ = api
    args = reuse_fixture(a, monkeypatch)
    replacement = {} if field == "accepted_http499_proofs" else "0" * 64
    for value in (args[0], args[1], args[3]):
        value["cache_successor_control"][field] = replacement
    with pytest.raises(ValueError, match="HTTP499"):
        a.validate_reuse(*args)


def test_retarget_uses_actual_new_primary_but_original_C100_pilot(coordinator):
    a, c = coordinator
    assert a.PRIMARY_SOURCE == "291283545c930b730bb257abb10c2a0814b055eb"
    assert a.PRIMARY_OWNER_PID == 29171
    assert set(c.original_jobs()) <= {"B", "C", "C0"}
    assert a.PILOT_SOURCE == "068a14a75e4d2b3bd83dda7783528956f7a2ca0f"
    assert a.PILOT_BINDING == "e11c7c7fe5c1fd7cae87df313fb5d42be7128b7906f8d716a486b0857bbdd37a"


def test_old_node_threshold_cannot_spend_independent_G100_or_descriptor_reserves(api):
    a, _ = api
    old_required = (20 * a.GIB + 205676085248 + 96 * a.GIB + 64 * a.MIB
                    + 12 * a.GIB + 6 * a.GIB + 4 * a.MIB + 16 * a.MIB + 12 * a.GIB)
    with pytest.raises(ValueError, match="liability"):
        a.node_budget(old_required + 1, **zero_paid())
    with pytest.raises(ValueError, match="liability"):
        a.node_budget(old_required + 6 * a.GIB + 4 * a.MIB, **zero_paid())
    extra = 6 * a.GIB + 4 * a.MIB + a.MIB
    result = a.node_budget(old_required + extra + 1, **zero_paid())
    assert result["remaining_bytes"] == 1
    assert result["conservative_G100_reserved_bytes_no_paid_credit"] == 6 * a.GIB + 4 * a.MIB
    assert result["descriptor_reserved_bytes_no_paid_credit"] == a.MIB
