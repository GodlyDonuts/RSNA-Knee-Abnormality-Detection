"""Local reserve/owner regressions; no remote calls or publication."""

import ast
import contextlib
import importlib.util
import json
import os
import types
import uuid
from pathlib import Path

import pytest

D = Path(__file__).resolve().parent


def functions():
    source = (D / "program.py.txt").read_text()
    nodes = ast.parse(source).body
    return {
        n.name: ast.get_source_segment(source, n) for n in nodes if isinstance(n, ast.FunctionDef)
    }


def accounting():
    scope = {}
    exec(functions()["shared_file_accounting"], scope)
    return scope["shared_file_accounting"]


def test_four_actual_owned_entries_are_charged_once():
    result = accounting()(5375, 4)
    assert result["remaining_future_shared_files"] == 372
    assert result["files_after_all_remaining_liabilities"] == 5003
    assert result["new_permanent_shared_files"] == result["new_shared_byte_reservation"] == 0


@pytest.mark.parametrize("free,owned", [(5371, 4), (5375, 3), (5375, 5), (5375, 8), (5375, True)])
def test_unfunded_or_unproven_file_credit_refused(free, owned):
    with pytest.raises(AssertionError):
        accounting()(free, owned)


def quiescence():
    spec = importlib.util.spec_from_file_location("v8_quiescence_test", D / "quiescence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root):
    journal = root / "artifacts/pointwise-final-deployment-20261020-v1"
    journal.mkdir(parents=True)
    (journal / "continuation-intent.json").write_text(
        json.dumps(
            {"PID": 72266, "original_source_git": "01cbe205080487b3fb39fbf7c0d28e112fcd7d77"}
        )
    )
    return journal


def test_healthy_owner_without_terminal_prevents_any_mirror(tmp_path):
    fixture(tmp_path)
    with pytest.raises(AssertionError, match="actual pointwise"):
        quiescence().check(tmp_path, processes="", own_pid=1)


@pytest.mark.parametrize(
    "processes",
    [
        "72266 python3 initiative/research/pointwise-final-deployment-20261020-v1/continue_verified_dataset.py",
        "40 python3 initiative/research/teacher-rank-final-deployment-20261020-v1/follow_once.py",
        "20168 python3 initiative/research/teacher-rank-delta-deployment-20261020-v1/continue_once.py",
        "41 ssh newton srun --job-name=rsna-pointwise-serving-status",
    ],
)
def test_terminal_file_cannot_override_live_account_process(tmp_path, processes):
    journal = fixture(tmp_path)
    (journal / "complete.json").write_text(json.dumps({"outcome": "scored"}))
    with pytest.raises(AssertionError):
        quiescence().check(tmp_path, processes=processes, own_pid=1)


def test_actual_terminal_and_no_account_processes_admitted(tmp_path):
    journal = fixture(tmp_path)
    (journal / "complete.json").write_text(json.dumps({"outcome": "scored"}))
    result = quiescence().check(
        tmp_path, processes="123 python3 predictive-full-fitting-20261020-v2/owner.py", own_pid=1
    )
    assert result["active_known_account_owners_or_transports"] == 0
    assert result["pointwise_terminal_name"] == "complete.json"


def test_both_locks_serialize_four_actual_owned_inodes_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    claim = tmp_path / "claim"
    hf = tmp_path / "publication.lock"
    state = {}

    def new(path, value):
        with path.open("x") as f:
            json.dump(value, f)

    @contextlib.contextmanager
    def publication_lock(path, timeout, poll):
        assert claim.is_dir() and (claim / "owner.json").is_file()
        assert timeout == 30 and poll == 1
        path.mkdir()
        new(path / "owner.json", {"pid": os.getpid(), "job_id": "synthetic"})
        try:
            yield
        finally:
            (path / "owner.json").unlink()
            path.rmdir()

    scope = dict(contextlib=contextlib, Path=Path, os=os, uuid=uuid, json=json)
    scope["subprocess"] = types.SimpleNamespace(
        check_output=lambda *a, **k: "/home/sa305415 0 1048576000 0 - 994625 1000000 0 -\n"
    )
    exec(functions()["shared_file_accounting"], scope)
    definition = functions()["serialized_milestone_locks"].replace(
        "/home/sa305415/rsna-knee/kaggle/coordinator-claim", str(claim)
    )
    exec("@contextlib.contextmanager\n" + definition, scope)
    pub = types.SimpleNamespace(
        atomic_new=new,
        LOCK_RUNNER=tmp_path / "runner.py",
        LOCK_RUNNER_SHA="pinned",
        PUBLICATION_LOCK=hf,
        load_pinned=lambda *a: types.SimpleNamespace(publication_lock=publication_lock),
    )
    with scope["serialized_milestone_locks"](pub) as value:
        state.update(value)
        assert claim.exists() and hf.exists()
    assert state["actually_owned_transient_files_already_charged"] == 4
    assert not claim.exists() and not hf.exists()


@pytest.mark.parametrize(
    "command",
    [
        "python3 initiative/research/predictive-c0-portable-placement-20261020-v1/deployment.py --source-git abc",
        "python3 /repo/initiative/research/predictive-v4-owner-handoff-20261020-v1/launch.py deployment --source-git abc",
        "ssh newton srun --job-name=rsna-predictive-final-qualification",
        "python3 initiative/research/predictive-v4-owner-handoff-20261020-v1/launch.py",
        "python3 initiative/research/predictive-v4-owner-handoff-20261020-v1/launch.py unknown",
        "python3 /repo/initiative/research/predictive-http-owner-handoff-20261020-v1/launch.py deployment --source-git abc",
        "python3 initiative/research/predictive-http-owner-handoff-20261020-v1/launch.py",
        "python3 initiative/research/predictive-http-owner-handoff-20261020-v1/launch.py unknown",
        "python3 /repo/initiative/research/predictive-source-correction-owner-handoff-20261020-v1/launch.py deployment --source-git abc",
        "python3 initiative/research/predictive-source-correction-owner-handoff-20261020-v1/launch.py",
        "python3 initiative/research/predictive-source-correction-owner-handoff-20261020-v1/launch.py unknown",
        "python3 /repo/initiative/research/predictive-http499-owner-handoff-20261020-v1/launch.py deployment --source-git abc",
        "python3 initiative/research/predictive-http499-owner-handoff-20261020-v1/launch.py",
        "python3 initiative/research/predictive-http499-owner-handoff-20261020-v1/launch.py unknown",
        "python3 initiative/research/predictive-evc27-deadline-placement-20261020-v1/owner.py deployment --source-git abc",
        "python3 initiative/research/predictive-evc27-deadline-placement-20261020-v1/owner.py",
    ],
)
def test_current_deployment_owners_and_transports_prevent_publication(tmp_path, command):
    journal = fixture(tmp_path)
    (journal / "complete.json").write_text(json.dumps({"outcome": "scored"}))
    with pytest.raises(AssertionError, match="Other account"):
        quiescence().check(tmp_path, processes="80996 " + command, own_pid=1)


@pytest.mark.parametrize("generation", ("v4", "http", "source-correction", "http499"))
def test_fitting_only_launcher_does_not_claim_account(tmp_path, generation):
    journal = fixture(tmp_path)
    (journal / "complete.json").write_text(json.dumps({"outcome": "scored"}))
    command = (
        f"80995 python3 /repo/initiative/research/predictive-{generation}-owner-handoff-20261020-v1/launch.py "
        "fitting --deployment-handoff /repo/accepted-deployment-stop.json"
    )
    result = quiescence().check(tmp_path, processes=command, own_pid=1)
    assert result["active_known_account_owners_or_transports"] == 0


def test_all27_fitting_mode_does_not_claim_account(tmp_path):
    journal = fixture(tmp_path)
    (journal / "complete.json").write_text(json.dumps({"outcome": "scored"}))
    command = (
        "80995 python3 /repo/initiative/research/"
        "predictive-evc27-deadline-placement-20261020-v1/owner.py fitting --source-git abc"
    )
    assert quiescence().check(tmp_path, processes=command, own_pid=1)[
        "active_known_account_owners_or_transports"
    ] == 0


def test_all_generations_are_physically_charged_once_and_false_directory_credit_refused(tmp_path):
    import stat

    scope = {"Path": Path, "os": os, "stat": stat}
    exec(functions()["paid_unique_npz"], scope)
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    p = old / "original.npz"
    p.write_bytes(b"original")
    os.link(p, new / "retained.npz")
    (new / "new.npz").write_bytes(b"new")
    assert scope["paid_unique_npz"]([old, new]) == len(b"originalnew")
    (new / "not-a-file.npz").mkdir()
    with pytest.raises(AssertionError):
        scope["paid_unique_npz"]([old, new])


def test_current_mirror_capacity_counts_all_roots_and_six_gib_scratch_without_new_funding():
    source = (D / "program.py.txt").read_text()
    nodes = ast.parse(source).body
    assigned = {}
    for node in nodes:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assigned[node.targets[0].id] = node.value
    raw_roots = ast.literal_eval(assigned["paid_raw"].args[0])
    context_roots = ast.literal_eval(assigned["paid_context"].args[0])
    assert "/tmp/rsna-predictive-full-raw-capacity-recovery-20261020-v1/cache" in raw_roots
    assert "/tmp/rsna-predictive-full-context-capacity-recovery-20261020-v1" in context_roots
    assert "/tmp/rsna-predictive-full-raw-http-recovery-20261020-v1/cache" in raw_roots
    assert "/tmp/rsna-predictive-full-context-http-recovery-20261020-v1" in context_roots
    assert "/tmp/rsna-predictive-full-raw-alias-recovery-20261020-v3/cache" in raw_roots
    assert "/tmp/rsna-predictive-full-context-alias-recovery-20261020-v3" in context_roots
    scope = {"paid_raw": 56_893_537_879, "paid_context": 16_043_022_828}
    required = eval(
        compile(ast.Expression(assigned["required_node"]), "resource_budget", "eval"), scope
    )
    expected = (
        (20 + 62 + 6) * 2**30
        + (594 + 4) * 2**20
        + 320 * 2**30
        - scope["paid_raw"]
        + 96 * 2**30
        - scope["paid_context"]
    )
    assert required == expected
    # Both preflight and inside-lock shared checks use the present standing scope.
    assert source.count("20 * 2**30 + 123 * 2**20") == 2
    compile(source, "unlaunched_mirror_worker", "exec")
