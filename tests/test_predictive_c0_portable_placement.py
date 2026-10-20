"""Operational placement checks; no MRI, remote actions or numerical training."""

import copy
import importlib.util
import io
import json
import os
import subprocess
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "initiative/research/predictive-c0-portable-placement-20261020-v1"
spec = importlib.util.spec_from_file_location("portable_policy_tests", CONTROL / "placement.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def stop():
    return {
        "status": "actual_idle_68792_stopped_before_full_fitting_intent",
        "old_owner_pid": 68792,
        "old_owner_source_git": p.SCIENCE,
        "old_owner_children": [],
        "old_local_phase_intents": [],
        "old_remote_admission_or_dispatch_exists": False,
        "predictive_full_GPU_jobs": [],
        "owner_process_absent_after_stop": True,
        "remote_jobs_cancelled": [],
        "ownership": {"B": "evc33", "C": "evc33", "C0": "evc27"},
        "original_C0_shared_log_envelope_reassigned": True,
        "remote_read_only_snapshot_sha256": "f" * 64,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("old_owner_children", [123]),
        ("old_local_phase_intents", ["admit-intent"]),
        ("old_remote_admission_or_dispatch_exists", True),
        ("predictive_full_GPU_jobs", ["999"]),
        ("owner_process_absent_after_stop", False),
        ("original_C0_shared_log_envelope_reassigned", False),
        ("ownership", {"B": "evc33", "C": "evc33", "C0": "evc48"}),
    ],
)
def test_never_override_an_active_or_advanced_owner(key, value):
    value_receipt = stop()
    p.handoff(value_receipt)
    value_receipt[key] = value
    with pytest.raises(ValueError):
        p.handoff(value_receipt)


def test_destination_scope_accounts_every_byte_without_deletion_credit():
    required = p.OTHER27 + 20 * p.GIB + p.CONTEXT_CAP + p.CONTROL_CAP + p.FIT_CAP
    with pytest.raises(ValueError):
        p.capacity(required)
    assert p.capacity(required + 1)["remaining_bytes"] == 1
    assert p.capacity(835340234752)["remaining_bytes"] == 492158087168
    paid = 12345
    assert p.capacity(required, contexts_paid=paid)["remaining_bytes"] == paid
    with pytest.raises(ValueError):
        p.capacity(required, contexts_paid=p.CONTEXT_CAP + 1)


def synthetic_rows(tmp_path, count=3):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for i in range(count):
        path = source / f"{i}.npz"
        data = (bytes([i]) + b"private_context") * (i + 2)
        path.write_bytes(data)
        path.chmod(0o600)
        rows.append(
            {
                "uid": str(i),
                "path": str(path),
                "proof": {
                    "sha256": p.sha(data),
                    "bytes": len(data),
                    "stat": p.signature(path),
                    "version": "unchanged_context_version",
                },
                "provenance": {
                    "module_sha256": {"cases.py": "old" if i == 0 else "corrected"},
                    "original_raw_cache_sha256": "a" * 64,
                },
            }
        )
    return rows


def test_context_stream_is_missing_only_and_keeps_every_original_provenance(tmp_path):
    rows = synthetic_rows(tmp_path)
    original = copy.deepcopy(rows)
    dest = tmp_path / "destination"
    (dest / "contexts").mkdir(parents=True)
    os.chmod(dest / "contexts", 0o700)
    stream = io.BytesIO()
    p.send_contexts(stream, rows, [0, 1])
    stream.seek(0)
    # Receive helper normally closes all rows; preexisting verified last member
    # demonstrates it is not resent by the sender.
    last = dest / "contexts" / p.context_name(2)
    last.write_bytes(Path(rows[2]["path"]).read_bytes())
    last.chmod(0o600)
    p.once(
        last.with_suffix(".verified.json"),
        {
            "bytes": rows[2]["proof"]["bytes"],
            "sha256": rows[2]["proof"]["sha256"],
            "stat": p.signature(last),
        },
    )
    prior = p.signature(last)
    old_umask = os.umask(0o077)
    try:
        missing = p.missing_contexts(dest, rows)
        assert missing == [0, 1]
        rebound, transferred = p.receive_contexts(stream, dest, rows, missing)
    finally:
        os.umask(old_umask)
    assert transferred == sum(r["proof"]["bytes"] for r in rows[:2])
    assert p.signature(last) == prior
    assert rows == original
    assert p.missing_contexts(dest, rows) == []
    for actual, expected in zip(rebound, rows):
        assert actual["provenance"] == expected["provenance"]
        assert actual["proof"]["sha256"] == expected["proof"]["sha256"]
        assert Path(actual["path"]).read_bytes() == Path(expected["path"]).read_bytes()


@pytest.mark.parametrize("failure", ["truncated", "corrupt", "extra", "source_changed"])
def test_interrupted_or_altered_transfer_cannot_be_admitted(tmp_path, failure):
    rows = synthetic_rows(tmp_path, count=1)
    dest = tmp_path / "destination"
    (dest / "contexts").mkdir(parents=True)
    stream = io.BytesIO()
    if failure == "source_changed":
        Path(rows[0]["path"]).write_bytes(b"changed")
        with pytest.raises(ValueError):
            p.send_contexts(stream, rows, [0])
        return
    p.send_contexts(stream, rows, [0])
    data = stream.getvalue()
    data = (
        data[:-3]
        if failure == "truncated"
        else (b"x" + data[1:] if failure == "corrupt" else data + b"x")
    )
    previous = os.umask(0o077)
    try:
        with pytest.raises(ValueError):
            p.receive_contexts(io.BytesIO(data), dest, rows, [0])
    finally:
        os.umask(previous)


def test_prior_intent_or_partial_is_never_retried(tmp_path):
    receipt = tmp_path / "intent.json"
    p.once(receipt, {"one": True})
    with pytest.raises(FileExistsError):
        p.once(receipt, {"two": True})
    assert json.loads(receipt.read_text()) == {"one": True}
    rows = synthetic_rows(tmp_path, 1)
    (tmp_path / "contexts").mkdir()
    (tmp_path / "contexts" / p.context_name(0)).with_suffix(".partial").write_bytes(b"prefix")
    with pytest.raises(ValueError, match="partial"):
        p.missing_contexts(tmp_path, rows)


def test_exact_original_owner_is_filtered_without_touching_science():
    source = subprocess.check_output(
        ["git", "show", p.SCIENCE + ":" + p.BASE + "/owner_remote.py.txt"], cwd=ROOT
    ).decode()
    for node, arms in (("evc33", ("B", "C")), ("evc27", ("C0",))):
        output = p.derived_remote(source, node, 123 * p.MIB, 376)
        ns = {"__name__": "local_test"}
        exec(compile(output, "derived_control", "exec"), ns)
        assert ns["ARMS"] == arms
        assert ns["PYTHON"] == (p.PY33 if node == "evc33" else p.PY27)
        assert ns["ROOT"] == (p.ROOT33 if node == "evc33" else p.ROOT27)
        assert "base_other(v) + len(ARMS) * 12 * GIB" in output
    prepared = json.loads((ROOT / p.BASE / "actual-cpu-prepared.json").read_text())
    for name in ("worker.py", "publish_epoch.py"):
        original = subprocess.check_output(
            ["git", "show", p.SCIENCE + ":" + p.BASE + "/" + name + ".txt"], cwd=ROOT
        )
        assert p.sha(original) == prepared["code_sha256"][name]


def test_pending_decision_does_not_authorize_launch():
    value = json.loads((CONTROL / "decision.pending.json").read_text())
    assert value["status"] == "prepared_only_not_accepted"
    assert value["placement_source_git"] is None
    assert value["shared_future_bytes"] is None
    assert value["shared_C0_log_envelope_reassigned_not_added"] is False


def test_real_frozen_dispatch_emits_only_one_C0_on27_and_refuses_second_intent(
    tmp_path, monkeypatch
):
    """Execute the actual old control function with local filesystem/fake scheduler."""
    original = subprocess.check_output(
        ["git", "show", p.SCIENCE + ":" + p.BASE + "/owner_remote.py.txt"], cwd=ROOT
    ).decode()
    ns = {"__name__": "dispatch_test"}
    exec(
        compile(
            p.derived_remote(original, "evc27", 123 * p.MIB, 376), "derived_exact_dispatch", "exec"
        ),
        ns,
    )
    root = tmp_path / "controls"
    admission = root / "full-admission"
    admission.mkdir(parents=True)
    binding = admission / "binding-C0.private.json"
    binding.write_text("{}")
    actual = {"arms": ["C0"], "binding_sha256": {"C0": p.file_sha(binding)}}
    (admission / "actual.json").write_text(json.dumps(actual))
    ns.update(
        ROOT=root,
        SCIENCE=p.SCIENCE,
        preparation=lambda _: None,
        check_node=lambda: "a" * 64,
        base_other=lambda _: 0,
        run_root=lambda _: tmp_path / "run-C0",
        LOG_ROOT=tmp_path / "logs",
    )
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == "sbatch"
        return types.SimpleNamespace(returncode=0, stdout="999000\n", stderr="")

    monkeypatch.setattr(
        ns["runpy"],
        "run_path",
        lambda *args, **kwargs: {
            "check_space": lambda *args, **kwargs: {"synthetic_capacity": True}
        },
    )
    monkeypatch.setattr(ns["subprocess"], "run", fake_run)
    monkeypatch.setenv("SLURM_JOB_ID", "900")
    value = {"prepared": {}, "admitted": actual, "owner_git": "f" * 40}
    result = ns["dispatch"](value)
    assert list(result["jobs"]) == ["C0"]
    assert result["status"] == "actual_predictive_placement_jobs_submitted_once"
    assert result["submitted_arm_count"] == 1
    assert result["ordinary_node_physical_GPU_limit"] == 2
    assert result["placement_total_GPU_limit"] == 3
    assert result["jobs"]["C0"]["node"] == "evc27"
    script = (root / "dispatch/C0.sh").read_text()
    assert "#SBATCH --nodelist=evc27" in script
    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in script
    assert p.PY27 in script
    assert p.file_sha(binding) in script
    with pytest.raises(AssertionError, match="Existing dispatch"):
        ns["dispatch"](value)
    assert len(calls) == 1


def test_full3887_manifest_gate_rejects_partial_and_reordered_roster(monkeypatch):
    rows = [
        {
            "uid": f"{i:04d}",
            "path": f"/tmp/source/{i}.npz",
            "proof": {"bytes": 1, "stat": [i, 1, 2, 3], "sha256": "a" * 64},
        }
        for i in range(3887)
    ]
    monkeypatch.setattr(p, "ROSTER", p.sha(p.canonical([r["uid"] for r in rows])))
    terminal = {
        "status": "actual_missing_only3887_exact_FP32_context_alias_recovery_accepted",
        "studies": 3887,
    }
    assert p.check_rows(rows, terminal) == 3887
    with pytest.raises(ValueError, match="complete"):
        p.check_rows(rows[:-1], terminal)
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="order"):
        p.check_rows(rows, terminal)


@pytest.fixture
def controls(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "placement", p)
    loaded = {}
    for name in ("owner", "deployment", "remote"):
        spec = importlib.util.spec_from_file_location(name, CONTROL / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return types.SimpleNamespace(**loaded)


def dispatch_job(arm, root, identity="999001"):
    name = "rsna-predictive-full2-" + arm
    return {
        "GPU_job": identity,
        "arm": arm,
        "source_git": p.SCIENCE,
        "node": "evc27" if arm == "C0" else "evc33",
        "job_name": name,
        "stdout": str(root / (name + "-" + identity + ".out")),
        "stderr": str(root / (name + "-" + identity + ".err")),
    }


def test_shared_log_scope_retires_only_existing_exact_private_job_paths(tmp_path):
    jobs = {
        arm: dispatch_job(arm, tmp_path, str(999001 + i)) for i, arm in enumerate(("B", "C", "C0"))
    }
    assert p.paid_shared_logs(jobs, tmp_path)["files"] == 0
    for arm in ("B", "C"):
        for field in ("stdout", "stderr"):
            path = Path(jobs[arm][field])
            path.write_bytes(b"log")
            path.chmod(0o600)
    actual = p.paid_shared_logs(jobs, tmp_path)
    assert actual["files"] == 4
    # Current free already charges four files. Missing C0 logs remain reserved.
    assert 5375 - 376 == 4999
    assert 5375 - (376 - actual["files"]) == 5003
    assert 5379 - 376 == 5003
    changed = copy.deepcopy(jobs)
    changed["C"]["stdout"] = jobs["B"]["stdout"]
    with pytest.raises(ValueError, match="path"):
        p.paid_shared_logs(changed, tmp_path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        p.paid_shared_logs(jobs, tmp_path)


def test_shared_file_funding_subtracts_logs_once_and_keeps_own8(controls):
    original = {
        "original_runs": {
            arm: {"run_root": arm, "license_path": "original"} for arm in ("B", "C", "C0")
        },
        "funding": {
            "other_shared_unpaid_files": 368,
            "new_shared_files": 8,
            "other_node_unpaid_bytes": 123456,
        },
    }
    paid = {"files": 6, "paths": [str(i) for i in range(6)], "originals_deleted": False}
    value = controls.deployment.rebind_config(original, paid)
    assert (
        value["funding"]["other_shared_unpaid_files"] + value["funding"]["new_shared_files"] == 370
    )
    assert value["funding"]["other_node_unpaid_bytes"] == 123456
    assert value["original_runs"]["B"] == original["original_runs"]["B"]
    assert value["original_runs"]["C0"]["run_root"] == str(p.RETURN33)
    assert original["funding"]["other_shared_unpaid_files"] == 368
    with pytest.raises(ValueError, match="one paid-log"):
        controls.deployment.rebind_config(value, paid)


def test_actual_idle_deployment_journal_allows_launch_intent_only(tmp_path, monkeypatch, controls):
    d = controls.deployment
    journal = tmp_path / "artifacts" / d.PARENT_DIR.name
    journal.mkdir(parents=True)
    for name in (
        "launch-intent.json",
        "owner-intent.json",
        "launcher.json",
        "stdout",
        "stderr",
        "frozen_phase_launcher.py",
        "frozen-phase-source.json",
    ):
        (journal / name).write_text("{}")
    (journal / "launch-intent.json").write_text(
        json.dumps(
            {
                "source_git": d.PARENT,
                "started_at": "2026-10-20T13:20:35.920779+00:00",
                "automatic_retry": False,
            }
        )
    )
    receipt = tmp_path / "stopped.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "actual_idle_predictive_deployment_owner_stopped_before_phase",
                "pid": 77985,
                "source_git": d.PARENT,
                "phase_intents": [],
                "children": [],
                "owner_process_absent_after_stop": True,
            }
        )
    )
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(d.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))
    assert d.stopped_before_phase(receipt)["pid"] == 77985
    (journal / "stage-qualification-intent.json").write_text("{}")
    with pytest.raises(ValueError, match="advanced"):
        d.stopped_before_phase(receipt)


def test_progress_only_retries_observation_loss_but_never_mutation(tmp_path, monkeypatch, controls):
    o = controls.owner
    args = types.SimpleNamespace(
        deadline_utc="2026-10-22T09:00:00+00:00", source_git="f" * 40, socket="unused"
    )
    obj = o.Owner(args, None, None, "", {}, tmp_path)
    monkeypatch.setattr(o.time, "time", lambda: 0)
    monkeypatch.setattr(o.time, "monotonic", lambda: 0)
    monkeypatch.setattr(o.time, "sleep", lambda _: None)
    calls = []

    def invoke(*args):
        calls.append(args)
        if len(calls) == 1:
            raise ConnectionError("read-only lost")
        return {"arms": {}}

    monkeypatch.setattr(obj, "_invoke_once", invoke)
    assert obj.invoke("evc33", "progress") == {"arms": {}}
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(ConnectionError):
        obj.invoke("evc33", "dispatch", "only-intent")
    assert len(calls) == 1
    calls.clear()

    def absent(*args):
        calls.append(args)
        raise subprocess.TimeoutExpired("ssh", 90)

    monkeypatch.setattr(obj, "_invoke_once", absent)
    times = iter((0, 1801))
    monkeypatch.setattr(o.time, "monotonic", lambda: next(times))
    with pytest.raises(ValueError, match="absent30"):
        obj.invoke("evc27", "progress")
    assert len(calls) == 1


def final_fixture(directory, original):
    for index, arm in enumerate(("B", "C", "C0")):
        final = {
            "status": original.FINAL,
            "arm": arm,
            "source_git": p.SCIENCE,
            "epochs": 6,
            "updates": 2916,
            "study_exposures": 23322,
            "all_original_fitting_labels_masks_unchanged": True,
            "protected_or_gold_cases_read": False,
            "actual_Slurm_terminal": {
                "state": "COMPLETED",
                "exit_code": "0:0",
                "queue_empty": True,
                "node": "evc27" if arm == "C0" else "evc33",
            },
            "GPU_job": str(999001 + index),
            "binding_sha256": "b" * 64,
            "initial_shared_tensor_sha256": p.INITIAL,
        }
        files = {
            name: {"bytes": 100 + index, "sha256": str(index) * 64}
            for name in (
                "model.pt",
                "provenance.json",
                "own-source.json",
                "source-permission.txt",
                "manifest.json",
            )
        }
        pub = {
            "status": original.VERIFIED,
            "objects": 5,
            "arm": arm,
            "epoch": 6,
            "source_git": p.SCIENCE,
            "GPU_job": final["GPU_job"],
            "canonical_shared_publication_lock_used": True,
            "revision": str(index) * 40,
            "prefix": f"checkpoints/predictive-full3887-{p.SCIENCE[:12]}/{arm}/epoch-006",
            "files": files,
        }
        p.once(directory / ("actual-fixed-final-" + arm + ".json"), final)
        p.once(directory / ("actual-owner-publish-" + arm + "-006.json"), pub)
    returned = {
        "status": "actual_C0_final_export_returned_to33_all_bytes_SHA_verified",
        "source_node": "evc27",
        "destination_node": "evc33",
        "original_training_GPU_job": "999003",
        "science_git": p.SCIENCE,
        "binding_sha256": "b" * 64,
        "mirror_run_root": str(p.RETURN33),
        "original_six_private_revisions": [str(i) * 40 for i in range(5)] + ["2" * 40],
        "originals_deleted": False,
        "new_publication": False,
        "objects": {"publication/epoch-006/" + name: expected for name, expected in files.items()},
    }
    p.once(directory / "actual-owner-return-C0.json", returned)


def test_deployment_requires_actual_node27_original_proofs_and_exact_return(tmp_path, controls):
    d = controls.deployment
    old = d.original_owner()
    assert d.portable_finals(old, tmp_path) is None
    final_fixture(tmp_path, old)
    value = d.portable_finals(old, tmp_path)
    assert value["finals"]["C0"]["actual_Slurm_terminal"]["node"] == "evc27"
    assert value["C0_return_receipt_sha256"] == p.file_sha(tmp_path / "actual-owner-return-C0.json")
    receipt = tmp_path / "actual-owner-return-C0.json"
    body = json.loads(receipt.read_text())
    body["objects"]["publication/epoch-006/model.pt"]["sha256"] = "f" * 64
    receipt.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="Returned final object"):
        d.portable_finals(old, tmp_path)
    receipt.unlink()
    assert d.portable_finals(old, tmp_path) is None
    final = tmp_path / "actual-fixed-final-C0.json"
    body = json.loads(final.read_text())
    body["actual_Slurm_terminal"]["node"] = "evc33"
    final.write_text(json.dumps(body))
    with pytest.raises(AssertionError):
        d.portable_finals(old, tmp_path)


def test_generated_observer_reads_only_actual33_mirror_and_original_paid_logs(
    tmp_path, monkeypatch, controls
):
    d = controls.deployment
    oldsource = d.fitting.git_file(
        d.PARENT, (d.PARENT_DIR / "observe.py.txt").relative_to(ROOT).as_posix()
    ).decode()
    returned = tmp_path / "return"
    returned.mkdir()
    proof = {"source_node": "evc27", "destination_node": "evc33"}
    p.once(returned / "return-verified.json", proof)
    (returned / "small-model.pt").write_bytes(b"model")
    monkeypatch.setattr(p, "RETURN33", returned)
    jobs = {
        arm: dispatch_job(arm, tmp_path, str(999001 + i)) for i, arm in enumerate(("B", "C", "C0"))
    }
    for job in jobs.values():
        for field in ("stdout", "stderr"):
            path = Path(job[field])
            path.write_bytes(b"log")
            path.chmod(0o600)
    module_source = (
        (CONTROL / "placement.py")
        .read_text()
        .replace(
            'root=Path("/home/sa305415/rsna-knee/kaggle")', "root=Path(" + repr(str(tmp_path)) + ")"
        )
    )
    source = d.observer_source(
        oldsource, p.file_sha(returned / "return-verified.json"), jobs, module_source
    )
    ns = {"__name__": "observer_test"}
    exec(compile(source, "actual_derived_observer", "exec"), ns)
    runs = {}
    for arm in ("B", "C"):
        path = tmp_path / arm
        path.mkdir()
        (path / "state.pt").write_bytes(b"x" * 1000)
        runs[arm] = path

    def path_map(raw):
        for arm, path in runs.items():
            if str(raw) == f"/tmp/rsna-predictive-full-{arm}-20261020-v2":
                return path
        return Path(raw)

    ns["Path"] = path_map
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(ns["os"], "uname", lambda: types.SimpleNamespace(nodename="evc33"))
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    class Session:
        headers = {}
        cookies = types.SimpleNamespace(clear=lambda: None)

        def post(self, url, **kwargs):
            if url.endswith("IntrospectToken"):
                body = {"active": True, "username": "godlydonuts"}
            elif url.endswith("GetAcceleratorQuotaStatistics"):
                body = {"gpuQuota": {"totalTimeAllowed": "4000s"}}
            else:
                body = {"numToday": 0, "numAllowedNow": 5}
            return types.SimpleNamespace(status_code=200, json=lambda: body)

    monkeypatch.setattr(ns["requests"], "Session", Session)
    actual = ns["main"](
        {
            "node": "evc33",
            "root": "/tmp/rsna-predictive-final-deployment-20261020-v1-test",
            "token": "fake",
        }
    )
    assert actual["paid_full_run_bytes"]["B"] == 1000
    assert actual["paid_full_run_bytes"]["C0"] == sum(q.stat().st_size for q in returned.iterdir())
    assert actual["paid_original_GPU_logs"]["files"] == 6
    assert actual["mutations"] == 0


def test_deployment_pending_plan_is_disabled_and_retains3360_seconds(controls):
    value = json.loads((CONTROL / "deployment-decision.pending.json").read_text())
    with pytest.raises(AssertionError):
        controls.deployment.original_owner().validate_decision(value)
    assert (
        value["runtime"]["qualification_cap_seconds"]
        + 3
        * (
            value["runtime"]["public_watchdog_seconds"]
            + value["runtime"]["public_VM_startup_allowance_seconds"]
        )
        == 3360
    )
    assert value["portable_placement_source_git"] is None


def test_sealed_portable_funding_uses_actual_net_file_liabilities(controls, monkeypatch):
    import ast

    # Execute the actual sealed funding expression, which the unchanged worker
    # later uses. Four charged B/C logs must not be reserved a second time.
    module = ast.parse((CONTROL / "remote.py").read_text())
    function = next(
        n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "portable_admit"
    )
    expression = next(
        n.value
        for n in function.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "fund" for t in n.targets)
    )
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    namespace = {
        "p": p,
        "os": os,
        "run_root": "/tmp/synthetic",
        "fresh": {"free_bytes": 400 * p.GIB},
        "capsule": {"context_bytes": 80 * p.GIB},
        "controls_paid": 10 * p.MIB,
        "shared": {"free_bytes": 30 * p.GIB, "free_files": 5375, "unpaid_shared_files": 372},
        "v": {
            "shared_bytes": 123 * p.MIB,
            "shared_files": 376,
            "placement_source_git": "f" * 40,
            "handoff_sha256": "a" * 64,
        },
    }
    actual = eval(compile(ast.Expression(expression), "actual_sealed_funding", "eval"), namespace)
    assert actual["shared_files_after_all_liabilities"] == 5003
    assert actual["no_deletion_credit"] is True
