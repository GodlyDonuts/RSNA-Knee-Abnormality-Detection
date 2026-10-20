"""Admission/transport boundaries, retained obligations and uncertain mutation state."""

import builtins
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("matched_test_" + name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


C, P = load("coordinate"), load("prepare")
NOW = datetime(2026, 10, 20, 20, tzinfo=timezone.utc)


def inputs():
    files = {name: (HERE / name).read_text() for name in ("worker_builder.py", "driver.py")}
    payload = dict(source_git="a" * 40, files=files,
                   sha256={n: C.B.sha(b.encode()) for n, b in files.items()},
                   control_sha256={"prepare.py": "b" * 64})
    decision = dict(accepted_at_UTC=NOW.isoformat(), finish_by_UTC=(NOW+timedelta(hours=2)).isoformat())
    funding = dict(observed_at_UTC=NOW.isoformat())
    return payload, {"prepare.py": "# frozen admission fixture"}, decision, funding


def test_management_admission_reuses_driver_controls_without_importing_Torch(monkeypatch):
    old_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"torch", "numpy"}, "Management Python must remain stdlib-only"
        return old_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    payload, *_ = inputs()
    builder, policy = P.controls(payload["files"], payload["sha256"])
    assert builder["UPDATES"] == 200 and policy["MAX_SECONDS"] == 5400
    policy["topology"]("NodeName=evc27 Gres=gpu:nvidia_h100_pcie:2(S:0-1)")
    with pytest.raises(ValueError, match="physical two-H100"):
        policy["topology"]("NodeName=evc27 Gres=gpu:nvidia_h100_pcie:8(S:0-1)")
    policy["pilot_completed"]("981620|COMPLETED|0:0|evc27|\n")
    with pytest.raises(ValueError):
        policy["pilot_completed"]("981620|PENDING|0:0||\n")


def test_funding_preserves_all_prior_scopes_and_refuses_stale_or_naive_clock():
    old = 205676085248 + 96*P.GIB + 36*P.GIB + 128*P.MIB
    before_contrast = old + 6*P.GIB + 4*P.MIB + 12*P.GIB + 16*P.MIB + P.MIB
    assert P.MINIMUM_OTHER_BYTES == before_contrast + 7*P.GIB + 4*P.MIB
    with pytest.raises(AssertionError):
        P.validate_other_funding(dict(other_unpaid_node_liabilities_bytes=before_contrast,
                                      prior_C0_and_full_C100_scopes_retained=True))
    P.validate_other_funding(dict(other_unpaid_node_liabilities_bytes=P.MINIMUM_OTHER_BYTES,
                                 prior_C0_and_full_C100_scopes_retained=True))
    _, _, decision, funding = inputs()
    P.decision_time(decision, funding, NOW)
    for value in ((NOW-timedelta(minutes=31)).isoformat(), NOW.replace(tzinfo=None).isoformat(),
                  (NOW+timedelta(seconds=1)).isoformat()):
        with pytest.raises(AssertionError):
            P.decision_time(decision, {"observed_at_UTC": value}, NOW)
    with pytest.raises(AssertionError):
        P.decision_time({**decision, "finish_by_UTC": (NOW+timedelta(minutes=89)).isoformat()}, funding, NOW)


def test_real_exclusive_staging_serialization_matches_driver_reference_hashes(tmp_path):
    payload, *_ = inputs()
    _, policy = P.controls(payload["files"], payload["sha256"])
    root = tmp_path / "new"
    decision, funding = {"actual": True}, {"bytes": 123}
    binding = {"source_git": "a"*40}
    result = {"status": "fixture"}
    P.stage(root, {"worker.py": "# own source\n"}, binding, decision, funding, result, policy["once"])
    assert result["binding_sha256"] == C.B.sha((root / "binding.private.json").read_bytes())
    expected = json.dumps(decision, sort_keys=True, allow_nan=False) + "\n"
    assert (root / "engineering-decision.json").read_text() == expected
    before = (root / "binding.private.json").read_bytes()
    with pytest.raises(AssertionError):
        P.stage(root, {}, {}, {}, {}, {}, policy["once"])
    assert (root / "binding.private.json").read_bytes() == before


def test_script_is_one_ordinary_GPU_no_shared_outputs_or_changed_runtime():
    _, _, decision, _ = inputs()
    body = C.script(decision)
    assert "#SBATCH --time=01:30:00" in body and "#SBATCH --cpus-per-task=2" in body
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in body
    assert "#SBATCH --nodelist=evc27" in body and "--dependency" not in body
    assert str(P.PORTABLE / "runtime/bin/python") in body
    for line in body.splitlines():
        if line.startswith(("#SBATCH --output", "#SBATCH --error", "#SBATCH --chdir")):
            assert str(C.B.ROOT) in line and "/home/" not in line


@pytest.mark.parametrize("failure", [None, "inspect", "admit", "sbatch"])
def test_one_flow_exact_string_envelope_and_no_retry_after_uncertainty(tmp_path, failure):
    payload, own, decision, funding = inputs()
    journal = tmp_path / "journal"
    calls = []
    transport = SimpleNamespace(COMMAND="stdin_receiver", envelope=lambda program, v: json.dumps([program, v]))

    def run(command, **kwargs):
        assert isinstance(kwargs["input"], str) and kwargs["text"] is True
        if "sbatch" in command[-1]:
            phase = "sbatch"
            assert (journal / "GPU-intent.json").is_file()
            output = "999999\n"
        else:
            assert "--gres=none" in command[-1] and "--cpus-per-task=1" in command[-1]
            program, v = json.loads(kwargs["input"])
            assert program == own["prepare.py"]
            phase = v["mode"]
            if phase == "inspect":
                assert not journal.exists()
                output = json.dumps(dict(status="actual_matched_generative_routing_readonly_preflight_passed",
                                         source_git=payload["source_git"], observed_at_UTC=NOW.isoformat()))
            else:
                assert (journal / "intent.json").is_file() and not (journal / "GPU-intent.json").exists()
                output = json.dumps(dict(status="actual_matched_generative_routing_CPU_admitted",
                                         source_git=payload["source_git"], CPU_job="999998", binding_sha256="c"*64))
        calls.append(phase)
        return SimpleNamespace(returncode=1 if phase == failure else 0, stdout=output, stderr="")

    if failure:
        with pytest.raises(AssertionError):
            C.execute(payload, own, decision, funding, journal, ["ssh", "fixture"], transport,
                      run=run, now=lambda: NOW)
    else:
        result = C.execute(payload, own, decision, funding, journal, ["ssh", "fixture"], transport,
                           run=run, now=lambda: NOW)
        assert result["GPU_job"] == "999999" and result["actual_optimizer_updates"] == 600
    assert calls == ["inspect", "admit", "sbatch"][:len(calls)]
    assert len(calls) == {None: 3, "inspect": 1, "admit": 2, "sbatch": 3}[failure]
    if failure != "inspect":
        before = list(calls)
        with pytest.raises(AssertionError, match="journal"):
            C.execute(payload, own, decision, funding, journal, ["ssh", "fixture"], transport,
                      run=run, now=lambda: NOW)
        assert calls == before
        assert (journal / "GPU-intent.json").exists() is (failure != "admit")


def test_current_primary_dispatch_must_match_original_phase_not_old_owner(tmp_path, monkeypatch):
    audit, journal = tmp_path / "audit", tmp_path / "journal"
    audit.mkdir()
    journal.mkdir()
    def put(path, value):
        path.write_text(json.dumps(value))
    original_raw = json.dumps(dict(source_git=C.PRIMARY_SOURCE, science_git=C.FULL_SCIENCE, pid=C.PRIMARY_OWNER_PID)).encode()
    (journal / "owner-intent.json").write_bytes(original_raw)
    original_sha = P.hashlib.sha256(original_raw).hexdigest()
    resumed = dict(source_git=P.CONTINUATION_SOURCE, pid=P.CONTINUATION_PID,
                   original_source_git=C.PRIMARY_SOURCE, original_owner_pid=C.PRIMARY_OWNER_PID,
                   original_startup_sha256={"owner-intent.json": original_sha},
                   original_owner_intent_retained=True, only_startup_marker_renamed=True, automatic_retry=False)
    continuation = tmp_path / "continuation.json"
    put(continuation, resumed)
    monkeypatch.setattr(C.P, "PRIMARY_INTENT_SHA", original_sha)
    monkeypatch.setattr(C.P, "CONTINUATION_INTENT_SHA", P.hashlib.sha256(continuation.read_bytes()).hexdigest())
    receipt = dict(status="actual_predictive_placement_jobs_submitted_once", source_git=C.FULL_SCIENCE,
                   jobs={a:dict(GPU_job=str(123+i),arm=a,node="evc27",source_git=C.FULL_SCIENCE)
                         for i,a in enumerate(("B","C","C0"))})
    put(audit / "actual-owner-dispatch-all27.json", receipt)
    put(journal / "dispatch-all27-result.json", receipt)
    intent = dict(mode="dispatch", node="evc27", placement_source_git=C.PRIMARY_SOURCE, program_sha256="d"*64)
    put(journal / "dispatch-all27-intent.json", intent)
    assert C.original_jobs(audit, journal, continuation) == receipt["jobs"]
    (audit / "actual-owner-dispatch-all27.json").unlink()
    (journal / "dispatch-all27-result.json").unlink()
    put(audit / "actual-known-dispatches-all27.json", receipt)
    put(journal / "known-dispatches-all27.json", receipt)
    reconciled = dict(status="actual_all27_original_dispatch_reconciled_readonly",
                      jobs=receipt["jobs"], new_GPU_jobs=0, submission_completeness_known=False)
    put(audit / "actual-dispatch-reconciled.json", reconciled)
    put(journal / "dispatch-reconciled.json", reconciled)
    assert C.original_jobs(audit, journal, continuation) == receipt["jobs"]
    put(audit / "actual-dispatch-reconciled.json", {**reconciled, "new_GPU_jobs": 1})
    with pytest.raises(AssertionError):
        C.original_jobs(audit, journal, continuation)
    put(audit / "actual-dispatch-reconciled.json", reconciled)
    put(journal / "dispatch-all27-intent.json", {**intent, "node": "evc33"})
    with pytest.raises(AssertionError):
        C.original_jobs(audit, journal, continuation)


def test_exact_actual_new_owner_and_old_owner_rejection():
    expected = P.primary_control()
    assert expected["source_git"] == "2a224ae05fc7c2fc78d4d508b6407344ac523a1c"
    assert expected["original_owner_pid"] == 32377
    assert expected["continuation_pid"] == 33107
    assert P.validate_primary_control(expected) == expected
    assert set(C.original_jobs()) <= {"B", "C", "C0"}
    old = {**expected, "source_git": "291283545c930b730bb257abb10c2a0814b055eb"}
    with pytest.raises(AssertionError, match="current primary"):
        P.validate_primary_control(old)
    original = (C.PRIMARY_JOURNAL / "owner-intent.json").read_bytes()
    continuation = (C.REPO / P.CONTINUATION_REL).read_bytes()
    assert P.validate_primary_ownership(original, continuation) == expected
    with pytest.raises(AssertionError, match="continuation intent"):
        P.validate_primary_ownership(original, continuation + b" ")


def test_primary_retarget_preserves_exact_original_pilot_and_numerical_sources():
    payload, *_ = inputs()
    builder, policy = P.controls(payload["files"], payload["sha256"])
    assert str(P.PILOT) == "/tmp/rsna-predictive-density100-pilot-20261020-v1"
    assert policy["PILOT_BINDING"] == "e11c7c7fe5c1fd7cae87df313fb5d42be7128b7906f8d716a486b0857bbdd37a"
    assert builder["SCIENCE"] == "39e315c65546aa5205518713bc27349ec63e7538"


def test_G100_other_funding_cannot_omit_existing_descriptor_scope():
    old_without_descriptor = (205676085248 + 96 * P.GIB + 36 * P.GIB + 128 * P.MIB
                              + 6 * P.GIB + 4 * P.MIB + 12 * P.GIB + 16 * P.MIB
                              + P.CONTRAST_RESERVED_BYTES)
    funding = {"prior_C0_and_full_C100_scopes_retained": True,
               "other_unpaid_node_liabilities_bytes": old_without_descriptor}
    with pytest.raises(AssertionError):
        P.validate_other_funding(funding)
    P.validate_other_funding({**funding, "other_unpaid_node_liabilities_bytes":
                             old_without_descriptor + P.MIB})
    assert P.DESCRIPTOR_RESERVED_BYTES == P.MIB


def test_G100_retains_primary_total_and_separate_contrast_scope():
    assert P.MINIMUM_OTHER_BYTES + 20*P.GIB + 6*P.GIB + 4*P.MIB == (
        394815078400 + 7*P.GIB + 4*P.MIB)
    funding = {"prior_C0_and_full_C100_scopes_retained": True,
               "other_unpaid_node_liabilities_bytes": P.MINIMUM_OTHER_BYTES - 24*P.GIB - 64*P.MIB}
    with pytest.raises(AssertionError):
        P.validate_other_funding(funding)
