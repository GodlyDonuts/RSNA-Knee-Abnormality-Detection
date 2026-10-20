"""Actual frozen policy with synthetic timings; no remote or GPU work."""

import copy
import importlib.util
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIN = "8fd43b9818c85cbbff9e608ae5ba7d3ae2c6a26f"


def local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def frozen(name, path):
    source = subprocess.check_output(["git", "show", PIN + ":" + path], cwd=ROOT)
    scope = {"__name__": name, "__file__": str(ROOT / path)}
    exec(compile(source, path, "exec"), scope)
    return SimpleNamespace(**scope)


new = local("runtime_continuation", HERE / "continuation.py")
owner = frozen("runtime_owner", "initiative/research/predictive-final-deployment-owner-20261020-v1/owner.py")
policy = frozen("runtime_policy", "initiative/research/predictive-final-deployment-20261020-v1/policy.py")
contract = frozen("runtime_serving", "initiative/research/predictive-final-serving-20261020-v1/contract.py")


def inputs():
    candidates = [{"arm": arm, "model": {"sha256": str(i + 1) * 64}}
                  for i, arm in enumerate(new.ARMS)]
    models = []
    for chosen, seconds in zip(candidates, (20, 50, 30)):
        measures = [
            dict(device=d, study_index=i, decode_seconds=seconds - 2,
                 render_seconds=1, FP16_seconds=1, FP32_seconds=2,
                 maximum_probability_gap=0.0001, windows=48, regions=4000)
            for d, i in ((0, 0), (0, 2), (1, 1), (1, 0))
        ]
        models.append(dict(
            arm=chosen["arm"], candidate_sha256=contract.digest(contract.canonical(chosen)),
            trained_export_sha256=chosen["model"]["sha256"],
            precision_accepted=True, memory_accepted=True, measurements=measures,
            FP16_FP32_tolerance=0.005, cross_device_FP16_tolerance=0.0001,
            cross_device_FP16_gap=0, device_peak_bytes=[2**30, 2**30], host_peak_bytes=4 * 2**30,
        ))
    qualification = dict(status=new.QUALIFIED, public_studies=3,
        GPU_names=["Tesla T4", "Tesla T4"], pinned_Docker=contract.DOCKER,
        submission_csv_absent=True, hidden_runtime_acceptance=False, models=models)
    decision = json.loads((ROOT / "initiative/research/predictive-http-owner-handoff-20261020-v1/deployment-decision.accepted.json").read_text())
    return decision, qualification, candidates


def evaluate(d, q, c):
    return new.evaluate(d, q, c, contract.validate_qualification, owner.runtime_plan, policy)


def test_slow_middle_arm_does_not_block_later_eligible_arm():
    d, q, c = inputs()
    result = evaluate(d, q, c)
    assert result["eligible"] == ["B", "C0"] and result["rejected"] == ["C"]
    assert result["roles"]["C"]["reason"] == new.RUNTIME_REJECTION
    assert result["roles"]["C"]["selected_estimate_seconds"] == 33100
    assert [result["roles"][a]["measurement"] for a in new.ARMS] == q["models"]
    assert result["roles"]["B"]["runtime_plan"]["public_watchdog_seconds"] == 600
    assert result["roles"]["B"]["runtime_plan"]["other_reserved_VM_seconds"] == 360


@pytest.mark.parametrize("kind", ("precision", "partial", "nonfinite", "source"))
def test_invalid_combined_qualification_always_stops_globally(kind):
    d, q, c = inputs()
    if kind == "precision":
        q["models"][2]["precision_accepted"] = False
    elif kind == "partial":
        q["models"].pop()
    elif kind == "source":
        q["models"][0]["candidate_sha256"] = "0" * 64
    else:
        q["models"][0]["measurements"][0]["FP16_seconds"] = float("nan")
    with pytest.raises(ValueError):
        evaluate(d, q, c)


def test_unrelated_policy_failure_is_never_a_skippable_runtime_result():
    d, q, c = inputs()
    def broken(*args):
        raise ValueError("wrong source or policy")
    with pytest.raises(ValueError, match="wrong source"):
        new.evaluate(d, q, c, contract.validate_qualification, broken, policy)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def boundary(tmp_path):
    d, q, candidates = inputs()
    journal = tmp_path / "original"
    journal.mkdir()
    body = (json.dumps(d, indent=3) + "\n").encode()
    import hashlib
    write(journal / "owner-intent.json", dict(pid=91866, placement_source_git="a" * 40,
          canonical_account_control_source_git=new.CANONICAL,
          decision_sha256=hashlib.sha256(body).hexdigest()))
    write(journal / "stopped.json", dict(error_type="ValueError", automatic_retry=False,
          preserve_original_phases_and_qualified_baseline=True))
    write(journal / "actual-trained-qualification.json", q)
    write(journal / "prepare-qualification-result.json",
          dict(status="actual_three_predictive_final_exports_prepared_CPU_only",
               bundle_identity="b" * 64, models=candidates))
    push = dict(state="predictive_final_pushed", slug="owner/exact-version", version=1, kernel_id=123)
    write(journal / "push-qualification-result.json", push)
    complete = dict(status="COMPLETE", receipt=q, hidden_score_claim=False,
                    **{k: push[k] for k in ("slug", "version", "kernel_id")})
    write(journal / "result-00008.json", complete)
    remote = dict(status="actual_predictive_runtime_continuation_boundary_read_only",
                  observed_at=time.time(), deployment_root=d["root"], canonical_claim_present=False,
                  qualification_sha256=new.digest(q), prepared_bundle_identity="b" * 64,
                  compiled_arm_directories=[], arm_ledger_rows=[], arm_kernel_pushes=[], arm_submissions=[])
    args = dict(source_git="a" * 40, pid=91866, decision=d, decision_bytes=body,
                process_command=lambda _: "", remote=remote,
                validate_qualification=contract.validate_qualification,
                runtime_plan=owner.runtime_plan, policy=policy)
    return journal, args


def test_original_receipt_bytes_and_complete_identity_admit_only_once_boundary(tmp_path):
    journal, args = boundary(tmp_path)
    accepted = new.admit_failure(journal, **args)
    assert accepted["evaluation"]["eligible"] == ["B", "C0"]
    assert accepted["new_model_dataset_or_qualification"] is False
    write(journal / "push-C-intent.json", {"uncertain": True})
    with pytest.raises(ValueError, match="public-arm"):
        new.admit_failure(journal, **args)


@pytest.mark.parametrize("changed", ("active", "remote_phase", "claim", "stale", "decision"))
def test_boundary_rejects_activity_uncertainty_or_stale_admission(tmp_path, changed):
    journal, args = boundary(tmp_path)
    if changed == "active":
        args["process_command"] = lambda _: "original live owner"
    elif changed == "remote_phase":
        args["remote"]["arm_ledger_rows"] = ["B"]
    elif changed == "claim":
        args["remote"]["canonical_claim_present"] = True
    elif changed == "stale":
        args["remote"]["observed_at"] -= 181
    else:
        args["decision_bytes"] += b" "
    with pytest.raises(ValueError):
        new.admit_failure(journal, **args)


def test_executable_tail_only_calls_eligible_original_phases_and_never_retries(tmp_path):
    journal, args = boundary(tmp_path)
    admitted = new.admit_failure(journal, **args)
    successor = tmp_path / "successor"
    successor.mkdir()
    calls = []
    def action(phase, arm):
        calls.append((phase, arm))
        if phase == "status":
            return {"status": "COMPLETE"}
        if phase == "outcome":
            return {"outcome": "scored", "publicScore": 0.5}
        return {"state": "submitted" if phase == "submit" else "original_phase_complete"}
    instance = SimpleNamespace(pin=new.CANONICAL, journal=successor, decision=args["decision"],
        action=action, sleep=lambda *args: pytest.fail("Unexpected wait"))
    result = new.continue_qualified(instance, admitted, once=owner.once, finish_deadline=owner.FINISH_END)
    assert [a for p, a in calls if p == "push"] == ["B", "C0"]
    assert not any(a == "C" for _, a in calls)
    assert not any(p in {"stage", "prepare", "dataset"} for p, _ in calls)
    assert instance.remaining_public == 1  # C's allowance remains conservatively reserved.
    assert set(result["entries"]) == {"B", "C0"}
    before = copy.deepcopy(calls)
    with pytest.raises(ValueError, match="Fresh continuation"):
        new.continue_qualified(instance, admitted, once=owner.once, finish_deadline=owner.FINISH_END)
    assert calls == before


def test_readonly_boundary_detects_compiled_or_canonical_model_actions(tmp_path):
    _, q, candidates = inputs()
    root = tmp_path / "original-root"
    root.mkdir()
    ledger = tmp_path / "account-ledger.sqlite3"
    prepared = dict(status="actual_three_predictive_final_exports_prepared_CPU_only",
                    bundle_identity="b" * 64, models=candidates)
    write(root / "prepared.json", prepared)
    write(root / "qualification-accepted.json", q)
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE entries (sha TEXT PRIMARY KEY, record TEXT NOT NULL)")
    before = ledger.read_bytes()
    assert new.read_boundary(root, ledger)["arm_ledger_rows"] == []
    assert ledger.read_bytes() == before
    with sqlite3.connect(ledger) as db:
        db.execute("INSERT INTO entries VALUES (?,?)",
                   (candidates[1]["model"]["sha256"], json.dumps({"submission_id": 123})))
    (root / "compiled-B").mkdir()
    observed = new.read_boundary(root, ledger)
    assert observed["arm_submissions"] == ["C"]
    assert observed["compiled_arm_directories"] == ["B"]
    assert observed["external_API_calls"] == observed["remote_writes"] == 0


def test_original_portable_adapter_changes_only_journal_identity_and_execution_tail():
    pin = "926b657770564c79b23509592e2d6a95761abd70"
    path = "initiative/research/predictive-c0-portable-placement-20261020-v1/deployment.py"
    source = subprocess.check_output(["git", "show", pin + ":" + path], cwd=ROOT).decode()
    derived = new.derive_deployment_source(source)
    assert "continue_runtime(PortableDeployment(), runtime_admission" in derived
    assert 'journal = ROOT / "artifacts" / ' + repr(new.JOURNAL_NAME) in derived
    assert source[:source.index("def run(")] == derived[:derived.index("def run(")]
    assert source[source.index("    class PortableDeployment(base):"):
                  source.index("    p.require(time.time() < original.LAUNCH_END")] == (
        derived[derived.index("    class PortableDeployment(base):"):
                derived.index("    p.require(time.time() < original.LAUNCH_END")]
    )
    with pytest.raises(ValueError):
        new.derive_deployment_source(derived)
