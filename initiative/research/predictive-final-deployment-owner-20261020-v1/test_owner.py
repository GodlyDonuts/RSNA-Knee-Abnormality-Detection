"""Local state-machine and original-policy tests; no credentials/network/GPU."""

import importlib.util
import json
import subprocess
import time
from pathlib import Path

import pytest

D = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


owner = load(D / "owner.py", "predictive_owner_test")
policy = load(owner.DEPLOY / "policy.py", "original_deployment_policy_test")


def decision():
    value = json.loads((D / "decision.json").read_text())
    value["status"] = "accepted_finite_predictive_final_deployment_owner"
    value["runtime_decision_accepted"] = True
    return value


def qualification(seconds=30):
    return {
        "status": "actual_trained_predictive_public_T4_qualification_accepted",
        "models": [
            {"arm": arm, "measurements": [
                {"study_index": i, "decode_seconds": seconds - 2,
                 "render_seconds": 1, "FP16_seconds": 1}
                for i in (0, 1, 2, 0)
            ]} for arm in owner.ARMS
        ],
    }


def make_finals(directory):
    for index, arm in enumerate(owner.ARMS):
        final = {
            "status": owner.FINAL, "arm": arm, "source_git": owner.SCIENCE,
            "epochs": 6, "updates": 2916, "study_exposures": 23322,
            "GPU_job": str(100 + index),
            "all_original_fitting_labels_masks_unchanged": True,
            "protected_or_gold_cases_read": False,
            "initial_shared_tensor_sha256": "a" * 64,
            "actual_Slurm_terminal": {
                "state": "COMPLETED", "exit_code": "0:0", "queue_empty": True, "node": "evc33"
            },
        }
        publication = {
            "status": owner.VERIFIED, "objects": 5, "arm": arm, "epoch": 6,
            "source_git": owner.SCIENCE, "GPU_job": str(100 + index),
            "canonical_shared_publication_lock_used": True, "revision": "b" * 40,
            "prefix": f"checkpoints/predictive-full3887-{owner.SCIENCE[:12]}/{arm}/epoch-006",
            "files": dict.fromkeys(
                ["model.pt", "provenance.json", "own-source.json", "source-permission.txt", "manifest.json"],
                {"sha256": "c" * 64, "bytes": 42},
            ),
        }
        (directory / f"actual-fixed-final-{arm}.json").write_text(json.dumps(final))
        (directory / f"actual-owner-publish-{arm}-006.json").write_text(json.dumps(publication))


def test_prepared_decision_blocks_execution():
    with pytest.raises(AssertionError, match="Root must accept"):
        owner.validate_decision(json.loads((D / "decision.json").read_text()))
    owner.validate_decision(decision())


def test_waits_without_all_three_original_final_proofs(tmp_path):
    assert owner.full_finals(tmp_path) is None
    make_finals(tmp_path)
    assert set(owner.full_finals(tmp_path)["finals"]) == set(owner.ARMS)
    (tmp_path / "actual-owner-publish-C0-006.json").unlink()
    assert owner.full_finals(tmp_path) is None


@pytest.mark.parametrize("field,value", [
    ("updates", 2915), ("epochs", 5), ("protected_or_gold_cases_read", True),
    ("source_git", "f" * 40),
])
def test_rejects_partial_or_different_original_final(tmp_path, field, value):
    make_finals(tmp_path)
    path = tmp_path / "actual-fixed-final-C.json"
    v = json.loads(path.read_text())
    v[field] = value
    path.write_text(json.dumps(v))
    with pytest.raises(AssertionError):
        owner.full_finals(tmp_path)


def test_unadmitted_other_node_and_failed_original_refused(tmp_path):
    make_finals(tmp_path)
    path = tmp_path / "actual-fixed-final-C0.json"
    v = json.loads(path.read_text())
    v["actual_Slurm_terminal"]["node"] = "evc27"
    path.write_text(json.dumps(v))
    with pytest.raises(AssertionError):
        owner.full_finals(tmp_path)
    (tmp_path / "actual-failure-C0.json").write_text("{}")
    with pytest.raises(RuntimeError, match="failed"):
        owner.full_finals(tmp_path)


def test_fixed_automatic_runtime_decision_uses_actual_receipt_and_tail_sensitivity():
    q = qualification(40)
    plan = owner.runtime_plan(decision(), q, "C", policy)
    estimate = plan["entry_decision"]["selected_estimate_seconds"]
    assert estimate == 600 + 1300 * 40 / 2
    assert plan["entry_decision"]["trained_qualification_sha256"] == owner.sha(owner.canonical(q))
    report = policy.hidden_projection(plan, q["models"][1]["measurements"],
                                     qualification_sha256=owner.sha(owner.canonical(q)))
    assert report["scenarios_seconds"]["tail_stress_two_lane"] == 39600
    assert estimate + 1800 <= 32400
    with pytest.raises(ValueError, match="nine-hour"):
        owner.runtime_plan(decision(), qualification(47), "C", policy)
    assert decision()["runtime"]["planning_study_count"] == 1300


def test_complete_case_measurements_not_gpu_only_or_duplicate_weighted():
    q = qualification(30)
    q["models"][0]["measurements"][-1]["decode_seconds"] = 38
    plan = owner.runtime_plan(decision(), q, "B", policy)
    assert plan["entry_decision"]["selected_estimate_seconds"] == 600 + 1300 * 40 / 2
    q["models"][0]["measurements"] = q["models"][0]["measurements"][:2]
    with pytest.raises(AssertionError):
        owner.runtime_plan(decision(), q, "B", policy)


def test_budget_preserves_vm_startup_beyond_watchdogs():
    plan = owner.runtime_plan(decision(), None, "qualification", policy)
    assert policy.budget(plan, {"totalTimeAllowed": "3359s"}, "qualification", 3)["available"] is False
    assert policy.budget(plan, {"totalTimeAllowed": "3360s"}, "qualification", 3)["available"] is True
    plan = owner.runtime_plan(decision(), qualification(), "C0", policy, remaining_public=1)
    assert policy.budget(plan, {"totalTimeAllowed": "720s"}, "competition", 1)["required_VM_seconds"] == 720


def test_known_no_mutation_wait_is_the_only_repeatable_mutation_response():
    assert owner.phase_result("push", "B", {"status": "waiting_existing_canonical_claim_no_mutation"}) == "wait"
    with pytest.raises(AssertionError):
        owner.phase_result("push", "B", {"status": "blocked", "error_type": "RemoteError"})
    with pytest.raises(AssertionError):
        owner.phase_result("dataset-reconcile", "qualification", {
            "status": "all_payload_bytes_verified", "version": 2, "private": True,
            "downloaded_all_bytes": True,
        })
    assert owner.retryable_poll_failure("status", {"returncode": 255})
    assert owner.retryable_poll_failure("outcome", {
        "returncode": 1, "stdout": '{"status":"blocked","error_type":"RemoteError"}'
    })
    for phase in owner.MUTATIONS:
        assert owner.retryable_poll_failure(phase, {"returncode": 255}) is False
    assert not owner.retryable_poll_failure("status", {
        "stdout": '{"status":"blocked","error_type":"ValueError"}'
    })


def test_frozen_launcher_does_not_read_mutable_current_script():
    source = (
        "from pathlib import Path\n"
        "def frozen(*args): return __frozen_runner_source__\n"
        "assert Path(__file__).read_text() == frozen('pin','path')\n"
        "actual = 'original'\n"
    )
    namespace = {"__name__": "__main__"}
    exec(owner.frozen_launcher(source), namespace)
    assert namespace["actual"] == "original"
    assert namespace["__frozen_runner_source__"] == source
    assert "actual = 'original'" in namespace["__frozen_runner_source__"]


def test_nested_original_poll_timeout_is_typed_and_mutation_timeout_is_not(monkeypatch, capsys):
    source = (
        "from pathlib import Path\n"
        "assert Path(__file__).read_text() == __frozen_runner_source__\n"
        "raise subprocess.TimeoutExpired('original SSH',1900)\n"
    )
    monkeypatch.setattr(owner.sys, "argv", ["frozen", "status"])
    with pytest.raises(SystemExit):
        exec(owner.frozen_launcher(source), {"__name__": "__main__"})
    value = capsys.readouterr().out
    assert owner.retryable_poll_failure("status", {"stdout": value})
    assert not owner.retryable_poll_failure("push", {"stdout": value})
    monkeypatch.setattr(owner.sys, "argv", ["frozen", "push"])
    with pytest.raises(subprocess.TimeoutExpired):
        exec(owner.frozen_launcher(source), {"__name__": "__main__"})


def test_all_fixed_entries_launch_before_waiting_for_any_hidden_score(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, "ROOT", tmp_path)
    monkeypatch.setattr(owner, "full_finals", lambda: {"three_actual_finals": True})
    obj = object.__new__(owner.Owner)
    obj.journal = tmp_path / "journal"
    obj.journal.mkdir()
    obj.decision, obj.policy = decision(), policy
    obj.qualification, obj.remaining_public = None, 3
    obj.observe = lambda *_: {"deployment_root_absent": True, "dataset_ready": True}
    calls = []

    def action(phase, mode="qualification"):
        calls.append((phase, mode))
        if phase == "prepare":
            return {"dataset_slug": "godlydonuts/exact-original"}
        if phase == "status":
            return {"status": "COMPLETE", "receipt": qualification()}
        if phase == "submit":
            return {"state": "submitted", "submission_id": 100 + owner.ARMS.index(mode)}
        if phase == "outcome":
            return {"outcome": "scored", "publicScore": "0.1"}
        return {}

    obj.action = action
    obj.run()
    assert [mode for phase, mode in calls if phase == "submit"] == ["B", "C", "C0"]
    first_outcome = next(i for i, row in enumerate(calls) if row[0] == "outcome")
    assert all(i < first_outcome for i, row in enumerate(calls) if row[0] == "submit")
    assert sum(phase == "dataset" for phase, _ in calls) == 1
    assert obj.remaining_public == 0
    assert (obj.journal / "complete.json").is_file()


def test_observation_timeout_can_recover_without_mutation(monkeypatch):
    calls = []
    obj = object.__new__(owner.Owner)
    def observed(_):
        calls.append("observe")
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("read-only", 340)
        return {"status": "actual_predictive_deployment_read_only"}
    obj._observe = observed
    obj.sleep = lambda *_: None
    assert obj.observe()["status"] == "actual_predictive_deployment_read_only"
    assert calls == ["observe", "observe"]


def test_no_entry_mutation_after_latest_launch_deadline(tmp_path, monkeypatch):
    obj = object.__new__(owner.Owner)
    obj.journal = tmp_path
    monkeypatch.setattr(time, "time", lambda: owner.LAUNCH_END + 1)
    obj.observe = lambda **_: pytest.fail("No late account operation permitted")
    with pytest.raises(AssertionError, match="No new launch"):
        obj.action("submit", "B")


def test_exclusive_owner_intent_preserves_original(tmp_path):
    path = tmp_path / "owner-intent.json"
    owner.once(path, {"original": 1})
    with pytest.raises(FileExistsError):
        owner.once(path, {"original": 2})
    assert json.loads(path.read_text()) == {"original": 1}
