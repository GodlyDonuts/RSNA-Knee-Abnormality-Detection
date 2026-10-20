"""No remote operations: actual-final admission, one-shot controls and runtime budgets."""

import ast
import copy
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

D = Path("initiative/research/predictive-final-deployment-20261020-v1")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


p = module("predictive_deployment_policy_test", D / "policy.py")
cpu = module("predictive_deployment_CPU_test", D / "cpu.py")
account = module("predictive_deployment_account_test", D / "account.py")
serving_tests = module("predictive_export_fixtures", Path("tests/test_predictive_final_serving.py"))
entries = serving_tests.entries


def now():
    return datetime.now(timezone.utc).isoformat()


def plan_and_probe():
    probe = dict(
        status="actual_public_MRI_predictive_two_T4_measurements_recorded",
        completed_device_count=2,
        device_errors=[],
        host22GiB_device13GiB_memory_checks=True,
        submission_csv_absent=True,
    )
    plan = dict(
        status="accepted_predictive_final_runtime_budget",
        checked_at=now(),
        probe_receipt_sha256=p.sha(p.canonical(probe)),
        qualification_cap_seconds=1200,
        public_watchdog_seconds=600,
        other_reserved_VM_seconds=3000,
        planning_study_count=5000,
        planning_count_basis="Illustrative planning assumption; hidden count is unknown",
        hidden_runtime_proven=False,
        public_tail_extrapolation_risk_accepted=True,
        startup_allowance_seconds=600,
        tail_sensitivity_factor=3,
        safety_margin_seconds=1200,
    )
    return plan, probe


def final_fixture(entries):
    _, values = entries
    value = values[0]
    # Original publisher includes actual runtime versions; synthetic fixture omitted it.
    serving_tests.rewrite_provenance(value, lambda x: x.update(runtime_versions={"torch": "test"}))
    chosen = serving_tests.c.candidate(**value)
    terminal = copy.deepcopy(chosen["provenance"])
    terminal.update(
        status="actual_fixed_final_full3887_predictive_arm_complete",
        epochs=6,
        updates=2916,
        study_exposures=3887 * 6,
        retained_full_checkpoints=7,
        immutable_epoch_sidecars=6,
        all_original_fitting_labels_masks_unchanged=True,
        protected_or_gold_cases_read=False,
        final_checkpoint_sha256="e" * 64,
        final_checkpoint_bytes=1000,
    )
    scheduler = dict(job_id=chosen["provenance"]["GPU_job"], state="COMPLETED", exit_code="0:0")
    return chosen, terminal, scheduler


def test_real_final_requires_publication_plus_full_original_terminal(entries):
    candidate, terminal, scheduler = final_fixture(entries)
    p.final_outcome(candidate, terminal, scheduler)
    assert terminal["final_checkpoint_sha256"] != candidate["provenance"]["checkpoint_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        "epoch",
        "updates",
        "pilot",
        "source",
        "parent",
        "GPU",
        "targets",
        "runtime",
        "failed",
        "exit",
        "masked",
    ],
)
def test_false_final_rejected(entries, change):
    candidate, terminal, scheduler = final_fixture(entries)
    if change in ("failed", "exit"):
        scheduler["state" if change == "failed" else "exit_code"] = (
            "FAILED" if change == "failed" else "1:0"
        )
    else:
        key, value = {
            "epoch": ("epochs", 5),
            "updates": ("updates", 2915),
            "pilot": ("study_exposures", 24),
            "source": ("source_git", "c" * 40),
            "parent": ("parent_weight_sha256", "f" * 64),
            "GPU": ("GPU_job", "different-original"),
            "targets": ("targets_sha256", "e" * 64),
            "runtime": ("runtime_versions", {}),
            "masked": ("protected_or_gold_cases_read", True),
        }[change]
        terminal[key] = value
    with pytest.raises(ValueError):
        p.final_outcome(candidate, terminal, scheduler)


def test_one_shot_remote_intent_blocks_even_without_a_returned_result(tmp_path):
    path = tmp_path / "intent.json"
    p.no_prior_intent(path, {}, "push")
    path.write_text("{}")
    with pytest.raises(ValueError, match="retry prohibited"):
        p.no_prior_intent(path, {}, "push")
    path.unlink()
    with pytest.raises(ValueError, match="retry prohibited"):
        p.no_prior_intent(path, {"push": True}, "push")


def test_authoritative_granted_total_matches_original_account_decision():
    quota = dict(
        minimumTimeAllowed="1000s",
        totalTimeAllowed="5000s",
        timeUsed="1000s",
        timeReserved="500s",
        isPayToScaleEnabled=True,
    )
    assert p.free_seconds(quota) == 3500
    plan, probe = plan_and_probe()
    p.runtime_plan(plan, probe)
    budget = p.budget(plan, quota, "qualification", 3)
    assert budget["required_VM_seconds"] == 6000
    assert budget["available"] is False
    assert budget["paid_credits_used"] is False
    plan["other_reserved_VM_seconds"] = 0
    assert p.budget(plan, quota, "qualification", 3)["available"] is True


@pytest.mark.parametrize(
    "change", ["stale", "receipt", "device", "memory", "cap", "watchdog", "hidden", "tail"]
)
def test_runtime_requires_actual_two_T4_evidence_and_budget(change):
    plan, probe = plan_and_probe()
    if change == "stale":
        plan["checked_at"] = "2000-01-01T00:00:00+00:00"
    elif change == "receipt":
        plan["probe_receipt_sha256"] = "0" * 64
    elif change in ("device", "memory"):
        probe[
            "completed_device_count"
            if change == "device"
            else "host22GiB_device13GiB_memory_checks"
        ] = 1 if change == "device" else False
        plan["probe_receipt_sha256"] = p.sha(p.canonical(probe))
    else:
        key, value = {
            "cap": ("qualification_cap_seconds", 32400),
            "watchdog": ("public_watchdog_seconds", 3600),
            "hidden": ("hidden_runtime_proven", True),
            "tail": ("public_tail_extrapolation_risk_accepted", False),
        }[change]
        plan[key] = value
    with pytest.raises(ValueError):
        p.runtime_plan(plan, probe)


def test_hidden_projection_uses_actual_pipeline_and_explicit_selected_scenario():
    plan, _ = plan_and_probe()
    fast = [
        dict(study_index=i, FP16_seconds=0.2, decode_seconds=0.1, render_seconds=0.2)
        for i in (0, 1, 2, 0)
    ]
    plan["entry_decision"] = dict(
        accepted=True,
        trained_qualification_sha256="a" * 64,
        selected_scenario="mean_two_lane",
        selected_estimate_seconds=1850,
        rationale="Choose mean with explicit 1200s headroom; hidden tails remain uncertain",
    )
    result = p.hidden_projection(plan, fast, qualification_sha256="a" * 64)
    assert result["scenarios_seconds"]["mean_two_lane"] == 1850
    assert result["hidden_runtime_proven"] is False
    slow = [
        dict(study_index=i, FP16_seconds=0.2, decode_seconds=3.0, render_seconds=5.0)
        for i in (0, 1, 2, 0)
    ]
    plan["entry_decision"]["selected_estimate_seconds"] = 21100
    result = p.hidden_projection(plan, slow, qualification_sha256="a" * 64)
    assert result["scenarios_seconds"]["tail_stress_two_lane"] > 32400
    # A stress scenario is reported, not an unsupported automatic veto.
    with pytest.raises(ValueError, match="cite this actual"):
        p.hidden_projection(plan, slow, qualification_sha256="b" * 64)
    plan["entry_decision"]["selected_scenario"] = "tail_stress_two_lane"
    plan["entry_decision"]["selected_estimate_seconds"] = result["scenarios_seconds"][
        "tail_stress_two_lane"
    ]
    with pytest.raises(ValueError, match="nine-hour"):
        p.hidden_projection(plan, slow, qualification_sha256="a" * 64)


def test_cross_device_precision_repeat_is_not_a_fourth_independent_case():
    plan, _ = plan_and_probe()
    measures = [
        dict(study_index=i, FP16_seconds=t, decode_seconds=1, render_seconds=1)
        for i, t in ((0, 1), (1, 1), (2, 1), (0, 4))
    ]
    # Three unique cases cost6,3,3 seconds; repeat weighting would incorrectly give3.75.
    plan["entry_decision"] = dict(
        accepted=True,
        trained_qualification_sha256="a" * 64,
        selected_scenario="mean_two_lane",
        selected_estimate_seconds=10600,
        rationale="Three unique measured cases, worst device per case; explicit headroom retained",
    )
    result = p.hidden_projection(plan, measures, qualification_sha256="a" * 64)
    assert result["measured_unique_studies"] == 3
    assert result["scenarios_seconds"]["mean_two_lane"] == 10600
    with pytest.raises(ValueError, match="three unique"):
        p.hidden_projection(plan, measures[:2], qualification_sha256="a" * 64)


def test_incremental_storage_is_not_old_training_funding():
    fund = dict(
        status="predictive_final_deployment_incremental_storage_funded",
        checked_at=now(),
        new_node_bytes=p.NODE_CAP,
        new_shared_bytes=p.SHARED_BYTES,
        new_shared_files=p.SHARED_FILES,
        existing_60GiB_or_574MiB_reused=False,
        other_node_unpaid_bytes=60 * 2**30,
        other_shared_unpaid_bytes=574 * 2**20,
        other_shared_unpaid_files=100,
    )
    args = dict(node_free=82 * 2**30, shared_free=21 * 2**30, shared_files=5116)
    p.storage(fund, **args)
    with pytest.raises(ValueError):
        p.storage(fund, **{**args, "node_free": 80 * 2**30})
    fund["existing_60GiB_or_574MiB_reused"] = True
    with pytest.raises(ValueError):
        p.storage(fund, **args)


def test_paid_scope_is_deducted_only_from_its_own_incremental_bound():
    fund = dict(
        status="predictive_final_deployment_incremental_storage_funded",
        checked_at=now(),
        new_node_bytes=p.NODE_CAP,
        new_shared_bytes=p.SHARED_BYTES,
        new_shared_files=p.SHARED_FILES,
        existing_60GiB_or_574MiB_reused=False,
        other_node_unpaid_bytes=0,
        other_shared_unpaid_bytes=0,
        other_shared_unpaid_files=0,
    )
    p.storage(fund, node_free=21 * 2**30, shared_free=21 * 2**30, shared_files=5016, used=2**30)
    with pytest.raises(ValueError):
        p.storage(
            fund,
            node_free=21 * 2**30,
            shared_free=21 * 2**30,
            shared_files=5016,
            used=p.NODE_CAP + 1,
        )


def test_request_exact_sources_separate_competition_and_combined_qualification():
    args = (
        "source",
        "qualification",
        "bundle",
        "godlydonuts/rsna-predictive-finals-123",
        "image",
        1200,
    )
    q = p.request(*args)
    b = p.request(args[0], "B", *args[2:])
    assert q["datasetDataSources"] == [p.PARENT_DATASET, args[3]]
    assert q["sessionTimeoutSeconds"] == 1200 and b["sessionTimeoutSeconds"] == 32400
    assert q["slug"] != b["slug"]
    assert q["enableInternet"] is False and q["isPrivate"] is True
    assert q["machineShape"] == "NvidiaTeslaT4" and not q["kernelDataSources"]


def test_observed_GPU_jobs_are_not_registered_as_new_pushes():
    assert account.observation({"x": [{"kernel_id": 123, "slug": "example"}]}) == {
        "x": [{"observed_kernel_id": 123, "slug": "example"}]
    }


def test_all_controls_import_without_remote_or_dispatch_side_effects():
    for name in ("policy.py", "cpu.py", "account.py", "run_once.py"):
        tree = ast.parse((D / name).read_text())
        assert not any(isinstance(n, ast.While) for n in tree.body)
    runner = module("predictive_deployment_runner_test", D / "run_once.py")
    assert runner.R.resolve() == Path.cwd()
    assert not hasattr(runner, "token")


def test_unsupported_cancel_RPC_and_full_tree_input_scan_are_absent():
    source = (D / "account.py").read_text()
    assert "CancelKernelSession" not in source
    assert "CreateCodeSubmission" in source
    assert '"fileName": "submission.csv"' in source
    assert "sdk_create_once" in source and "publication.reconcile(" in source
    assert "csv.download_csv(" in source and "csv.verify_csv(" in source
    assert not any(t in source for t in ("with_retry", "upload_submission", "sleep("))


def test_partial_preparation_is_never_claimed_complete(tmp_path):
    cpu.write(tmp_path / "prepare-intent.json", {"state": "started"})
    with pytest.raises(FileExistsError):
        cpu.write(tmp_path / "prepare-intent.json", {"state": "retry"})
    assert not (tmp_path / "prepared.json").exists()
    assert json.loads((tmp_path / "prepare-intent.json").read_text())["state"] == "started"


@pytest.mark.parametrize("uncertain", [False, True])
def test_actual_dataset_phase_calls_create_once_then_refuses_repeat(
    tmp_path, monkeypatch, uncertain
):
    """Exercise the actual canonical phase with a fake service, including lost response."""
    from types import SimpleNamespace

    root = tmp_path / "deployment"
    (root / "upload").mkdir(parents=True)
    (root / "upload/bundle.zip").write_bytes(b"fake-test-archive")
    cpu.write(
        root / "prepared.json",
        dict(
            bundle_identity="a" * 64,
            dataset_slug="godlydonuts/test-only",
            files={"test": {"sha256": "b" * 64, "bytes": 1}},
            archive={"sha256": cpu.digest(root / "upload/bundle.zip")},
        ),
    )
    calls = []

    class FakeRemote:
        def call(self, service, method, body):
            assert method == "IntrospectToken"
            return {"active": True, "username": "godlydonuts"}

        def dataset(self, slug):
            assert slug == p.PARENT_DATASET
            return {"id": p.PARENT_ID, "currentVersionNumber": 1}

    workflow = module("predictive_fake_remote_workflow", Path("scripts/kaggle/workflow.py"))
    workflow.Remote = FakeRemote

    def create(*args):
        calls.append("CreateDataset")
        if uncertain:
            raise TimeoutError("synthetic lost response")
        return {"returned": True, "final_create_rpc_calls": 1}

    publication = SimpleNamespace(
        require_absent=lambda *a: {"method": "synthetic-owned-listing"},
        validate_zip=lambda *a: None,
        sdk_create_once=create,
    )

    def loader(name, path):
        if str(path).endswith("workflow.py"):
            return workflow
        if str(path).endswith("publication.py"):
            return publication
        return SimpleNamespace()

    monkeypatch.setattr(account.cpu, "private_path", lambda path: Path(path))
    monkeypatch.setattr(account.cpu, "check_controls", lambda path: {"source_git": "a" * 40})
    monkeypatch.setattr(account.cpu, "funding", lambda *a: None)
    monkeypatch.setattr(account.cpu, "load", loader)
    monkeypatch.setattr(account, "LEDGER", tmp_path / "canonical/account-ledger.sqlite3")
    account.LEDGER.parent.mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    args = dict(
        root=str(root),
        source_git="a" * 40,
        token="synthetic-test-token",
        phase="dataset",
        mode="qualification",
    )
    if uncertain:
        with pytest.raises(TimeoutError):
            account.execute(args)
        assert (account.LEDGER.parent / "coordinator-claim").exists()
        assert account.execute(args)["status"] == "waiting_existing_canonical_claim_no_mutation"
    else:
        assert account.execute(args)["new_Dataset_calls"] == 1
        assert not (account.LEDGER.parent / "coordinator-claim").exists()
        with pytest.raises(ValueError, match="retry prohibited"):
            account.execute(args)
    assert calls == ["CreateDataset"]
    assert (root / "dataset-intent.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("alias_regression_receipt_sha256", "f" * 64),
        ("original_numerical_source_git", "f" * 40),
        ("accepted_renderer_sha256", []),
        ("renderer_provenance_counts", {"original": 0, "corrected": 3887}),
    ],
)
def test_final_outcome_keeps_actual_alias_lineage(entries, field, value):
    candidate, terminal, scheduler = final_fixture(entries)
    terminal[field] = value
    with pytest.raises(ValueError, match="Final outcome and preserved export differ"):
        p.final_outcome(candidate, terminal, scheduler)
