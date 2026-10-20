"""Alias recovery changes admission lineage, never the predictive fitting math."""

import ast
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

D = Path("initiative/research/predictive-full-fitting-20261020-v2")
OLD = Path("initiative/research/predictive-full-fitting-20261020-v1")
FAILURE = Path("initiative/research/predictive-alias-failure-20261020-v1")


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(module)
    return module


alias = load("alias_provenance_v2_test", D / "alias_provenance.py")
admit = load("alias_admit_v2_test", D / "admit_cpu.py.txt")
owner = load("alias_owner_v2_test", D / "owner.py")
remote = load("alias_owner_remote_v2_test", D / "owner_remote.py.txt")


def manifests():
    original = json.loads((OLD / "actual-cpu-prepared.json").read_text())["code_sha256"]
    current = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in original
        if name.startswith("rsna/")
    }
    # Original standalone source intentionally uses empty namespace initializers.
    for name in ("rsna/__init__.py", "rsna/competitive/__init__.py"):
        current[name] = hashlib.sha256(b"").hexdigest()
    regression = json.loads((FAILURE / "actual-regression.json").read_text())
    return original, current, regression


def valid_terminal(renderers):
    return {
        "status": "actual_missing_only3887_exact_FP32_context_alias_recovery_accepted",
        "original_context_job": "978414",
        "original_raw_job": "978318",
        "successor_raw_job": "999991",
        "studies": 3887,
        "exact_both_view_images_targets_metadata_priors": True,
        "global_RNG_unchanged": True,
        "all_originals_preserved": True,
        "original_renderer_sha256": renderers[0],
        "corrected_renderer_sha256": renderers[1],
        "alias_regression_receipt_sha256": alias.REGRESSION_SHA,
        "prefix_admission_receipt_sha256": alias.PREFIX_SHA,
    }


def test_actual_source_delta_and_actual_regression_are_exactly_bound():
    original, current, regression = manifests()
    accepted = alias.accepted_renderers(original, current, regression)
    assert accepted[0][alias.CASES] == alias.ORIGINAL_CASES
    assert accepted[1][alias.CASES] == alias.CORRECTED_CASES
    assert (
        hashlib.sha256((FAILURE / "actual-regression.json").read_bytes()).hexdigest()
        == alias.REGRESSION_SHA
    )
    assert [k for k in current if current[k] != original[k]] == [alias.CASES]


@pytest.mark.parametrize(
    "name",
    ["model", "train", "inputs", "supports", "density", "context_cache"],
)
def test_no_other_numerical_change_can_be_hidden_behind_alias_compatibility(name):
    original, current, regression = manifests()
    current[f"rsna/competitive/predictive/{name}.py"] = "f" * 64
    with pytest.raises(ValueError, match="other than"):
        alias.accepted_renderers(original, current, regression)


@pytest.mark.parametrize(
    "field",
    [
        "no_source_downloads",
        "prior_successful_contexts_unchanged",
        "retained_failed_case_corrected",
    ],
)
def test_regression_plan_or_partial_result_cannot_replace_actual_acceptance(field):
    original, current, regression = manifests()
    regression[field] = False
    with pytest.raises(ValueError, match="regression proof"):
        alias.accepted_renderers(original, current, regression)


def test_old_and_new_rows_keep_their_actual_provenance():
    accepted = alias.accepted_renderers(*manifests())
    rows = [
        {"provenance": {"module_sha256": accepted[0]}, "proof": {"sha256": "a" * 64}},
        {"provenance": {"module_sha256": accepted[1]}, "proof": {"sha256": "b" * 64}},
    ]
    original = copy.deepcopy(rows)
    assert alias.renderer_counts(rows, accepted) == {"original": 1, "corrected": 1}
    assert rows == original
    rows[0]["provenance"]["module_sha256"] = {alias.CASES: alias.CORRECTED_CASES}
    with pytest.raises(ValueError, match="per-row"):
        alias.renderer_counts(rows, accepted)


def test_terminal_must_bind_both_renderers_and_exact_actual_regression():
    accepted = alias.accepted_renderers(*manifests())
    terminal = valid_terminal(accepted)
    alias.validate_terminal(terminal, accepted, alias.REGRESSION_SHA)
    terminal["original_renderer_sha256"] = accepted[1]
    with pytest.raises(ValueError, match="binding differs"):
        alias.validate_terminal(terminal, accepted, alias.REGRESSION_SHA)
    with pytest.raises(ValueError, match="CPU979537"):
        alias.validate_terminal(valid_terminal(accepted), accepted, "f" * 64)


def test_owner_stop_receipt_cannot_hide_an_inflight_phase():
    receipt = json.loads((FAILURE / "actual-old-fitting-owner-stopped.json").read_text())
    alias.validate_owner_stop(receipt)
    receipt["phase_intents"] = 1
    with pytest.raises(ValueError, match="old-owner stop"):
        alias.validate_owner_stop(receipt)


def test_purged_terminal_is_checked_with_account_queue_not_invalid_job_lookup(monkeypatch):
    calls = []

    def output(args, **kwargs):
        calls.append(args)
        if args[0] == "sacct":
            return "978318|FAILED|1:0\n"
        assert args == ["squeue", "-h", "-u", "sa305415", "-o", "%i|%T"]
        return "975297|RUNNING\n"  # Unrelated healthy job must not block reconciliation.

    monkeypatch.setattr(admit.subprocess, "check_output", output)
    assert admit.terminal_job("978318", ("FAILED",))["exit_code"] == "1:0"
    assert len(calls) == 2


def test_same_requested_job_still_in_queue_is_not_terminal(monkeypatch):
    monkeypatch.setattr(
        admit.subprocess,
        "check_output",
        lambda args, **kwargs: (
            "978318|FAILED|1:0\n" if args[0] == "sacct" else "978318|COMPLETING\n"
        ),
    )
    with pytest.raises(ValueError, match="actually be complete"):
        admit.terminal_job("978318", ("FAILED",))


def test_observer_handles_purged_completed_arm_among_other_running_jobs(monkeypatch):
    monkeypatch.setattr(
        remote.subprocess,
        "check_output",
        lambda args, **kwargs: (
            "999991|COMPLETED|0:0|rsna-predictive-full2-B|evc33\n"
            if args[0] == "sacct"
            else "975297|RUNNING\n999992|RUNNING\n"
        ),
    )
    assert remote.accounting(["999991"])["999991"]["queue_empty"]


def test_initialization_and_actual_optimizer_loop_are_unchanged():
    def function(source, name):
        return next(
            n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name
        )

    old_admit = subprocess.check_output(
        ["git", "show", alias.ORIGINAL_SCIENCE + ":" + str(OLD / "admit_cpu.py.txt")], text=True
    )
    assert ast.dump(function(old_admit, "initialize")) == ast.dump(
        function((D / "admit_cpu.py.txt").read_text(), "initialize")
    )
    old_worker = subprocess.check_output(
        ["git", "show", alias.ORIGINAL_SCIENCE + ":" + str(OLD / "worker.py.txt")], text=True
    )
    new_worker = (D / "worker.py.txt").read_text()
    # The complete fit loop, per-epoch serialization, LR/counter/RNG handling and
    # final receipt calculation after model.cuda are untouched.
    marker = "    model = model.cuda()\n"
    assert old_worker.split(marker, 1)[1] == new_worker.split(marker, 1)[1]


def test_duplicate_successor_mutation_never_contacts_ssh(tmp_path, monkeypatch):
    owner.once(tmp_path / "dispatch-intent.json", {"original": True})
    monkeypatch.setattr(owner.subprocess, "run", lambda *a, **k: pytest.fail("Duplicate SSH"))
    frozen = {
        "owner_git": "f" * 40,
        "program": "pass",
        "prepared_sha256": "a" * 64,
    }
    with pytest.raises(FileExistsError):
        owner.transport_call(
            frozen, SimpleNamespace(socket="/sock"), {}, "dispatch", "dispatch", tmp_path
        )


def test_new_owner_uses_alias_v2_collection_and_distinct_output_roots():
    assert owner.COLLECT.name == "predictive-full-cache-alias-recovery-20261020-v3"
    assert str(remote.ROOT).endswith("predictive-full-fitting-20261020-v2")
    assert str(remote.run_root("C0")).endswith("predictive-full-C0-20261020-v2")
    assert remote.SHARED_FUTURE == 121 * 2**20
    assert '"shared_unpaid_files": 376' in (D / "owner_remote.py.txt").read_text()
