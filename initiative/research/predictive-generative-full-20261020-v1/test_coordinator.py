"""Future G100 controls must retain existing jobs, inputs and independent budgets."""

import copy
import importlib.util
import subprocess
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


coordinator = load("future_G100_coordinator_test", HERE / "coordinator.py")
helpers = load("future_G100_worker_test_helpers", HERE / "test_full_worker.py")
controls = load("future_G100_control_test", HERE / "controls.py")


@pytest.fixture(scope="module")
def prepared():
    ns = coordinator.namespace()
    texts = controls.sources(REPO, helpers.c)
    operations = types.ModuleType("future_G100_operations")
    source = texts["operations.py"].replace("import admission as a", "a = ADMISSION")
    operations.ADMISSION = ns["a"]
    exec(compile(source, "derived_G100_operations", "exec"), operations.__dict__)
    return ns, texts, operations


def test_distinct_paths_pilot_and_original_controller_bindings(prepared):
    ns, texts, _ = prepared
    a = ns["a"]
    assert str(a.ROOT) == helpers.c.ROOT and str(a.RUN) == helpers.c.RUN
    assert str(a.PILOT / "GPU-accepted.json") == helpers.c.PILOT_PATH
    assert a.PILOT_SOURCE == helpers.c.PILOT_SOURCE and a.PILOT_BINDING == helpers.c.PILOT_BINDING
    assert a.recipe() == helpers.c.base.recipe()
    assert a.primary_control()["continuation_pid"] == 33107
    assert a.primary_control()["source_git"] == "2a224ae05fc7c2fc78d4d508b6407344ac523a1c"
    assert ns["JOURNAL"] == REPO / "artifacts" / HERE.name
    assert 'scheduler("986011")' in texts["operations.py"]
    assert 'scheduler("981620")' not in texts["operations.py"]
    assert 'binding["generative_pilot"]' in texts["operations.py"]


def test_original_c100_full_scope_is_not_reassigned_to_g100(prepared):
    a = prepared[0]["a"]
    old = types.ModuleType("original_C100_funding")
    raw = subprocess.check_output(
        ["git", "show", controls.PIN + ":" + controls.ORIGINAL + "admission.py"], cwd=REPO
    )
    exec(compile(raw, "original_C100_admission", "exec"), old.__dict__)
    args = dict(
        contexts_paid=0,
        C0_controls_paid=0,
        C0_fit_paid=0,
        pilot_checkpoints_paid=0,
        pilot_controls_paid=0,
    )
    original = old.node_budget(1000 * 2**30, **args)
    future = a.node_budget(1000 * 2**30, **args)
    assert future["required_bytes"] == original["required_bytes"] + 12 * 2**30 + 16 * 2**20
    assert future["C100_full_reserved_bytes_no_paid_credit"] == 12 * 2**30 + 16 * 2**20
    assert future["C100_pilot_reserved_bytes_no_paid_credit"] == 6 * 2**30 + 4 * 2**20
    with pytest.raises(ValueError):
        a.node_budget(original["required_bytes"] + 1, **args)


def test_actual_matched_pilot_and_explicit_full_decision_before_mutation(prepared):
    a = prepared[0]["a"]
    b, pilot, decision, _, _ = helpers.fixture(helpers.d.build(REPO))
    decision.pop("full_G100_agent_decision_sha256")
    decision["status"] = "accepted_separate_full_G100_after_actual_pilot_review"
    a.validate_decision(
        decision, pilot, b["generative_pilot"]["sha256"], b["source_git"], b["code_sha256"]
    )
    bad = copy.deepcopy(decision)
    bad["pilot_scheduler"]["state"] = "RUNNING"
    with pytest.raises(ValueError):
        a.validate_decision(
            bad, pilot, b["generative_pilot"]["sha256"], b["source_git"], b["code_sha256"]
        )
    bad = copy.deepcopy(decision)
    bad["engineering_predictions_used_for_selection"] = True
    with pytest.raises(ValueError):
        a.validate_decision(
            bad, pilot, b["generative_pilot"]["sha256"], b["source_git"], b["code_sha256"]
        )


def test_one_gpu_script_preserves_all_three_original_start_dependencies(prepared):
    op = prepared[2]
    text = op.script("a" * 64, ["111", "222", "333"])
    for required in (
        "--dependency=after:111:222:333",
        "--cpus-per-task=2",
        "--gres=gpu:nvidia_h100_pcie:1",
        "--nodelist=evc27",
        "--job-name=rsna-predictive-full-G100",
        "--time=08:00:00",
        "--deadline=2026-10-22T04:00:00",
        helpers.c.ROOT,
    ):
        assert required in text
    assert "scancel" not in text and "full-C100" not in text
    with pytest.raises(ValueError):
        op.script("a" * 64, ["111", "222"])


def test_precommit_source_cannot_be_admitted_by_frozen_coordinator(prepared):
    ns = prepared[0]
    args = types.SimpleNamespace(source_git="4f2e0127e0aaff777469d0b0780d5712fbd5322c")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        ns["frozen"](args)
