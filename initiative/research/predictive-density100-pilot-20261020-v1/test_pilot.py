"""Bounded coefficient intervention, recovery rejection, and no shared-log regression."""

import ast
import copy
import importlib.util
import json
import subprocess
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


C = module("C100_test_contract", "worker.py")
D = module("C100_test_driver", "driver.py")
R = module("C100_test_runner", "run_once.py")
P = module("C100_test_prepare", "prepare.py")


def original_worker():
    return subprocess.check_output([
        "git", "show", C.SCIENCE
        + ":initiative/research/predictive-stable-warmup-20261020-v1/worker.py.txt",
    ], cwd=REPO).decode()


def test_only_declared_worker_literals_change_not_numerical_function_bodies():
    old = original_worker()
    new = C.build(old)
    a, b = ast.parse(old), ast.parse(new)
    def definitions(tree):
        return [
            ast.dump(n, include_attributes=False) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        ]
    assert definitions(a) == definitions(b)
    restored = new.replace(str(C.ROOT), str(C.ORIGINAL))
    restored = restored.replace('assert arm == "C" and phase in',
                                'assert arm in ("B", "C", "C0") and phase in')
    restored = restored.replace("density_weight=100.0,", 'density_weight=0 if arm == "C0" else 0.1,')
    restored = restored.replace("< 6 * 2**30", "< 12 * 2**30")
    assert restored == old
    with pytest.raises(AssertionError):
        C.build(old + "\n")
    assert "epoch_snapshots=False" in new and "stop_after=1" in new


def test_capacity_preserves_entire_original_C0_and_other_future_scopes():
    required = 349628792832
    with pytest.raises(AssertionError):
        C.node_capacity(required)
    assert C.node_capacity(required + 1)["required_bytes"] == required
    assert C.node_capacity(required, 1)["remaining_bytes"] == 1
    with pytest.raises(AssertionError):
        C.node_capacity(required + 10, C.CHECKPOINT_CAP + 1)
    quota = " /home/sa305415 1026232988 1048576000 1059061760 - 994621 1000000 1010000 -\n"
    assert C.quota_capacity(quota)["free_files"] == 5379
    with pytest.raises(AssertionError):
        C.quota_capacity(quota.replace("994621", "994625"))


def test_latest_start_and_scheduler_bound_and_no_shared_logs():
    C.admit_time(C.FINISH_BY - timedelta(minutes=30))
    with pytest.raises(AssertionError):
        C.admit_time(C.FINISH_BY - timedelta(minutes=29, seconds=59))
    script = R.script()
    assert "--time=00:30:00" in script and "--deadline=2026-10-21T17:15:00" in script
    assert "--begin" not in script and "--gres=gpu:nvidia_h100_pcie:1" in script
    for line in script.splitlines():
        if line.startswith(("#SBATCH --output=", "#SBATCH --error=")):
            assert str(C.ROOT) in line and "/home/" not in line


def test_complete_state_comparison_catches_optimizer_rng_and_prediction_changes():
    state = {"model": {"w": torch.tensor([1.0])}, "optimizer": {"m": torch.tensor([0.02])},
             "rng": (np.array([1, 2]), [1, 2]), "history": [{"step": 24}]}
    D.equal(state, copy.deepcopy(state))
    for key in ("model", "optimizer"):
        wrong = copy.deepcopy(state)
        next(iter(wrong[key].values())).add_(0.1)
        with pytest.raises(AssertionError):
            D.equal(state, wrong)
    wrong = copy.deepcopy(state)
    wrong["rng"][0][0] += 1
    with pytest.raises(AssertionError):
        D.equal(state, wrong)


def example_state():
    original = json.loads((REPO / "initiative/research/predictive-stable-warmup-20261020-v1/actual-accepted.json").read_bytes())
    arm = next(a for a in original["arms"] if a["arm"] == "C")
    recipe = {
        "arm": "C", "epochs": 12, "effective_batch": 8, "encoder_lr": 3e-5,
        "head_lr": 3e-5, "weight_decay": 0.02, "warmup_fraction": 0.05,
        "end_lr_fraction": 0.1, "density_weight": 100.0, "head_only_updates": 3,
        "clip": 1.0, "seed": 20261020, "schedule_updates": 2916,
    }
    history = copy.deepcopy(arm["real_optimizer_history"])
    for row in history:
        row["loss"] = row["exam_loss"] + 100 * row["density_loss"]
    return {"recipe": recipe, "binding": {"fixture": True}, "step": 24, "cohort_size": 12,
            "epoch_snapshots": False, "history": history}, arm


def test_large_gaussian_constant_is_not_automatic_failure():
    state, original_c = example_state()
    assert state["history"][0]["loss"] > 90
    D.check_history(state, {"fixture": True})
    summary = C.scientific_warnings(state["history"], original_c["terminal"], original_c)
    assert summary["weighted_fixed_Gaussian_constant"] == pytest.approx(91.89385332046727)
    assert summary["last_six_exam_mean_over_twice_original"] is False
    assert summary["maximum_exam_over_twice_original"] is False
    assert summary["maximum_BF_over_ten_times_original"] is False
    for row in state["history"][-6:]:
        row["exam_loss"] *= 3
    summary = C.scientific_warnings(state["history"], original_c["terminal"], original_c)
    assert summary["last_six_exam_mean_over_twice_original"] is True
    D.check_history(state, {"fixture": True})  # Warning only; no clinical/model-selection gate.


@pytest.mark.parametrize("change", ["nonfinite", "encoder", "tail", "coefficient", "cursor"])
def test_actual_engineering_invariants_are_hard_failures(change):
    state, _ = example_state()
    if change == "nonfinite":
        state["history"][2]["head_gradient_norm"] = float("nan")
    elif change == "encoder":
        state["history"][3]["encoder_updated"] = False
    elif change == "tail":
        state["history"][-1]["studies"] = 8
    elif change == "coefficient":
        state["recipe"]["density_weight"] = 0.1
    else:
        state["step"] = 23
    with pytest.raises(AssertionError):
        D.check_history(state, {"fixture": True})


def test_exclusive_intent_preserves_original_uncertain_attempt(tmp_path):
    intent = tmp_path / "GPU-intent.json"
    C.once(intent, {"one": True})
    original = intent.read_bytes()
    with pytest.raises(FileExistsError):
        C.once(intent, {"two": True})
    assert intent.read_bytes() == original


def test_resolved_system_python_may_have_other_owner_but_not_unsafe_permissions(tmp_path, monkeypatch):
    path = tmp_path / "python"
    path.write_bytes(b"executable")
    path.chmod(0o755)
    owner = path.stat().st_uid
    monkeypatch.setattr(P.os, "getuid", lambda: owner + 1)
    assert P.signature(path, require_owned=False)[1] == len(b"executable")
    with pytest.raises(AssertionError):
        P.signature(path)
    path.chmod(0o777)
    with pytest.raises(AssertionError):
        P.signature(path, require_owned=False)
