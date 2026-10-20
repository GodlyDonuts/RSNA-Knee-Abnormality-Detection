"""Matched execution ordering, hard failures, recovery and fresh funding boundaries."""

import ast
import copy
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location("routing_test_" + name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


D, W = module("driver"), module("worker_builder")


def prerequisites():
    binding = dict(branch=D.BRANCH, roles=["C100", "G100"], source_git="a" * 40,
                   original_numerical_source_git=D.SCIENCE, recipe=D.recipe(),
                   density100_pilot=dict(path=str(D.PILOT), sha256="1" * 64),
                   code_sha256={"worker.py": "2" * 64})
    history = [dict(step=i + 1, studies=8 if i % 2 == 0 else 4, loss=93., exam_loss=.7,
                    density_loss=.923, encoder_gradient_norm=1., head_gradient_norm=2.,
                    gradient_norm=2.) for i in range(24)]
    pilot = dict(status="actual_C100_full_stop_resume_engineering_complete", GPU_job="981620",
                 source_git=D.PILOT_SOURCE, binding_sha256=D.PILOT_BINDING, branch="C100",
                 original_numerical_source_git=D.SCIENCE,
                 full_stop_resume_complete_state_and_prediction_bytes_equal=True,
                 terminal=dict(steps=24, density_weight=100),
                 automatic_full_training_promotion=False, clinical_or_held_out_accuracy_claim=False,
                 real_optimizer_history=history, engineering_warnings={"fixture": True})
    decision = dict(status="accepted_matched_generative_routing_engineering",
                    source_git=binding["source_git"], code_sha256=binding["code_sha256"],
                    original_numerical_source_git=D.SCIENCE, pilot_acceptance_sha256="1" * 64,
                    pilot_warnings_reviewed=pilot["engineering_warnings"], recipe=D.recipe(),
                    roles=["C100", "G100"], phases=[list(p) for p in D.PHASES],
                    actual_optimizer_updates=600, maximum_GPU_seconds=5400,
                    engineering_predictions_used_for_selection=False, automatic_recipe_retuning=False,
                    primary_B_C_C0_C100_unchanged=True, automatic_full_training_promotion=False,
                    pilot_scheduler=dict(GPU_job="981620", state="COMPLETED", exit_code="0:0", node="evc27"))
    funding = dict(status="actual_matched_generative_routing_scope_funded", branch=D.BRANCH,
                   source_git=binding["source_git"], node="evc27", new_checkpoint_cap_bytes=6 * D.GIB,
                   new_control_and_log_cap_bytes=4 * D.MIB, standing_free_reserve_bytes=20 * D.GIB,
                   standing_free_file_reserve=5000, prior_C100_pilot_scope_retained=True,
                   private_descriptor_scale_control_bytes=D.MIB,
                   other_unpaid_node_liabilities_bytes=400 * D.GIB,
                   shared_future_bytes=123 * D.MIB, shared_future_files=376,
                   new_shared_files=0, new_shared_bytes=0)
    return binding, pilot, decision, funding


def test_original_pilot_must_be_actual_completed_not_pending_or_another_job():
    D.pilot_completed("981620|COMPLETED|0:0|evc27|\n981620.batch|FAILED|1:0|evc27|\n")
    for row in ("981620|PENDING|0:0||", "981620|COMPLETED|1:0|evc27|",
                "981620|COMPLETED|0:0|evc33|", "9816201|COMPLETED|0:0|evc27|"):
        with pytest.raises(ValueError):
            D.pilot_completed(row)


@pytest.mark.parametrize("target,key,value", [
    (1, "status", "PENDING"), (1, "GPU_job", "981621"),
    (1, "full_stop_resume_complete_state_and_prediction_bytes_equal", False),
    (2, "pilot_warnings_reviewed", {}), (2, "maximum_GPU_seconds", 5500),
    (2, "engineering_predictions_used_for_selection", True),
    (3, "prior_C100_pilot_scope_retained", False), (3, "new_checkpoint_cap_bytes", 5 * D.GIB),
    (3, "private_descriptor_scale_control_bytes", 0),
])
def test_missing_or_changed_admission_is_not_execution_authority(target, key, value):
    values = prerequisites()
    D.validate_prerequisites(*values)
    values[target][key] = value
    with pytest.raises(ValueError):
        D.validate_prerequisites(*values)


def state_fixture():
    binding = prerequisites()[0]
    history = []
    for i in range(200):
        fraction = i / 2915
        factor = min(1., (i + 1) / 145.8) if fraction < .05 else (
            .1 + .9 * (1 + math.cos(math.pi * (fraction - .05) / .95)) / 2)
        history.append(dict(step=i + 1, epoch=i // 2 + 1, studies=8 if i % 2 == 0 else 4,
                            loss=93., exam_loss=.7, density_loss=.923, encoder_gradient_norm=1.,
                            head_gradient_norm=2., gradient_norm=2., encoder_updated=i >= 3,
                            LR=[3e-5 * factor] * 2))
    return dict(recipe=D.recipe(), step=200, cohort_size=12, epoch_snapshots=False,
                binding={**binding, "matched_worker": {"role": "G100"}}, history=history,
                model={"weight": torch.tensor([1., 2.])},
                optimizer={"state": {0: {"exp_avg": torch.tensor([.1, .2])}}},
                rng={"numpy": np.array([1, 2]), "python": (3, (4, 5)), "torch": torch.tensor([1])})


def test_exact_production_peak_and_all_exposures_are_checked_without_loss_quality_gate():
    state = state_fixture()
    h = D.check_history(state, prerequisites()[0], "G100")
    assert sum(r["studies"] for r in h) == 1200
    assert h[145]["LR"] == [3e-5] * 2 and h[-1]["LR"][0] < 3e-5
    state["history"][-1]["exam_loss"] = 1000  # Bad finite training loss is reported, not hidden tuning.
    D.check_history(state, prerequisites()[0], "G100")
    for mutation in (lambda s: s["history"][145].update(LR=[2.9e-5] * 2),
                     lambda s: s["history"][-1].update(studies=8),
                     lambda s: s["history"][3].update(encoder_updated=False),
                     lambda s: s["history"][20].update(head_gradient_norm=float("nan")),
                     lambda s: s["optimizer"]["state"][0]["exp_avg"].fill_(float("inf"))):
        changed = copy.deepcopy(state)
        mutation(changed)
        with pytest.raises(ValueError):
            D.check_history(changed, prerequisites()[0], "G100")


def test_full_recovery_equality_includes_optimizer_rng_history_and_tensor_dtype():
    state = state_fixture()
    D.equal(state, copy.deepcopy(state))
    for mutation in (lambda s: s["optimizer"]["state"][0]["exp_avg"].add_(.1),
                     lambda s: s["rng"]["numpy"].fill(0),
                     lambda s: s["rng"].update(python=(3, (4, 6))),
                     lambda s: s["history"][-1].update(loss=94.),
                     lambda s: s["model"].update(weight=s["model"]["weight"].double())):
        changed = copy.deepcopy(state)
        mutation(changed)
        with pytest.raises(ValueError):
            D.equal(state, changed)


def terminal(role, phase):
    return dict(role=role, arm="C", phase=phase, steps=1 if phase == "stop1" else 200,
                seed=W.SEED, training_epochs=100, planned_updates=200, density_weight=100,
                production_LR_horizon_updates=2916,
                fit_study_exposures={"full": 1200, "stop1": 8, "resume": 1192}[phase],
                exam_evidence_gradient=role == "C100", initial_shared_tensor_sha256=W.SHARED_INITIAL_SHA,
                original_worker_sha256=W.ORIGINAL_WORKER_SHA, routing_sha256=W.ROUTING_SHA,
                train_sha256=W.NUMERICAL_SHA["rsna/competitive/predictive/train.py"],
                density_sha256=W.NUMERICAL_SHA["rsna/competitive/predictive/density.py"],
                initial_model_sha256="a" * 64, initial_encoder_sha256="b" * 64,
                initial_head_sha256="c" * 64, final_encoder_sha256="d" * 64,
                initial_encoder_parameters_sha256="a" * 64,
                final_encoder_parameters_sha256=("a" if phase == "stop1" else "b") * 64,
                initial_head_parameters_sha256="a" * 64, final_head_parameters_sha256="b" * 64,
                final_head_sha256="e" * 64, encoder_parameters_changed=phase != "stop1",
                head_parameters_changed=True, training_only_numerical_ranges={"maximum_absolute_log_bayes_factor": 1.},
                optimizer_seconds=1., peak_GPU_bytes=10)


def test_four_phases_once_and_no_retry_after_any_uncertain_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(D.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=100))
    called = []
    def run(args, **kwargs):
        role, phase = args[-2:]
        called.append((role, phase))
        assert 0 < kwargs["timeout"] <= 5400 and kwargs["check"] is True
        path = tmp_path / "engineering" / role / phase
        path.mkdir(parents=True)
        (path / "terminal.json").write_text(json.dumps(terminal(role, phase)))
    results = D.run_phases(tmp_path, W, lambda: None, run=run)
    assert called == list(D.PHASES) and len(results) == 4
    assert sum(r["fit_study_exposures"] for r in results) == 3600
    with pytest.raises(ValueError, match="Existing phase"):
        D.run_phases(tmp_path, W, lambda: None, run=run)
    assert len(called) == 4
    failed = tmp_path / "failed"
    failed.mkdir()
    attempts = []
    def uncertain(args, **kwargs):
        attempts.append(args[-2:])
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])
    with pytest.raises(subprocess.TimeoutExpired):
        D.run_phases(failed, W, lambda: None, run=uncertain)
    assert attempts == [["C100", "full"]]


def test_role_initializer_mismatch_stops_before_stop_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(D.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=100))
    attempts = []
    def run(args, **kwargs):
        role, phase = args[-2:]
        attempts.append((role, phase))
        item = terminal(role, phase)
        if role == "G100":
            item["initial_model_sha256"] = "f" * 64
        path = tmp_path / "engineering" / role / phase
        path.mkdir(parents=True)
        (path / "terminal.json").write_text(json.dumps(item))
    with pytest.raises(ValueError, match="initialization differs"):
        D.run_phases(tmp_path, W, lambda: None, run=run)
    assert attempts == [("C100", "full"), ("G100", "full")]


def test_physical_topology_excludes_four_eight_and_wrong_gpu_nodes():
    D.topology("NodeName=evc27 Gres=gpu:nvidia_h100_pcie:2(S:0) CPUTot=64")
    for gres in ("gpu:nvidia_h100_pcie:4", "gpu:nvidia_h100_pcie:8", "gpu:nvidia_h200:2",
                 "gpu:nvidia_h100_pcie:2,gpu:nvidia_h100_pcie:2"):
        with pytest.raises(ValueError):
            D.topology("NodeName=evc27 Gres=" + gres)


def test_scope_requires_fresh_space_and_credits_only_actual_owned_bytes(tmp_path, monkeypatch):
    funding = prerequisites()[3]
    required = 20 * D.GIB + funding["other_unpaid_node_liabilities_bytes"] + 6 * D.GIB + 4 * D.MIB
    monkeypatch.setattr(D.shutil, "disk_usage", lambda _: SimpleNamespace(free=required))
    with pytest.raises(ValueError):
        D.storage(tmp_path, funding)
    (tmp_path / "binding.private.json").write_bytes(b"one byte credit")
    assert D.storage(tmp_path, funding)["controls_and_logs_bytes"] == 15
    quota = " /home/sa305415 1026232988 1048576000 1059061760 - 994627 1000000 1010000 -\n"
    with pytest.raises(ValueError):
        D.shared_capacity(quota, funding)
    assert D.shared_capacity(quota, funding, 6)["exact_paid_original_log_files"] == 6


def test_paid_log_credit_never_counts_missing_files_or_aliases(tmp_path):
    name = "rsna-predictive-full2-C0"
    job = dict(arm="C0", GPU_job="123", source_git="44cc559e947560a635c34fee7b1f2a3b7be3a241",
               node="evc27", job_name=name, stdout=str(tmp_path / (name + "-123.out")),
               stderr=str(tmp_path / (name + "-123.err")))
    assert D.paid_log_count({"C0": job}, tmp_path) == 0
    path = Path(job["stdout"])
    path.write_bytes(b"actual log")
    path.chmod(0o600)
    assert D.paid_log_count({"C0": job}, tmp_path) == 1
    Path(job["stderr"]).symlink_to(path)
    with pytest.raises(ValueError):
        D.paid_log_count({"C0": job}, tmp_path)


def test_binding_reference_hash_and_immutable_intent(tmp_path):
    path = tmp_path / "ref.json"
    D.once(path, {"one": True})
    path.chmod(0o600)
    raw = path.read_bytes()
    assert D.read_reference(dict(path=str(path), sha256=D.sha(raw))) == {"one": True}
    with pytest.raises(FileExistsError):
        D.once(path, {"one": False})
    assert path.read_bytes() == raw
    with pytest.raises(ValueError):
        D.read_reference(dict(path=str(path), sha256="f" * 64))


@pytest.mark.parametrize("role,phase", D.PHASES)
def test_actual_generated_worker_terminal_expression_matches_driver_interface(role, phase, monkeypatch):
    original = (HERE.parent / "predictive-stable-warmup-20261020-v1/worker.py.txt").read_text()
    tree = ast.parse(W.build(original))
    def expression(name):
        return ast.Expression(next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                  and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)))
    initial = {k: "a" * 64 for k in ("model", "shared", "encoder", "head", "encoder_parameters", "head_parameters")}
    initial["shared"] = W.SHARED_INITIAL_SHA
    final = {**initial, "encoder": "b" * 64, "head": "b" * 64,
             "head_parameters": "b" * 64,
             "encoder_parameters": ("a" if phase == "stop1" else "b") * 64}
    tiny = torch.nn.Linear(1, 1)
    scope = dict(role=role, phase=phase, arm="C", seed=W.SEED, initial_digests=initial,
                 final_digests=final, state={"step": 1 if phase == "stop1" else 200},
                 elapsed=1., fit_load_seconds=.1,
                 fit_loads={"full": 1200, "stop1": 8, "resume": 1192}[phase],
                 model=SimpleNamespace(encoder=tiny, parameters=tiny.parameters),
                 initial_metrics={"exam_loss": torch.tensor(.7)}, end_loss=None if phase == "stop1" else 93.,
                 case={"contexts": [{"targets": torch.zeros(4, 64)}]},
                 recipe=SimpleNamespace(density_weight=100.), torch=torch,
                 numerics={"maximum_absolute_log_bayes_factor": 1.})
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 10)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=100))
    scope["matched_worker"] = eval(compile(expression("matched_worker"), "actual_worker_binding", "eval"), scope)
    actual_terminal = eval(compile(expression("out"), "actual_worker_terminal", "eval"), scope)
    D.validate_terminal(actual_terminal, role, phase, W)
