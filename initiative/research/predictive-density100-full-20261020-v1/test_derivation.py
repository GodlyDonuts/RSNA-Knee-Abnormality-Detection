"""CPU-only exact-source, admission, publication, and passive-observer checks."""

import ast
import copy
import importlib.util
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d = load("C100_derivation_test", HERE / "derive.py")
t = load("C100_telemetry_test", HERE / "telemetry.py")


@pytest.fixture(scope="module")
def sources():
    return d.build(REPO)


def original(name):
    return subprocess.check_output(
        ["git", "show", d.BASE + ":" + d.ORIGINAL + name], cwd=REPO
    )


def function_tree(source, name):
    return ast.dump(next(n for n in ast.parse(source).body
                         if isinstance(n, ast.FunctionDef) and n.name == name),
                    include_attributes=False)


def fixture():
    binding = dict(
        arm="C", branch="C100", base_science_git=d.BASE, node="evc27",
        source_git="a" * 40, code_root=d.ROOT + "/code", run_root=d.RUN,
        code_sha256=dict(d.NUMERICAL_SHA), expected_initial_shared_tensor_sha256=d.INITIAL,
        density100_pilot={"path": d.PILOT_PATH, "sha256": "1" * 64},
        density100_pilot_source_git=d.PILOT_SOURCE,
        original_portable_admission_sha256="2" * 64, original_C0_binding_sha256="3" * 64,
    )
    pilot = dict(
        status="actual_C100_full_stop_resume_engineering_complete",
        GPU_job="981620", source_git=d.PILOT_SOURCE, binding_sha256=d.PILOT_BINDING,
        branch="C100", full_stop_resume_complete_state_and_prediction_bytes_equal=True,
        automatic_full_training_promotion=False, clinical_or_held_out_accuracy_claim=False,
        primary_B_C_C0_pipeline_unchanged=True,
        original_numerical_source_git="39e315c65546aa5205518713bc27349ec63e7538",
        objective_measurement_sha256=
            "6f10fc5a21e8149731e1600088e601b97c16be42b0a9e944baa05c67f6920e2a",
        original_engineering_receipt_sha256=
            "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10",
        engineering_warnings={"synthetic_only": True},
        terminal={"density_weight": 100.0, "steps": 24},
        real_optimizer_history=[
            dict(step=i, studies=8 if i % 2 else 4, encoder_updated=i > 3,
                 **dict.fromkeys(("loss", "exam_loss", "density_loss", "encoder_gradient_norm",
                                  "head_gradient_norm", "gradient_norm"), 0.5))
            for i in range(1, 25)
        ],
    )
    decision = dict(
        status="accepted_full_predictive_fitting_recipe",
        pilot_scheduler={"GPU_job": "981620", "state": "COMPLETED",
                         "exit_code": "0:0", "node": "evc27"},
        pilot_acceptance_sha256=binding["density100_pilot"]["sha256"],
        source_git=binding["source_git"], code_sha256=binding["code_sha256"],
        recipe=d.recipe(), pilot_warnings_reviewed=pilot["engineering_warnings"],
        observed_C_C0_gradient_risk_reviewed=True, primary_B_C_C0_unchanged=True,
        primary_Kaggle_reserve_seconds=3360, engineering_predictions_used_for_selection=False,
        automatic_recipe_retuning=False, agent_reasoning="Synthetic admission test only. " * 4,
    )
    bind_decision(decision)
    funding = dict(branch="C100", arm="C", run_root=d.RUN, pilot_scope_retained=True)
    admitted = dict(
        branch="C100", source_git=binding["source_git"], code_sha256=binding["code_sha256"],
        new_MRI_transferred_bytes=0, original_portable_admission_sha256="2" * 64,
        original_C0_binding_sha256="3" * 64,
    )
    return binding, pilot, decision, funding, admitted


def bind_decision(decision):
    value = {k: v for k, v in decision.items() if k != "full_C100_agent_decision_sha256"}
    value["status"] = "accepted_separate_full_C100_after_actual_pilot_review"
    decision["full_C100_agent_decision_sha256"] = d.sha(d.canonical(value))


def test_exact_science_only_density_changes_and_original_publication_preserved(sources):
    actual = json.loads((REPO / d.ORIGINAL / "actual-cpu-prepared.json").read_text())
    assert actual["CPU_job"] == "979667" and actual["source_git"] == d.BASE
    expected = {n: h for n, h in actual["code_sha256"].items()
                if n.startswith("rsna/") or n == "alias_provenance.py"}
    assert d.NUMERICAL_SHA == expected
    for name, expected in d.NUMERICAL_SHA.items():
        assert d.sha(sources[name].encode()) == expected
    old = ast.parse(original("worker.py.txt"))
    new = ast.parse(sources["worker.py"])
    calls = []
    for tree in (old, new):
        calls.append(next(n for n in ast.walk(tree)
                          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                          and n.func.id == "Recipe"))
    before, after = calls
    before.keywords = [k for k in before.keywords if k.arg != "density_weight"]
    density = next(k.value for k in after.keywords if k.arg == "density_weight")
    after.keywords = [k for k in after.keywords if k.arg != "density_weight"]
    assert ast.literal_eval(density) == 100
    assert ast.dump(before) == ast.dump(after)
    for name in ("validate_state", "budget", "upload_and_verify"):
        assert function_tree(original("publish_epoch.py.txt"), name) == function_tree(
            sources["publish_epoch.py"], name
        )
    assert "predictive-C100-full3887-" in sources["publish_epoch.py"]
    assert "**branch" in sources["publish_epoch.py"]
    assert 'monitor.finish_update(record["step"], record["studies"])' in sources["worker.py"]


@pytest.mark.parametrize("kind", ["worker", "publisher"])
def test_different_source_cannot_be_derived(kind):
    raw = original("worker.py.txt" if kind == "worker" else "publish_epoch.py.txt")
    with pytest.raises(ValueError, match="Exact"):
        (d.derive_worker if kind == "worker" else d.derive_publisher)(raw + b"\n")


def test_actual_pilot_decision_binding_and_clean_provenance():
    values = fixture()
    result = d.validate(*values)
    assert set(result) == set(d.PROVENANCE_KEYS)
    assert result["density100_pilot_GPU_job"] == "981620"
    assert not any("path" in key or "label" in key or "prediction" in key for key in result)


@pytest.mark.parametrize("which,key,value", [
    (0, "branch", "C"),
    (0, "expected_initial_shared_tensor_sha256", "x" * 64),
    (1, "status", "PENDING"),
    (1, "source_git", "b" * 40),
    (1, "full_stop_resume_complete_state_and_prediction_bytes_equal", False),
    (2, "pilot_acceptance_sha256", "f" * 64),
    (2, "engineering_predictions_used_for_selection", True),
    (3, "branch", "C0"),
    (4, "original_C0_binding_sha256", "e" * 64),
])
def test_reject_unaccepted_or_changed_branch(which, key, value):
    values = fixture()
    values[which][key] = value
    with pytest.raises(ValueError):
        d.validate(*values)


def test_reject_changed_math_or_pilot_nonfinite():
    values = fixture()
    values[0]["code_sha256"]["rsna/competitive/predictive/model.py"] = "f" * 64
    with pytest.raises(ValueError, match="numerical"):
        d.validate(*values)
    values = fixture()
    values[1]["real_optimizer_history"][0]["density_loss"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        d.validate(*values)


def test_actual_reference_bytes_read_not_metadata_only(tmp_path, monkeypatch):
    values = list(fixture())
    binding = values[0]
    for name, item in zip(
        ("density100_pilot", "engineering_decision", "funding", "input_admission"), values[1:]
    ):
        if name == "engineering_decision":
            item["pilot_acceptance_sha256"] = binding["density100_pilot"]["sha256"]
            bind_decision(item)
        raw = d.canonical(item)
        p = tmp_path / (name + ".json")
        p.write_bytes(raw)
        p.chmod(0o600)
        binding[name] = {"path": str(p), "sha256": d.sha(raw)}
    monkeypatch.setattr(d, "PILOT_PATH", binding["density100_pilot"]["path"])
    assert d.validate_references(binding)["branch"] == "C100"
    Path(binding["density100_pilot"]["path"]).write_text('{"status":"PENDING"}')
    with pytest.raises(ValueError, match="reference changed"):
        d.validate_references(binding)


def test_generated_publisher_demands_branch_fields_and_density100(sources):
    ns = {"__name__": "generated_C100_publisher"}
    exec(compile(sources["publish_epoch.py"], "derived_publisher", "exec"), ns)
    binding, pilot, decision, funding, admitted = fixture()
    branch = d.validate(binding, pilot, decision, funding, admitted)
    ns["c100_provenance"] = lambda b: d.validate(b, pilot, decision, funding, admitted)
    binding.update(
        schema="predictive_full3887_one_arm_v1", source_revision=ns["SOURCE_REVISION"],
        runtime_versions={"torch": "synthetic"}, renderer_sha256={},
        accepted_renderer_sha256=[{}], renderer_provenance_counts={"synthetic": 3887},
        source_context_index_sha256="4" * 64, alias_regression={"sha256": "5" * 64},
    )
    for name, value in (
        ("parent_weight", ns["PARENT"]), ("parent_source", ns["PARENT_SOURCE"]),
        ("license", ns["PERMISSION"]), ("targets", ns["TARGETS"]),
        *((name, "6" * 64) for name in ("context_index", "runtime_receipt",
                                      "engineering_decision", "funding", "input_admission")),
    ):
        binding[name] = {"sha256": value}
    start = dict(
        **branch, arm="C", source_git=binding["source_git"], binding_sha256="7" * 64,
        recipe=d.recipe(), parent_weight_sha256=ns["PARENT"],
        parent_source_sha256=ns["PARENT_SOURCE"], source_revision=ns["SOURCE_REVISION"],
        targets_sha256=ns["TARGETS"], fitting_order_sha256=ns["ROSTER"],
        license_sha256=ns["PERMISSION"], arm_cap_bytes=12 * 2**30,
        parameter_count=ns["PARAMETERS"]["C"], clinical_accuracy_or_Kaggle_score=False,
        initial_shared_tensor_sha256=d.INITIAL, model_schema_sha256="8" * 64,
        runtime_versions=binding["runtime_versions"], renderer_sha256={},
        accepted_renderer_sha256=[{}], renderer_provenance_counts={"synthetic": 3887},
        alias_regression_receipt_sha256="5" * 64, source_context_index_sha256="4" * 64,
        original_numerical_source_git="b361ee40939d875d3adf82d5e04dbf2d7d62b0a0",
    )
    start.update({n + "_sha256": "6" * 64 for n in (
        "context_index", "runtime_receipt", "engineering_decision", "funding", "input_admission"
    )})
    side = dict(
        **start, status="immutable_full_state_epoch_ready_for_private_publication",
        epoch=6, completed_updates=2916, cohort_size=3887, checkpoint_name="epoch-006.pt",
        checkpoint_bytes=100, checkpoint_stat=[1, 100, 2, 3],
        contains_model_optimizer_recipe_binding_counters_history_all_RNG=True,
        independent_private_publication_verified=False,
    )
    ns["validate_binding"](binding, start, side, 6, "7" * 64)
    for key, changed in (("branch", "C"), ("density100_pilot_sha256", "f" * 64),
                         ("recipe", {**d.recipe(), "density_weight": 0.1})):
        bad_start, bad_side = {**start, key: changed}, {**side, key: changed}
        with pytest.raises(ValueError):
            ns["validate_binding"](binding, bad_start, bad_side, 6, "7" * 64)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b) and a.dtype == b.dtype
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def test_passive_monitor_preserves_complete_training_optimizer_rng_and_outputs(tmp_path):
    from rsna.competitive.predictive.density import Supports
    from rsna.competitive.predictive.model import Predictive
    from rsna.competitive.predictive.train import Recipe, fit, restore_rng, rng_state

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(23)
        model = Predictive(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 4)),
                           4, hidden=4, classes=2, descriptor_width=3)
        support = Supports(torch.tensor([0, 1, 2]), torch.tensor([0, 1]), torch.zeros(2, 2), 2)
        contexts = [dict(
            images=torch.rand(2, 3, 2, 2), token_metadata=torch.rand(2, 8),
            region_metadata=torch.rand(2, 8), targets=torch.rand(2, 3) / 20,
            observed=torch.tensor([True, True]), supports=support,
        ) for _ in range(2)]
        initial = copy.deepcopy(rng_state())
        recipe = Recipe("C", epochs=3, effective_batch=2, head_lr=3e-5,
                        head_only_updates=3, density_weight=100, schedule_updates=2916)
        results, summaries = [], []
        for enabled in (False, True):
            restore_rng(initial)
            current = copy.deepcopy(model)
            monitor = t.Monitor(current, selected=(1, 4), epoch_steps=2) if enabled else None
            history = []

            class Data:
                def __len__(self):
                    return 3

                def case(self, index, _epoch):
                    random.random()
                    np.random.rand()
                    torch.rand(1)
                    labels, mask = torch.tensor([[index % 2, 0.25]]), torch.tensor([[True, False]])
                    if monitor:
                        monitor.begin_case(len(history) + 1, labels, mask)
                    return dict(contexts=contexts, labels=labels, mask=mask)

            def on_update(record):
                history.append(record)
                if monitor:
                    result = monitor.finish_update(record["step"], record["studies"])
                    if result:
                        summary = result["C100_passive_evidence"]
                        assert summary["BCE_full"] == pytest.approx(record["exam_loss"], abs=1e-7)
                        summaries.append(summary)

            state = fit(current, Data(), recipe, binding={"synthetic_only": True},
                        output=tmp_path / str(enabled), on_update=on_update)
            if monitor:
                monitor.close()
            results.append(copy.deepcopy(state))
        equal(*results)
        assert len(summaries) == 4
        assert summaries[0]["studies"] == 2 and summaries[1]["studies"] == 1
        assert summaries[-1]["contexts"] == 2
    finally:
        torch.set_num_threads(old_threads)


def test_passive_metrics_use_mean_posterior_and_masked_BCE():
    class Emitting(torch.nn.Module):
        def forward(self, a, bf):
            return {"context_logits": a, "evidence": SimpleNamespace(log_bayes_factor=bf[None])}

    model = Emitting()
    monitor = t.Monitor(model)
    monitor.begin_case(1, torch.tensor([[1., float("nan")]]), torch.tensor([[True, False]]))
    first = model(torch.tensor([-2., 0.]), torch.tensor([4., 1.]))
    model(torch.tensor([1., 0.]), torch.tensor([-3., -1.]))
    row = monitor.finish_update(1, 1)["C100_passive_evidence"]
    assert row["BCE_full"] == pytest.approx(-np.log(0.5), abs=1e-7)
    assert row["BF_RMS"] == pytest.approx((27 / 4)**0.5)
    assert row["BF_absolute_max"] == 4 and row["observed_targets"] == 1
    assert torch.equal(first["context_logits"], torch.tensor([-2., 0.]))
    monitor.close()
