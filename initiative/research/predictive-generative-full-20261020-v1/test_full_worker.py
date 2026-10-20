"""Prepared full worker's loss interface, immutable inputs, and distinct receipts."""

import ast
import copy
import hashlib
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rsna.competitive.predictive import train
from rsna.competitive.predictive.density import two_mask_loss
from rsna.competitive.predictive.generative_routing import generative_two_mask_loss

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("G100_full_derivation", HERE / "derive.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
c = d.local_contract(REPO)


@pytest.fixture(scope="module")
def files():
    return d.build(REPO)


def test_original_science_graph_forward_initialization_and_fit_are_preserved(files):
    for name, digest in c.base.NUMERICAL_SHA.items():
        assert hashlib.sha256(files[name].encode()).hexdigest() == digest
    raw = subprocess.check_output(
        ["git", "show", c.base.BASE + ":" + c.base.ORIGINAL + "worker.py.txt"], cwd=REPO
    )
    old = ast.parse(c.base.derive_worker(raw))
    new = ast.parse(files["worker.py"])

    def main(tree):
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")

    for name in ("Encoder", "Cases"):
        nodes = [
            next(n for n in main(t).body if isinstance(n, ast.ClassDef) and n.name == name)
            for t in (old, new)
        ]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1])
    for name in ("model", "recipe", "state"):
        nodes = [
            next(
                n
                for n in main(t).body
                if isinstance(n, ast.Assign)
                and any(isinstance(x, ast.Name) and x.id == name for x in n.targets)
            )
            for t in (old, new)
        ]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1])
    assert "COHORT = 3887" in files["worker.py"] and "EPOCHS = 6" in files["worker.py"]
    assert "predictive-G100-full3887-" in files["publish_epoch.py"]
    assert "predictive-C100-full3887-" not in files["publish_epoch.py"]
    assert 'funding["branch"] == "G100"' in files["publish_epoch.py"]


def test_exact_new_full_worker_calls_real_positional_routing_interface(files):
    tree = ast.parse(files["worker.py"])
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    function = next(
        n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "routed_two_mask_loss"
    )
    ns = {"generative_two_mask_loss": generative_two_mask_loss}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "actual_full_G100_adapter", "exec"),
        ns,
    )
    helper = REPO / "tests/test_predictive_generative_routing.py"
    spec = importlib.util.spec_from_file_location("G100_test_inputs", helper)
    inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inputs)
    args = inputs.example()
    original, original_parts = two_mask_loss(*args[:5], 2, 100.0)
    actual, parts = ns["routed_two_mask_loss"](*args[:5], 2, 100.0)
    assert torch.equal(actual, original)
    assert all(torch.equal(parts[k], original_parts[k]) for k in parts)
    assert all(
        v is None
        for v in torch.autograd.grad(
            parts["exam_loss"], args[-1], allow_unused=True, retain_graph=True
        )
    )
    observed = torch.autograd.grad(actual, args[-1], retain_graph=True)
    expected = torch.autograd.grad(100 * original_parts["density_loss"], args[-1])
    assert all(torch.equal(a, b) for a, b in zip(observed, expected))
    with pytest.raises(AssertionError):
        ns["routed_two_mask_loss"](*args[:5], 2, 0.1)
    assert train.two_mask_loss is two_mask_loss  # Local inspection never patches the real trainer.


def test_publisher_loads_its_pinned_helper_without_ambient_python_path(files, tmp_path):
    tree = ast.parse(files["publish_epoch.py"])
    helper = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "g100_provenance"
    )
    bodies = {
        "base_C100_derivation.py": 'MARKER = "original helper"\n',
        "generative_contract.py": (
            "import base_C100_derivation as base\n"
            "def validate_references(binding):\n"
            '    return {"branch": "G100", "helper": base.MARKER}\n'
        ),
    }
    for name, body in bodies.items():
        (tmp_path / name).write_text(body)
    binding = dict(
        code_root=str(tmp_path),
        code_sha256={
            name: hashlib.sha256(body.encode()).hexdigest() for name, body in bodies.items()
        },
    )

    def reference(ref):
        path = Path(ref["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]
        return path

    prior = sys.modules.get("base_C100_derivation")
    ns = dict(Path=Path, reference=reference, importlib=importlib, sys=sys)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "actual_publisher_helper", "exec"), ns)
    assert ns["g100_provenance"](binding) == {"branch": "G100", "helper": "original helper"}
    assert sys.modules.get("base_C100_derivation") is prior
    (tmp_path / "base_C100_derivation.py").write_text("MARKER = 'changed'\n")
    with pytest.raises(AssertionError):
        ns["g100_provenance"](binding)


def fixture(files):
    history = []
    for i in range(200):
        fraction = i / 2915
        scale = (
            min(1.0, (i + 1) / (0.05 * 2916))
            if fraction < 0.05
            else (0.1 + 0.9 * (1 + math.cos(math.pi * (fraction - 0.05) / 0.95)) / 2)
        )
        history.append(
            dict(
                step=i + 1,
                studies=8 if i % 2 == 0 else 4,
                encoder_updated=i >= 3,
                LR=[3e-5 * scale] * 2,
                **dict.fromkeys(
                    (
                        "loss",
                        "exam_loss",
                        "density_loss",
                        "gradient_norm",
                        "encoder_gradient_norm",
                        "head_gradient_norm",
                    ),
                    0.5,
                ),
            )
        )
    pilot = dict(
        status="actual_matched_C100_G100_through_peak_engineering_complete",
        GPU_job="986011",
        source_git=c.PILOT_SOURCE,
        binding_sha256=c.PILOT_BINDING,
        branch="matched_C100_G100_generative_routing",
        actual_optimizer_updates=600,
        fresh_initialization_equal=True,
        G100_full_stop_resume_complete_state_and_prediction_bytes_equal=True,
        primary_B_C_C0_C100_pipeline_unchanged=True,
        automatic_full_training_promotion=False,
        clinical_or_held_out_accuracy_claim=False,
        roles=[
            dict(role=r, real_optimizer_history=copy.deepcopy(history)) for r in ("C100", "G100")
        ],
    )
    binding = dict(
        arm="C",
        branch="G100",
        base_science_git=c.base.BASE,
        node="evc27",
        source_git="a" * 40,
        run_root=c.RUN,
        code_root=c.ROOT + "/code",
        expected_initial_shared_tensor_sha256=c.base.INITIAL,
        generative_pilot=dict(path=c.PILOT_PATH, sha256="b" * 64),
        code_sha256={n: hashlib.sha256(v.encode()).hexdigest() for n, v in files.items()},
        original_portable_admission_sha256="c" * 64,
        original_C0_binding_sha256="d" * 64,
    )
    scheduler = dict(GPU_job="986011", state="COMPLETED", exit_code="0:0", node="evc27")
    decision = dict(
        status="accepted_full_predictive_fitting_recipe",
        branch="G100",
        pilot_scheduler=scheduler,
        pilot_acceptance_sha256="b" * 64,
        source_git="a" * 40,
        code_sha256=binding["code_sha256"],
        recipe=c.base.recipe(),
        observed_C_C0_gradient_risk_reviewed=True,
        matched_peak_LR_diagnostics_reviewed=True,
        primary_B_C_C0_unchanged=True,
        primary_Kaggle_reserve_seconds=3360,
        engineering_predictions_used_for_selection=False,
        automatic_recipe_retuning=False,
        agent_reasoning="Synthetic fixture only; no actual admission or remote run. " * 3,
    )
    original_decision = dict(decision)
    original_decision["status"] = "accepted_separate_full_G100_after_actual_pilot_review"
    decision["full_G100_agent_decision_sha256"] = c.base.sha(c.base.canonical(original_decision))
    funding = dict(
        branch="G100",
        arm="C",
        run_root=c.RUN,
        arm_cap_bytes=12 * 2**30,
        pilot_scope_retained=True,
        all_other_obligations_retained=True,
        no_deletion_credit=True,
    )
    admitted = dict(
        branch="G100",
        source_git="a" * 40,
        code_sha256=binding["code_sha256"],
        new_MRI_transferred_bytes=0,
        original_portable_admission_sha256="c" * 64,
        original_C0_binding_sha256="d" * 64,
    )
    return binding, pilot, decision, funding, admitted


def test_distinct_g100_provenance_and_no_relabelling_old_admission(files):
    args = fixture(files)
    provenance = c.validate(*args)
    assert provenance["branch"] == "G100"
    assert provenance["exam_gradients_through_Bayes_factor"] is False
    assert provenance["proper_conditional_density_gradients_attached"] is True
    for index, field, value in [
        (0, "branch", "C100"),
        (3, "arm_cap_bytes", 6 * 2**30),
        (4, "branch", "C100"),
        (2, "engineering_predictions_used_for_selection", True),
    ]:
        bad = copy.deepcopy(args)
        bad[index][field] = value
        with pytest.raises(ValueError):
            c.validate(*bad)


@pytest.mark.parametrize(
    "failure", ["pending", "wrong_job", "partial", "nonfinite", "wrong_lr", "no_recovery"]
)
def test_original_actual_pilot_required(files, failure):
    args = list(fixture(files))
    p = args[1]
    s = args[2]["pilot_scheduler"]
    if failure == "pending":
        s["state"] = "PENDING"
    if failure == "wrong_job":
        p["GPU_job"] = "981620"
    if failure == "partial":
        p["roles"][1]["real_optimizer_history"].pop()
    if failure == "nonfinite":
        p["roles"][1]["real_optimizer_history"][145]["exam_loss"] = float("nan")
    if failure == "wrong_lr":
        p["roles"][1]["real_optimizer_history"][145]["LR"][0] = 1e-4
    if failure == "no_recovery":
        p["G100_full_stop_resume_complete_state_and_prediction_bytes_equal"] = False
    with pytest.raises(ValueError):
        c.validate(*args)
