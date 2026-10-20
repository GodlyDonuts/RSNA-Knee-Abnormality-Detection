"""Exact paired-worker derivation and the real positional objective interface."""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from rsna.competitive.predictive import train
from rsna.competitive.predictive.density import Supports, two_mask_loss
from rsna.competitive.predictive.model import Predictive

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("routed_worker_builder", HERE / "worker_builder.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def original():
    return (REPO / "initiative/research/predictive-stable-warmup-20261020-v1/worker.py.txt"
            ).read_text()


def namespace(source):
    result = dict(hashlib=hashlib, json=json)
    exec(compile(source, "prepared_worker_helper", "exec"), result)
    return result


@pytest.fixture(autouse=True)
def isolated_cpu_rng_and_threads():
    count = torch.get_num_threads()
    with torch.random.fork_rng():
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(count)


def test_exact_source_and_scientific_modules_pinned_no_silent_rebase():
    body = original()
    assert B.sha(body.encode()) == B.ORIGINAL_WORKER_SHA
    for name, expected in B.NUMERICAL_SHA.items():
        assert B.sha((REPO / name).read_bytes()) == expected
    assert B.sha((REPO / "rsna/competitive/predictive/generative_routing.py").read_bytes()
                 ) == B.ROUTING_SHA
    with pytest.raises(AssertionError, match="Exact original worker"):
        B.build(body + "\n")
    new = B.build(body)
    for before, after in reversed(B.replacements()):
        assert new.count(after) == 1
        new = new.replace(after, before, 1)
    assert new == body


def assignment(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def test_recipe_roles_paths_and_recovery_calls_preserve_original_math():
    old, new = ast.parse(original()), ast.parse(B.build(original()))
    assert ast.literal_eval(assignment(new, "arm").value) == "C"
    recipe_call = assignment(new, "recipe").value
    recipe = eval(compile(ast.Expression(recipe_call), "prepared_recipe", "eval"),
                  dict(Recipe=train.Recipe, arm="C"))
    assert recipe == train.Recipe("C", epochs=100, effective_batch=8, head_only_updates=3,
                                 schedule_updates=2916, head_lr=3e-5, density_weight=100)
    assert (12 + recipe.effective_batch - 1) // recipe.effective_batch * recipe.epochs == 200
    assert recipe.seed == B.SEED and recipe.schedule_updates == B.SCHEDULE_UPDATES
    for tree in (old, new):
        call = assignment(tree, "state").value
        # Only descriptive callback tagging changes; optimizer/recovery uses
        # the original engine, output phase, stop cursor and snapshot policy.
        call.keywords = [k for k in call.keywords if k.arg != "on_update"]
    assert ast.dump(assignment(old, "state")) == ast.dump(assignment(new, "state"))
    for role in B.ROLES:
        paths = []
        for phase in ("full", "stop1", "resume"):
            root = eval(compile(ast.Expression(assignment(new, "root").value),
                                "prepared_root", "eval"), dict(D=B.ROOT, role=role))
            paths.append(root / phase)
        assert len(set(paths)) == 3 and all(role in str(p) for p in paths)
    assert 'resume=root / "stop1/stopped.pt" if phase == "resume" else None' in B.build(original())
    assert 'for i in range(12, 16):' in B.build(original())
    assert 'assert 0 <= index_number < 16 and 0 <= epoch < 100' in B.build(original())


def test_encoder_initialization_and_existing_diagnostics_are_not_changed():
    old, new = ast.parse(original()), ast.parse(B.build(original()))
    for name in ("FixedEncoder", "mean_hook", "shift_hook", "output_hook"):
        definitions = [next(n for n in tree.body
                            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
                       for tree in (old, new)]
        assert ast.dump(definitions[0]) == ast.dump(definitions[1])
    for name in ("seed", "cls", "model", "projection"):
        assert ast.dump(assignment(old, name)) == ast.dump(assignment(new, name))
    body = B.build(original())
    assert body.count('training_engine.two_mask_loss = routed_two_mask_loss') == 1
    assert 'if role == "G100":' in body
    assert 'binding = {**binding, "matched_worker": matched_worker}' in body


def test_canonical_digest_matches_original44_algorithm_without_rng_or_state_change():
    digest = namespace(B.DIGEST_FUNCTION)["state_digests"]
    model = nn.Module()
    model.encoder = nn.Linear(2, 2)
    model.encoder.register_buffer("counter", torch.tensor(0))
    model.prior = nn.Linear(2, 1)
    model.shift = nn.Linear(2, 1)
    state, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    measured = digest(model)
    reference = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if name.split(".")[0] in {"encoder", "project", "position", "attention", "norm", "prior"}:
            reference.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                                        sort_keys=True, separators=(",", ":")).encode())
            reference.update(tensor.contiguous().numpy().tobytes())
    assert measured["shared"] == reference.hexdigest()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(t, state[n]) for n, t in model.state_dict().items())
    model.encoder.counter.add_(1)
    changed = digest(model)
    assert changed["encoder"] != measured["encoder"]
    assert changed["encoder_parameters"] == measured["encoder_parameters"]
    with torch.no_grad():
        model.encoder.weight.add_(1)
    assert digest(model)["encoder_parameters"] != measured["encoder_parameters"]


def fixture_case():
    torch.manual_seed(71)
    model = Predictive(nn.Sequential(nn.Flatten(), nn.Linear(12, 8)), 8,
                       hidden=8, descriptor_width=2).eval()
    with torch.no_grad():
        model.negative.weight.normal_(std=0.03)
        model.shift.weight.normal_(std=0.04)
    supports = Supports(torch.tensor([0, 2, 4]), torch.tensor([0, 1, 2, 3]),
                        torch.zeros(12, 2), 4)
    contexts = [dict(images=torch.rand(2, 3, 2, 2), token_metadata=torch.rand(2, 8),
                     region_metadata=torch.rand(4, 8), targets=torch.rand(4, 2),
                     observed=torch.tensor(o), supports=supports)
                for o in ([True, False, True, False], [False, True, False, True])]
    return model, dict(contexts=contexts, labels=torch.full((1, 12), 0.3),
                       mask=torch.ones(1, 12, dtype=torch.bool))


def test_real_positional_train_objective_routes_only_evidence_exam_gradients(monkeypatch):
    model, case = fixture_case()
    recipe = train.Recipe("C", epochs=100, density_weight=100, schedule_updates=2916,
                          head_only_updates=3, head_lr=3e-5)
    original_loss, original_metrics = train.objective(model, case, recipe, torch.device("cpu"))
    monkeypatch.setattr(train, "two_mask_loss", two_mask_loss)
    ns = {"role": "G100"}
    exec(B.ROUTING_ADAPTER, ns)
    assert train.two_mask_loss is ns["routed_two_mask_loss"]
    routed_loss, routed_metrics = train.objective(model, case, recipe, torch.device("cpu"))
    assert torch.equal(original_loss, routed_loss)
    assert all(torch.equal(original_metrics[k], routed_metrics[k]) for k in original_metrics)
    parameters = tuple(model.parameters())
    original_grad = torch.autograd.grad(original_metrics["exam_loss"], parameters,
                                       allow_unused=True, retain_graph=True)
    exam_grad = torch.autograd.grad(routed_metrics["exam_loss"], parameters,
                                   allow_unused=True, retain_graph=True)
    total_grad = torch.autograd.grad(routed_loss, parameters, allow_unused=True, retain_graph=True)
    density_grad = torch.autograd.grad(100 * routed_metrics["density_loss"], parameters,
                                      allow_unused=True)
    for i, (name, _) in enumerate(model.named_parameters()):
        if name.startswith(("negative.", "shift.")):
            assert original_grad[i] is not None and original_grad[i].abs().sum() > 0
            assert exam_grad[i] is None
            assert torch.equal(total_grad[i], density_grad[i])
        if name.startswith("prior."):
            assert torch.equal(original_grad[i], exam_grad[i])
    with pytest.raises(AssertionError):
        ns["routed_two_mask_loss"](None, None, None, None, None, 64, 0.1)


def test_C100_keeps_original_objective_and_implicit_coefficient_is_rejected(monkeypatch):
    monkeypatch.setattr(train, "two_mask_loss", two_mask_loss)
    ns = {"role": "C100"}
    exec(B.ROUTING_ADAPTER, ns)
    assert train.two_mask_loss is two_mask_loss
    with pytest.raises(AssertionError):
        ns["routed_two_mask_loss"](None, None, None, None, None)
