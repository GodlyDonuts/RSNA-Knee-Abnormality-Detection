import ast
import base64
import copy
from dataclasses import asdict

import pytest
import test_licensed_importance_production as fixture_source

from rsna.competitive import encoder_lr_epoch as validator
from rsna.competitive.licensed_importance import VERSION, Recipe
from rsna.competitive.licensed_parent_train import lr_factor
from scripts.kaggle import encoder_lr_production as treatment
from scripts.kaggle import licensed_importance_production as control


@pytest.fixture
def matched_inputs(monkeypatch):
    return fixture_source.inputs.__wrapped__(monkeypatch)


def prior(meta, epoch):
    return {
        "status": treatment.STATUS,
        "role": "influence",
        "segment": epoch,
        "completed": 486 * epoch,
        "worker_sha256": meta["worker_sha256"],
        "binding": meta["binding"],
        "checkpoint": {"bytes": 100, "sha256": "a" * 64},
        "run": {"bytes": 50, "sha256": "b" * 64},
    }


def test_only_worker_recipe_and_provenance_change(matched_inputs):
    old_source, _ = control.build(*matched_inputs, "influence", 1)
    source, meta = treatment.build(*matched_inputs, segment=1)
    old_values, values = treatment.literals(old_source), treatment.literals(source)
    old_worker = base64.b64decode(old_values["WORKER"]).decode()
    worker = base64.b64decode(values["WORKER"]).decode()
    restored = worker.replace(
        "_BINDING=" + repr(values["BINDING"]), "_BINDING=" + repr(old_values["BINDING"])
    )
    restored = restored.replace(
        "Recipe(epochs=3,role=_ROLE,encoder_lr=3e-5)", "Recipe(epochs=3,role=_ROLE)"
    )
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(old_worker))
    assert meta["maximum_GPU_seconds"] == 6000
    assert values["BINDING"]["actual_new_LR_recovery_claim"] is False
    second, m2 = treatment.build(*matched_inputs, segment=2, previous=prior(meta, 1))
    _, m3 = treatment.build(*matched_inputs, segment=3, previous=prior(m2, 2))
    assert meta["worker_sha256"] == m2["worker_sha256"] == m3["worker_sha256"]
    assert meta["binding"] == m2["binding"] == m3["binding"]
    assert treatment.literals(second)["SEGMENT"] == 2 and m3["completed_target"] == 1458


@pytest.mark.parametrize("bad", ["control", "binding", "role", "missing", "sha", "start", "epoch"])
def test_no_cross_experiment_or_unbound_resume(matched_inputs, bad):
    _, meta = treatment.build(*matched_inputs, segment=1)
    p = prior(meta, 1)
    segment = 2
    if bad == "control":
        p["status"] = "actual_matched_IID16_full_parent_epoch_session_complete"
    elif bad == "binding":
        p["binding"] = {}
    elif bad == "role":
        p["role"] = "uniform"
    elif bad == "missing":
        p.pop("checkpoint")
    elif bad == "sha":
        p["checkpoint"]["sha256"] = "z" * 64
    elif bad == "start":
        segment = 1
    else:
        segment = 4
    with pytest.raises(ValueError):
        treatment.build(*matched_inputs, segment=segment, previous=p)


def state():
    recipe = Recipe(epochs=3, role="influence", encoder_lr=3e-5)
    return {
        "binding": {
            "production_plan": validator.PLAN,
            "role": "influence",
            "version": VERSION,
            "recipe": asdict(recipe),
            "population_sha256": validator.POPULATION_SHA,
            "single_factor": {"encoder_lr_control": 3e-6, "encoder_lr_treatment": 3e-5},
            "proposal": {
                "proposal_table": {
                    "bytes": 2923152,
                    "sha256": "e812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e",
                }
            },
            "actual_recovery": {"kernel_id": 139130771},
        },
        "history": [
            {
                "encoder_lr": 3e-5 * lr_factor(i, 1458, recipe),
                "head_lr": 3e-5 * lr_factor(i, 1458, recipe),
            }
            for i in range(486)
        ],
    }


def test_validator_passes_complete_state_to_original_strict_checker(monkeypatch):
    s = state()
    called = []

    def complete(a, b, c, **kwargs):
        called.append(kwargs)
        assert a is s
        return {"completed": 486}

    monkeypatch.setattr(validator, "complete", complete)
    assert validator.verify(s, {}, {"role": "influence"})["encoder_lr"] == 3e-5
    assert called == [
        {
            "expected_parameters": 73136096,
            "expected_windows": 16,
            "expected_status": treatment.STATUS,
        }
    ]


@pytest.mark.parametrize("bad", ["LR", "head", "history", "proposal", "plan"])
def test_validator_rejects_other_recipes_before_state_admission(monkeypatch, bad):
    s = copy.deepcopy(state())

    def forbidden(*args, **kwargs):
        raise AssertionError("wrong recipe reached complete-state admission")

    monkeypatch.setattr(validator, "complete", forbidden)
    if bad == "LR":
        s["binding"]["recipe"]["encoder_lr"] = 3e-6
    elif bad == "head":
        s["binding"]["recipe"]["head_lr"] = 3e-4
    elif bad == "history":
        s["history"][5]["encoder_lr"] /= 10
    elif bad == "proposal":
        s["binding"]["proposal"] = {}
    else:
        s["binding"]["production_plan"] = "other"
    with pytest.raises(ValueError):
        validator.verify(s, {}, {"role": "influence"})
