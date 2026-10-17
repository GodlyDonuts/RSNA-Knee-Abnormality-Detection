import base64
import copy

import pytest

from scripts.kaggle import licensed_importance_production as production


@pytest.fixture
def inputs(monkeypatch):
    worker = 'def _whole_model(base):return base\ndef main():raise RuntimeError("diagnostic only")\nif __name__=="__main__":main()\n'
    source = (
        "WORKER="
        + repr(base64.b64encode(worker.encode()).decode())
        + "\nPERMISSION="
        + repr(base64.b64encode(b"synthetic permission").decode())
        + "\n"
    )
    monkeypatch.setattr(production, "RECOVERY_SHA", production.digest(source))
    monkeypatch.setattr(production, "WORKER_SHA", production.digest(worker))
    files = {
        n: "# synthetic compile fixture\n"
        for n in (*production.ACCEPTED_MODULES, *production.POPULATION_MODULES)
    }
    monkeypatch.setattr(
        production,
        "ACCEPTED_MODULES",
        {n: production.digest(files[n]) for n in production.ACCEPTED_MODULES},
    )
    monkeypatch.setattr(
        production,
        "POPULATION_MODULES",
        {n: production.digest(files[n]) for n in production.POPULATION_MODULES},
    )
    monkeypatch.setattr(production, "validate_proofs", lambda x: None)
    r = {
        "status": "actual_matched_uniform_influence_full_encoder_twoT4_MRI_recovery_accepted",
        "worker_source_sha256": production.WORKER_SHA,
        "source_module_sha256": production.ACCEPTED_MODULES,
        "fit_studies": 16,
        "disjoint_engineering_studies": 16,
        "updates_per_role": 4,
        "same_initial_complete_model_and_allrank_RNG_between_roles": True,
        "distinct_final_learned_tensors": 3,
        "roles": {},
    }
    for role in ("uniform", "influence"):
        r["roles"][role] = {
            "complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal": True,
            "all16_full94_two_view_prediction_bytes_equal": True,
            "devices": [
                {
                    "GPU": "Tesla T4",
                    "trainable_parameters": 73136096,
                    "all_BN_buffers_equal_parent": True,
                }
                for _ in range(2)
            ],
        }
    actual = {
        "status": "COMPLETE",
        "kernel_id": 139130771,
        "version": 1,
        "source_sha256": production.RECOVERY_SHA,
        "receipt": r,
    }
    proposal = {
        "status": "all3887_fixed_fitting_teacher_influence_proposals_CPU_accepted",
        "population": 3887,
        "population_sha256": production.POPULATION_SHA,
        "shape": [3887, 94],
        "dtype": "float64",
        "uniform_fraction": 0.25,
        "proposal_table": {"bytes": 2923152, "sha256": production.PROPOSAL_SHA},
        "proposal_module_sha256": production.ACCEPTED_MODULES["rsna/competitive/mil_sampling.py"],
        "clinical_labels_reports_Gold_confirmation_read": False,
        "new_CNN_GPU_or_optimizer": False,
    }
    return source, actual, files, [], proposal, "b" * 40


def prior(meta, segment):
    return {
        "status": "actual_matched_IID16_full_parent_epoch_session_complete",
        "role": meta["role"],
        "segment": segment,
        "completed": 486 * segment,
        "worker_sha256": meta["worker_sha256"],
        "binding": meta["binding"],
        "checkpoint": {"bytes": 100, "sha256": "a" * 64},
        "run": {"bytes": 50, "sha256": "b" * 64},
    }


def test_checkpoint_continuous_worker_and_recipe_stay_fixed_per_role(inputs):
    for role in ("uniform", "influence"):
        source, first = production.build(*inputs, role, 1)
        _, second = production.build(*inputs, role, 2, previous=prior(first, 1))
        _, third = production.build(*inputs, role, 3, previous=prior(second, 2))
        assert first["worker_sha256"] == second["worker_sha256"] == third["worker_sha256"]
        assert first["binding"] == second["binding"] == third["binding"]
        assert first["maximum_GPU_seconds"] == 6000
        assert third["completed_target"] == 1458
        compile(source, "synthetic_compilation_only", "exec")


@pytest.mark.parametrize(
    "bad",
    [
        "pending_GPU",
        "different_initial",
        "partial_model",
        "proposals",
        "module",
        "cross_role",
        "missing_checkpoint",
        "first_resume",
    ],
)
def test_unqualified_or_cross_role_inputs_cannot_compile_production(inputs, bad):
    values = list(copy.deepcopy(inputs))
    source, meta = production.build(*values, "uniform", 1)
    previous = prior(meta, 1)
    segment = 2
    if bad == "pending_GPU":
        values[1]["status"] = "RUNNING"
    elif bad == "different_initial":
        values[1]["receipt"]["same_initial_complete_model_and_allrank_RNG_between_roles"] = False
    elif bad == "partial_model":
        values[1]["receipt"]["roles"]["influence"]["devices"][0]["trainable_parameters"] = 1024
    elif bad == "proposals":
        values[4]["proposal_table"]["sha256"] = "a" * 64
    elif bad == "module":
        values[2][next(iter(values[2]))] += "# changed numerical module\n"
    elif bad == "cross_role":
        previous["role"] = "influence"
    elif bad == "missing_checkpoint":
        previous.pop("checkpoint")
    else:
        segment = 1
    with pytest.raises(ValueError):
        production.build(*values, "uniform", segment, previous=previous)
