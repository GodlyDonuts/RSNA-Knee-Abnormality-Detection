"""Synthetic publication contract checks; no production/GPU/accuracy claim."""

import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path("initiative/research/predictive-full-fitting-20261020-v1/publish_epoch.py.txt")
loader = importlib.machinery.SourceFileLoader("predictive_epoch_publisher", str(PATH))
m = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(m)


def contract(arm="C"):
    binding = {
        "schema": "predictive_full3887_one_arm_v1",
        "arm": arm,
        "source_git": "a" * 40,
        "expected_initial_shared_tensor_sha256": "b" * 64,
        "runtime_versions": {"torch": "exact"},
        "renderer_sha256": {"inputs.py": "c" * 64},
        "source_revision": m.SOURCE_REVISION,
        "source_context_index_sha256": "d" * 64,
    }
    refs = {
        "parent_weight": m.PARENT,
        "parent_source": m.PARENT_SOURCE,
        "license": m.PERMISSION,
        "targets": m.TARGETS,
        **{
            name: "e" * 64
            for name in (
                "context_index",
                "runtime_receipt",
                "engineering_decision",
                "funding",
                "input_admission",
            )
        },
    }
    binding.update({name: {"sha256": sha} for name, sha in refs.items()})
    start = {
        "arm": arm,
        "source_git": binding["source_git"],
        "GPU_job": "synthetic",
        "binding_sha256": "f" * 64,
        "recipe": m.expected_recipe(arm),
        "parent_weight_sha256": m.PARENT,
        "parent_source_sha256": m.PARENT_SOURCE,
        "source_revision": m.SOURCE_REVISION,
        "targets_sha256": m.TARGETS,
        "fitting_order_sha256": m.ROSTER,
        "license_sha256": m.PERMISSION,
        "arm_cap_bytes": 12 * m.GIB,
        "parameter_count": m.PARAMETERS[arm],
        "clinical_accuracy_or_Kaggle_score": False,
        "initial_shared_tensor_sha256": binding["expected_initial_shared_tensor_sha256"],
        "model_schema_sha256": "1" * 64,
        "runtime_versions": binding["runtime_versions"],
        "renderer_sha256": binding["renderer_sha256"],
        "source_context_index_sha256": binding["source_context_index_sha256"],
    }
    start.update(
        {
            name + "_sha256": binding[name]["sha256"]
            for name in (
                "context_index",
                "runtime_receipt",
                "engineering_decision",
                "funding",
                "input_admission",
            )
        }
    )
    side = {
        **start,
        "status": "immutable_full_state_epoch_ready_for_private_publication",
        "epoch": 1,
        "completed_updates": 486,
        "cohort_size": 3887,
        "checkpoint_name": "epoch-001.pt",
        "checkpoint_bytes": 100,
        "checkpoint_stat": [1, 100, 2, 3],
        "checkpoint_sha256": "2" * 64,
        "contains_model_optimizer_recipe_binding_counters_history_all_RNG": True,
        "independent_private_publication_verified": False,
    }
    return binding, start, side


@pytest.mark.parametrize("arm", ["B", "C", "C0"])
def test_exact_worker_sidecar_contract(arm):
    binding, start, side = contract(arm)
    m.validate_binding(binding, start, side, 1, "f" * 64)


@pytest.mark.parametrize(
    "key,value",
    [
        ("cohort_size", 12),
        ("completed_updates", 485),
        ("checkpoint_name", "../final.pt"),
        ("checkpoint_stat", [1, 101, 2, 3]),
        ("independent_private_publication_verified", True),
        ("status", "writing"),
        ("epoch", 2),
    ],
)
def test_rejects_unsealed_or_wrong_epoch(key, value):
    binding, start, side = contract()
    side[key] = value
    with pytest.raises(ValueError):
        m.validate_binding(binding, start, side, 1, "f" * 64)


@pytest.mark.parametrize(
    "key,value",
    [
        ("parent_weight_sha256", "stanford"),
        ("parent_source_sha256", "different"),
        ("targets_sha256", "changed"),
        ("initial_shared_tensor_sha256", "bad"),
        ("fitting_order_sha256", "shuffled"),
        ("parameter_count", 73136096),
        ("recipe", {**m.expected_recipe("C"), "schedule_updates": 24}),
    ],
)
def test_rejects_changed_lineage_inputs_initialization_and_schedule(key, value):
    binding, start, side = contract()
    start[key] = side[key] = value
    with pytest.raises(ValueError):
        m.validate_binding(binding, start, side, 1, "f" * 64)


def synthetic_state(arm="C"):
    binding, _, side = contract(arm)
    history = []
    for step in range(1, 487):
        import math

        f = (step - 1) / 2915
        factor = (
            min(1.0, step / (0.05 * 2916))
            if f < 0.05
            else 0.1 + 0.9 * (1 + math.cos(math.pi * (f - 0.05) / 0.95)) / 2
        )
        history.append(
            dict(
                step=step,
                epoch=1,
                studies=7 if step == 486 else 8,
                encoder_updated=step > 3,
                LR=[3e-5 * factor] * 2,
                **dict.fromkeys(
                    (
                        "loss",
                        "exam_loss",
                        "density_loss",
                        "encoder_gradient_norm",
                        "head_gradient_norm",
                        "gradient_norm",
                    ),
                    0.2,
                ),
            )
        )
    model = {f"encoder.backbone.parameter{i}": torch.ones(1) for i in range(101)}
    names = (
        ("reconstruct.weight", "target_project.weight", "vote.weight", "weight.weight")
        if arm == "B"
        else ("negative.weight", "shift.weight", "fixed_log_variance")
    )
    model.update({name: torch.zeros(1) for name in names})
    schema = {name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name, t in model.items()}
    side["model_schema_sha256"] = hashlib.sha256(m.canonical(schema)).hexdigest()
    side["history_sha256"] = hashlib.sha256(m.canonical(history)).hexdigest()
    optimizer = {
        "state": {
            i: dict(
                step=torch.tensor(483 if i == 0 else 486),
                exp_avg=torch.zeros(1),
                exp_avg_sq=torch.zeros(1),
            )
            for i in (0, 1)
        },
        "param_groups": [
            dict(params=[i], weight_decay=0.02, lr=history[-1]["LR"][i]) for i in (0, 1)
        ],
    }
    return (
        binding,
        side,
        dict(
            recipe=side["recipe"],
            binding=binding,
            cohort_size=3887,
            epoch_snapshots=True,
            model=model,
            optimizer=optimizer,
            step=486,
            history=history,
            rng=dict(python=(), numpy=(), torch=torch.zeros(1), cuda=[torch.zeros(1)]),
        ),
    )


@pytest.mark.parametrize("arm", ["B", "C", "C0"])
def test_predictive_schema_accepts_complete_synthetic_state(arm):
    binding, side, state = synthetic_state(arm)
    assert m.validate_state(state, binding, side) is state["model"]


@pytest.mark.parametrize(
    "change",
    [
        "old_schema",
        "optimizer_missing",
        "optimizer_nonfinite",
        "model_missing",
        "model_nonfinite",
        "history_tail",
        "RNG_missing",
    ],
)
def test_rejects_incomplete_or_nonfinite_state(change):
    binding, side, state = synthetic_state()
    if change == "old_schema":
        state = {"binding": binding, "epoch": 1, "completed": 486, "model": state["model"]}
    elif change == "optimizer_missing":
        state["optimizer"]["state"].pop(0)
    elif change == "optimizer_nonfinite":
        state["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
    elif change == "model_missing":
        state["model"].pop("shift.weight")
    elif change == "model_nonfinite":
        state["model"]["shift.weight"][0] = float("nan")
    elif change == "history_tail":
        state["history"][-1]["studies"] = 8
        side["history_sha256"] = hashlib.sha256(m.canonical(state["history"])).hexdigest()
    else:
        state["rng"].pop("cuda")
    with pytest.raises(ValueError):
        m.validate_state(state, binding, side)


class FakeAPI:
    def __init__(self, *, exists=False, corrupt=False):
        self.exists, self.corrupt, self.commits, self.downloads = exists, corrupt, [], []
        self.payload = {}

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(private=True, sha="parent")

    def get_paths_info(self, *args, **kwargs):
        return ["exists"] if self.exists else []

    def create_commit(self, *args, **kwargs):
        self.commits.append(kwargs)
        self.payload = {
            op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in kwargs["operations"]
        }
        return SimpleNamespace(oid="revision")

    def hf_hub_download(self, repo, name, **kwargs):
        assert repo == m.REPO and kwargs["force_download"] and kwargs["revision"] == "revision"
        self.downloads.append(name)
        path = Path(kwargs["local_dir"]) / "object"
        path.write_bytes(self.payload[name] + (b"bad" if self.corrupt else b""))
        return path


def atomic_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle)


def test_upload_verifies_every_original_object_and_manifest(tmp_path):
    api = FakeAPI()
    payload = tmp_path / "model.pt"
    payload.write_bytes(b"synthetic model only")
    result = m.upload_and_verify(
        api,
        SimpleNamespace(atomic_new=atomic_new),
        tmp_path,
        "private/epoch-001",
        {"model.pt": payload},
        SimpleNamespace,
        {"arm": "C", "epoch": 1},
    )
    assert len(api.commits) == 1 and len(api.downloads) == result["objects"] == 2
    assert (tmp_path / "intent.json").exists() and (tmp_path / "commit.json").exists()
    assert api.commits[0]["parent_commit"] == "parent"
    with pytest.raises(ValueError):
        m.upload_and_verify(
            api,
            SimpleNamespace(atomic_new=atomic_new),
            tmp_path,
            "private/epoch-001",
            {"model.pt": payload},
            SimpleNamespace,
            {},
        )
    assert len(api.commits) == 1


def test_existing_private_path_refuses_before_intent(tmp_path):
    api = FakeAPI(exists=True)
    with pytest.raises(ValueError):
        m.upload_and_verify(
            api,
            SimpleNamespace(atomic_new=atomic_new),
            tmp_path,
            "private/epoch-001",
            {},
            SimpleNamespace,
            {},
        )
    assert not api.commits and not (tmp_path / "intent.json").exists()


def test_failed_independent_download_preserves_intent_and_commit_without_verified(tmp_path):
    api = FakeAPI(corrupt=True)
    payload = tmp_path / "model.pt"
    payload.write_bytes(b"synthetic")
    with pytest.raises(ValueError):
        m.upload_and_verify(
            api,
            SimpleNamespace(atomic_new=atomic_new),
            tmp_path,
            "private/epoch-001",
            {"model.pt": payload},
            SimpleNamespace,
            {},
        )
    assert len(api.commits) == 1 and (tmp_path / "intent.json").exists()
    assert (tmp_path / "commit.json").exists() and not (tmp_path / "verified.json").exists()


def test_worker_sidecar_names_and_schema_match_template():
    source = PATH.with_name("worker.py.txt").read_text()
    for key in (
        "checkpoint_stat",
        "model_schema_sha256",
        "expected_initial_shared_tensor_sha256",
        "parent_source_sha256",
        "source_revision",
        "input_admission_sha256",
        "parameter_count",
    ):
        assert key in source
    assert 'root / f"epoch-{epoch:03d}.json"' in source
