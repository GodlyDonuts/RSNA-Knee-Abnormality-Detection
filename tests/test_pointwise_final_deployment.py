"""Operational pointwise tests; no network, GPU, model training or clinical claim."""

import ast
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import json
import sqlite3
import types
from pathlib import Path

import pytest

D = Path("initiative/research/pointwise-final-deployment-20261020-v1")
OLD = Path("initiative/research/S6-Kaggle-production-route-20261019-v1")


def module(name):
    spec = importlib.util.spec_from_file_location("test_pointwise_" + name, D / (name + ".py"))
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


@pytest.fixture
def plan():
    return dict(
        status="accepted_pointwise_final_runtime_budget",
        qualification_cap_seconds=600,
        public_watchdog_seconds=300,
        predictive_reserved_VM_seconds=3000,
        actual_same_graph_timing_receipt_sha256="a" * 64,
        same32_case_checks_and_inference_unchanged=True,
    )


def generated_policy():
    text = (D / "policy_functions.py").read_text()
    funcs = {
        n.name: ast.get_source_segment(text, n)
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef)
    }
    source = module("adapter").policy((OLD / "serving_policy.py.txt").read_text(), funcs)
    policy = types.ModuleType("synthetic_pointwise_policy")
    exec(source, policy.__dict__)
    return source, policy


def generated_program(tmp_path):
    source = module("adapter").program(
        (OLD / "serving_cloud.py.txt").read_text(), (D / "helpers.py").read_text()
    )
    source = source.replace("/home/sa305415/rsna-knee/kaggle", str(tmp_path / "kaggle"))
    ns = {"__name__": "synthetic_pointwise_account"}
    exec(source, ns)
    ns["D"] = tmp_path / "new-deployment"
    ns["F"] = ns["D"] / "account-controls"
    ns["LEDGER"].parent.mkdir()
    with sqlite3.connect(ns["LEDGER"]) as db:
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE events (at TEXT, sha TEXT, event TEXT)")
    return source, ns


def put(path, value):
    with path.open("x") as f:
        json.dump(value, f)


def test_pinned_original_sources_refuse_unreviewed_change():
    adapter = module("adapter")
    with pytest.raises(ValueError, match="Exact original"):
        adapter.program((OLD / "serving_cloud.py.txt").read_text() + "\n", "")
    with pytest.raises(ValueError, match="Exact original"):
        adapter.policy((OLD / "serving_policy.py.txt").read_text() + "\n", {})


def test_exact_docker_and_complete_offline_request(plan):
    _, policy = generated_policy()
    policy.request.__globals__["PLAN"] = plan
    training = dict(
        epoch=3, status="COMPLETE", kernel_id=None, training_Slurm_job="975297", role="pointwise"
    )
    dataset = dict(
        status="all_payload_bytes_verified",
        private=True,
        version=1,
        downloaded_all_bytes=True,
        slug="godlydonuts/private-final",
    )
    q = policy.request(3, "qualification", "source", training, dataset=dataset)
    assert q["dockerImage"] == module("adapter").DOCKER
    assert len(q["dockerImage"].split("sha256:")[1]) == 64
    assert q["sessionTimeoutSeconds"] == 600 and q["datasetDataSources"] == [dataset["slug"]]
    assert q["kernelDataSources"] == [policy.OFFLINE, policy.PIXELS32]
    assert not q["enableInternet"] and q["isPrivate"] and q["machineShape"] == "NvidiaTeslaT4"
    c = policy.request(3, "competition", "source", training, dataset=dataset)
    assert c["sessionTimeoutSeconds"] == 32400 and c["kernelDataSources"] == [policy.OFFLINE]
    assert c["competitionDataSources"] == ["rsna-knee-abnormality-detection"]


@pytest.mark.parametrize(
    "field,value",
    [("epoch", 2), ("kernel_id", 1), ("role", "teacher_rank"), ("training_Slurm_job", "old")],
)
def test_wrong_actual_model_binding_rejected(plan, field, value):
    _, policy = generated_policy()
    policy.request.__globals__["PLAN"] = plan
    training = dict(
        epoch=3, status="COMPLETE", kernel_id=None, training_Slurm_job="975297", role="pointwise"
    )
    training[field] = value
    with pytest.raises(ValueError):
        policy.request(3, "qualification", "source", training, dataset={})


@pytest.mark.parametrize(
    "free,mode,accepted,required",
    [
        (4166.2086641, "qualification", True, 3900),
        (3900, "qualification", True, 3900),
        (3899, "qualification", False, 3900),
        (3300, "competition", True, 3300),
        (3299, "competition", False, 3300),
    ],
)
def test_fresh_free_budget_preserves_predictive_lane(plan, free, mode, accepted, required):
    _, policy = generated_policy()
    policy.budget.__globals__["PLAN"] = plan
    result = policy.budget(
        dict(
            totalTimeAllowed="162000s",
            timeUsed=str(162000 - free) + "s",
            timeReserved="0s",
            minimumTimeAllowed="108000s",
            isPayToScaleEnabled=True,
        ),
        3,
        mode,
        latest_training_epoch=3,
    )
    assert result["available"] is accepted and result["required_VM_seconds"] == required
    assert result["hidden_session_timeout_seconds"] == 32400
    assert not result["paid_credits_allowed"] and not result["minimumTimeAllowed_used_as_ceiling"]


def test_public_watchdog_depends_on_exact_roster_and_leaves_qualification_unchanged(monkeypatch):
    _, policy = generated_policy()
    original = "def main():\n    return 'same inference'\nif __name__ == '__main__':\n    main()\n"
    assert policy.public_bounded_source(original, "qualification") == original
    source = policy.public_bounded_source(original, "competition")
    assert "count==3" not in source and "Timer(300," in source and policy.ROSTER in source
    records = []

    class Timer:
        def __init__(self, seconds, action):
            records.append(seconds)

        def start(self):
            records.append("started")

    fake = {
        "__name__": "test",
        "find_test_root": lambda: Path("/none"),
        "threading": types.SimpleNamespace(Timer=Timer),
    }
    exec(source, fake)
    import pandas as pd

    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *a, **k: types.SimpleNamespace(
            StudyInstanceUID=types.SimpleNamespace(tolist=lambda: ["different", "three", "cases"])
        ),
    )
    fake["_actual_public_only_watchdog"]()
    assert records == []

    # Execute the exact public branch independently of inaccessible private source identifiers.
    class PublicHash:
        def hexdigest(self):
            return policy.ROSTER

    monkeypatch.setattr(hashlib, "sha256", lambda x: PublicHash())
    fake["_actual_public_only_watchdog"]()
    assert records == [300, "started"]
    assert fake["main"]() == "same inference"


def test_real_temporary_stage_then_prepare_and_duplicate_refusal(tmp_path, monkeypatch, plan):
    _, ns = generated_program(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    ns["bounds"] = lambda v: None
    original_dir = tmp_path / "preserved-original"
    original_dir.mkdir()
    source = "def main():\n    pass\nif __name__ == '__main__':\n    main()\n"
    outputs = {}
    for mode in ("qualification", "competition"):
        p = original_dir / (mode + ".private.py")
        p.write_text(source)
        outputs[mode] = dict(
            bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()
        )
    training = dict(
        status="COMPLETE",
        epoch=3,
        kernel_id=None,
        training_Slurm_job="975297",
        role="pointwise",
        source_sha256="s" * 64,
    )
    compiled = dict(model_sha256=ns["MODEL_SHA"], outputs=outputs)
    ns["original_pointwise"] = lambda v: (training, {"model_sha256": ns["MODEL_SHA"]}, compiled)
    ns["ORIGINAL_SERVING"] = original_dir
    packed = []
    ns["prepare_bundle"] = lambda original: packed.append(original)
    value = dict(
        phase="stage",
        epoch=3,
        sources={"program.py": "pass\n"},
        source_sha256={"program.py": hashlib.sha256(b"pass\n").hexdigest()},
        source_git="frozen",
        runtime_plan=plan,
        funding={"approved": True},
        original_compiled=compiled,
    )
    result = ns["execute"](value)
    assert result["status"] == "actual_twoT4_epoch_serving_account_control_staged"
    assert (ns["F"] / "code/program.py").read_text() == "pass\n" and len(packed) == 1
    with pytest.raises(AssertionError, match="reconciliation"):
        ns["execute"](value)
    _, policy = generated_policy()

    class Remote:
        def call(self, service, method, args):
            if method == "GetAcceleratorQuotaStatistics":
                return {
                    "gpuQuota": dict(totalTimeAllowed="4166s", timeUsed="0s", timeReserved="0s")
                }
            if method == "SaveKernel":
                return {"ref": "/code/" + args["slug"], "versionNumber": 1, "kernelId": 123}
            return dict(active=True, username="godlydonuts")

        def kernel(self, *args):
            return {"status": "COMPLETE"}

        def require_absent_kernel(self, slug):
            return {"absent": True}

    @contextlib.contextmanager
    def ledger(path, account):
        with sqlite3.connect(path) as db:
            yield db

    workflow = types.SimpleNamespace(
        Remote=Remote,
        ledger=ledger,
        verify_account=lambda *a: None,
        new_json=put,
        object_hash=lambda obj: "a" * 64,
        utcnow=lambda: "synthetic",
        SAVE_KERNEL_INVALID=[],
    )
    ns["load"] = lambda name: {
        "workflow": workflow,
        "policy": policy,
        "gpu_capacity": types.SimpleNamespace(admission=lambda *a, **kw: {"available": True}),
    }[name]
    ns["verified_dataset"] = lambda remote: dict(
        slug="godlydonuts/verified-private",
        status="all_payload_bytes_verified",
        private=True,
        version=1,
        downloaded_all_bytes=True,
    )
    prepared = ns["execute"]({**value, "phase": "prepare", "token": "synthetic"})
    assert prepared["status"] == "actual_preserved_epoch_serving_prepared_without_model_reupload"
    with sqlite3.connect(ns["LEDGER"]) as db:
        row = json.loads(
            db.execute("SELECT value FROM meta WHERE key=?", (ns["MODEL_KEY"],)).fetchone()[0]
        )
    assert row["training"]["kernel_id"] is None and row["candidate"]["role"] == "pointwise"
    assert not (ns["LEDGER"].parent / "coordinator-claim").exists()
    assert (ns["D"] / "account-phase-v1/qualification.private.py").read_text() == source
    pushed = ns["execute"]({**value, "phase": "push", "token": "synthetic"})
    assert pushed["status"] == "one_exact_target_epoch_qualification_pushed"
    assert pushed["kernel_id"] == 123 and pushed["role"] == "pointwise"
    assert not (ns["LEDGER"].parent / "coordinator-claim").exists()
    with pytest.raises(AssertionError):
        ns["execute"]({**value, "phase": "push", "token": "synthetic"})


def test_actual_pure_validator_can_load_without_torch(monkeypatch):
    original = Path("rsna/competitive/teacher_pair_serving.py").read_text()
    assert (
        hashlib.sha256(original.encode()).hexdigest()
        == "d887431cc07a9ac2f1b441b44ee52c3f7ba0720c524f350b8e5ceb202680a147"
    )
    imported = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("Management Python must not import Torch")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    scope = {}
    exec(original.replace("\nimport torch\n", "\n"), scope)
    sha = "a" * 64
    binding = copy.deepcopy(scope["EXPECTED_BINDING"])
    training = dict(
        status="COMPLETE",
        epoch=3,
        kernel_id=None,
        training_Slurm_job=scope["GPU_JOB"],
        source_sha256=scope["WORKER_SHA"],
        role="pointwise",
        receipt=dict(
            status="actual_teacher_pair_fixed_final_model_for_serving",
            completed=1458,
            binding=binding,
            epoch=dict(bytes=292867251, sha256=sha),
        ),
    )
    publication = dict(
        status="actual_teacher_pair_immutable_epoch_private_all_objects_download_size_SHA_verified",
        GPU_job=scope["GPU_JOB"],
        source_git=scope["SCIENCE"],
        role="pointwise",
        epoch=3,
        completed_updates=1458,
        objects=6,
        canonical_shared_publication_lock_used=True,
        prefix="checkpoints/teacher-pair-full3887-"
        + scope["SCIENCE"][:12]
        + "/pointwise/epoch-003",
        model_sha256=sha,
        model_bytes=292867251,
        provenance=dict(
            scientific_source_git=scope["SCIENCE"],
            GPU_job=scope["GPU_JOB"],
            role="pointwise",
            epoch=3,
            completed_updates=1458,
            binding=binding,
            snapshot_sha256=sha,
            trainable_parameters_from_actual_accepted_trainer=73136096,
            source_permission_sha256=binding["source_binding"]["source_permission_sha256"],
            original_MRI_labels_UIDs_private_arrays_included=False,
        ),
    )
    assert scope["provenance"](training, publication, sha) == 3
    training["receipt"]["completed"] = 1457
    with pytest.raises(ValueError):
        scope["provenance"](training, publication, sha)


def test_dataset_intent_precedes_rpc_and_uncertainty_cannot_repeat(tmp_path):
    _, ns = generated_program(tmp_path)
    ns["D"].mkdir()
    (ns["D"] / "upload").mkdir()
    archive = ns["D"] / "upload/bundle.zip"
    archive.write_bytes(b"synthetic")
    prepared = dict(
        files={"epoch-003.pt": {"bytes": 9, "sha256": "a" * 64}},
        archive=dict(sha256=ns["sha"](archive)),
    )
    put(ns["D"] / "bundle-prepared.json", prepared)
    calls = []
    with sqlite3.connect(ns["LEDGER"]) as db:

        def create(*args):
            calls.append(1)
            assert (ns["D"] / "dataset-intent.json").is_file()
            with sqlite3.connect(ns["LEDGER"]) as independent:
                assert independent.execute(
                    "SELECT value FROM meta WHERE key=?", (ns["MODEL_KEY"] + ":dataset",)
                ).fetchone()
            raise RuntimeError("Synthetic unknown response")

        pub = types.SimpleNamespace(
            require_absent=lambda *a: {"absent": True},
            validate_zip=lambda *a: None,
            sdk_create_once=create,
        )
        ns["load"] = lambda name: pub
        with pytest.raises(RuntimeError):
            ns["dataset_phase"](
                dict(phase="dataset", token="synthetic"),
                db,
                object(),
                types.SimpleNamespace(new_json=put),
            )
        assert (ns["D"] / "dataset-uncertain.json").is_file()
        with pytest.raises(AssertionError, match="never recreate"):
            ns["dataset_phase"](
                dict(phase="dataset", token="synthetic"),
                db,
                object(),
                types.SimpleNamespace(new_json=put),
            )
    assert len(calls) == 1


def test_complete_generated_controls_are_locally_compilable(monkeypatch):
    runner = module("run_once")
    monkeypatch.setattr(runner, "git_bytes", lambda pin, path: path.read_bytes())
    controls = runner.controls("synthetic-uncommitted-test-only")
    for name, source in controls.items():
        compile(source, name, "exec")
    assert sum(len(source.encode()) for source in controls.values()) < 256 * 2**10
