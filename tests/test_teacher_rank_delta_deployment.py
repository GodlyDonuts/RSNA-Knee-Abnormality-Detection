"""One-version account orchestration with synthetic services and genuine local SQLite."""

import contextlib
import copy
import hashlib
import importlib.util
import json
import sqlite3
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "initiative/research/teacher-rank-delta-deployment-20261020-v1"
spec = importlib.util.spec_from_file_location("rank_delta_coordinator", D / "run_once.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(scope="module")
def controls():
    own = D / "policy_functions.py"
    serving = ROOT / runner._delta.SERVING
    return runner._delta.sources(
        ROOT, json.loads((D / "config.json").read_text()),
        (D / "engineering-decision.json").read_text(), own.read_text(),
        {
            "rank_delta_adapter.py" if name == "adapter.py" else name: (serving / name).read_text()
            for name in ("adapter.py", "publiccheck.py", "prelude.py")
        },
        (ROOT / runner._delta.INHERITED_PATH).read_text(),
    )


@pytest.fixture
def policy(controls):
    module = types.ModuleType("synthetic_rank_delta_policy")
    exec(controls["policy.py"], module.__dict__)
    module.PLAN = json.loads((D / "config.json").read_text())["runtime_plan"]
    return module


@pytest.mark.parametrize("free,available", [(3620, True), (3619.99, False), (3672.7366641, True)])
def test_fresh_quota_funds_startup_check_and_predictive_reserve(policy, free, available):
    result = policy.budget(
        dict(totalTimeAllowed="162000s", timeUsed=f"{162000-free}s", timeReserved="0s"),
        3, "competition", latest_training_epoch=3,
    )
    assert result["available"] is available and result["required_VM_seconds"] == 3620
    assert result["predictive_reserved_VM_seconds"] == 3360
    assert result["public_VM_setup_reserve_seconds"] == 80
    assert not result["paid_credits_allowed"]


@pytest.mark.parametrize("mode", ["qualification", "diagnostic", ""])
def test_no_qualification_or_diagnostic_request(policy, mode):
    with pytest.raises(ValueError):
        policy.budget({}, 3, mode, latest_training_epoch=3)
    with pytest.raises(ValueError):
        policy.request(3, mode, "", {}, dataset={})


@pytest.mark.parametrize("key,value", [
    ("predictive_reserved_VM_seconds", 3000),
    ("public_watchdog_seconds", 179),
    ("public_VM_setup_reserve_seconds", 0),
    ("required_free_VM_seconds", 3360),
    ("predictive_reserved_VM_seconds", True),
])
def test_changed_budget_rejected_before_intent(key, value):
    config = json.loads((D / "config.json").read_text())
    config["runtime_plan"][key] = value
    with pytest.raises(AssertionError):
        runner._delta.accepted(config, (D / "engineering-decision.json").read_text())


def test_exact_dependency_pins_model_key_and_single_mode(controls, policy):
    ns = {"__name__": "synthetic_bound_program"}
    exec(controls["program.py"], ns)
    rank, parent, _ = runner._delta.rank_parent(ROOT)
    for name in ("workflow", "components", "gpu_capacity", "output_csv", "publication"):
        assert controls[name + ".py"] == parent["scripts/kaggle/" + name + ".py"]
    assert ns["MODEL_SHA"] == runner._delta.MODEL
    assert ns["MODEL_KEY"] == "licensed-parent-full-epoch-serving:teacher_rank-final:" + ns["MODEL_SHA"]
    assert ns["D"].name == "rsna-" + D.name
    assert ns["COMPILED_SHA"]["competition"] == "dd3b7ea13383c5132a3d02c60e87aa65767a21efe8032de43c2bd524718f455e"
    assert policy.CAPS == {"competition": 32400}
    assert "full_parent_epoch_qualification_accepted" not in controls["program.py"]
    assert "other_node_unpaid += max(0, 640 * 2**20 - pointwise_paid)" in controls["program.py"]
    assert 'f["shared_other_liabilities_bytes"] >= 122 * 2**20' in controls["program.py"]
    assert sum(len(v.encode()) for v in controls.values()) < 256 * 2**10
    assert rank["PARENT"] == "01cbe205080487b3fb39fbf7c0d28e112fcd7d77"


def fixture_receipt(policy, source):
    metadata = {
        "status": "prepared_teacher_rank_single_public_run_delta_source",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "original_competition_sha256": "dd3b7ea13383c5132a3d02c60e87aa65767a21efe8032de43c2bd524718f455e",
        "model_sha256": runner._delta.MODEL,
        "training_source_sha256": "ffc1fbf914b9e1a97476ac40b3d4cc0c58b7948c45061079a497f1135c2df461",
        "inherited32_receipt_sha256": runner._delta.INHERITED_SHA,
        "public_roster_sha256": policy.ROSTER,
        "adapter_sha256": "a" * 64, "publiccheck_sha256": "b" * 64, "prelude_sha256": "c" * 64,
        "public_watchdog_seconds": 180, "public_VM_setup_reserve_seconds": 80,
        "total_reserved_VM_seconds": 260, "hidden_session_timeout_seconds": 32400,
        "qualification_scope": runner._delta.SCOPE,
    }
    receipt = {
        "status": "actual_teacher_pair_final_epoch_MRI_test_roster_CSV_complete",
        "epoch": 3, "role": "teacher_rank", "model_sha256": runner._delta.MODEL,
        "training_source_sha256": metadata["training_source_sha256"],
        "test_roster_sha256": policy.ROSTER, "studies": 3, "columns": 13,
        "training_windows": 16, "serving_windows": 94,
        "proposal_sha256": "e812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e",
        "competition_entry": False, "clinical_accuracy_claim": False,
        "devices": ["Tesla T4", "Tesla T4"], "CSV_sha256": "f" * 64, "CSV_bytes": 470,
        "public_total_elapsed_seconds": 120,
        "prediction_counts": {"started": 6, "completed": 6, "failed": 0, "decode_errors": 0, "empty_views": 0},
        "inline_new_weight_acceptance": {
            **metadata, "status": "actual_teacher_rank_public3_bothT4_delta_qualified",
            "views_per_study": 2, "windows_per_view": 94,
            "actual_new_weight_studies_per_device": 3, "inherited_code_qualification_studies": 32,
            "inherited32_reexecuted": False, "elapsed_seconds": 80,
            "devices": [
                dict(index=i, GPU="Tesla T4", MRI_studies=3,
                     both_original94_input_views_bit_equal=True,
                     complete_learned_graph_tensor_values_equal=True,
                     maximum_original_class_vs_training_graph_probability_error=0,
                     peak_allocated_bytes=2 * 2**30, input_views_aggregate_sha256="e" * 64)
                for i in (0, 1)
            ],
        },
    }
    return metadata, receipt


@pytest.fixture
def account(tmp_path, monkeypatch, controls, policy):
    source = controls["program.py"].replace(
        "/home/sa305415/rsna-knee/kaggle", str(tmp_path / "kaggle")
    )
    source = source.replace(
        "/tmp/rsna-teacher-rank-final-deployment-20261020-v1", str(tmp_path / "old-unused")
    )
    ns = {"__name__": "synthetic_rank_account"}
    exec(source, ns)
    ns["D"], ns["F"] = tmp_path / "new", tmp_path / "new/account-controls"
    ns["LEDGER"].parent.mkdir()
    with sqlite3.connect(ns["LEDGER"]) as db:
        db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT)")
        db.execute("CREATE TABLE events (at TEXT,sha TEXT,event TEXT)")
    ns["bounds"] = lambda v: None
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    original = tmp_path / "original"
    original.mkdir()
    original_source = "pass\n"
    (original / "competition.private.py").write_text(original_source)
    actual_source = "# synthetic candidate\npass\n"
    metadata, receipt = fixture_receipt(policy, actual_source)
    compiled = dict(model_sha256=ns["MODEL_SHA"], outputs={"competition": {
        "bytes": len(original_source), "sha256": hashlib.sha256(original_source.encode()).hexdigest()
    }})
    training = dict(epoch=3, status="COMPLETE", role="teacher_rank", kernel_id=None,
                    training_Slurm_job="975297", source_sha256=metadata["training_source_sha256"])
    ns["ORIGINAL_SERVING"] = original
    ns["original_teacher_rank"] = lambda v: (training, {}, compiled)
    bundle_calls = []
    ns["prepare_bundle"] = lambda original: bundle_calls.append(original)
    policy.__file__ = str(ns["F"] / "code/policy.py")
    adapted = []

    def adapt(text, mode):
        assert text == original_source and mode == "competition"
        adapted.append(mode)
        return actual_source, copy.deepcopy(metadata)

    policy.adapt_source = adapt
    calls = []
    state = {"slots": 1, "kernel": None, "receipt": receipt, "free": 3672.7366641}

    class Remote:
        def call(self, service, method, args):
            calls.append(method)
            if method == "IntrospectToken":
                return dict(active=True, username="godlydonuts")
            if method == "GetAcceleratorQuotaStatistics":
                return dict(gpuQuota=dict(totalTimeAllowed=f'{state["free"]}s', timeUsed="0s", timeReserved="0s"))
            assert method == "SaveKernel"
            state["kernel"] = copy.deepcopy(args)
            return {"ref": "/code/" + args["slug"], "versionNumber": 1, "kernelId": 123}

        def kernel(self, method, slug, version):
            assert version == 1
            if method == "GetKernelSessionStatus":
                return {"status": "COMPLETE"}
            assert method == "GetKernel"
            return dict(metadata={**state["kernel"], "id": 123}, blob={"source": state["kernel"]["text"]})

        def require_absent_kernel(self, slug):
            return {"absent": True}

        def output_json(self, slug, version, name):
            assert version == 1 and name == "licensed-IID16-epoch-public-accepted.json"
            return state["receipt"]

        def account(self):
            calls.append("fresh_allowance")
            return {"synthetic": True}

        def submissions(self):
            return []

        def competition(self, method, value):
            calls.append(method)
            assert method == "CreateCodeSubmission" and value["kernelVersion"] == 1
            assert value["kernelSlug"] == state["kernel"]["slug"].split("/")[1]
            assert (ns["D"] / "account-phase-v1/entry-intent.json").is_file()
            return {"ref": 456}

    def put(path, obj):
        with path.open("x") as handle:
            json.dump(obj, handle)

    @contextlib.contextmanager
    def ledger(path, account):
        with sqlite3.connect(path) as db:
            yield db

    workflow = types.SimpleNamespace(
        Remote=Remote, ledger=ledger, verify_account=lambda *a: None,
        new_json=put, object_hash=lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest(),
        utcnow=lambda: "synthetic", SAVE_KERNEL_INVALID=[],
        allowance=lambda *a: state["slots"],
    )
    modules = {
        "workflow": workflow, "policy": policy,
        "gpu_capacity": types.SimpleNamespace(admission=lambda *a, **k: {"available": True}),
        "output_csv": types.SimpleNamespace(
            download_csv=lambda *a: b"synthetic",
            verify_csv=lambda *a: {"CSV_sha256": "f" * 64, "bytes": 470, "test_roster_sha256": policy.ROSTER},
        ),
    }
    ns["load"] = lambda name: modules[name]
    ns["verified_dataset"] = lambda remote: dict(
        status="all_payload_bytes_verified", private=True, version=1, downloaded_all_bytes=True,
        slug=ns["DATASET_SLUG"],
    )
    config = json.loads((D / "config.json").read_text())
    value = dict(
        **config, epoch=3, mode="competition", token="synthetic", source_git="synthetic",
        original_compiled=compiled, sources=controls,
        source_sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in controls.items()},
    )
    return types.SimpleNamespace(
        ns=ns, value=value, state=state, calls=calls, policy=policy, metadata=metadata,
        bundle_calls=bundle_calls, adapted=adapted,
        act=lambda phase: ns["execute"]({**value, "phase": phase}),
    )


def test_real_stage_preflight_before_bundle_and_prepare_only_competition(account):
    assert account.act("stage")["delta_preflight"]["source_sha256"] == account.metadata["source_sha256"]
    assert len(account.adapted) == len(account.bundle_calls) == 1
    assert account.calls == []
    account.act("prepare")
    assert account.adapted == ["competition", "competition"]
    target = account.ns["D"] / "account-phase-v1"
    assert (target / "competition.private.py").is_file()
    assert not (target / "qualification.private.py").exists()
    with pytest.raises(AssertionError):
        account.act("stage")


def test_bad_actual_compiler_interface_stops_before_bundle_or_any_account_call(account):
    def reject(*args):
        raise ValueError("Synthetic actual-source mismatch")
    account.policy.adapt_source = reject
    with pytest.raises(ValueError):
        account.act("stage")
    assert account.bundle_calls == account.calls == []


def test_single_public_push_and_same_version_submission_without_qualification(account):
    account.act("stage")
    account.act("prepare")
    push = account.act("push")
    assert push["kernel_id"] == 123 and push["status"] == "one_exact_target_epoch_competition_pushed"
    request = account.state["kernel"]
    assert request["kernelDataSources"] == [account.policy.OFFLINE]
    assert request["datasetDataSources"] == [account.ns["DATASET_SLUG"]]
    assert request["dockerImage"] == account.policy.DOCKER
    assert request["machineShape"] == "NvidiaTeslaT4" and not request["enableInternet"]
    with pytest.raises(AssertionError):
        account.act("push")
    accepted = account.act("status")
    assert accepted["status"] == "COMPLETE" and accepted["CSV_verified_in_RAM"]["CSV_sha256"] == "f" * 64
    account.state["slots"] = 0
    assert account.act("submit")["status"] == "waiting_authoritative_live_account_allowance_no_mutation"
    assert not (account.ns["D"] / "account-phase-v1/entry-intent.json").exists()
    account.state["slots"] = 1
    assert account.act("submit")["submission_id"] == 456
    with pytest.raises(AssertionError):
        account.act("submit")
    assert account.calls.count("SaveKernel") == account.calls.count("CreateCodeSubmission") == 1
    assert account.calls.count("fresh_allowance") == 2
    assert not (account.ns["LEDGER"].parent / "coordinator-claim").exists()


@pytest.mark.parametrize("bad", ["missing_inline", "graph", "CSV", "version_source"])
def test_bad_new_weight_or_csv_evidence_cannot_accept_or_submit(account, bad):
    account.act("stage")
    account.act("prepare")
    account.act("push")
    if bad == "missing_inline":
        del account.state["receipt"]["inline_new_weight_acceptance"]
    elif bad == "graph":
        account.state["receipt"]["inline_new_weight_acceptance"]["devices"][1][
            "complete_learned_graph_tensor_values_equal"
        ] = False
    elif bad == "CSV":
        account.state["receipt"]["CSV_sha256"] = "0" * 64
    else:
        account.state["kernel"]["text"] += "\n# different"
    with pytest.raises((ValueError, AssertionError, KeyError)):
        account.act("status")
    with pytest.raises(AssertionError):
        account.act("submit")
    assert "CreateCodeSubmission" not in account.calls


def test_old_prepared_route_and_canonical_key_both_prevent_new_stage(account):
    old = account.ns["D"].parent / "old-unused"
    old.mkdir()
    with pytest.raises(AssertionError, match="Original prepared"):
        account.act("inspect")
    old.rmdir()
    with sqlite3.connect(account.ns["LEDGER"]) as db:
        db.execute("INSERT INTO meta VALUES (?,?)", (account.ns["MODEL_KEY"], "{}"))
    with pytest.raises(AssertionError):
        account.act("stage")
    assert account.calls == [] and not account.ns["D"].exists()


def test_finite_owner_has_one_gpu_and_no_qualification_action():
    text = runner._delta.owner_source(ROOT)
    assert 'for mode in ("competition",):' in text
    assert 'def action(phase, mode="competition"):' in text
    assert "maximum_new_GPU_sessions=1" in text and "maximum_new_entries=1" in text
    assert "automatic_mutation_retry=False" in text
    assert 'for mode in ("qualification", "competition"):' not in text
    compile(text, "finite_delta_owner", "exec")
