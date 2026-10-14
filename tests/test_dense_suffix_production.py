"""Fixed-ten matched dense-suffix production launcher and private epoch publisher: fail closed."""

import contextlib
import copy
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.newton import dense_suffix_epoch_publisher as pub
from scripts.newton import run_dense_suffix_production as launch

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "initiative/audits/dense-suffix-admission-root-review-20261014-v1"
SUPERVISION = {"folds_sha256": "f", "overlay_sha256": "o", "fitting_uid_sha256": "u"}
COHORT = {"index_sha256": "i" * 64, "members": 3403}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return launch.sha(path)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(
        launch.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**15, total=0, used=0)
    )
    adm, root, art = tmp_path / "admission", tmp_path / "root", tmp_path / "artifact"
    code = adm / "code"
    files = {"rsna/x.py": "x = 1\n"}
    for name in launch.ARMS.values():
        files[f"{launch.RECIPE}/{name}"] = (REPO / launch.RECIPE / name).read_text()
    wrapper = "scripts/newton/run_dense_suffix_recovery.py"
    files[wrapper] = (REPO / wrapper).read_text()
    manifest = {}
    for name, text in files.items():
        (code / name).parent.mkdir(parents=True, exist_ok=True)
        (code / name).write_text(text)
        manifest[name] = launch.sha(code / name)
    manifest_sha = write(adm / "recovery-source-manifest.json", manifest)
    spec = dict(json.loads((REPO / launch.RECIPE / "dense-config.json").read_text())["encoder"])
    spec.pop("suffix_adapter")
    spec_sha = write(adm / "encoder-spec.json", spec)
    fold = art / "initiative/experiments/competitive-v1/folds/fold.csv"
    fold.parent.mkdir(parents=True)
    fold.write_text("study_uid,fold\na,0\n")
    overlay = tmp_path / "overlay.json"
    overlay_sha = write(overlay, {"overlay": True})
    code_sha = {"x.py": manifest["rsna/x.py"]}
    cpu = {
        "status": launch.CPU_STATUS,
        "CPU_job": "899766",
        "source_git": launch.PINS["source_git"],
        "GPU_acceptance_inferred": False,
        "recipe_sha256": launch.PINS["recipe_sha256"],
        "recovery_source_manifest_sha256": manifest_sha,
        "wrapper_sha256": manifest[wrapper],
        "full_fitting_studies": 2247,
        "full_outer_studies": 1156,
        "protected_labels_used_for_recipe_or_metrics": False,
        "arms": {
            arm: {
                "encoder_learned_parameters": p["trainable_parameters"],
                "encoder_lr": p["lr"],
            }
            for arm, p in launch.PINNED.items()
        },
        "future_obligations": {
            "production_fixed10": {
                "checkpoint_upper_bound_bytes": launch.PINS["checkpoint_upper_bound_bytes"]
            }
        },
        "code_sha256": code_sha,
        "folds_sha256": launch.sha(fold),
        "release_sha256": "r" * 64,
        "supervision_sha256": launch.canonical(SUPERVISION),
        "overlay_sha256": overlay_sha,
        "public_cohort": COHORT,
        "GPU_wrapper_arguments": {
            "artifact-root": str(art),
            "cache": str(tmp_path / "cache"),
            "overlay": str(overlay),
            "output": str(adm / "recovery"),
        },
    }
    cpu_sha = write(adm / "CPU-accepted.json", cpu)
    pins = dict(
        launch.PINS,
        CPU_receipt_sha256=cpu_sha,
        recovery_source_manifest_sha256=manifest_sha,
        encoder_spec_path_sha256=spec_sha,
    )
    arms = {}
    for arm, p in launch.PINNED.items():
        full = adm / "recovery" / arm / "full"
        full.mkdir(parents=True)
        (full / "checkpoint.pt").write_bytes(arm.encode())
        (full / "outer_predictions.csv").write_text("study_uid\n")
        write(full / "run.json", {"binding": {"device_type": "cuda", "code_sha256": code_sha}})
        arms[arm] = {
            "exact_complete_state_history_and_prediction_recovery": True,
            "trainable_encoder_parameters": p["trainable_parameters"],
            "finite_updates": [{"loss": 0.5, "optimizer_steps": i + 1} for i in range(4)],
            "checkpoint_sha256": launch.sha(full / "checkpoint.pt"),
            "prediction_sha256": launch.sha(full / "outer_predictions.csv"),
        }
    recovery = {
        "status": launch.RECOVERY_STATUS,
        "device": "cuda",
        "GPUs_used_concurrently": 1,
        "source_manifest_sha256": manifest_sha,
        "recipe_sha256": launch.PINS["recipe_sha256"],
        "competition_submission": False,
        "diagnostic_only": True,
        "arms": arms,
    }
    recovery_sha = write(adm / "recovery" / "accepted.json", recovery)
    root.mkdir()
    shutil.copy(REPO / "scripts/newton/run_dense_suffix_production.py", root / "launch.py")
    shutil.copy(REPO / "scripts/newton/dense_suffix_epoch_publisher.py", root / "publish.py")
    shutil.copy(
        REPO / "scripts/newton/final_fit_epoch_publisher.py", root / "publication-transport.py"
    )
    prepared = launch.prepare(root, adm, pins)
    checkpoint = sum(launch.PINS["checkpoint_upper_bound_bytes"].values())
    funding = {
        "funded_by": "root",
        "production_funded": True,
        "node": "evc31",
        "parent_job": "899766",
        "assigned_GPU": "3",
        "CPUs": 2,
        "source_git": launch.PINS["source_git"],
        "CPU_receipt_sha256": cpu_sha,
        "GPU_recovery_receipt_sha256": recovery_sha,
        "launcher_sha256": launch.sha(root / "launch.py"),
        "publisher_sha256": launch.sha(root / "publish.py"),
        "config_sha256": prepared["config_sha256"],
        "folds_sha256": cpu["folds_sha256"],
        "release_sha256": cpu["release_sha256"],
        "supervision_sha256": cpu["supervision_sha256"],
        "overlay_sha256": overlay_sha,
        "checkpoint_bytes": checkpoint,
        "runtime_bytes": 2**30,
        "publication_bytes": 5 * 2**30,
        "existing_liabilities_bytes": 300 * 2**30,
        "reserve_bytes": launch.RESERVE,
    }
    funding.update(
        required_free_bytes=300 * 2**30 + checkpoint + 2**30 + 5 * 2**30 + launch.RESERVE,
        shared_future_bytes=0,
    )
    write(root / "production-funding.json", funding)
    return SimpleNamespace(
        root=root, adm=adm, code=code, pins=pins, cpu=cpu, funding=funding, fold=fold
    )


def binding(env, arm):
    values = json.loads((env.root / f"{arm}-config.json").read_text())
    suffix = dict(
        values["encoder"]["suffix_adapter"],
        trainable_parameters=launch.PINNED[arm]["trainable_parameters"],
    )
    return {
        "config": dict(values, encoder=dict(values["encoder"], suffix_adapter=suffix)),
        "code_sha256": env.cpu["code_sha256"],
        "device_type": "cuda",
        "world_size": 1,
        "experimental_fitting_supervision": SUPERVISION,
        "public_input_cache": COHORT,
        "release": {"manifest_sha256": "r" * 64, "binding_sha256": "b" * 64},
    }


class Trainer:
    """Stands in for the canonical trainer's exact on-disk contract."""

    def __init__(self, env, plan=None, steps=2810):
        self.env, self.plan, self.calls, self.steps = env, plan or {}, [], steps

    def __call__(self, config, out, extra, logs, seconds):
        arm = out.name
        self.calls.append((arm, list(extra)))
        behaviour = self.plan.get(arm, ["complete"]).pop(0) if self.plan.get(arm) else "complete"
        if behaviour == "fail":
            return 1
        b = binding(self.env, arm)
        out.mkdir(parents=True, exist_ok=True)
        if not extra:
            write(
                out / "run.json",
                {
                    "binding": b,
                    "diagnostic_run": False,
                    "training_studies": 2247,
                    "outer_studies": 1156,
                    "encoder_trainable_parameters": launch.PINNED[arm]["trainable_parameters"],
                    "held_out_confirmation_and_gold_used": False,
                },
            )
        else:
            assert extra == ["--resume", str(out / "checkpoint.pt")]
        (out / "checkpoint.pt").write_bytes(f"{arm}-{len(self.calls)}".encode())
        if behaviour == "stop":
            (out / "invocation.json").unlink(missing_ok=True)
            write(
                out / "invocation.json",
                {
                    "status": "checkpointed_for_resume",
                    "exit_code": 75,
                    "binding_sha256": launch.canonical(b),
                    "checkpoint_sha256": launch.sha(out / "checkpoint.pt"),
                },
            )
            return 75
        for e in range(1, 11):
            write(out / "epoch-checkpoints" / f"epoch-{e:03d}.json", {"epoch": e})
        (out / "outer_predictions.csv").write_text("study_uid\n")
        write(
            out / "result.json",
            {
                "status": "completed",
                "diagnostic_run": False,
                "optimizer_steps": self.steps,
                "training_studies": 2247,
                "outer_studies": 1156,
                "checkpoint_selection": "fixed_final_epoch_no_outer_selection",
                "binding_sha256": launch.canonical(b),
                "checkpoint_sha256": launch.sha(out / "checkpoint.pt"),
                "predictions_sha256": launch.sha(out / "outer_predictions.csv"),
            },
        )
        return 0


def go(env, trainer):
    return launch.run(
        env.root,
        env.adm,
        execute=trainer,
        placement=lambda funding: "NVIDIA H100 PCIe",
        remaining=lambda job: 48 * 3600.0,
        pins=env.pins,
    )


def refund(env, **edit):
    funding = dict(env.funding, **edit)
    (env.root / "production-funding.json").unlink()
    write(env.root / "production-funding.json", funding)


def test_pins_match_the_actual_committed_CPU_admission():
    record = json.loads((AUDIT / "actual-CPU-accepted.json").read_text())
    receipt = record["receipt"]
    assert record["receipt_sha256"] == launch.PINS["CPU_receipt_sha256"]
    assert receipt["status"] == launch.CPU_STATUS
    assert receipt["CPU_job"] == launch.PINS["CPU_job"]
    assert receipt["source_git"] == launch.PINS["source_git"]
    assert receipt["recipe_sha256"] == launch.PINS["recipe_sha256"]
    assert receipt["wrapper_sha256"] == launch.PINS["recovery_wrapper_sha256"]
    assert (
        receipt["recovery_source_manifest_sha256"] == launch.PINS["recovery_source_manifest_sha256"]
    )
    assert receipt["encoder_spec"]["path_sha256"] == launch.PINS["encoder_spec_path_sha256"]
    assert (
        receipt["future_obligations"]["production_fixed10"]["checkpoint_upper_bound_bytes"]
        == launch.PINS["checkpoint_upper_bound_bytes"]
    )
    assert receipt["GPU_acceptance_inferred"] is False
    for arm, name in launch.ARMS.items():
        assert launch.sha(REPO / launch.RECIPE / name) == launch.PINS["recipe_sha256"][arm]
        assert (
            receipt["arms"][arm]["encoder_learned_parameters"]
            == (launch.PINNED[arm]["trainable_parameters"])
        )
        assert receipt["arms"][arm]["encoder_lr"] == launch.PINNED[arm]["lr"]
    assert (
        launch.sha(REPO / "scripts/newton/run_dense_suffix_recovery.py")
        == (launch.PINS["recovery_wrapper_sha256"])
    )
    assert launch.sha(REPO / "scripts/newton/final_fit_epoch_publisher.py") == pub.TRANSPORT_SHA


def test_production_configs_are_the_unchanged_recipes():
    for name in launch.ARMS.values():
        recipe = json.loads((REPO / launch.RECIPE / name).read_text())
        spec = {k: v for k, v in recipe["encoder"].items() if k != "suffix_adapter"}
        assert launch.production_config(recipe, spec) == recipe


def test_both_arms_run_sequentially_once_to_fixed_final(env):
    trainer = Trainer(env)
    assert go(env, trainer) == 0
    assert trainer.calls == [("suffix_r32", []), ("dense", [])]
    terminal = json.loads((env.root / "terminal.json").read_text())
    assert terminal["status"] == launch.TERMINAL_STATUS and terminal["accuracy_claimed"] is False
    admission = json.loads((env.root / "production-admission.json").read_text())
    assert admission["GPU_recovery_receipt_sha256"] == env.funding["GPU_recovery_receipt_sha256"]
    assert admission["CPU_receipt_sha256"] == env.pins["CPU_receipt_sha256"]
    assert go(env, trainer) == 0 and len(trainer.calls) == 2


def test_exit75_is_resumed_exactly_only_by_a_later_launch(env):
    trainer = Trainer(env, {"suffix_r32": ["stop", "complete"]})
    assert go(env, trainer) == 75
    assert trainer.calls == [("suffix_r32", [])]
    journal = env.root / "journal" / "suffix_r32"
    terminal = json.loads((journal / "invocation-01-terminal.json").read_text())
    assert terminal["returncode"] == 75 and terminal["resumable"] is True
    assert go(env, trainer) == 0
    out = env.root / "production" / "suffix_r32"
    assert trainer.calls[1] == ("suffix_r32", ["--resume", str(out / "checkpoint.pt")])
    assert trainer.calls[2] == ("dense", [])
    intent = json.loads((journal / "invocation-02-intent.json").read_text())
    assert intent["mode"] == "resume" and intent["resume_checkpoint_sha256"] is not None
    assert json.loads((journal / "invocation-01-intent.json").read_text())["mode"] == "fresh"


def test_intent_without_terminal_is_uncertain_replay(env):
    write(env.root / "journal" / "suffix_r32" / "invocation-01-intent.json", {"arm": "r32"})
    trainer = Trainer(env)
    with pytest.raises(ValueError, match="uncertain replay"):
        go(env, trainer)
    assert trainer.calls == []


def test_changed_exit75_checkpoint_is_never_resumed(env):
    trainer = Trainer(env, {"suffix_r32": ["stop"]})
    assert go(env, trainer) == 75
    (env.root / "production" / "suffix_r32" / "checkpoint.pt").write_bytes(b"other")
    with pytest.raises(ValueError, match="exact resume refused"):
        go(env, trainer)
    assert len(trainer.calls) == 1
    assert not (env.root / "journal" / "suffix_r32" / "invocation-02-intent.json").exists()


def test_failed_trainer_is_preserved_and_never_retried(env):
    trainer = Trainer(env, {"suffix_r32": ["fail"]})
    with pytest.raises(ValueError, match="originals preserved"):
        go(env, trainer)
    with pytest.raises(ValueError, match="no automatic retry"):
        go(env, trainer)
    assert len(trainer.calls) == 1


def test_wrong_completion_counters_refused(env):
    with pytest.raises(ValueError, match="2810-update"):
        go(env, Trainer(env, steps=2809))
    assert not (env.root / "terminal.json").exists()


def _source(env):
    (env.code / "rsna/x.py").write_text("x = 2\n")


def _recipe(env):
    path = env.code / launch.RECIPE / "dense-config.json"
    path.write_text(path.read_text().replace('"encoder_lr": 1e-05', '"encoder_lr": 1e-04'))


def _config(env):
    path = env.root / "dense-config.json"
    value = json.loads(path.read_text())
    path.unlink()
    write(path, dict(value, epochs=9))


def _roster(env):
    env.fold.write_text("study_uid,fold\na,1\n")


def _funded_roster(env):
    refund(env, folds_sha256="0" * 64)


def _base(env):
    path = env.adm / "encoder-spec.json"
    value = json.loads(path.read_text())
    path.unlink()
    write(path, dict(value, weights_sha256="0" * 64))


def _missing_gpu(env):
    (env.adm / "recovery" / "accepted.json").unlink()


def _prepared_receipt(env):
    path = env.adm / "recovery" / "accepted.json"
    value = json.loads(path.read_text())
    path.unlink()
    refund(env, GPU_recovery_receipt_sha256=write(path, dict(value, status="prepared_not_run")))


def _unadmitted_gpu(env):
    refund(env, GPU_recovery_receipt_sha256="0" * 64)


def _recovery_evidence(env):
    (env.adm / "recovery" / "dense" / "full" / "checkpoint.pt").write_bytes(b"invented")


def _engineering_funding(env):
    refund(env, checkpoint_bytes=8 * 2**30, runtime_bytes=2**30, publication_bytes=2**30)


def _launcher(env):
    with (env.root / "launch.py").open("a") as stream:
        stream.write("\n# edited\n")


def _cpu_receipt(env):
    path = env.adm / "CPU-accepted.json"
    path.write_text(path.read_text() + " ")


@pytest.mark.parametrize(
    "edit",
    [
        _source,
        _recipe,
        _config,
        _roster,
        _funded_roster,
        _base,
        _missing_gpu,
        _prepared_receipt,
        _unadmitted_gpu,
        _recovery_evidence,
        _engineering_funding,
        _launcher,
        _cpu_receipt,
    ],
)
def test_changed_or_unadmitted_preconditions_refused_before_any_trainer(env, edit):
    edit(env)
    trainer = Trainer(env)
    with pytest.raises(ValueError):
        go(env, trainer)
    assert trainer.calls == []
    assert not (env.root / "production").exists()
    assert not (env.root / "production-admission.json").exists()


class Api:
    def __init__(self, fail_commit=False, corrupt=False):
        self.store, self.fail_commit, self.corrupt, self.commits = {}, fail_commit, corrupt, 0

    def repo_info(self, repo, repo_type, revision=None):
        return SimpleNamespace(private=True, sha="parent")

    def get_paths_info(self, repo, paths, repo_type, revision):
        return [SimpleNamespace(path=p) for p in paths if p in self.store]

    def create_commit(self, repo, repo_type, parent_commit, commit_message, operations, **kw):
        self.commits += 1
        if self.fail_commit:
            raise TimeoutError("lost response")
        for op in operations:
            self.store[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
        return SimpleNamespace(oid="rev1")

    def hf_hub_download(self, repo, filename, repo_type, revision, local_dir, force_download):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.store[filename] + (b"x" if self.corrupt else b""))
        return str(path)


@pytest.fixture
def epoch(env, monkeypatch, tmp_path):
    license_file = tmp_path / "LICENSE.md"
    license_file.write_text("license\n")
    original = pub.transport

    def transport(root):
        core = original(root)
        core.LICENSE, core.LICENSE_SHA = license_file, launch.sha(license_file)
        return core

    monkeypatch.setattr(pub, "transport", transport)
    assert go(env, Trainer(env, {"suffix_r32": ["stop"]})) == 75
    out = env.root / "production" / "suffix_r32"
    b = json.loads((out / "run.json").read_text())["binding"]
    state = {
        "schema_version": 2,
        "binding": b,
        "epoch": 1,
        "next_update": 0,
        "step": 281,
        "scheduled_epoch": True,
        "optimizer_boundary": True,
        "model_storage": {
            "format": "frozen_base_delta_v1",
            "learned_tensors": {"encoder.lora": {}, "head.w": {}},
            "frozen_tensors": {"encoder.base": {}},
        },
        "model": {"encoder.lora": torch.zeros(7_077_888), "head.w": torch.ones(2)},
        "optimizer": {"state": {0: {"step": 281}}},
        "rng_by_rank": [{"torch": 1, "cuda": [1], "python": 1, "numpy": 1}],
        "amp_scaler": {"scale": 128.0},
    }

    def save(**edit):
        value = copy.deepcopy(state)
        value.update(edit)
        snapshots = out / "epoch-checkpoints"
        snapshots.mkdir(exist_ok=True)
        checkpoint = snapshots / "epoch-001.pt"
        torch.save(value, checkpoint)
        (snapshots / "epoch-001.json").unlink(missing_ok=True)
        write(snapshots / "epoch-001.json", pub.expected_snapshot(launch, b, 1, checkpoint))

    save()
    return save


def publish(env, api):
    return pub.publish(
        "suffix_r32",
        1,
        env.root,
        env.adm,
        stdin=io.StringIO(json.dumps({"token": "hf_test"})),
        api_factory=lambda token: api,
        operation=lambda path_in_repo, path_or_fileobj: SimpleNamespace(
            path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj
        ),
        lock=contextlib.nullcontext(),
        pins=env.pins,
        placement=lambda node: None,
    )


def test_epoch_published_once_with_complete_redownload(env, epoch):
    api = Api()
    receipt = publish(env, api)
    assert receipt["file_count"] == 5 and receipt["revision"] == "rev1"
    assert receipt["prefix"] == pub.prefix_for("suffix_r32", 1)
    assert all(p.startswith(pub.PREFIX + "/suffix_r32/epoch-001/") for p in api.store)
    with pytest.raises(ValueError, match="read-only reconciliation"):
        publish(env, api)
    assert api.commits == 1


def test_uncertain_publication_is_never_retried(env, epoch):
    api = Api(fail_commit=True)
    with pytest.raises(RuntimeError, match="uncertain"):
        publish(env, api)
    journal = env.root / "epoch-publication" / "suffix_r32" / "epoch-001"
    assert (journal / "uncertain.json").exists() and (journal / "intent.json").exists()
    before = sorted(p.name for p in journal.iterdir())
    with pytest.raises(ValueError, match="read-only reconciliation"):
        publish(env, Api())
    result = pub.reconcile(
        "suffix_r32",
        1,
        env.root,
        stdin=io.StringIO(json.dumps({"token": "hf_test"})),
        api_factory=lambda token: api,
    )
    assert result["status"] == "uncertain_requires_owner_review"
    assert sorted(p.name for p in journal.iterdir()) == before
    assert api.commits == 1


def test_redownload_mismatch_is_not_verified(env, epoch):
    with pytest.raises(ValueError, match="re-download"):
        publish(env, Api(corrupt=True))
    journal = env.root / "epoch-publication" / "suffix_r32" / "epoch-001"
    assert (journal / "commit.json").exists() and not (journal / "verified.json").exists()


@pytest.mark.parametrize(
    "edit",
    [
        {"step": 280},
        {"scheduled_epoch": False},
        {
            "model": {
                "encoder.lora": torch.full((7_077_888,), float("nan")),
                "head.w": torch.ones(2),
            }
        },
        {"model": {"encoder.lora": torch.zeros(10), "head.w": torch.ones(2)}},
        {"rng_by_rank": [{"torch": 1, "cuda": None, "python": 1, "numpy": 1}]},
    ],
)
def test_incomplete_epoch_refused_before_intent(env, epoch, edit):
    if "model" in edit:
        edit = dict(
            edit,
            model_storage={
                "format": "frozen_base_delta_v1",
                "learned_tensors": dict.fromkeys(edit["model"], {}),
                "frozen_tensors": {},
            },
        )
    epoch(**edit)
    api = Api()
    with pytest.raises(ValueError):
        publish(env, api)
    assert not (env.root / "epoch-publication").exists() and api.commits == 0


def test_publisher_requires_actual_gpu_acceptance(env, epoch):
    (env.adm / "recovery" / "accepted.json").unlink()
    with pytest.raises(ValueError, match="GPU recovery"):
        publish(env, Api())
    assert not (env.root / "epoch-publication").exists()
