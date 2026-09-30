"""Rank-64 preservation must not accept another experiment or diagnostic weights."""

import hashlib
import importlib.util
import io
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsna.competitive.backbones import _encoder_provenance

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/hf" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = load("rank64_publisher_test", "publish_adapter_a64_epoch.py")
dispatcher = load("rank64_dispatcher_test", "dispatch_adapter_a64_epoch.py")


def actual_recipe():
    config = json.loads(
        (
            ROOT
            / "initiative/experiments/competitive-v1/adapter-rank-capacity-v1/rank64-production-delta-config.json"
        ).read_text()
    )
    architecture = json.loads(
        (ROOT / "configs/competitive/weights/dinov3-large.config.json").read_text()
    )
    config["encoder"] = _encoder_provenance(config["encoder"], architecture, "unused_for_vit")
    return config


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    code = tmp_path / "code"
    source = code / "rsna" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic source fixture, not training evidence.\n")
    monkeypatch.setattr(
        publisher.subprocess,
        "check_output",
        lambda *args, **kwargs: (publisher.TRAINING_GIT + "\n").encode(),
    )
    license_file = tmp_path / "LICENSE.md"
    license_file.write_text("Synthetic license fixture.\n")
    monkeypatch.setattr(publisher, "LICENSE_SHA", publisher.sha(license_file))
    run = tmp_path / "run"
    folder = run / "epoch-checkpoints"
    folder.mkdir(parents=True)
    checkpoint = folder / "epoch-001.pt"
    checkpoint.write_bytes(b"Synthetic payload fixture; never trained or submission eligible.")
    original_binding = json.loads(
        (
            ROOT / "initiative/audits/dinov3-delta-recovery-launch-v1/production-start/run.json"
        ).read_text()
    )["binding"]
    binding = {
        **original_binding,
        "config": actual_recipe(),
        "world_size": 1,
        "code_sha256": {"example.py": publisher.sha(source)},
        "release": {"binding_sha256": publisher.RELEASE_SHA},
        "cache_side": 256,
        "cache_version": "series-u16-v3",
    }
    record = {
        "binding": binding,
        "training_studies": 2247,
        "outer_studies": 1156,
        "encoder_trainable_parameters": 20_447_232,
    }
    meta = {
        "status": "scheduled_epoch_snapshot",
        "epoch": 1,
        "optimizer_steps": 281,
        "checkpoint_file": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": publisher.sha(checkpoint),
        "fold": 0,
        "seed": 20260922,
        "diagnostic_only": False,
        "submission_eligible": True,
        "submission_every_epochs": 1,
        "release_binding_sha256": publisher.RELEASE_SHA,
    }
    snapshot = folder / "epoch-001.json"
    return run, snapshot, code, license_file, record, meta


def write_candidate(candidate):
    run, snapshot, code, license_file, record, meta = candidate
    binding = record["binding"]
    meta["binding_sha256"] = hashlib.sha256(publisher.canonical(binding).encode()).hexdigest()
    meta["release_binding_sha256"] = binding["release"]["binding_sha256"]
    (run / "run.json").write_text(json.dumps(record))
    snapshot.write_text(json.dumps(meta))
    return snapshot, run, license_file, code


def test_rank64_contract_matches_actual_factory_provenance_and_dispatcher():
    assert publisher.canonical(actual_recipe()) == publisher.canonical(
        publisher.expected_training_config()
    )
    assert dispatcher.PREFIX == publisher.PREFIX
    assert dispatcher.PUBLISHER_SHA == publisher.sha(
        ROOT / "scripts/hf/publish_adapter_a64_epoch.py"
    )
    assert "whole-series" not in dispatcher.RUN
    assert dispatcher.RUN.endswith("adapter-a64-sampled-fold0-c9fcc7e-production-v1")


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "rank32",
        "whole_series",
        "all_data",
        "diagnostic",
        "wrong_projection",
        "schedule",
        "changed_base",
        "changed_release",
        "old_cache",
        "source_changed",
        "boolean_batch",
        "partial_epoch",
        "full_storage",
        "changed_membership",
        "changed_target_order",
        "changed_runtime",
    ],
)
def test_rank64_private_epoch_requires_exact_recipe_and_completed_state(candidate, fault):
    run, snapshot, code, license_file, record, meta = candidate
    config = record["binding"]["config"]
    if fault == "rank32":
        config["encoder"]["adapter"]["rank"] = 32
    elif fault == "whole_series":
        config["slice_sampling"] = config["series_sampling"] = "all"
    elif fault == "all_data":
        config["fit_scope"] = "all_development"
        record.update(training_studies=3403, outer_studies=0)
    elif fault == "diagnostic":
        config["diagnostic_study_limit"] = 16
        meta["diagnostic_only"] = True
        meta["submission_eligible"] = False
    elif fault == "wrong_projection":
        config["encoder"]["adapter"]["target_modules"][0] = "wrong.projection"
    elif fault == "schedule":
        config["head_lr"] = 0.0001
    elif fault == "changed_base":
        config["encoder"]["weights_sha256"] = "0" * 64
    elif fault == "changed_release":
        record["binding"]["release"]["binding_sha256"] = "0" * 64
    elif fault == "old_cache":
        record["binding"]["cache_version"] = "series-u16-v2"
    elif fault == "source_changed":
        record["binding"]["code_sha256"]["example.py"] = "0" * 64
    elif fault == "boolean_batch":
        config["batch_size"] = True  # Python equality alone would accept True == 1.
    elif fault == "partial_epoch":
        meta["optimizer_steps"] = 280
    elif fault == "full_storage":
        config["checkpoint_format"] = "full"
    elif fault == "changed_membership":
        record["binding"]["training_uids_sha256"] = "0" * 64
    elif fault == "changed_target_order":
        record["binding"]["targets"] = list(reversed(record["binding"]["targets"]))
    elif fault == "changed_runtime":
        record["binding"]["torch_version"] = "synthetic_other_runtime"
    inputs = write_candidate(candidate)
    if fault is not None:
        with pytest.raises(ValueError):
            publisher.source_files(*inputs)
    else:
        manifest, files = publisher.source_files(*inputs)
        assert manifest["experiment"] == "matched_sampled_rank64"
        assert manifest["adapter"]["trainable_parameters"] == 20_447_232
        assert manifest["checkpoint_format"] == "frozen_base_delta"
        assert manifest["slice_sampling"] == "uniform"
        assert manifest["series_sampling"] == "selected"
        assert manifest["requires_pinned_original_weights"] is True
        assert set(files) == {"checkpoint.pt", "snapshot.json", "DINOv3-LICENSE.md"}


@pytest.mark.parametrize("private", [False, True])
def test_rank64_refuses_public_repo_or_existing_intent_without_upload(
    candidate, tmp_path, monkeypatch, private
):
    snapshot, run, license_file, code = write_candidate(candidate)
    journal = tmp_path / "journal"
    epoch = journal / "epoch-001"
    epoch.mkdir(parents=True)
    (epoch / "intent.json").write_text("{}")
    mutations = []

    class ReadOnlyFakeApi:
        def __init__(self, **kwargs):
            pass

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(private=private, sha="synthetic_parent")

        def create_commit(self, *args, **kwargs):
            mutations.append("commit")
            raise AssertionError("Publication must stop before this mutation")

    monkeypatch.setenv("SLURM_JOB_ID", "synthetic_cpu_allocation")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setattr(publisher.sys, "stdin", io.StringIO('{"token":"hf_synthetic_fixture"}'))
    monkeypatch.setattr(publisher, "HfApi", ReadOnlyFakeApi)
    args = Namespace(
        mode="publish",
        snapshot=snapshot,
        run=run,
        license_file=license_file,
        code_root=code,
        journal=journal,
    )
    with pytest.raises(ValueError, match="private dataset|already attempted"):
        publisher.run(args)
    assert mutations == []
