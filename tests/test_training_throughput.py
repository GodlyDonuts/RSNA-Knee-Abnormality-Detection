import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from rsna.competitive import throughput
from rsna.competitive.loss import effective_batch_loss, observed_counts
from rsna.competitive.sampling import EffectiveBatchSampler
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS
from test_competitive_training import SyntheticStudies

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "initiative/experiments/competitive-v1/public-label-source-v1"
SETTINGS = ROOT / "initiative/experiments/competitive-v1/throughput-profile-v1/settings.json"


def _recipe(name: str) -> TrainConfig:
    config = TrainConfig(**json.loads((RECIPES / f"{name}-config.json").read_text()))
    config.validate()
    return config


def _script():
    spec = importlib.util.spec_from_file_location(
        "profile_training_throughput", ROOT / "scripts/newton/profile_training_throughput.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_settings_file_matches_the_module_proposal():
    settings = json.loads(SETTINGS.read_text())
    assert settings["status"] == "proposal_only_not_dispatched"
    assert {name: value["overrides"] for name, value in settings["settings"].items()} == (
        throughput.CANDIDATE_SETTINGS
    )


@pytest.mark.parametrize("setting", sorted(throughput.CANDIDATE_SETTINGS))
def test_every_setting_keeps_both_matched_arms_identical_except_label_source(setting):
    control, public = _recipe("control"), _recipe("public")
    overrides = throughput.CANDIDATE_SETTINGS[setting]
    first, second = (throughput.apply_setting(arm, overrides) for arm in (control, public))
    for arm in (first, second):
        assert arm.batch_size * arm.accumulation_steps == throughput.CANONICAL_EFFECTIVE_STUDIES
        assert set(throughput.recipe_differences(control, first)) <= set(
            throughput.THROUGHPUT_FIELDS
        )
    assert throughput.recipe_differences(first, second) == throughput.recipe_differences(
        control, public
    )
    assert set(throughput.recipe_differences(control, public)) == {"fitting_supervision"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 4},
        {"batch_size": 2, "accumulation_steps": 2},
        {"workers": 3},
        {"precision": "bf16"},
        {"warmup_updates": 1},
        {"encode_chunk_size": True},
    ],
)
def test_recipe_or_effective_batch_changes_fail_closed(overrides):
    with pytest.raises(ValueError):
        throughput.apply_setting(_recipe("control"), overrides)


def test_numerical_audit_never_claims_bit_equality_for_batch_or_chunk_changes():
    base = _recipe("control")
    for name, overrides in throughput.CANDIDATE_SETTINGS.items():
        audit = throughput.numerical_audit(base, throughput.apply_setting(base, overrides))
        only_workers = set(audit["changed_fields"]) <= {"workers"}
        assert audit["expectation"] == (
            throughput.BIT_IDENTICAL if only_workers else throughput.NOT_BIT_IDENTICAL
        ), name
        if not only_workers:
            assert any("dropout" in reason for reason in audit["reasons"])


def test_microbatch_partition_keeps_update_membership_and_masked_normalization():
    first = list(EffectiveBatchSampler(21, 1, 8, seed=3, epoch=0))
    second = list(EffectiveBatchSampler(21, 4, 2, seed=3, epoch=0))
    flat = [[key[:2] for batch in first[i * 8 : i * 8 + 8] for key in batch] for i in range(3)]
    grouped = [[key[:2] for batch in second[i * 2 : i * 2 + 2] for key in batch] for i in range(3)]
    assert flat == grouped
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(8, len(TARGETS), generator=generator, dtype=torch.float64)
    labels = (torch.rand(8, len(TARGETS), generator=generator) > 0.5).double()
    mask = (torch.rand(8, len(TARGETS), generator=generator) > 0.6).double()
    mask[:, 0] = 0
    one = observed_counts(list(mask.split(1)))
    four = observed_counts(list(mask.split(4)))
    assert torch.equal(one, four)
    single = sum(
        effective_batch_loss(*parts, one)
        for parts in zip(logits.split(1), labels.split(1), mask.split(1))
    )
    pairs = sum(
        effective_batch_loss(*parts, four)
        for parts in zip(logits.split(4), labels.split(4), mask.split(4))
    )
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    observed = one > 0
    expected = ((bce * mask).sum(0)[observed] / one[observed]).mean()
    torch.testing.assert_close(single, expected)
    torch.testing.assert_close(pairs, expected)


def test_import_has_no_gpu_or_secret_side_effects():
    code = (
        "import sys, torch\n"
        "import rsna.competitive.throughput\n"
        "import importlib.util as u\n"
        "s=u.spec_from_file_location('p','scripts/newton/profile_training_throughput.py')\n"
        "u.module_from_spec(s); s.loader.exec_module(u.module_from_spec(s))\n"
        "assert not torch.cuda.is_initialized()\n"
        "assert not {'transformers','huggingface_hub'} & set(sys.modules)\n"
    )
    env = {**os.environ, "HF_TOKEN": "must-not-be-read", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "must-not-be-read" not in result.stdout + result.stderr


def test_phase_clock_synchronizes_exactly_at_boundaries(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    clock = throughput.PhaseClock(torch.device("cuda"))
    assert clock.measure("optimizer", lambda: calls.append("work") or 5) == 5
    assert calls == ["sync", "work", "sync"]
    assert clock.totals["optimizer"] >= 0 and set(clock.totals) == set(throughput.PHASES)
    with pytest.raises(ValueError):
        clock.measure("evaluation", lambda: None)


@pytest.mark.parametrize(
    "caps",
    [
        (0, 1, 10.0),
        (throughput.HARD_MAX_MEASURED_UPDATES + 1, 1, 10.0),
        (2, 0, 10.0),
        (2, throughput.HARD_MAX_WARMUP_UPDATES + 1, 10.0),
        (2, 1, 0.0),
        (2, 1, float("inf")),
        (2, 1, throughput.HARD_MAX_WALL_SECONDS + 1),
        (2, 1, True),
    ],
)
def test_resource_caps_fail_closed(caps):
    with pytest.raises(ValueError):
        throughput.validate_caps(*caps)


def _fixture(tmp_path, monkeypatch):
    from rsna.competitive import train

    rows = []
    for i in range(14):
        row = {"study_uid": f"study{i:02}", "leakage_group": f"group{i}", "split": "train"}
        for j, target in enumerate(TARGETS):
            row[target], row[target + "__mask"] = i % 2, int(j < 6 or i % 3 == 0)
            row[target + "__provenance"] = "synthetic_weak"
        rows.append(row)
    manifest = pd.DataFrame(rows)
    overlay = manifest[["study_uid", "leakage_group"]].copy()
    overlay["fold"] = [0] * 5 + [1] * 9
    series = tmp_path / "series.csv"
    series.write_text("SeriesInstanceUID\na\n")
    binding = {
        "manifest_sha256": "m" * 64,
        "series_sha256": hashlib.sha256(series.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    monkeypatch.setattr(train, "load_verified_release", lambda *args: (manifest.copy(), binding))
    config = TrainConfig(
        diagnostic_study_limit=20,
        epochs=2,
        batch_size=2,
        accumulation_steps=2,
        slices=2,
        side=16,
        max_series=3,
        hidden_dim=16,
        freeze_encoder=False,
        min_free_gib=0,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(dataclasses.asdict(config)))
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "settings": {
                    "reference": {"overrides": {}},
                    "mb1-acc4": {"overrides": {"batch_size": 1, "accumulation_steps": 4}},
                }
            }
        )
    )
    args = argparse.Namespace(
        config=config_path,
        settings=settings,
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=tmp_path / "unused",
        series=series,
        cache=tmp_path,
        receipt=tmp_path / "receipt.json",
        setting="reference",
        measured_updates=1,
        warmup_updates=1,
        max_wall_seconds=600.0,
        expected_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        expected_manifest_sha256=binding["manifest_sha256"],
        expected_series_sha256=binding["series_sha256"],
        expected_source_revision="synthetic",
        expected_torch_version=str(torch.__version__),
        expected_device_name="cpu",
        evict_page_cache=False,
        device="cpu",
    )
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.data import collate_studies
    from rsna.competitive.model import CompetitiveStudyNet

    factories = dict(
        create_encoder=create_encoder,
        dataset_class=SyntheticStudies,
        model_class=CompetitiveStudyNet,
        load_folds=lambda *args: overlay.copy(),
        collate=collate_studies,
    )
    return train, config, args, factories


def test_profiled_updates_match_the_real_trainer_and_receipt_is_bound(tmp_path, monkeypatch):
    train, config, args, factories = _fixture(tmp_path, monkeypatch)
    train_args = argparse.Namespace(
        studies=args.studies,
        artifact_root=args.artifact_root,
        folds=args.folds,
        series=args.series,
        cache=args.cache,
        output=tmp_path / "trainer",
        resume=None,
        stop_after_updates=2,
    )
    status = train._run_locked(
        train_args,
        config,
        [False],
        0,
        1,
        torch.device("cpu"),
        factories["create_encoder"],
        factories["dataset_class"],
        factories["model_class"],
        factories["load_folds"],
    )
    assert status == 75
    trained = torch.load(train_args.output / "checkpoint.pt", weights_only=True)["model"]

    state = throughput.build_profile_state(
        args,
        config,
        torch.device("cpu"),
        factories["create_encoder"],
        factories["dataset_class"],
        factories["model_class"],
        factories["load_folds"],
    )
    caps = throughput.validate_caps(1, 1, 600.0)
    profile = throughput.profile_updates(
        state, config, torch.device("cpu"), caps, factories["collate"]
    )
    assert profile["optimizer_steps"] == 2 and profile["stop_reason"] == (
        "measured_updates_completed"
    )
    profiled = state["model"].state_dict()
    assert profiled.keys() == trained.keys()
    for name, value in trained.items():
        assert torch.equal(profiled[name], value), name
    for record in profile["records"]:
        assert record["studies"] == 4 and record["microbatch_studies"] == [2, 2]
        assert record["views"] == 4 * 3 * 2
        assert set(record["phase_seconds"]) == set(throughput.PHASES)
        assert all(value >= 0 for value in record["phase_seconds"].values())
        assert record["phase_seconds"]["encoder_forward"] <= record["phase_seconds"]["forward"]
        assert sum(record["phase_seconds"][p] for p in ("data_wait", "host_to_device", "forward",
                   "backward", "optimizer")) <= record["wall_seconds"]
        assert record["item_seconds_sum"] <= record["phase_seconds"]["data_wait"]


def test_wall_cap_stops_at_an_update_boundary(tmp_path, monkeypatch):
    _, config, args, factories = _fixture(tmp_path, monkeypatch)
    state = throughput.build_profile_state(
        args, config, torch.device("cpu"), *[factories[n] for n in (
            "create_encoder", "dataset_class", "model_class", "load_folds")]
    )
    times = iter([0.0, 0.0, 1000.0])
    profile = throughput.profile_updates(
        state,
        config,
        torch.device("cpu"),
        throughput.validate_caps(2, 1, 10.0),
        factories["collate"],
        clock=lambda: next(times),
    )
    assert profile["stop_reason"] == "wall_cap" and len(profile["records"]) == 1


def test_script_writes_one_bound_non_evidence_receipt_per_fresh_process(tmp_path, monkeypatch):
    _, _, args, factories = _fixture(tmp_path, monkeypatch)
    script = _script()
    args.setting = "mb1-acc4"
    assert script.run(args, factories) == 0
    receipt = json.loads(args.receipt.read_text())
    assert receipt["setting"] == "mb1-acc4"
    assert receipt["overrides"]["batch_size"] == 1
    assert receipt["overrides"]["accumulation_steps"] == 4
    assert receipt["bindings"]["config_sha256"] == args.expected_config_sha256
    assert receipt["bindings"]["fresh_process"] is True
    assert receipt["release"]["manifest_sha256"] == args.expected_manifest_sha256
    assert receipt["caps"]["max_studies"] == 16
    assert receipt["numerical_audit"]["expectation"] == throughput.NOT_BIT_IDENTICAL
    assert receipt["code_sha256"] == throughput.code_tree_sha256()
    assert receipt["summary"]["measured_updates"] == 1
    assert receipt["summary"]["studies"] == 4
    assert receipt["summary"]["peak_reserved_bytes"] is None
    assert receipt["evidence_eligible"] is False
    assert {"not_cuda", "diagnostic_study_limit", "input_decomposition_unavailable"} <= set(
        receipt["evidence_blockers"]
    )
    assert receipt["accuracy_claim"] is False and receipt["production_acceptance"] is False
    assert receipt["checkpoint_written"] is False
    with pytest.raises(ValueError, match="distinct"):
        script.run(args, factories)


@pytest.mark.parametrize(
    "field",
    [
        "expected_config_sha256",
        "expected_manifest_sha256",
        "expected_series_sha256",
        "expected_source_revision",
        "expected_torch_version",
        "expected_device_name",
    ],
)
def test_script_refuses_any_unbound_source_runtime_or_input(tmp_path, monkeypatch, field):
    _, _, args, factories = _fixture(tmp_path, monkeypatch)
    setattr(args, field, "different")
    with pytest.raises(ValueError):
        _script().run(args, factories)
    assert not args.receipt.exists()


def test_script_refuses_a_process_that_already_initialized_cuda(tmp_path, monkeypatch):
    _, _, args, factories = _fixture(tmp_path, monkeypatch)
    args.device = "cuda"
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="fresh process"):
        _script().run(args, factories)
