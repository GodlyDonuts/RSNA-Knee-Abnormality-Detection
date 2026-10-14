"""Profiler admission of the public-triplet source-slot/window-bag recipes.

Engineering only: CPU fixtures prove setup/first-update parity with the unchanged
trainer; they are never throughput evidence or an accuracy result.
"""

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_source_matched_slot import (
    CPU,
    ENGINEERING,
    OFFICIAL,
    assert_same,
    cohort,  # noqa: F401  (module fixture)
    fake_create_encoder,
)
from test_source_matched_slot import RECIPE as SLOT
from test_window_bag import RECIPE as BAG

from rsna.competitive import public_input as pi
from rsna.competitive import throughput
from rsna.competitive.checkpoint import capture_rng
from rsna.competitive.data import collate_studies
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS
from scripts.research.build_label_source_overlay import build as build_overlay

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {
    "slot": SLOT / "r32-config.json",
    "bag-mean": BAG / "mean-r32-config.json",
    "bag-query": BAG / "query-r32-config.json",
}
WINDOWS = {"slot": 1, "bag-mean": 4, "bag-query": 4}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def supervised(cohort):  # noqa: F811
    """A real ``label_source_v1`` overlay bound to the fixture's release and folds."""
    base = cohort["base"] / "profiler-supervision"
    (base / "folds").mkdir(parents=True)
    studies, folds = base / "studies.parquet", base / "folds/fold.csv"
    cohort["manifest"].to_parquet(studies)
    cohort["folds"].to_csv(folds, index=False)
    fitting = cohort["folds"].study_uid[cohort["folds"].fold != 0].tolist()
    source = base / "source.csv"
    pd.DataFrame(
        [
            {
                "StudyInstanceUID": uid,
                **{t: 0.25 + 0.5 * ((i + j) % 2) for j, t in enumerate(TARGETS)},
            }
            for i, uid in enumerate(fitting[:10])
        ]
    ).to_csv(source, index=False)
    overlay = base / "overlay.json"
    build_overlay(
        studies=studies,
        folds=folds,
        fold=0,
        source=source,
        output=overlay,
        source_name="synthetic",
        source_revision="rev",
    )
    return dict(
        overlay=overlay,
        folds=base / "folds",
        release=dict(cohort["release"], manifest_sha256=_sha(studies)),
        policy={"enabled": True, "kind": "label_source_v1", "sha256": _sha(overlay)},
    )


def recipe(arm, supervision=None):
    values = dict(json.loads(RECIPES[arm].read_text()), **ENGINEERING)
    values["fitting_supervision"] = copy.deepcopy(supervision)
    return values


def _patch_release(monkeypatch, cohort, supervised):  # noqa: F811
    from rsna.competitive import train

    monkeypatch.setattr(
        train,
        "load_verified_release",
        lambda *_: (cohort["manifest"].copy(), dict(supervised["release"])),
    )


def run_trainer(cohort, supervised, values, output, stop, seen):  # noqa: F811
    """Unchanged ``_run_locked``; records its initial learned/base/RNG/optimizer state."""
    from rsna.competitive import train

    args = argparse.Namespace(
        studies=cohort["base"] / "unused",
        artifact_root=cohort["base"],
        folds=supervised["folds"],
        series=cohort["series"],
        cache=cohort["base"] / "cache",
        output=output,
        resume=None,
        stop_after_updates=stop,
        fitting_supervision=supervised["overlay"] if values["fitting_supervision"] else None,
    )

    def snapshot(args, model, guard):
        seen.update(model=copy.deepcopy(model.state_dict()), rng=capture_rng(CPU))
        return None

    class RecordingAdamW(torch.optim.AdamW):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen["optimizer"] = copy.deepcopy(self.state_dict())

    patch = pytest.MonkeyPatch()
    _patch_release(patch, cohort, supervised)
    patch.setattr(train, "initialize_private_mrnet", snapshot)
    patch.setattr(torch.optim, "AdamW", RecordingAdamW)
    output.mkdir(parents=True, exist_ok=True)
    try:
        return train._run_locked(
            args,
            TrainConfig(**values),
            [False],
            0,
            1,
            CPU,
            fake_create_encoder,
            None,
            CompetitiveStudyNet,
            lambda *_: cohort["folds"].copy(),
        )
    finally:
        patch.undo()


def profiler_args(cohort, supervised, values, tmp_path):  # noqa: F811
    config = tmp_path / "config.json"
    config.write_text(json.dumps(values))
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "settings": {
                    "reference": {"overrides": {}},
                    "mb4-acc2-chunk32-w0": {
                        "overrides": throughput.CANDIDATE_SETTINGS["mb4-acc2-chunk32-w0"]
                    },
                }
            }
        )
    )
    return argparse.Namespace(
        config=config,
        settings=settings,
        studies=cohort["base"] / "unused",
        artifact_root=cohort["base"],
        folds=supervised["folds"],
        series=cohort["series"],
        cache=cohort["base"] / "cache",
        receipt=tmp_path / "receipt.json",
        setting="reference",
        measured_updates=1,
        warmup_updates=1,
        max_wall_seconds=600.0,
        expected_config_sha256=_sha(config),
        expected_manifest_sha256=supervised["release"]["manifest_sha256"],
        expected_series_sha256=supervised["release"]["series_sha256"],
        expected_source_revision="synthetic",
        expected_torch_version=str(torch.__version__),
        expected_device_name="cpu",
        evict_page_cache=False,
        device="cpu",
        fitting_supervision=supervised["overlay"] if values["fitting_supervision"] else None,
        expected_public_index_sha256=_sha(
            cohort["base"] / "cache" / pi.CACHE_DIRECTORY / "index.json"
        ),
    )


def factories(cohort):  # noqa: F811
    return dict(
        create_encoder=fake_create_encoder,
        dataset_class=None,
        model_class=CompetitiveStudyNet,
        load_folds=lambda *_: cohort["folds"].copy(),
        collate=collate_studies,
    )


def build_state(cohort, supervised, values, tmp_path, monkeypatch):  # noqa: F811
    _patch_release(monkeypatch, cohort, supervised)
    made = factories(cohort)
    args = profiler_args(cohort, supervised, values, tmp_path)
    return args, throughput.build_profile_state(
        args,
        TrainConfig(**values),
        CPU,
        *[made[n] for n in ("create_encoder", "dataset_class", "model_class", "load_folds")],
    )


def _script():
    spec = importlib.util.spec_from_file_location(
        "profile_training_throughput", ROOT / "scripts/newton/profile_training_throughput.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "arm,supervise",
    [("slot", False), ("slot", True), ("bag-mean", False), ("bag-query", True)],
)
def test_profiler_matches_unchanged_trainer_setup_and_first_updates(
    cohort,  # noqa: F811
    supervised,
    tmp_path,
    monkeypatch,
    arm,
    supervise,
):
    values = recipe(arm, supervised["policy"] if supervise else None)
    seen: dict = {}
    trainer = tmp_path / "trainer"
    assert run_trainer(cohort, supervised, values, trainer, 2, seen) == 75
    _, state = build_state(cohort, supervised, values, tmp_path, monkeypatch)

    # Initial learned + frozen base parameters, RNG and optimizer groups.
    assert_same(state["model"].state_dict(), seen["model"], "initial_model")
    assert_same(capture_rng(CPU), seen["rng"], "initial_rng")
    assert_same(state["optimizer"].state_dict(), seen["optimizer"], "initial_optimizer")
    backbone = state["model"].encoder.backbone.state_dict()
    assert all(torch.equal(backbone[k], v) for k, v in OFFICIAL.items())
    if supervise:
        assert state["supervision_receipt"]["enabled"] is True
        assert state["supervision_receipt"]["applied_cells"] > 0
    assert state["public_receipt"]["index_sha256"] == _sha(
        cohort["base"] / "cache" / pi.CACHE_DIRECTORY / "index.json"
    )

    config = TrainConfig(**values)
    profile = throughput.profile_updates(
        state, config, CPU, throughput.validate_caps(1, 1, 600.0), collate_studies
    )
    assert profile["optimizer_steps"] == 2
    checkpoint = torch.load(trainer / "checkpoint.pt", weights_only=True)
    assert (checkpoint["epoch"], checkpoint["next_update"], checkpoint["step"]) == (1, 0, 2)
    stored = state["delta_guard"].checkpoint_model(state["model"])
    for key in ("model", "model_storage"):
        assert_same(stored[key], checkpoint[key], key)
    assert_same(state["optimizer"].state_dict(), checkpoint["optimizer"], "optimizer")
    assert_same(state["scaler"].state_dict(), checkpoint["amp_scaler"], "scaler")
    assert_same(capture_rng(CPU), checkpoint["rng_by_rank"][0], "rng")
    history = json.loads((trainer / "invocation.json").read_text())["updates"]
    fields = ("loss", "gradient_norm", "observed_cells", "optimizer_steps")
    assert [tuple(r[f] for f in fields) for r in profile["records"]] == [
        tuple(e[f] for f in fields) for e in history
    ]

    for record in profile["records"]:
        assert record["studies"] == 8 and record["microbatch_studies"] == [1] * 8
        # True acquired-window counts, equal to what the encoder actually received.
        assert record["views"] == record["encoder_images"] > 0
        assert record["views"] % WINDOWS[arm] == 0
        assert record["phase_seconds"]["cpu_transform"] > 0
        assert 0 < record["phase_seconds"]["encoder_forward"] <= record["phase_seconds"]["forward"]
        assert record["phase_seconds"]["encoder_backward"] > 0
    assert "forward_features" not in vars(state["model"].encoder)


def test_acquired_window_counts_follow_slot_masks():
    batch = {
        "sample_valid": torch.tensor([True, True, False]),
        "series_mask": torch.tensor([[1, 0, 1, 0, 0, 0], [1] * 6, [1] * 6], dtype=torch.bool),
        "slice_mask": torch.ones(3, 6, 10, dtype=torch.bool),
    }
    slot = json.loads(RECIPES["slot"].read_text())["classifier_strategy"]
    bag = json.loads(RECIPES["bag-mean"].read_text())["classifier_strategy"]
    assert throughput.encoded_views(batch, slot) == 8
    assert throughput.encoded_views(batch, bag) == 32
    assert throughput.encoded_views(batch, None) == 120


def test_microbatch4_chunk32_profiles_same_update_membership(
    cohort,  # noqa: F811
    supervised,
    tmp_path,
    monkeypatch,
):
    base = TrainConfig(**recipe("bag-query"))
    candidate = throughput.apply_setting(base, throughput.CANDIDATE_SETTINGS["mb4-acc2-chunk32-w0"])
    _, state = build_state(cohort, supervised, dataclasses.asdict(candidate), tmp_path, monkeypatch)
    (tmp_path / "ref").mkdir()
    reference = throughput.profile_updates(
        build_state(cohort, supervised, recipe("bag-query"), tmp_path / "ref", monkeypatch)[1],
        base,
        CPU,
        throughput.validate_caps(1, 1, 600.0),
        collate_studies,
    )
    profile = throughput.profile_updates(
        state, candidate, CPU, throughput.validate_caps(1, 1, 600.0), collate_studies
    )
    for first, second in zip(reference["records"], profile["records"]):
        assert second["microbatch_studies"] == [4, 4]
        assert first["study_uids"] == second["study_uids"]
        assert first["observed_cells"] == second["observed_cells"]
        assert first["views"] == second["views"] == second["encoder_images"]
    audit = throughput.numerical_audit(base, candidate)
    assert audit["expectation"] == throughput.NOT_BIT_IDENTICAL
    assert any("head dropout" in reason for reason in audit["reasons"])


def test_script_writes_bound_public_receipt_with_public_decomposition(
    cohort,  # noqa: F811
    supervised,
    tmp_path,
    monkeypatch,
):
    values = recipe("bag-query", supervised["policy"])
    _patch_release(monkeypatch, cohort, supervised)
    args = profiler_args(cohort, supervised, values, tmp_path)
    args.setting = "mb4-acc2-chunk32-w0"
    assert _script().run(args, factories(cohort)) == 0
    receipt = json.loads(args.receipt.read_text())
    bindings = receipt["bindings"]
    assert bindings["input_format"] == throughput.PUBLIC_FORMAT
    assert bindings["public_input_cache"]["index_sha256"] == args.expected_public_index_sha256
    assert bindings["fitting_supervision"]["enabled"] is True
    assert bindings["classifier_strategy"] == values["classifier_strategy"]
    decomposition = receipt["input_decomposition"]
    assert decomposition["format"] == throughput.PUBLIC_FORMAT
    index = json.loads((cohort["base"] / "cache" / pi.CACHE_DIRECTORY / "index.json").read_text())
    assert decomposition["studies"] == 16
    assert decomposition["bytes"] == sum(
        index["members"][uid]["bytes"]
        for uid in cohort["folds"].study_uid[cohort["folds"].fold != 0]
    )
    assert all(decomposition[f"{n}_seconds"] > 0 for n in ("read", "checksum", "decompress"))
    assert decomposition["transform_seconds"] > 0
    assert "input_decomposition_unavailable" not in receipt["evidence_blockers"]
    assert {"not_cuda", "diagnostic_study_limit"} <= set(receipt["evidence_blockers"])
    assert receipt["numerical_audit"]["expectation"] == throughput.NOT_BIT_IDENTICAL
    assert receipt["checkpoint_written"] is False and receipt["accuracy_claim"] is False
    assert receipt["production_acceptance"] is False
    assert set(receipt["summary"]["phase_seconds"]) == set(throughput.PHASES)


@pytest.mark.parametrize("value", [None, "0" * 64])
def test_script_refuses_unbound_public_index(
    cohort,  # noqa: F811
    supervised,
    tmp_path,
    monkeypatch,
    value,
):
    values = recipe("slot")
    _patch_release(monkeypatch, cohort, supervised)
    args = profiler_args(cohort, supervised, values, tmp_path)
    args.expected_public_index_sha256 = value
    with pytest.raises(ValueError, match="index"):
        _script().run(args, factories(cohort))
    assert not args.receipt.exists()


def test_decomposition_refuses_incompatible_formats(
    cohort,  # noqa: F811
    supervised,
    tmp_path,
    monkeypatch,
):
    _, state = build_state(cohort, supervised, recipe("slot"), tmp_path, monkeypatch)
    data = state["training_data"]
    assert throughput.input_format(data) == throughput.PUBLIC_FORMAT
    with pytest.raises(ValueError, match="format"):
        throughput.decompose_inputs(data, [0], False, throughput.NATIVE_FORMAT)
    with pytest.raises(ValueError, match="format"):
        throughput.decompose_inputs(object(), [0], False, throughput.PUBLIC_FORMAT)
    result = throughput.decompose_inputs(data, [0, 1], False, throughput.PUBLIC_FORMAT)
    assert result["format"] == throughput.PUBLIC_FORMAT and result["studies"] == 2


@pytest.mark.parametrize("setting", sorted(throughput.CANDIDATE_SETTINGS))
def test_settings_keep_matched_source_arms_identical_except_their_contrast(setting):
    overrides = throughput.CANDIDATE_SETTINGS[setting]
    pairs = [
        (recipe("bag-mean"), recipe("bag-query")),
        (
            recipe("slot"),
            recipe("slot", {"enabled": True, "kind": "label_source_v1", "sha256": "c" * 64}),
        ),
    ]
    for first, second in pairs:
        base_first, base_second = TrainConfig(**first), TrainConfig(**second)
        a, b = (throughput.apply_setting(arm, overrides) for arm in (base_first, base_second))
        assert throughput.recipe_differences(a, b) == throughput.recipe_differences(
            base_first, base_second
        )
        assert a.batch_size * a.accumulation_steps == throughput.CANONICAL_EFFECTIVE_STUDIES
        audit = throughput.numerical_audit(base_first, a)
        if set(audit["changed_fields"]) <= {"workers"}:
            assert audit["expectation"] == throughput.BIT_IDENTICAL
        else:
            assert audit["expectation"] == throughput.NOT_BIT_IDENTICAL
            assert any("window/augmentation draws are" in r for r in audit["reasons"])
            if "batch_size" in audit["changed_fields"]:
                assert any("head dropout" in r for r in audit["reasons"])
