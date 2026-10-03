import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset

from rsna.competitive.checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from rsna.competitive.loss import effective_batch_loss, observed_counts
from rsna.competitive.sampling import EffectiveBatchSampler, SeededDataset
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS


def test_accumulation_matches_one_effective_batch_with_uneven_masks():
    torch.manual_seed(5)
    logits = torch.randn(7, 12, requires_grad=True)
    labels = torch.randint(0, 2, logits.shape).float()
    masks = torch.randint(0, 2, logits.shape).float()
    masks[:2] = 0
    masks[:, 11] = 0
    counts = observed_counts([masks[:2], masks[2:6], masks[6:]])
    whole = effective_batch_loss(logits, labels, masks, counts)
    accumulated = sum(
        effective_batch_loss(logits[part], labels[part], masks[part], counts)
        for part in (slice(0, 2), slice(2, 6), slice(6, 7))
    )
    expected = torch.autograd.grad(whole, logits, retain_graph=True)[0]
    actual = torch.autograd.grad(accumulated, logits)[0]
    torch.testing.assert_close(accumulated, whole)
    torch.testing.assert_close(actual, expected)
    assert not actual[masks == 0].any()


def test_hidden_labels_including_nan_never_enter_the_loss():
    logits = torch.zeros(2, 12, requires_grad=True)
    mask = torch.zeros_like(logits)
    mask[0, 0] = 1
    labels = torch.full_like(logits, float("nan"))
    labels[0, 0] = 1
    loss = effective_batch_loss(logits, labels, mask, observed_counts([mask]))
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] < 0
    assert not logits.grad[mask == 0].any()


def test_all_masked_loss_is_zero_and_differentiable():
    logits = torch.randn(2, 12, requires_grad=True)
    mask = torch.zeros_like(logits)
    loss = effective_batch_loss(logits, torch.zeros_like(logits), mask, observed_counts([mask]))
    loss.backward()
    assert loss == 0 and logits.grad is not None and not logits.grad.any()


def _ddp_rank(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(8)
    inputs, labels = torch.randn(8, 3), torch.randint(2, (8, 12)).float()
    masks = torch.zeros_like(labels)
    masks[2, :3] = 1
    masks[4:6, :8] = 1
    masks[6, :11] = 1
    reference = nn.Linear(3, 12)
    effective_batch_loss(reference(inputs), labels, masks, observed_counts([masks])).backward()
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = nn.Linear(3, 12)
        model.load_state_dict(reference.state_dict())
        wrapped = DistributedDataParallel(model)
        pieces = [slice(rank * 2, rank * 2 + 2), slice(4 + rank * 2, 6 + rank * 2)]
        counts = observed_counts([masks[piece] for piece in pieces])
        with wrapped.no_sync():
            effective_batch_loss(
                wrapped(inputs[pieces[0]]), labels[pieces[0]], masks[pieces[0]], counts
            ).backward()
        effective_batch_loss(
            wrapped(inputs[pieces[1]]), labels[pieces[1]], masks[pieces[1]], counts
        ).backward()
        torch.testing.assert_close(model.weight.grad, reference.weight.grad)
        torch.testing.assert_close(model.bias.grad, reference.bias.grad)
    finally:
        dist.destroy_process_group()


def test_cpu_two_rank_accumulation_matches_global_gradient(tmp_path: Path):
    mp.spawn(_ddp_rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


@pytest.mark.parametrize("size", [1, 7, 8, 9, 19])
def test_distributed_padding_has_no_extra_study_exposure(size):
    keys = []
    for rank in range(2):
        sampler = EffectiveBatchSampler(size, 2, 2, seed=7, epoch=2, rank=rank, world_size=2)
        keys.extend(key for batch in sampler for key in batch)
    real = [key[0] for key in keys if key[1]]
    assert sorted(real) == list(range(size))
    assert len(keys) % 8 == 0


def test_resume_sampler_jumps_to_exact_next_update():
    whole = list(EffectiveBatchSampler(31, 2, 3, seed=7, epoch=2, rank=1, world_size=2))
    resumed = list(
        EffectiveBatchSampler(31, 2, 3, seed=7, epoch=2, rank=1, world_size=2, start_update=1)
    )
    assert resumed == whole[3:]


class RandomDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "image": torch.rand(2),
            "python": random.random(),
            "numpy": np.random.random(),
            "label_mask": torch.ones(12),
        }


def test_item_randomness_is_replayable_and_does_not_mutate_parent_rng():
    dataset = SeededDataset(RandomDataset())
    before = capture_rng(torch.device("cpu"))
    first, second = dataset[(0, True, 42)], dataset[(0, True, 42)]
    torch.testing.assert_close(first["image"], second["image"], rtol=0, atol=0)
    assert first["python"] == second["python"] and first["numpy"] == second["numpy"]
    after = capture_rng(torch.device("cpu"))
    assert before["python"] == after["python"] and before["numpy"] == after["numpy"]
    torch.testing.assert_close(before["torch"], after["torch"], rtol=0, atol=0)
    assert not dataset[(0, False, 42)]["label_mask"].any()


def test_cpu_augmentation_never_calls_the_cuda_seeding_api(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dataset must not reset model CUDA generators")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    assert SeededDataset(RandomDataset())[(0, True, 42)]["sample_valid"]


def test_checkpoint_restores_all_rng_and_rejects_changed_world(tmp_path):
    device = torch.device("cpu")
    binding = {"world_size": 1, "code": "fixed"}
    rng = capture_rng(device)
    checkpoint = {
        "schema_version": 1,
        "binding": binding,
        "optimizer_boundary": True,
        "rng_by_rank": [rng],
    }
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, checkpoint, min_free_gib=0)
    expected = [torch.rand(2), random.random(), np.random.random()]
    loaded = load_checkpoint(path, binding)
    restore_rng(loaded["rng_by_rank"][0], device)
    actual = [torch.rand(2), random.random(), np.random.random()]
    torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
    assert expected[1:] == actual[1:]
    with pytest.raises(ValueError, match="differs"):
        load_checkpoint(path, {"world_size": 2, "code": "fixed"})


def test_failed_checkpoint_reserve_preserves_previous_file(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous")
    with pytest.raises(OSError, match="reserve"):
        save_checkpoint(path, {"x": torch.ones(2)}, min_free_gib=10**9)
    assert path.read_bytes() == b"previous"


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"diagnostic_study_limit": 2, "batch_size": True},
        {"diagnostic_study_limit": 2, "precision": "fp16"},
        {"diagnostic_study_limit": 2, "head_lr": float("nan")},
        {"diagnostic_study_limit": 2, "fit_scope": "include_gold"},
        {"diagnostic_study_limit": 2, "slice_sampling": "silent_truncate"},
        {"diagnostic_study_limit": 2, "series_sampling": "omit_axial"},
        {"diagnostic_study_limit": 2, "slice_context": "neighbor3", "pooling": "patch_attention"},
        {
            "diagnostic_study_limit": 2,
            "pooling": "patch_attention",
            "encoder": {"architecture": "dinov2_b14_reg"},
        },
    ],
)
def test_invalid_training_config_fails_closed(changes):
    with pytest.raises(ValueError):
        TrainConfig(**changes).validate()


def test_diagnostic_configuration_is_explicit():
    TrainConfig(diagnostic_study_limit=4, slices=2, side=32, max_series=3).validate()
    TrainConfig(
        diagnostic_study_limit=4,
        slices=2,
        side=32,
        max_series=3,
        pooling="patch_attention",
    ).validate()


class SyntheticStudies(Dataset):
    """Narrow training-loop fixture; real release acceptance is a separate run."""

    cache_side = 256
    source_revision = "synthetic"

    def __init__(self, studies, series, cache, augment=False, **kwargs):
        self.studies = studies.reset_index(drop=True)
        self.variable = kwargs.get("slice_sampling") == "all"

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        row = self.studies.iloc[index]
        s, k = (3 + index % 2, 2 + index % 3) if self.variable else (3, 2)
        return {
            "images": torch.rand(s, k, 1, 16, 16),
            "features": torch.eye(s, 5),
            "series_mask": torch.ones(s, dtype=torch.bool),
            "slice_mask": torch.ones(s, k, dtype=torch.bool),
            "positions": torch.zeros(s, k, 2),
            "labels": torch.tensor([float(row[t]) for t in TARGETS]),
            "label_mask": torch.tensor([float(row[t + "__mask"]) for t in TARGETS]),
            "study_uid": row.study_uid,
        }


def _assert_tree_equal(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_tree_equal(left, right)
    else:
        assert first == second


@pytest.mark.parametrize("fit_scope", ["outer_fold", "all_development"])
@pytest.mark.parametrize(
    "slice_context,relation_case,witness_weight",
    [
        ("none", None, None),
        ("neighbor3", None, None),
        ("none", "physical", None),
        ("none", "shuffled", None),
        ("none", "zero", None),
        ("none", None, 0),
        ("none", None, 1),
    ],
)
def test_training_mid_epoch_resume_matches_every_checkpoint_value(
    tmp_path, monkeypatch, fit_scope, slice_context, relation_case, witness_weight
):
    from rsna.competitive import train
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.model import CompetitiveStudyNet

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
    series_path = tmp_path / "series.csv"
    series_path.write_text("SeriesInstanceUID\na\n")
    release_binding = {
        "series_sha256": hashlib.sha256(series_path.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    monkeypatch.setattr(
        train, "load_verified_release", lambda *args: (manifest.copy(), release_binding)
    )
    config = TrainConfig(
        slice_context=slice_context,
        slice_sampling="all" if slice_context == "neighbor3" else "uniform",
        series_sampling="all" if slice_context == "neighbor3" else "selected",
        fit_scope=fit_scope,
        diagnostic_study_limit=20,
        epochs=2,
        batch_size=2,
        accumulation_steps=2,
        slices=2,
        side=16,
        max_series=3,
        hidden_dim=16,
        freeze_encoder=False,
        augment=True,
        min_free_gib=0,
        witness=(
            None
            if witness_weight is None
            else dict(patches=4, slices=2, seed=20261003, weight=witness_weight)
        ),
        relation=(
            None
            if relation_case is None
            else dict(
                weight=0 if relation_case == "zero" else 0.05,
                max_pairs=3,
                target_mode="shuffled" if relation_case == "shuffled" else "physical",
                hidden_dim=12,
                head_seed=19,
            )
        ),
    )
    args = argparse.Namespace(
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=tmp_path / "unused",
        series=series_path,
        cache=tmp_path,
        output=tmp_path / "full",
        resume=None,
        stop_after_updates=None,
    )

    def execute():
        class GeometryStudies(SyntheticStudies):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                item["positions"][..., 0] = torch.arange(item["positions"].shape[1]) * 0.03
                item["positions"][..., 1] = 1
                return item

        return train._run_locked(
            args,
            config,
            [False],
            0,
            1,
            torch.device("cpu"),
            create_encoder,
            SyntheticStudies if relation_case is None else GeometryStudies,
            CompetitiveStudyNet,
            lambda *args: overlay.copy(),
        )

    assert execute() == 0
    full = torch.load(args.output / "checkpoint.pt", weights_only=True)
    prediction_file = args.output / "outer_predictions.csv"
    expected_predictions = prediction_file.read_bytes() if prediction_file.exists() else None
    if fit_scope == "all_development":
        result = json.loads((args.output / "result.json").read_text())
        assert result["training_studies"] == 14 and result["outer_studies"] == 0
        assert result["metrics"] is None and result["evaluation_source"] is None
        assert result["optimizer_steps"] == 8  # ceil(14/4) per epoch, including masked tail
        assert expected_predictions is None
    args.output = tmp_path / "resumed"
    args.stop_after_updates = 1
    assert execute() == 75
    partial = torch.load(args.output / "checkpoint.pt", weights_only=True)
    assert partial["epoch"] == 0 and partial["next_update"] == 1
    assert not (args.output / "outer_predictions.csv").exists()
    args.resume = args.output / "checkpoint.pt"
    args.stop_after_updates = None
    assert execute() == 0
    actual = torch.load(args.output / "checkpoint.pt", weights_only=True)
    _assert_tree_equal(full, actual)
    assert expected_predictions == (
        (args.output / "outer_predictions.csv").read_bytes() if fit_scope == "outer_fold" else None
    )
    assert (args.output / "outer_predictions.csv").exists() == (fit_scope == "outer_fold")

    # The Slurm supervisor's durable request must stop the real trainer at an
    # optimizer boundary even if a worker misses the OS signal. A later resume
    # session ignores the old request and reaches the exact uninterrupted state.
    args.output = tmp_path / "external-stop"
    args.output.mkdir()
    args.resume = None
    monkeypatch.setenv("RSNA_STOP_SESSION", "current-session")
    (args.output / train.STOP_REQUEST_NAME).write_text(
        '{"status":"checkpoint_requested","session":"current-session",'
        '"checkpoint_name":"checkpoint.pt","requested_wall":1.25}'
    )
    assert execute() == 75
    assert not (args.output / "outer_predictions.csv").exists()
    args.resume = args.output / "checkpoint.pt"
    monkeypatch.setenv("RSNA_STOP_SESSION", "next-session")
    assert execute() == 0
    _assert_tree_equal(full, torch.load(args.output / "checkpoint.pt", weights_only=True))
    assert expected_predictions == (
        (args.output / "outer_predictions.csv").read_bytes() if fit_scope == "outer_fold" else None
    )
    assert (args.output / "outer_predictions.csv").exists() == (fit_scope == "outer_fold")


def test_all_development_membership_preserves_masks_and_excludes_reserved_data():
    from rsna.competitive.train import select_development

    rows = []
    for i, split in enumerate(
        ("train", "train", "train", "weak_validation", "gold_evaluation", "quarantine")
    ):
        row = {"study_uid": str(i), "leakage_group": str(i), "split": split}
        for target in TARGETS:
            row[target] = i % 2
            row[target + "__mask"] = int(i != 2)
            row[target + "__provenance"] = "official_gold" if split == "gold_evaluation" else "weak"
        rows.append(row)
    frame = pd.DataFrame(rows)
    folds = pd.DataFrame({"study_uid": ["0", "1", "2"], "fold": [0, 1, 1]})
    pool, outer = select_development(frame, folds, 0, "all_development")
    assert pool.study_uid.tolist() == ["0", "1"] and outer.empty
    pd.testing.assert_frame_equal(pool.drop(columns="fold"), frame.iloc[:2].reset_index(drop=True))
    poisoned = frame.copy()
    poisoned.loc[0, TARGETS[0] + "__provenance"] = "official_gold"
    with pytest.raises(ValueError, match="Official labels"):
        select_development(poisoned, folds, 0, "all_development")
    leaked = frame.copy()
    leaked.loc[3, "leakage_group"] = "0"
    with pytest.raises(ValueError, match="held-out split"):
        select_development(leaked, folds, 0, "all_development")


@pytest.mark.parametrize("enabled", [False, True])
def test_fitting_overlay_real_trainer_resume_and_bound_admission(tmp_path, monkeypatch, enabled):
    from rsna.competitive import train
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.infer import validate_inference_checkpoint
    from rsna.competitive.model import CompetitiveStudyNet
    from rsna.competitive.supervision import roster_sha

    rows = []
    for i in range(12):
        row = dict(
            study_uid=f"s{i:02}",
            leakage_group=f"g{i}",
            split="train",
            report_supervision_allowed=True,
        )
        for target in TARGETS:
            row[target], row[target + "__mask"] = i % 2, 1
            row[target + "__provenance"] = "synthetic_weak"
        rows.append(row)
    manifest = pd.DataFrame(rows)
    manifest.loc[5, "Effusion__mask"] = 0
    original_manifest = manifest.copy(deep=True)
    folds = manifest[["study_uid", "leakage_group"]].copy()
    folds["fold"] = [0] * 4 + [1] * 8
    folds_path = tmp_path / "folds.csv"
    folds.to_csv(folds_path, index=False)
    series_path = tmp_path / "series.csv"
    series_path.write_text("SeriesInstanceUID\na\n")
    binding = dict(
        manifest_sha256="d" * 64,
        source_revision="synthetic",
        series_sha256=hashlib.sha256(series_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(train, "load_verified_release", lambda *args: (manifest.copy(), binding))
    fitting, outer = train.select_development(manifest, folds, 0)
    proposal = dict(
        schema_version=1,
        status="source_review_accepted_experiment_only",
        target="Effusion",
        fit_scope="outer_fold",
        fold=0,
        release_sha256="d" * 64,
        folds_sha256=hashlib.sha256(folds_path.read_bytes()).hexdigest(),
        fitting_uid_sha256=roster_sha(fitting),
        outer_uid_sha256=roster_sha(outer),
        cell_count=1,
        cells=[
            dict(
                study_uid="s05",
                target="Effusion",
                disposition="conditional_training_only_reuse_recommendation",
                first_reader_state="positive",
                independent_reader_state="positive",
                source_report_sha256="a" * 64,
                current_annotation_sha256="b" * 64,
                assessment_sha256="c" * 64,
                original_hold_id="hold-original",
                original_obligation_id="ob-original",
            )
        ],
        reviews=dict.fromkeys(
            ["first_reader", "independent_reader", "original_context", "root_assessment"],
            "e" * 64,
        ),
    )
    artifact = tmp_path / "proposal.json"
    artifact.write_text(json.dumps(proposal))
    config = TrainConfig(
        epochs=2,
        batch_size=2,
        accumulation_steps=2,
        side=16,
        slices=2,
        max_series=3,
        hidden_dim=16,
        freeze_encoder=False,
        min_free_gib=0,
        # Admission uses the full roster even though the engineering trainer
        # consumes only a subset; its binding must retain the complete cohort.
        diagnostic_study_limit=3,
        fitting_supervision=dict(
            kind="effusion_review_v1",
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            enabled=enabled,
        ),
    )
    config.validate()
    args = argparse.Namespace(
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=folds_path,
        series=series_path,
        cache=tmp_path,
        output=tmp_path / "full",
        resume=None,
        stop_after_updates=None,
        fitting_supervision=artifact,
    )
    initial_states, input_frames = [], []

    def build_model(*args, **kwargs):
        model = CompetitiveStudyNet(*args, **kwargs)
        initial_states.append({k: v.clone() for k, v in model.state_dict().items()})
        return model

    class CapturedStudies(SyntheticStudies):
        def __init__(self, studies, *args, **kwargs):
            input_frames.append(studies.copy(deep=True))
            super().__init__(studies, *args, **kwargs)

    def execute():
        return train._run_locked(
            args,
            config,
            [False],
            0,
            1,
            torch.device("cpu"),
            create_encoder,
            CapturedStudies,
            build_model,
            lambda *args: folds.copy(),
        )

    assert execute() == 0
    full = torch.load(args.output / "checkpoint.pt", weights_only=True)
    csv = (args.output / "outer_predictions.csv").read_bytes()
    receipt = full["binding"]["experimental_fitting_supervision"]
    assert receipt["applied_cells"] == int(enabled)
    assert receipt["fitting_uid_sha256"] == roster_sha(fitting)
    expected = fitting.head(3).copy()
    expected.loc[expected.study_uid == "s05", "Effusion__mask"] = int(enabled)
    pd.testing.assert_frame_equal(input_frames[0], expected)
    pd.testing.assert_frame_equal(input_frames[1], outer.head(3))
    pd.testing.assert_frame_equal(manifest, original_manifest)
    with pytest.raises(ValueError, match="not admitted"):
        validate_inference_checkpoint(full, allow_diagnostic=True)
    args.output, args.stop_after_updates = tmp_path / "resumed", 1
    assert execute() == 75
    args.resume, args.stop_after_updates = args.output / "checkpoint.pt", None
    assert execute() == 0
    _assert_tree_equal(full, torch.load(args.output / "checkpoint.pt", weights_only=True))
    assert csv == (args.output / "outer_predictions.csv").read_bytes()
    for state in initial_states:
        _assert_tree_equal(state, initial_states[0])
    config.fitting_supervision["enabled"] = not enabled
    with pytest.raises(ValueError, match="differs"):
        execute()


@pytest.mark.parametrize("extra", [{"fit_scope": "all_development"}, {"witness": {}}])
def test_fitting_supervision_requires_isolated_outer_fold(extra):
    with pytest.raises(ValueError):
        TrainConfig(
            fitting_supervision=dict(kind="effusion_review_v1", sha256="a" * 64, enabled=True),
            **extra,
        ).validate()
