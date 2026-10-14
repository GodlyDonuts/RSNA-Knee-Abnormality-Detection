"""Whole-window-bag source-slot variant (issue #49): engineering only, never accuracy.

Pool tests adapt PR #38's ``tests/test_focal_window.py`` (Devin AI, "Add focal
disease-query window pooling and window-level teacher supervision protocol") to the
integrated, teacher-free source-slot bag.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from test_public_triplet_training import row
from torch import nn

from rsna.competitive import source_slot as ss
from rsna.competitive import window_bag as wb
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "initiative/experiments/competitive-v1/source-window-bag-v1"
SLOT_RECIPE = ROOT / "initiative/experiments/competitive-v1/source-matched-slot-v1"
CPU = torch.device("cpu")
T = len(TARGETS)
# The reviewed one-window classifier that the live source-slot recovery runs from.
SOURCE_SLOT_SHA256 = "3e4e1bb95881a6521ff4c2fc30e4175e5b23e2e5dcaab5a225c2be552b4066e6"


def configs():
    return {
        arm: json.loads((RECIPE / f"{arm}-r32-config.json").read_text())
        for arm in ("mean", "query")
    }


STRATEGIES = {arm: values["classifier_strategy"] for arm, values in configs().items()}
SLOT_STRATEGY = json.loads((SLOT_RECIPE / "r32-config.json").read_text())["classifier_strategy"]


class Backbone(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 16, 16)
        self.cls = nn.Linear(3, dim)

    def forward_features(self, images):
        self.calls.append(len(images))
        patches = self.patch(images).permute(0, 2, 3, 1)
        return self.cls(images.mean((2, 3))), patches


def encoder():
    module = Backbone()
    module.calls = []
    return module


def model(arm, seed=3, strategy=None):
    torch.manual_seed(seed)
    return wb.WindowBagSlotStudyNet(encoder(), 8, strategy or STRATEGIES[arm]).eval()


def bag(views, absent=(1, 4), batch=2, side=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mask = torch.ones(batch, 6, dtype=torch.bool)
    mask[:, list(absent)] = False
    images = torch.randint(0, 256, (batch, 6, views, 3, side, side), generator=generator) / 255
    images = images * mask[:, :, None, None, None, None]
    return (
        images,
        torch.zeros(batch, 6, 5),
        mask,
        mask[:, :, None].expand(batch, 6, views),
        torch.zeros(batch, 6, views, 2),
    )


def pool_inputs(b=2, w=10, h=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(b, w, T, generator=generator) * 3,
        torch.randn(b, w, T, h, generator=generator),
        torch.randn(T, h, generator=generator),
        torch.ones(b, w, dtype=torch.bool),
    )


def mixture(logits, beta):
    return (beta * logits.sigmoid()).sum(1)


# --- Live source and default compatibility ------------------------------------------


def test_live_source_slot_numerical_source_is_byte_unchanged():
    source = (ROOT / "rsna/competitive/source_slot.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SLOT_SHA256
    assert wb.classifier_module(SLOT_STRATEGY) is ss
    assert wb.classifier_module(STRATEGIES["mean"]) is wb
    assert wb.classifier_module(None) is ss
    assert TrainConfig().classifier_strategy is None


def test_strategies_reuse_the_reviewed_slot_policy_and_differ_only_in_weighting():
    mean, query = (wb.validate_strategy(STRATEGIES[arm]) for arm in ("mean", "query"))
    assert {k for k in mean if mean[k] != query[k]} == {"window_weighting"}
    assert (mean["window_weighting"], query["window_weighting"]) == ("plain_mean", "disease_query")
    assert wb.slot_strategy(mean) == ss.validate_strategy(SLOT_STRATEGY)
    reused = {"source_sha256", "reference_node_text_sha256", "training_node_text_sha256"}
    for key in reused | {"pool", "prior", "hidden", "dropout", "augmentation"}:
        assert mean[key] == SLOT_STRATEGY[key]
    with pytest.raises(ValueError, match="Exact source-matched"):
        ss.validate_strategy(mean)


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(window_weighting="learned"),
        lambda v: v.update(window_weighting=None),
        lambda v: v.update(uniform_floor=0.0),
        lambda v: v.update(uniform_floor=0),
        lambda v: v.update(bag_pool="logit_mean"),
        lambda v: v.update(training_windows=ss.TRAINING_WINDOWS),
        lambda v: v.update(evaluation_windows=ss.EVALUATION_WINDOWS),
        lambda v: v.update(kind=ss.KIND),
        lambda v: v.update(hidden=128),
        lambda v: v.update(prior=True),
        lambda v: v.update(dropout=0),
        lambda v: v.update(augmentation="flip"),
        lambda v: v.update(source_sha256="0" * 64),
        lambda v: v.update(teacher="window_v1"),
        lambda v: v.pop("window_weighting"),
    ],
)
def test_strategy_tampering_is_refused(change):
    for arm in ("mean", "query"):
        values = copy.deepcopy(configs()[arm])
        change(values["classifier_strategy"])
        with pytest.raises(ValueError, match="Exact"):
            TrainConfig(**values).validate()
        with pytest.raises(ValueError, match="Exact"):
            wb.WindowBagSlotStudyNet(encoder(), 8, values["classifier_strategy"])


def test_fixed_final_configs_change_only_the_classifier_strategy():
    slot = json.loads((SLOT_RECIPE / "r32-config.json").read_text())
    manifest = json.loads((ROOT / "initiative/manifest.json").read_text())
    for values in configs().values():
        assert {k: v for k, v in values.items() if k != "classifier_strategy"} == {
            k: v for k, v in slot.items() if k != "classifier_strategy"
        }
        TrainConfig(**values).validate()
    for values in (slot, dict(slot, public_input=None)):
        values = copy.deepcopy(values)
        values["classifier_strategy"] = STRATEGIES["query"]
        if values["public_input"] is None:
            with pytest.raises(ValueError, match="public rows"):
                TrainConfig(**values).validate()
    for path in sorted(RECIPE.iterdir()):
        key = str(path.relative_to(ROOT / "initiative"))
        assert manifest[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


# --- Arm equality --------------------------------------------------------------------


def test_both_arms_have_identical_parameter_graphs_and_initial_values():
    mean, query = model("mean"), model("query")
    a, b = mean.state_dict(), query.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert torch.equal(a["window_query"], torch.zeros(T, 256))
    torch.manual_seed(3)
    slot = ss.SourceSlotStudyNet(encoder(), 8, SLOT_STRATEGY)
    assert set(a) - set(slot.state_dict()) == {"window_query"}
    assert all(torch.equal(v, a[k]) for k, v in slot.state_dict().items())
    inputs = bag(10)
    assert torch.equal(mean(*inputs), query(*inputs))


def test_target_window_features_expose_the_head_without_changing_its_logits():
    net = model("query").train()
    x, mask = torch.randn(5, 6, 16), (torch.rand(5, 6) > 0.3).float()
    mask[:, 0] = 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        expected = net.head(x, mask)
    torch.manual_seed(1)
    context, logits = wb.target_window_features(net.head, x, mask)
    assert torch.equal(logits, expected) and context.shape == (5, T, 256)


def test_plain_mean_is_fixed_probability_averaging_of_the_ten_windows():
    net = model("mean")
    with torch.no_grad():
        net.window_query.normal_()
    torch.manual_seed(3)
    slot = ss.SourceSlotStudyNet(encoder(), 8, SLOT_STRATEGY).eval()
    slot.load_state_dict({k: v for k, v in net.state_dict().items() if k != "window_query"})
    inputs = bag(10)
    study, _, beta = net.bag_outputs(*inputs)
    assert torch.equal(beta, torch.full_like(beta, 0.1))
    torch.testing.assert_close(
        torch.sigmoid(study), slot.window_probabilities(*inputs), rtol=0, atol=2e-7
    )


def test_learned_weighting_changes_only_the_query_arm_and_mean_query_gets_no_gradient():
    inputs = bag(4)
    outputs = {}
    for arm in ("mean", "query"):
        net = model(arm)
        with torch.no_grad():
            net.window_query.copy_(torch.randn(T, 256, generator=torch.Generator().manual_seed(5)))
        study = net(*inputs)
        study.sum().backward()
        outputs[arm] = study.detach()
        gradient = net.window_query.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.count_nonzero() == 0 if arm == "mean" else gradient.count_nonzero() > 0
    assert not torch.equal(outputs["mean"], outputs["query"])


def test_zero_initialized_query_still_receives_gradient_in_the_learned_arm():
    net = model("query")
    net(*bag(4)).sum().backward()
    assert net.window_query.grad.abs().sum() > 0


# --- Bounded pool (adapted from PR #38) ----------------------------------------------


@pytest.mark.parametrize("selection", [0.0, 1.0])
def test_pool_logit_is_the_exact_probability_mixture(selection):
    logits, context, query, valid = pool_inputs()
    study, beta = wb.bag_pool(logits, context, query, valid, selection=selection)
    torch.testing.assert_close(beta.sum(1), torch.ones(2, T))
    torch.testing.assert_close(torch.sigmoid(study), mixture(logits, beta), rtol=0, atol=1e-6)


def test_uniform_floor_bounds_every_window_weight_under_an_extreme_query():
    logits, context, query, valid = pool_inputs()
    _, beta = wb.bag_pool(logits, context, query * 1e6, valid, selection=1.0)
    rho, w = wb.UNIFORM_FLOOR, logits.shape[1]
    assert beta.min() >= rho / w - 1e-7 and beta.max() <= 1 - rho + rho / w + 1e-7


def test_focal_positive_window_is_preserved_by_selection_not_by_the_mean():
    logits = torch.full((1, 10, T), -6.0)
    logits[0, 7] = 6.0
    context = torch.zeros(1, 10, T, 4)
    context[0, 7] = 1.0
    query = torch.full((T, 4), 50.0)
    valid = torch.ones(1, 10, dtype=torch.bool)
    mean, _ = wb.bag_pool(logits, context, query, valid, selection=0.0)
    focal, _ = wb.bag_pool(logits, context, query, valid, selection=1.0)
    assert (torch.sigmoid(mean) < 0.11).all() and (torch.sigmoid(focal) > 0.75).all()


def test_masked_windows_never_contribute_even_when_nan_poisoned():
    logits, context, query, _ = pool_inputs(b=1)
    valid = torch.zeros(1, 10, dtype=torch.bool)
    valid[0, [0, 3, 6, 9]] = True
    poisoned_logits, poisoned_context = logits.clone(), context.clone()
    poisoned_logits[~valid], poisoned_context[~valid] = float("nan"), float("inf")
    for selection in (0.0, 1.0):
        study, beta = wb.bag_pool(
            poisoned_logits, poisoned_context, query, valid, selection=selection
        )
        subset, _ = wb.bag_pool(
            logits[:, valid[0]],
            context[:, valid[0]],
            query,
            torch.ones(1, 4, dtype=torch.bool),
            selection=selection,
        )
        assert torch.isfinite(study).all() and (beta[~valid] == 0).all()
        torch.testing.assert_close(study, subset, rtol=0, atol=1e-6)


@pytest.mark.parametrize("value", [1e4, -1e4, 80.0, -80.0])
def test_extreme_window_logits_keep_finite_values_and_gradients(value):
    logits, context, query, valid = pool_inputs()
    logits = torch.full_like(logits, value)
    logits[:, 0] = -value
    logits.requires_grad_(True)
    query.requires_grad_(True)
    for selection in (0.0, 1.0):
        study, _ = wb.bag_pool(logits, context, query, valid, selection=selection)
        study.sum().backward()
        assert torch.isfinite(study).all()
        assert torch.isfinite(logits.grad).all() and torch.isfinite(query.grad).all()
        logits.grad = query.grad = None


def test_pool_refusals():
    logits, context, query, valid = pool_inputs()
    with pytest.raises(ValueError, match="empty bags"):
        wb.bag_pool(logits, context, query, valid.clone().fill_(False), selection=1.0)
    with pytest.raises(ValueError, match="finite"):
        wb.bag_pool(logits.clone().fill_(float("nan")), context, query, valid, selection=1.0)
    with pytest.raises(ValueError, match="finite"):
        wb.bag_pool(logits, context, query.clone().fill_(float("nan")), valid, selection=0.0)
    with pytest.raises(ValueError, match="Aligned"):
        wb.bag_pool(logits, context[..., 0], query, valid, selection=1.0)
    with pytest.raises(ValueError, match="selection"):
        wb.bag_pool(logits, context, query, valid, selection=0.5)
    with pytest.raises(ValueError, match="Aligned"):
        wb.bag_pool(logits, context, query, valid.float(), selection=1.0)


def test_pool_runs_in_fp32_under_autocast():
    logits, context, query, valid = pool_inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        study, beta = wb.bag_pool(logits.bfloat16(), context.bfloat16(), query, valid, selection=1)
    assert study.dtype == beta.dtype == torch.float32


# --- Model, masking, inference ---------------------------------------------------------


def test_missing_slots_never_reach_the_encoder_and_cost_scales_with_views():
    for views, absent in ((4, (1, 4)), (10, (1, 4)), (10, (1, 2, 3, 4, 5))):
        net = model("query")
        inputs = bag(views, absent=absent)
        net(*inputs)
        assert sum(net.encoder.calls) == 2 * views * (6 - len(absent))
    poisoned = list(bag(4))
    poisoned[0] = poisoned[0].clone()
    poisoned[0][:, 1] = 0.5
    with pytest.raises(ValueError, match="Absent"):
        model("query")(*poisoned)
    with pytest.raises(ValueError, match="four source groups"):
        model("query")(*bag(1))
    with pytest.raises(ValueError, match="ten original"):
        model("query").window_probabilities(*bag(4))


def test_no_extra_sigmoid_between_logits_and_probabilities():
    net = model("query")
    with torch.no_grad():
        net.window_query.normal_()
    inputs = bag(10)
    study, window_logits, beta = net.bag_outputs(*inputs)
    probabilities = net.window_probabilities(*inputs)
    assert torch.equal(probabilities, torch.sigmoid(study))
    torch.testing.assert_close(probabilities, mixture(window_logits, beta), rtol=0, atol=1e-6)
    assert not torch.equal(probabilities, torch.sigmoid(probabilities))


def test_training_bag_is_the_four_source_groups_with_the_one_window_augmentation():
    source = [row(seed, absent=(2, 5)) for seed in range(2)]
    images = torch.stack([torch.from_numpy(r["images"])[None].float() / 255 for r in source])
    windows = torch.stack([images[:, 0, :, s : s + 3] for s in ss.EVALUATION_VIEWS], 2)
    mask = torch.stack([torch.from_numpy(r["slot_mask"]) for r in source])
    batch = dict(
        images=windows[:, :, :, :, :64, :64].contiguous(),
        features=torch.zeros(2, 6, 5),
        series_mask=mask,
        slice_mask=mask[:, :, None].expand(2, 6, 10),
        positions=torch.zeros(2, 6, 10, 2),
        study_uid=["a", "b"],
    )
    updates = {ss.training_group(1, 2, update): update for update in range(64)}
    for augmentation in (ss.AUGMENT, "none"):
        strategy = dict(STRATEGIES["mean"], augmentation=augmentation)
        bagged = wb.training_inputs(batch, CPU, strategy, seed=1, epoch=2, update=0)
        assert bagged[0].shape == (2, 6, 4, 3, 64, 64) and bagged[3].shape == (2, 6, 4)
        assert torch.equal(
            bagged[0], wb.training_inputs(batch, CPU, strategy, seed=1, epoch=2, update=9)[0]
        )
        slot = dict(SLOT_STRATEGY, augmentation=augmentation)
        for group, update in sorted(updates.items()):
            one = ss.training_inputs(batch, CPU, slot, seed=1, epoch=2, update=update)
            assert torch.equal(bagged[0][:, :, group : group + 1], one[0])
        assert (bagged[0][~mask] == 0).all()


def test_public_mri_only_deployment_matches_the_bag_evaluation():
    source = row(4, absent=(1, 3))
    images, mask = torch.from_numpy(source["images"]), torch.from_numpy(source["slot_mask"])
    torch.manual_seed(2)
    net = wb.WindowBagSlotStudyNet(_Strided(), 8, STRATEGIES["query"]).eval()
    with torch.no_grad():
        net.window_query.normal_()
    deployed = wb.deployment_probabilities(net, images, mask, CPU, "fp32")
    windows = ss.source_row_windows(images)[None]
    inputs = (
        windows,
        torch.zeros(1, 6, 5),
        mask[None],
        mask[None, :, None].expand(1, 6, 10),
        torch.zeros(1, 6, 10, 2),
    )
    with torch.no_grad():
        assert torch.equal(deployed, net.window_probabilities(*inputs)[0])
    assert deployed.shape == (T,) and ((deployed > 0) & (deployed < 1)).all()
    slot = ss.SourceSlotStudyNet(_Strided(), 8, SLOT_STRATEGY)
    with pytest.raises(ValueError, match="window-bag student"):
        wb.deployment_probabilities(slot, images, mask, CPU, "fp32")


class _Strided(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 8, 48, 48)
        self.cls = nn.Linear(3, 8)

    def forward_features(self, images):
        return self.cls(images.mean((2, 3))), self.patch(images).permute(0, 2, 3, 1)


def test_student_has_no_teacher_reference_or_variability_dependency():
    source = (ROOT / "rsna/competitive/window_bag.py").read_text()
    assert "tools." not in source and "focal_window" not in source
    assert "teacher" not in source.replace("No labels, reports, teacher", "")


# --- Canonical trainer on the engineering encoder (no transformers needed) -------------


def _diagnostic_run(tmp_path, arm, output, stop=None, resume=None, every=0, precision="fp16"):
    import argparse

    from test_public_triplet_training import POLICY, build, cohort

    from rsna.competitive import train
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.model import CompetitiveStudyNet

    manifest, folds = cohort()
    if not (tmp_path / "cache").exists():
        build(tmp_path / "cache", manifest, folds)
        (tmp_path / "series.csv").write_text("SeriesInstanceUID\na\n")
    series = tmp_path / "series.csv"
    release = {
        "series_sha256": hashlib.sha256(series.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    if every:
        # Scheduled epoch snapshots bind the release identity.
        release["binding_sha256"] = hashlib.sha256(
            json.dumps(release, sort_keys=True).encode()
        ).hexdigest()
    config = TrainConfig(
        epochs=2,
        batch_size=1,
        accumulation_steps=2,
        slices=10,
        side=336,
        max_series=6,
        freeze_encoder=False,
        min_free_gib=0,
        diagnostic_study_limit=16,
        precision=precision,
        public_input=POLICY,
        classifier_strategy=copy.deepcopy(STRATEGIES[arm]),
        submission_every_epochs=every,
    )
    args = argparse.Namespace(
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=tmp_path / "unused",
        series=series,
        cache=tmp_path / "cache",
        output=output,
        resume=resume,
        stop_after_updates=stop,
    )
    output.mkdir(parents=True, exist_ok=True)
    patch = pytest.MonkeyPatch()
    patch.setattr(train, "load_verified_release", lambda *_: (manifest.copy(), release))
    try:
        return train._run_locked(
            args,
            config,
            [False],
            0,
            1,
            CPU,
            create_encoder,
            None,
            CompetitiveStudyNet,
            lambda *_: folds.copy(),
        )
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def diagnostic(tmp_path_factory):
    root = tmp_path_factory.mktemp("window-bag-diagnostic")
    runs = {}
    for arm in ("mean", "query"):
        base = root / arm
        assert _diagnostic_run(root, arm, base / "full") == 0
        assert _diagnostic_run(root, arm, base / "replay", stop=1) == 75
        first = json.loads((base / "replay/invocation.json").read_text())["updates"]
        resume = base / "replay/checkpoint.pt"
        assert _diagnostic_run(root, arm, base / "replay", resume=resume) == 0
        runs[arm] = dict(base=base, first=first)
    return runs


@pytest.mark.parametrize("arm", ["mean", "query"])
def test_diagnostic_trainer_full_and_stop1_resume_are_exact(diagnostic, arm):
    from test_competitive_training import _assert_tree_equal

    base = diagnostic[arm]["base"]
    full = torch.load(base / "full/checkpoint.pt", weights_only=True)
    replay = torch.load(base / "replay/checkpoint.pt", weights_only=True)
    _assert_tree_equal(full, replay)
    assert "amp_scaler" in full and set(full["rng_by_rank"][0]) >= {"torch", "python", "numpy"}
    assert full["binding"]["config"]["classifier_strategy"] == STRATEGIES[arm]
    query = full["model"]["window_query"]
    assert (query.count_nonzero() == 0) == (arm == "mean")
    assert (base / "full/outer_predictions.csv").read_bytes() == (
        base / "replay/outer_predictions.csv"
    ).read_bytes()
    history = json.loads((base / "full/invocation.json").read_text())["updates"]
    resumed = json.loads((base / "replay/invocation.json").read_text())["updates"]

    def numeric(events):
        return [{k: v for k, v in e.items() if k != "elapsed_seconds"} for e in events]

    assert numeric(history) == numeric(diagnostic[arm]["first"] + resumed)


def test_diagnostic_arms_share_first_update_loss_and_rng(diagnostic):
    mean, query = (
        json.loads((diagnostic[arm]["base"] / "full/invocation.json").read_text())["updates"]
        for arm in ("mean", "query")
    )
    assert mean[0]["loss"] == query[0]["loss"]
    a, b = (
        torch.load(diagnostic[arm]["base"] / "full/checkpoint.pt", weights_only=True)
        for arm in ("mean", "query")
    )
    assert set(a["model"]) == set(b["model"])
    from test_competitive_training import _assert_tree_equal

    _assert_tree_equal(a["rng_by_rank"], b["rng_by_rank"])
