"""Source-public ordered triplets in matched DINOv3 training; engineering only, not accuracy."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rsna.competitive import public_input as pi
from rsna.competitive.backbones import _GrayEncoder, create_encoder
from rsna.competitive.data import collate_studies
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "initiative/experiments/competitive-v1/public-input-recipe-v1"
COMPARISON = ROOT / "initiative/audits/public-dinov2-real-inputs-20261013-v2"
PIXEL_GROUP = json.loads((COMPARISON / "actual-input-comparison.json").read_text())["provenance"][
    "member_pixel_group"
]
POLICY = dict(
    kind=pi.KIND,
    renderer=pi.RENDERER,
    source_sha256=pi.SOURCE_SHA256,
    node_text_sha256=pi.NODE_TEXT_SHA256,
    pixel_group_sha256=pi.canonical_sha256(PIXEL_GROUP),
    adapter_sha256=hashlib.sha256(pi.ADAPTER_PATH.read_bytes()).hexdigest(),
)


def row(seed, absent=(5,)):
    generator = np.random.default_rng(seed)
    mask = np.ones(6, dtype=bool)
    mask[list(absent)] = False
    images = generator.integers(0, 256, (6, 12, 336, 336), dtype=np.uint8)
    features = generator.random((6, 5), dtype=np.float32)
    positions = np.zeros((6, 12, 2), dtype=np.float32)
    positions[:, :, 0] = np.arange(12, dtype=np.float32) * 0.03 + seed
    positions[:, :, 1] = 1
    positions[1, 4:7] = 0  # unknown geometry stays explicit, never fabricated
    images[~mask], features[~mask], positions[~mask] = 0, 0, 0
    return dict(images=images, slot_mask=mask, features=features, positions=positions)


def cohort(count=8):
    rows = []
    for i in range(count):
        values = dict(study_uid=f"s{i}", leakage_group=f"g{i}", split="train")
        for t in TARGETS:
            values[t], values[t + "__mask"] = i % 2, float((i + len(t)) % 3 != 0)
            values[t + "__provenance"] = "synthetic_weak"
        rows.append(values)
    manifest = pd.DataFrame(rows)
    folds = manifest[["study_uid", "leakage_group"]].copy()
    folds["fold"] = [0] * (count // 2) + [1] * (count - count // 2)
    return manifest, folds


def build(tmp_path, manifest, folds, rows=None):
    fitting = set(folds.study_uid[folds.fold != 0])
    evaluation = set(folds.study_uid[folds.fold == 0])
    rows = rows or {uid: row(i) for i, uid in enumerate(manifest.study_uid)}
    pi.write_public_triplet_cache(
        tmp_path,
        rows,
        public_input=POLICY,
        source_revision="synthetic",
        fitting_uids=fitting,
        evaluation_uids=evaluation,
    )
    return rows


def dataset(tmp_path, studies, policy=POLICY, **changes):
    values = (
        dict(
            public_input=policy,
            source_revision="synthetic",
            slices=10,
            side=336,
            max_series=6,
        )
        | changes
    )
    return pi.PublicTripletStudyDataset(studies, pd.DataFrame(), tmp_path, **values)


def rewrite_index(tmp_path, edit):
    path = tmp_path / pi.CACHE_DIRECTORY / "index.json"
    index = json.loads(path.read_text())
    edit(index)
    path.write_text(json.dumps(index))


def test_pixels_channels_masks_geometry_and_frame_labels_are_source_exact(tmp_path):
    manifest, folds = cohort()
    rows = build(tmp_path, manifest, folds)
    before = manifest.copy(deep=True)
    data = dataset(tmp_path, manifest)
    for index, uid in enumerate(manifest.study_uid):
        item, source = data[index], rows[uid]
        pixels = torch.from_numpy(source["images"])
        assert item["images"].shape == (6, 10, 3, 336, 336)
        assert item["images"].dtype == torch.float32
        for view in range(10):
            # Channel c of view w is original slice w+c, divided by 255 exactly once.
            assert torch.equal(item["images"][:, view], pixels[:, view : view + 3].float() / 255)
        assert torch.equal(item["series_mask"], torch.from_numpy(source["slot_mask"]))
        assert torch.equal(item["slice_mask"], item["series_mask"][:, None].expand(6, 10))
        assert torch.equal(item["features"], torch.from_numpy(source["features"]))
        assert torch.equal(item["positions"], torch.from_numpy(source["positions"][:, 1:11]))
        assert item["positions"][1, 3:6, 1].count_nonzero() == 0
        assert item["images"][5].count_nonzero() == 0
        frame = manifest.iloc[index]
        mask = torch.tensor([frame[t + "__mask"] for t in TARGETS], dtype=torch.float32)
        labels = torch.tensor([frame[t] for t in TARGETS], dtype=torch.float32)
        assert torch.equal(item["label_mask"], mask)
        assert torch.equal(item["labels"], labels * mask)
    pd.testing.assert_frame_equal(manifest, before)
    batch = collate_studies([data[0], data[1]])
    assert batch["images"].shape == (2, 6, 10, 3, 336, 336)


def test_cache_stores_each_source_row_once_as_uint8(tmp_path):
    manifest, folds = cohort()
    build(tmp_path, manifest, folds)
    members = sorted((tmp_path / pi.CACHE_DIRECTORY / "members").iterdir())
    assert len(members) == len(manifest)
    for path in members:
        with np.load(path, allow_pickle=False) as archive:
            assert archive["images"].dtype == np.uint8
            assert not any(archive[name].dtype == np.float64 for name in archive.files)
        assert path.stat().st_size < 6 * 12 * 336 * 336 + 4096


@pytest.mark.parametrize(
    "tamper",
    [
        "member_bytes",
        "member_missing",
        "pixel_group",
        "pixel_rules",
        "pixel_slots",
        "renderer",
        "source_revision",
        "roster_swap",
        "cohort_extra",
        "path",
        "member_sha",
        "adapter",
        "extra_array",
        "float_images",
        "nonzero_absent",
        "fabricated_geometry",
    ],
)
def test_cache_source_and_roster_tampering_refused(tmp_path, tamper):
    manifest, folds = cohort()
    rows = build(tmp_path, manifest, folds)
    root = tmp_path / pi.CACHE_DIRECTORY
    policy = dict(POLICY)
    train, outer = (manifest[folds.fold != 0], manifest[folds.fold == 0])

    def member(uid, **arrays):
        payload = rows[uid] | arrays
        path = root / f"members/{uid}.npz"
        np.savez(path.with_suffix(""), **payload)
        rewrite_index(
            tmp_path,
            lambda index: index["members"][uid].update(
                bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        )

    if tamper == "member_bytes":
        path = root / "members/s0.npz"
        payload = bytearray(path.read_bytes())
        payload[-100] ^= 1
        path.write_bytes(bytes(payload))
    elif tamper == "member_missing":
        (root / "members/s1.npz").unlink()
    elif tamper == "pixel_group":
        rewrite_index(tmp_path, lambda index: index["pixel_group"].update(crop_mm=161.25))
    elif tamper == "pixel_rules":
        rewrite_index(tmp_path, lambda index: index["pixel_group"]["rules"].update(lat="none"))
    elif tamper == "pixel_slots":
        rewrite_index(tmp_path, lambda index: index["pixel_group"]["slots"].reverse())
    elif tamper == "renderer":
        rewrite_index(tmp_path, lambda index: index.update(source_sha256="0" * 64))
    elif tamper == "source_revision":
        rewrite_index(tmp_path, lambda index: index.update(source_revision="other"))
    elif tamper == "roster_swap":
        train, outer = (
            pd.concat([train.iloc[1:], outer.iloc[:1]]),
            pd.concat([outer.iloc[1:], train.iloc[:1]]),
        )
    elif tamper == "cohort_extra":
        rewrite_index(tmp_path, lambda index: index["members"].update(x=index["members"]["s0"]))
    elif tamper == "path":
        rewrite_index(tmp_path, lambda index: index["members"]["s0"].update(path="../s0.npz"))
    elif tamper == "member_sha":
        rewrite_index(tmp_path, lambda index: index["members"]["s0"].update(sha256="0" * 64))
    elif tamper == "adapter":
        policy["adapter_sha256"] = "0" * 64
    elif tamper == "extra_array":
        member("s0", labels=np.ones(12, dtype=np.float32))
    elif tamper == "float_images":
        member("s0", images=rows["s0"]["images"].astype(np.float32) / 255)
    elif tamper == "nonzero_absent":
        images = rows["s0"]["images"].copy()
        images[5, 0, 0, 0] = 1
        member("s0", images=images)
    elif tamper == "fabricated_geometry":
        positions = rows["s0"]["positions"].copy()
        positions[1, 5, 0] = 0.5
        member("s0", positions=positions)
    with pytest.raises((ValueError, FileNotFoundError)):
        pi.verify_public_cohort(tmp_path, policy, train, outer, "synthetic")
        data = dataset(tmp_path, manifest, policy)
        for index in range(len(data)):
            data[index]


@pytest.mark.parametrize(
    "changes",
    [
        dict(side=384),
        dict(slices=16),
        dict(max_series=5),
        dict(augment=True),
        dict(slice_sampling="random"),
        dict(series_sampling="random"),
    ],
)
def test_dataset_refuses_native_or_altered_geometry_substitution(tmp_path, changes):
    manifest, folds = cohort()
    build(tmp_path, manifest, folds)
    with pytest.raises(ValueError):
        dataset(tmp_path, manifest, **changes)


class Capture(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.seen = []

    def forward(self, pixel_values):
        self.seen.append(pixel_values.detach().clone())
        return type("Output", (), {"pooler_output": pixel_values.mean((2, 3)) * self.weight})()


def test_encoder_normalizes_once_with_honest_triplet_contract(tmp_path):
    manifest, folds = cohort()
    rows = build(tmp_path, manifest, folds)
    item = dataset(tmp_path, manifest)[0]
    backbone = Capture()
    _GrayEncoder(backbone, 16, input_channels=3)(item["images"][0])
    expected = torch.from_numpy(rows["s0"]["images"][0]).unfold(0, 3, 1).permute(0, 3, 1, 2)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(backbone.seen[0], (expected.float() / 255 - mean) / std, atol=1e-6)
    tiny = dict(architecture="diagnostic_tiny")
    _, _, public = create_encoder(tiny, input_channels=3, input_contract="public_ordered_triplet")
    _, _, neighbor = create_encoder(tiny, input_channels=3)
    _, _, gray = create_encoder(tiny)
    assert public["input_contract"] == "public_ordered_triplet_0_1_v1"
    assert neighbor["input_contract"] == "neighbor_delta_0_1_v1"
    assert gray["input_contract"] == "gray_0_1_v1"
    with pytest.raises(ValueError):
        create_encoder(tiny, input_contract="public_ordered_triplet")
    with pytest.raises(ValueError):
        create_encoder(tiny, input_channels=3, input_contract="neighbor_delta_v2")


def test_common_initialized_learned_state_and_rng_match_control():
    states = []
    for channels in (1, 3):
        torch.manual_seed(20260922)
        encoder = _GrayEncoder(Capture(), 16, input_channels=channels)
        model = CompetitiveStudyNet(encoder, 3, hidden_dim=12, input_channels=channels)
        states.append((copy.deepcopy(model.state_dict()), torch.get_rng_state()))
    assert states[0][0].keys() == states[1][0].keys()
    assert all(torch.equal(states[0][0][key], states[1][0][key]) for key in states[0][0])
    assert torch.equal(states[0][1], states[1][1])


def matched_configs():
    control = json.loads((RECIPE / "control-config.json").read_text())
    public = json.loads((RECIPE / "public-triplet-config.json").read_text())
    return control, public


def test_first_bundled_comparison_differs_only_in_its_input_bundle():
    control, public = matched_configs()
    accepted = json.loads(
        (
            ROOT / "initiative/experiments/competitive-v1/public-label-source-v1/public-config.json"
        ).read_text()
    )
    assert control == accepted
    assert {
        key for key in public.keys() | control.keys() if public.get(key) != control.get(key)
    } == {
        "public_input",
        "side",
        "slices",
    }
    assert public["public_input"] == POLICY
    assert (public["side"], public["slices"], public["max_series"]) == (336, 10, 6)
    assert public["batch_size"] * public["accumulation_steps"] == 8
    assert (public["epochs"], public["fold"], public["fit_scope"]) == (10, 0, "outer_fold")
    assert public["fitting_supervision"]["kind"] == "label_source_v1"
    TrainConfig(**control).validate()
    TrainConfig(**public).validate()


@pytest.mark.parametrize(
    "changes",
    [
        dict(fit_scope="all_development"),
        dict(side=384),
        dict(slices=16),
        dict(augment=True),
        dict(neighbor_channels=dict(mode="physical", strength=0.5)),
        dict(cache_contrast=dict(mode="p01_p99")),
        dict(witness=dict(weight=1)),
        dict(slice_context="neighbor3"),
        dict(fitting_supervision=dict(kind="effusion_v1", enabled=True, sha256="0" * 64)),
        dict(public_input=dict(POLICY, adapter_sha256="bad")),
        dict(public_input=dict(POLICY, renderer="native384")),
    ],
)
def test_unrelated_experiments_and_final_fit_mixing_rejected(changes):
    _, public = matched_configs()
    with pytest.raises(ValueError):
        TrainConfig(**(public | changes)).validate()


def test_old_defaults_and_bindings_do_not_gain_public_policy():
    assert TrainConfig().public_input is None
    TrainConfig(diagnostic_study_limit=4).validate()


def test_inference_refuses_public_triplet_research_weights():
    from rsna.competitive import infer

    with pytest.raises(ValueError, match="Public ordered-triplet"):
        infer.validate_inference_checkpoint(
            {
                "schema_version": 2,
                "optimizer_boundary": True,
                "binding": {"config": {"public_input": POLICY}},
            }
        )


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_actual_canonical_trainer_full_stop_resume_state_scaler_and_labels(
    tmp_path, monkeypatch, precision
):
    from test_competitive_training import _assert_tree_equal

    from rsna.competitive import train

    manifest, folds = cohort()
    build(tmp_path / "cache", manifest, folds)
    original = manifest.copy(deep=True)
    series_path = tmp_path / "series.csv"
    series_path.write_text("SeriesInstanceUID\na\n")
    release = {
        "series_sha256": hashlib.sha256(series_path.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    monkeypatch.setattr(train, "load_verified_release", lambda *args: (manifest.copy(), release))
    config = TrainConfig(
        epochs=2,
        batch_size=1,
        accumulation_steps=2,
        slices=10,
        side=336,
        max_series=6,
        hidden_dim=12,
        freeze_encoder=False,
        min_free_gib=0,
        diagnostic_study_limit=16,
        precision=precision,
        public_input=POLICY,
    )
    args = argparse.Namespace(
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=tmp_path / "unused",
        series=series_path,
        cache=tmp_path / "cache",
        output=tmp_path / "full",
        resume=None,
        stop_after_updates=None,
    )

    class NativeStudies:  # must never be used under the public policy
        def __init__(self, *args, **kwargs):
            raise AssertionError("native dataset substituted")

    def execute():
        return train._run_locked(
            args,
            config,
            [False],
            0,
            1,
            torch.device("cpu"),
            create_encoder,
            NativeStudies,
            CompetitiveStudyNet,
            lambda *args: folds.copy(),
        )

    assert execute() == 0
    full = torch.load(args.output / "checkpoint.pt", weights_only=True)
    csv = (args.output / "outer_predictions.csv").read_bytes()
    binding = full["binding"]
    assert binding["config"]["public_input"] == POLICY
    assert binding["private_research_only"] is True
    assert binding["public_input_cache"]["members"] == 8
    assert binding["config"]["encoder"]["input_contract"] == "public_ordered_triplet_0_1_v1"
    assert ("amp_scaler" in full) == (precision == "fp16")
    predictions = pd.read_csv(args.output / "outer_predictions.csv")
    assert set(predictions.study_uid) == set(folds.study_uid[folds.fold == 0])
    args.output = tmp_path / "resume"
    args.stop_after_updates = 1
    assert execute() == 75
    args.resume = args.output / "checkpoint.pt"
    args.stop_after_updates = None
    assert execute() == 0
    _assert_tree_equal(full, torch.load(args.output / "checkpoint.pt", weights_only=True))
    assert csv == (args.output / "outer_predictions.csv").read_bytes()
    pd.testing.assert_frame_equal(manifest, original)


def test_policy_constants_match_reviewed_source_attribution():
    review = json.loads((pi.ADAPTER_PATH.parent / "review.json").read_text())
    comparison = json.loads((COMPARISON / "actual-input-comparison.json").read_text())
    assert review["node_text_sha256"] == pi.NODE_TEXT_SHA256
    assert comparison["provenance"]["source_sha256"] == pi.SOURCE_SHA256
    assert comparison["provenance"]["node_text_sha256"] == pi.NODE_TEXT_SHA256


def test_admitted_pixel_group_is_the_actual_member_group():
    assert pi.ADMITTED_PIXEL_GROUP == PIXEL_GROUP
    assert pi.canonical_sha256(PIXEL_GROUP) == pi.ADMITTED_PIXEL_GROUP_SHA256
    assert POLICY["pixel_group_sha256"] == pi.ADMITTED_PIXEL_GROUP_SHA256


@pytest.mark.parametrize(
    "edit",
    [
        lambda g: g["slots"].reverse(),
        lambda g: g["slots"].__setitem__(5, "AX_T1"),
        lambda g: g["rules"].update(slot_fallback=True),
        lambda g: g["rules"].update(extra="x"),
        lambda g: g.update(crop_mm=130),
    ],
)
def test_other_slots_or_rules_refused(edit):
    group = copy.deepcopy(PIXEL_GROUP)
    edit(group)
    with pytest.raises(ValueError, match="pixel group"):
        pi._validate_pixel_group(group)
    with pytest.raises(ValueError, match="policy"):
        pi.validate_public_input(POLICY | {"pixel_group_sha256": pi.canonical_sha256(group)})


def tree(path):
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


@pytest.mark.parametrize(
    "uid", ["../outside", "..", ".", "a/b", "a\\b", "/abs", "", " s", "s..t", 7, "x" * 129]
)
@pytest.mark.parametrize("roster", ["fitting", "evaluation", "rows"])
def test_unsafe_uids_refused_before_any_write(tmp_path, uid, roster):
    cache = tmp_path / "cache"
    fitting, evaluation = {"s0", "s1"}, {"s2"}
    rows = {u: row(i) for i, u in enumerate(sorted(fitting | evaluation))}
    if roster == "rows":
        rows[uid] = row(9)
    elif roster == "fitting":
        fitting.add(uid)
    else:
        evaluation.add(uid)
    before = tree(tmp_path)
    with pytest.raises(ValueError):
        pi.write_public_triplet_cache(
            cache,
            rows,
            public_input=POLICY,
            source_revision="synthetic",
            fitting_uids=fitting,
            evaluation_uids=evaluation,
        )
    assert tree(tmp_path) == before or roster == "rows"
    assert not (tmp_path / "outside.npz").exists() and not Path("/abs.npz").exists()
    assert not (cache / pi.CACHE_DIRECTORY).exists()
    if roster == "rows":  # unknown row UIDs stop before their own write; nothing is published
        assert not any(p.name.startswith(("outside", "abs", "..")) for p in tmp_path.rglob("*"))


@pytest.mark.parametrize(
    "fitting,evaluation",
    [({"s0", "s1"}, {"s1", "s2"}), (set(), {"s0"}), (["s0", "s0"], ["s1"])],
)
def test_overlapping_or_invalid_rosters_refused_before_writes(tmp_path, fitting, evaluation):
    rows = {u: row(i) for i, u in enumerate(["s0", "s1", "s2"])}
    with pytest.raises(ValueError, match="disjoint"):
        pi.write_public_triplet_cache(
            tmp_path / "cache",
            rows,
            public_input=POLICY,
            source_revision="synthetic",
            fitting_uids=fitting,
            evaluation_uids=evaluation,
        )
    assert not (tmp_path / "cache").exists()


def test_incomplete_or_existing_cache_never_published(tmp_path):
    kwargs = dict(
        public_input=POLICY,
        source_revision="synthetic",
        fitting_uids={"s0", "s1"},
        evaluation_uids={"s2"},
    )
    with pytest.raises(ValueError, match="cover"):
        pi.write_public_triplet_cache(tmp_path, {"s0": row(0), "s2": row(2)}, **kwargs)
    assert not (tmp_path / pi.CACHE_DIRECTORY).exists()
    with pytest.raises(FileExistsError):
        pi.write_public_triplet_cache(tmp_path, {}, **kwargs)
    bad = row(1)
    bad["images"][5, 0, 0, 0] = 1  # absent slot carrying pixels is refused before writing
    with pytest.raises(ValueError):
        pi.write_public_triplet_cache(
            tmp_path / "other", {"s0": row(0), "s1": bad, "s2": row(2)}, **kwargs
        )
    assert not (tmp_path / "other" / pi.CACHE_DIRECTORY).exists()
    assert not (tmp_path / "other" / f"{pi.CACHE_DIRECTORY}.partial" / "members/s1.npz").exists()


def source_namespace():
    from tools.research.public_dinov2_input_reference import reference

    namespace, _ = reference.load(reference.SOURCE_PATH.read_bytes(), [])
    reference.configure(namespace, copy.deepcopy(PIXEL_GROUP))
    return namespace, reference


def source_studies(tmp_path):
    from tools.research.public_dinov2_input_reference import synthetic

    root = tmp_path / "train_series"
    studies = {
        name: synthetic.write_case(root, name, synthetic.MIRROR) for name in ("left", "right")
    }
    plane_map = {}
    for name, study in studies.items():
        plane_map |= synthetic.plane_map(study, synthetic.MIRROR[name])
    return root, studies, plane_map


def test_bounded_producer_stores_exact_source_rows(tmp_path):
    namespace, reference = source_namespace()
    root, studies, plane_map = source_studies(tmp_path)
    fitting, evaluation = [studies["left"].name], [studies["right"].name]
    digest = pi.produce_public_triplet_cache(
        tmp_path / "cache",
        public_input=POLICY,
        namespace=namespace,
        study_root=root,
        plane_map=plane_map,
        source_revision="synthetic",
        fitting_uids=fitting,
        evaluation_uids=evaluation,
        max_studies=2,
    )
    index, index_sha = pi.load_public_index(tmp_path / "cache", POLICY, "synthetic")
    assert index_sha == digest and index["pixel_group"] == PIXEL_GROUP
    adapter = pi.load_adapter(POLICY)
    labels, mask = torch.rand(len(TARGETS)), torch.ones(len(TARGETS))
    for name, study in studies.items():
        original = reference.render_original(namespace, [study], plane_map)
        member = np.load(tmp_path / "cache" / pi.CACHE_DIRECTORY / f"members/{study.name}.npz")
        assert member["images"].dtype == np.uint8
        np.testing.assert_array_equal(member["images"], original["cache"][0])
        np.testing.assert_array_equal(member["slot_mask"], original["mask"][0].astype(bool))
        expected = adapter.render_inputs(namespace, study, plane_map, labels, mask)
        stored = adapter.study_inputs(
            *(torch.from_numpy(member[k]) for k in pi.MEMBER_ARRAYS),
            labels,
            mask,
            study_uid=study.name,
        )
        for key in ("images", "series_mask", "features", "positions", "labels"):
            assert torch.equal(stored[key], expected[key]), (name, key)
    assert set(index["members"]) == set(fitting) | set(evaluation)


def test_producer_bounds_source_and_roster_refusals(tmp_path):
    namespace, _ = source_namespace()
    root, studies, plane_map = source_studies(tmp_path)
    uids = sorted(s.name for s in studies.values())
    base = dict(
        public_input=POLICY,
        namespace=namespace,
        study_root=root,
        plane_map=plane_map,
        source_revision="synthetic",
        fitting_uids=uids[:1],
        evaluation_uids=uids[1:],
        max_studies=2,
    )
    rules = dict(namespace["RULES"], slot_fallback=True)
    for change, error in [
        (dict(max_studies=1), ValueError),
        (dict(max_studies=2.0), ValueError),
        (dict(namespace=namespace | {"RULES": rules}), ValueError),
        (dict(namespace=namespace | {"CROP_MM": 161.25}), ValueError),
        (dict(namespace=namespace | {"SLOTS": namespace["SLOTS"][::-1]}), ValueError),
        (dict(evaluation_uids=uids), ValueError),
        (dict(evaluation_uids=["1.2.826.0.1.999"]), FileNotFoundError),
        (dict(fitting_uids=["../train_series/" + uids[0]]), ValueError),
        (dict(public_input=POLICY | {"adapter_sha256": "0" * 64}), ValueError),
    ]:
        with pytest.raises(error):
            pi.produce_public_triplet_cache(tmp_path / "cache", **(base | change))
        assert not (tmp_path / "cache").exists()


def test_cache_producer_cli_refuses_off_owning_node(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setattr("os.uname", lambda: argparse.Namespace(nodename="elsewhere"))
    script = importlib.import_module("scripts.newton.produce_public_triplet_cache")
    names = ("studies", "artifact-root", "folds", "series", "dicom-root", "source")
    names += ("member-manifest", "checkpoint", "out-cache")
    args = [f"--{k}={tmp_path}" for k in names]
    with pytest.raises(ValueError, match="owning-node"):
        script.main(args + ["--max-studies=3403"])


@pytest.mark.parametrize("uid", ["../escaped", "/absolute", "nested/name", "back\\slash", "", ".."])
def test_writer_rejects_unsafe_uids_before_any_write(tmp_path, uid):
    with pytest.raises(ValueError, match="safe individual"):
        pi.write_public_triplet_cache(
            tmp_path,
            {uid: row(0)},
            public_input=POLICY,
            source_revision="synthetic",
            fitting_uids=[uid],
            evaluation_uids=[],
        )
    assert list(tmp_path.iterdir()) == []


def test_writer_rejects_overlapping_rosters_before_any_write(tmp_path):
    with pytest.raises(ValueError, match="disjoint unique"):
        pi.write_public_triplet_cache(
            tmp_path,
            {"study": row(0)},
            public_input=POLICY,
            source_revision="synthetic",
            fitting_uids=["study"],
            evaluation_uids=["study"],
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,value", [("slots", ["arbitrary"] * 6), ("rules", {})])
def test_first_policy_rejects_unadmitted_pixel_recipe(field, value):
    group = copy.deepcopy(PIXEL_GROUP)
    group[field] = value
    with pytest.raises(ValueError, match="pixel group"):
        pi._validate_pixel_group(group)
