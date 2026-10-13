"""Streaming source-bound public triplet producer; engineering only, not accuracy."""

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_public_triplet_training import PIXEL_GROUP, POLICY, cohort, dataset, row

from rsna.competitive import public_input as pi
from rsna.schema import TARGETS


def rosters(count=8):
    manifest, folds = cohort(count)
    fitting = list(folds.study_uid[folds.fold != 0])
    evaluation = list(folds.study_uid[folds.fold == 0])
    return manifest, folds, fitting, evaluation


class Renderer:
    """Deterministic rows; optionally fails once at a given study to simulate a crash."""

    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def __call__(self, uid):
        self.calls.append(uid)
        if uid == self.fail_at:
            self.fail_at = None
            raise RuntimeError("interrupted")
        return row(int(uid[1:]))


def stream(cache, render, fitting, evaluation, **changes):
    values = dict(
        pixel_group=copy.deepcopy(PIXEL_GROUP),
        source_revision="synthetic",
        fitting_uids=fitting,
        evaluation_uids=evaluation,
    )
    return pi.stream_public_triplet_cache(cache, render, **(values | changes))


def tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_streamed_cache_matches_in_memory_writer_and_trainer_admission(tmp_path):
    manifest, folds, fitting, evaluation = rosters()
    render = Renderer()
    digest = stream(tmp_path / "a", render, fitting, evaluation)
    assert render.calls == sorted(fitting + evaluation)
    rows = {uid: row(int(uid[1:])) for uid in manifest.study_uid}
    memory = pi.write_public_triplet_cache(
        tmp_path / "b",
        rows,
        pixel_group=copy.deepcopy(PIXEL_GROUP),
        source_revision="synthetic",
        fitting_uids=fitting,
        evaluation_uids=evaluation,
    )
    assert digest == memory
    assert tree(tmp_path / "a") == tree(tmp_path / "b")
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == [pi.CACHE_DIRECTORY]
    train, outer = manifest[folds.fold != 0], manifest[folds.fold == 0]
    receipt = pi.verify_public_cohort(tmp_path / "a", POLICY, train, outer, "synthetic")
    assert receipt["members"] == 8 and receipt["index_sha256"] == digest
    data = dataset(tmp_path / "a", manifest)
    item = data[3]
    source = torch.from_numpy(rows["s3"]["images"])
    assert torch.equal(item["images"][:, 2], source[:, 2:5].float() / 255)
    assert torch.equal(item["positions"], torch.from_numpy(rows["s3"]["positions"])[:, 1:11])


def test_interrupted_run_resumes_to_identical_bytes_without_rerendering(tmp_path):
    _, _, fitting, evaluation = rosters()
    order = sorted(fitting + evaluation)
    clean = stream(tmp_path / "clean", Renderer(), fitting, evaluation)
    first = Renderer(fail_at=order[5])
    with pytest.raises(RuntimeError):
        stream(tmp_path / "run", first, fitting, evaluation)
    assert not (tmp_path / "run" / pi.CACHE_DIRECTORY).exists()
    with pytest.raises(ValueError, match="resume"):
        stream(tmp_path / "run", Renderer(), fitting, evaluation)
    second = Renderer()
    assert stream(tmp_path / "run", second, fitting, evaluation, resume=True) == clean
    assert second.calls == order[5:]
    assert tree(tmp_path / "run") == tree(tmp_path / "clean")
    with pytest.raises(ValueError, match="never overwritten"):
        stream(tmp_path / "run", Renderer(), fitting, evaluation, resume=True)


def test_uncommitted_member_and_torn_journal_line_are_rerendered(tmp_path):
    _, _, fitting, evaluation = rosters()
    order = sorted(fitting + evaluation)
    clean = stream(tmp_path / "clean", Renderer(), fitting, evaluation)
    with pytest.raises(RuntimeError):
        stream(tmp_path / "run", Renderer(fail_at=order[4]), fitting, evaluation)
    partial = tmp_path / "run" / f".{pi.CACHE_DIRECTORY}.partial"
    members = partial / pi.CACHE_DIRECTORY / "members"
    (members / f"{order[4]}.npz").write_bytes(b"uncommitted")
    (members / f"{order[5]}.npz.partial").write_bytes(b"torn")
    journal = partial / pi.PRODUCER_JOURNAL
    lines = journal.read_bytes().split(b"\n")
    journal.write_bytes(b"\n".join(lines[:3]) + b"\n" + lines[3][:20])
    second = Renderer()
    assert stream(tmp_path / "run", second, fitting, evaluation, resume=True) == clean
    assert second.calls == order[3:]


@pytest.mark.parametrize(
    "tamper", ["member_bytes", "journal_order", "pixel_group", "revision", "roster", "extra"]
)
def test_resume_refuses_tampered_state_or_a_different_request(tmp_path, tamper):
    _, _, fitting, evaluation = rosters()
    order = sorted(fitting + evaluation)
    with pytest.raises(RuntimeError):
        stream(tmp_path, Renderer(fail_at=order[4]), fitting, evaluation)
    partial = tmp_path / f".{pi.CACHE_DIRECTORY}.partial"
    changes = {}
    if tamper == "member_bytes":
        path = partial / pi.CACHE_DIRECTORY / "members" / f"{order[1]}.npz"
        payload = bytearray(path.read_bytes())
        payload[-100] ^= 1
        path.write_bytes(bytes(payload))
    elif tamper == "journal_order":
        journal = partial / pi.PRODUCER_JOURNAL
        lines = journal.read_bytes().splitlines()
        journal.write_bytes(b"\n".join([lines[1], lines[0], *lines[2:]]) + b"\n")
    elif tamper == "pixel_group":
        changes["pixel_group"] = copy.deepcopy(PIXEL_GROUP) | {"crop_mm": 161.25}
    elif tamper == "revision":
        changes["source_revision"] = "other"
    elif tamper == "roster":
        fitting, evaluation = fitting[1:], [*evaluation, fitting[0]]
    elif tamper == "extra":
        journal = partial / pi.PRODUCER_JOURNAL
        entry = json.loads(journal.read_bytes().splitlines()[0])
        journal.write_bytes(journal.read_bytes() + json.dumps(entry | {"labels": 1}).encode())
        journal.write_bytes(journal.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        stream(tmp_path, Renderer(), fitting, evaluation, resume=True, **changes)
    assert not (tmp_path / pi.CACHE_DIRECTORY).exists()


@pytest.mark.parametrize(
    "change", ["empty_mask", "nonzero_absent", "fabricated_geometry", "float_images", "labels"]
)
def test_rows_violating_the_source_contract_are_never_committed(tmp_path, change):
    _, _, fitting, evaluation = rosters()

    def render(uid):
        arrays = row(int(uid[1:]))
        if uid == "s2":
            if change == "empty_mask":
                arrays["slot_mask"][:] = False
                arrays["images"][:] = arrays["features"][:] = arrays["positions"][:] = 0
            elif change == "nonzero_absent":
                arrays["images"][5, 0, 0, 0] = 1
            elif change == "fabricated_geometry":
                arrays["positions"][1, 5, 0] = 0.5
            elif change == "float_images":
                arrays["images"] = arrays["images"].astype(np.float32) / 255
            else:
                arrays["labels"] = np.ones(12, dtype=np.float32)
        return arrays

    with pytest.raises(ValueError, match="source"):
        stream(tmp_path, render, fitting, evaluation)
    members = tmp_path / f".{pi.CACHE_DIRECTORY}.partial" / pi.CACHE_DIRECTORY / "members"
    assert sorted(p.name for p in members.iterdir()) == ["s0.npz", "s1.npz"]


def test_source_renderer_rows_equal_original_render_and_hashed_adapter_inputs(tmp_path):
    from tools.research.public_dinov2_input_reference import cache_producer, reference, synthetic
    from tools.research.public_dinov2_input_reference.dino_adapter import render_inputs

    namespace, receipt = reference.load(reference.SOURCE_PATH.read_bytes(), [])
    reference.configure(namespace, PIXEL_GROUP)
    split = tmp_path / "train_series"
    studies, planes = {}, {}
    for side in ("left", "right"):
        study = synthetic.write_case(split, side, synthetic.MIRROR)
        studies[side] = study.name
        planes.update(synthetic.plane_map(study, synthetic.MIRROR[side]))
    render = cache_producer.source_row_renderer(
        POLICY, namespace, receipt, copy.deepcopy(PIXEL_GROUP), split, planes
    )
    uids = sorted(studies.values())
    stream(tmp_path / "cache", render, uids[:1], uids[1:])
    with pytest.raises(ValueError, match="unavailable"):
        render("absent")
    frame = pd.DataFrame([{"study_uid": uid, **{t: 0.0 for t in TARGETS}} for uid in uids])
    for t in TARGETS:
        frame[t + "__mask"] = 1.0
    data = dataset(tmp_path / "cache", frame)
    for index, uid in enumerate(uids):
        study = split / uid
        public = reference.render_study(namespace, study, planes)
        expected = render_inputs(
            namespace, study, planes, torch.zeros(len(TARGETS)), torch.ones(len(TARGETS))
        )
        item = data[index]
        member = data._member(uid)
        assert torch.equal(member["images"], public["images"])
        assert torch.equal(member["slot_mask"], public["slot_mask"])
        for key in ("images", "series_mask", "slice_mask", "features", "positions"):
            assert torch.equal(item[key], expected[key]), key
        assert torch.equal(item["label_mask"], torch.ones(len(TARGETS)))


@pytest.mark.parametrize("field", ["source_sha256", "node_text_sha256", "pixel_group", "slots"])
def test_source_renderer_refuses_unbound_namespace_or_pixel_group(tmp_path, field):
    from tools.research.public_dinov2_input_reference import cache_producer, reference

    namespace, receipt = reference.load(reference.SOURCE_PATH.read_bytes(), [])
    reference.configure(namespace, PIXEL_GROUP)
    group = copy.deepcopy(PIXEL_GROUP)
    if field == "pixel_group":
        group["crop_mm"] = 161.25
    elif field == "slots":
        namespace = dict(namespace, SLOTS=namespace["SLOTS"][::-1])
    else:
        receipt = dict(receipt, **{field: "0" * 64})
    with pytest.raises(ValueError):
        cache_producer.source_row_renderer(POLICY, namespace, receipt, group, tmp_path, {})


def test_policy_adapter_bytes_unchanged_by_producer():
    assert hashlib.sha256(pi.ADAPTER_PATH.read_bytes()).hexdigest() == POLICY["adapter_sha256"]
    config = json.loads(
        (
            pi.ADAPTER_PATH.parents[3]
            / "initiative/experiments/competitive-v1/public-input-recipe-v1"
            / "public-triplet-config.json"
        ).read_text()
    )
    assert config["public_input"] == POLICY


def test_prepared_producer_script_binds_release_rosters_and_reports_no_ids(tmp_path, monkeypatch):
    import argparse
    import importlib.util

    path = pi.ADAPTER_PATH.parents[3] / "scripts/newton/produce_public_triplet_cache.py"
    spec = importlib.util.spec_from_file_location("produce_public_triplet_cache", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    manifest, folds, _, _ = rosters()
    manifest["study_uid"] = folds["study_uid"] = [f"s{i}-uid" for i in range(8)]
    series = tmp_path / "series.csv"
    series.write_text("SeriesInstanceUID,Anatomical_Plane\n1,Sagittal\n")
    for name in ("source", "manifest", "member.pt"):
        (tmp_path / name).write_bytes(b"x")
    binding = {
        "series_sha256": hashlib.sha256(series.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    seen = {}

    def renderer(policy, namespace, receipt, pixel_group, split_root, plane_map):
        seen.update(plane_map=plane_map, split_root=split_root)
        return lambda uid: row(int(uid[1]))

    def development(frame, f, fold, scope):
        assert (fold, scope) == (0, "outer_fold")
        return manifest[folds.fold != 0], manifest[folds.fold == 0]

    receipt = {"source_ref": "r", "source_sha256": "s", "node_text_sha256": "n"}
    for name, value in dict(
        EXPECTED_POPULATION=(4, 4),
        file_sha256=lambda p: script.provenance.CHECKPOINT_SHA256,
        member_pixel_group=lambda raw: copy.deepcopy(PIXEL_GROUP),
        load_verified_release=lambda *a: (manifest, binding),
        load_folds=lambda *a: folds,
        select_development=development,
    ).items():
        monkeypatch.setattr(script, name, value)
    monkeypatch.setattr(script.reference, "load", lambda raw, messages: ({}, receipt))
    monkeypatch.setattr(script.reference, "configure", lambda namespace, group: None)
    monkeypatch.setattr(script.cache_producer, "source_row_renderer", renderer)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    (tmp_path / "dicom" / "train_series").mkdir(parents=True)
    args = argparse.Namespace(
        source=str(tmp_path / "source"),
        member_manifest=str(tmp_path / "manifest"),
        checkpoint=str(tmp_path / "member.pt"),
        studies="studies",
        artifact_root="artifacts",
        folds="folds",
        series=str(series),
        cache=str(tmp_path / "cache"),
        dicom_root=str(tmp_path / "dicom"),
        resume=False,
    )
    report = script.run(args)
    assert report["members"] == 8 and report["status"] == "produced_not_admitted"
    assert seen == {"plane_map": {"1": "Sagittal"}, "split_root": tmp_path / "dicom/train_series"}
    assert not any(uid in json.dumps(report) for uid in manifest.study_uid)
    with pytest.raises(ValueError, match="never overwritten"):
        script.run(args)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        script.run(args)
