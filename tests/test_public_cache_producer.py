"""Streaming source-bound public triplet cache producer; synthetic DICOM, engineering only."""

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch

from rsna.competitive import public_cache_producer as pc
from rsna.competitive import public_input as pi
from rsna.schema import TARGETS
from tools.research.public_dinov2_input_reference import reference, synthetic
from tools.research.public_dinov2_input_reference.dino_adapter import render_inputs

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "initiative/audits/public-dinov2-real-inputs-20261013-v2"
PIXEL_GROUP = json.loads((COMPARISON / "actual-input-comparison.json").read_text())["provenance"][
    "member_pixel_group"
]
POLICY = json.loads(
    (
        ROOT / "initiative/experiments/competitive-v1/public-input-recipe-v1/"
        "public-triplet-config.json"
    ).read_text()
)["public_input"]
SIZES = (1, 1)
CAPS = dict(max_total_bytes=4 * pc.member_bytes(), min_free_bytes=1)


@pytest.fixture(scope="module")
def renderer():
    return pc.Renderer(POLICY, copy.deepcopy(PIXEL_GROUP))


def write_study(root, name):
    """Original-layout study: train_series/<study>/<series>/<sop>.dcm plus its records."""
    study = synthetic.write_case(root / pc.SPLIT, name)
    records = []
    for path in sorted(study.rglob("*.dcm")):
        sop = str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)
        target = path.with_name(f"{sop}.dcm")
        path.rename(target)
        records.append(record(root, target))
    return study.name, records, synthetic.plane_map(study, synthetic.CASES[name])


def record(root, path):
    header = pydicom.dcmread(path, stop_before_pixels=True)
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "study_uid": str(header.StudyInstanceUID),
        "series_uid": str(header.SeriesInstanceUID),
        "sop_uid": str(header.SOPInstanceUID),
    }


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "source"
    fit = write_study(source, "complete_right")
    held = write_study(source, "absent_slots")
    cache = tmp_path / "cache"
    cache.mkdir()
    plane_map = fit[2] | held[2]
    studies = {fit[0]: fit[1], held[0]: held[1]}
    return dict(
        source=source, cache=cache, fit=fit[0], held=held[0], records=studies, plane_map=plane_map
    )


def intent(s, **changes):
    values = (
        dict(
            policy=POLICY,
            pixel_group=copy.deepcopy(PIXEL_GROUP),
            source_revision="synthetic",
            fitting_uids=[s["fit"]],
            evaluation_uids=[s["held"]],
            source_studies={u: pc.study_source_sha256(u, r) for u, r in s["records"].items()},
            caps=CAPS,
            cohort_sizes=SIZES,
        )
        | changes
    )
    return pc.create_intent(s["cache"], **values)


def stage(s, uid, renderer):
    return pc.stage_member(
        s["cache"], uid, s["source"], s["records"][uid], s["plane_map"], renderer=renderer
    )


def tree(cache):
    root = cache / pi.CACHE_DIRECTORY
    return {
        p.relative_to(root).as_posix(): (
            p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class NoRender:
    def __init__(self, real):
        self.binding = real.binding

    def render(self, *args):
        raise AssertionError("completed member must never be re-rendered")


def frame(uids):
    rows = []
    for uid in uids:
        row = {"study_uid": uid}
        for target in TARGETS:
            row[target], row[target + "__mask"] = 1, 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def dataset(cache, uids):
    return pi.PublicTripletStudyDataset(
        frame(uids),
        pd.DataFrame(),
        cache,
        public_input=POLICY,
        source_revision="synthetic",
        slices=10,
        side=336,
        max_series=6,
    )


def test_actual_render_to_adapter_roundtrip_missing_slots_and_exact_normalization(setup, renderer):
    s = setup
    intent(s)
    first = stage(s, s["fit"], renderer)
    second = stage(s, s["held"], renderer)
    assert not first["reused"] and not second["reused"]
    assert first["labels_stored"] is False and second["occupied_slots"] < 6
    index_sha = pc.finalize(s["cache"], policy=POLICY)
    data = dataset(s["cache"], [s["fit"], s["held"]])
    assert data.index_sha256 == index_sha
    ns, _ = reference.load()
    reference.configure(ns, copy.deepcopy(PIXEL_GROUP))
    for position, uid in enumerate([s["fit"], s["held"]]):
        item = data[position]
        study_dir = s["source"] / pc.SPLIT / uid
        direct = render_inputs(ns, study_dir, s["plane_map"], torch.zeros(12), torch.ones(12))
        assert torch.equal(item["images"], direct["images"])
        assert torch.equal(item["series_mask"], direct["series_mask"])
        assert torch.equal(item["positions"], direct["positions"])
        assert torch.equal(item["features"], direct["features"])
        public = reference.render_study(ns, study_dir, s["plane_map"])
        expected = reference.windows(ns, public["images"]).permute(1, 0, 2, 3, 4).float() / 255
        assert torch.equal(item["images"], expected)
        assert item["images"].dtype == torch.float32 and item["images"].max() <= 1
        absent = ~item["series_mask"]
        assert item["images"][absent].count_nonzero() == 0
        assert item["features"][absent].count_nonzero() == 0
    assert absent.any()
    root = s["cache"] / pi.CACHE_DIRECTORY
    for uid in (s["fit"], s["held"]):
        with np.load(root / pc.MEMBERS / f"{uid}.npz", allow_pickle=False) as archive:
            assert set(archive.files) == set(pi.MEMBER_ARRAYS)
        receipt = json.loads((root / pc.RECEIPTS / f"{uid}.json").read_text())
        assert receipt["source"]["records_sha256"] == pc.study_source_sha256(uid, s["records"][uid])
        assert not any("label" in key for key in receipt if key != "labels_stored")
    with pytest.raises(pc.ProducerError, match="finalized"):
        stage(s, s["fit"], renderer)
    with pytest.raises(pc.ProducerError, match="already finalized"):
        pc.finalize(s["cache"], policy=POLICY)


def test_intent_requires_exact_disjoint_unique_cohort_and_bindings(setup):
    s = setup
    with pytest.raises(pc.ProducerError, match="Disjoint"):
        intent(s, cohort_sizes=pc.COHORT_SIZES)
    with pytest.raises(pc.ProducerError, match="Disjoint"):
        intent(s, evaluation_uids=[s["fit"]])
    with pytest.raises(pc.ProducerError, match="Disjoint"):
        intent(s, fitting_uids=[s["fit"], s["fit"]], cohort_sizes=(2, 1))
    with pytest.raises(pc.ProducerError, match="original-source digest"):
        intent(s, source_studies={s["fit"]: "0" * 64})
    with pytest.raises(ValueError):
        intent(s, policy=dict(POLICY, adapter_sha256="0" * 64))
    with pytest.raises(ValueError):
        intent(s, policy=dict(POLICY, source_sha256="0" * 64))
    with pytest.raises(ValueError):
        intent(s, policy=dict(POLICY, renderer="native384"))
    with pytest.raises(ValueError):
        intent(s, pixel_group=dict(PIXEL_GROUP, slots=list(reversed(PIXEL_GROUP["slots"]))))
    assert not (s["cache"] / pi.CACHE_DIRECTORY).exists()
    intent(s)
    with pytest.raises(pc.ProducerError, match="already exists"):
        intent(s)


def test_renderer_refuses_other_source_bytes_or_pixel_group():
    with pytest.raises(Exception):
        pc.Renderer(POLICY, copy.deepcopy(PIXEL_GROUP), reference.SOURCE_PATH.read_bytes() + b" ")
    with pytest.raises(ValueError):
        pc.Renderer(dict(POLICY, node_text_sha256="0" * 64), copy.deepcopy(PIXEL_GROUP))


def test_source_hash_and_uid_mismatches_refused_before_any_write(setup, renderer):
    s = setup
    intent(s)
    before = tree(s["cache"])
    records = s["records"][s["fit"]]
    path = s["source"] / records[0]["path"]
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(pc.ProducerError, match="bytes differ"):
        stage(s, s["fit"], renderer)
    path.write_bytes(original)
    header = pydicom.dcmread(path)
    header.SOPInstanceUID = header.SOPInstanceUID + "9"
    header.save_as(path, enforce_file_format=True)
    tampered = [record(s["source"], path) | {"sop_uid": records[0]["sop_uid"]}, *records[1:]]
    with pytest.raises(pc.ProducerError, match="digest"):
        pc.stage_member(
            s["cache"], s["fit"], s["source"], tampered, s["plane_map"], renderer=renderer
        )
    path.write_bytes(original)
    with pytest.raises(pc.ProducerError):
        pc.stage_member(
            s["cache"],
            s["fit"],
            s["source"],
            [dict(records[0], study_uid=s["held"]), *records[1:]],
            s["plane_map"],
            renderer=renderer,
        )
    assert tree(s["cache"]) == before


def test_actual_header_uid_drift_refused(setup, renderer):
    s = setup
    records = s["records"][s["fit"]]
    path = s["source"] / records[0]["path"]
    header = pydicom.dcmread(path)
    header.SOPInstanceUID = header.SOPInstanceUID + "9"
    header.save_as(path, enforce_file_format=True)
    s["records"][s["fit"]] = [record(s["source"], path) | {"sop_uid": records[0]["sop_uid"]}]
    s["records"][s["fit"]] += records[1:]
    intent(s)
    with pytest.raises(pc.ProducerError, match="UIDs differ"):
        stage(s, s["fit"], renderer)


@pytest.mark.parametrize("uid", ["../x", "a/b", "", ".", "..", "x\x00"])
def test_malicious_study_ids_refused(uid):
    with pytest.raises(pc.ProducerError):
        pc.study_source_sha256(uid, [])


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: dict(r, path="/" + r["path"]),
        lambda r: dict(r, path=r["path"].replace(pc.SPLIT, "test_series")),
        lambda r: dict(r, path=pc.SPLIT + "/../" + r["path"]),
        lambda r: dict(r, path=r["path"][:-4] + ".npz"),
        lambda r: dict(r, bytes=float(r["bytes"])),
        lambda r: dict(r, sha256="F" * 64),
        lambda r: dict(r, extra=1),
    ],
)
def test_malicious_or_drifted_records_refused(setup, edit):
    s = setup
    records = s["records"][s["fit"]]
    with pytest.raises(pc.ProducerError):
        pc.study_source_sha256(s["fit"], [edit(records[0]), *records[1:]])


@pytest.mark.parametrize("kind", ["symlink", "extra", "missing"])
def test_source_tree_substitution_refused(setup, renderer, kind):
    s = setup
    intent(s)
    first = s["source"] / s["records"][s["fit"]][0]["path"]
    if kind == "symlink":
        moved = first.with_suffix(".moved")
        first.rename(moved)
        first.symlink_to(moved)
        moved.rename(s["source"] / "outside.bin")
        first.unlink()
        first.symlink_to(s["source"] / "outside.bin")
    elif kind == "extra":
        (first.parent / "native.npz").write_bytes(b"x")
    else:
        first.unlink()
    with pytest.raises(pc.ProducerError):
        stage(s, s["fit"], renderer)
    assert not any((s["cache"] / pi.CACHE_DIRECTORY / pc.MEMBERS).iterdir())


def test_interrupted_link_resumes_without_render_and_reconcile_is_read_only(
    setup, renderer, monkeypatch
):
    s = setup
    intent(s)
    real_link = os.link
    monkeypatch.setattr(pc.os, "link", lambda *a: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        stage(s, s["fit"], renderer)
    monkeypatch.setattr(pc.os, "link", real_link)
    before = tree(s["cache"])
    status = pc.reconcile(s["cache"])
    assert status["states"] == {"receipt_unlinked": [s["fit"]], "pending": [s["held"]]}
    assert tree(s["cache"]) == before
    with pytest.raises(pc.ProducerError, match="Complete membership"):
        pc.finalize(s["cache"], policy=POLICY)
    resumed = stage(s, s["fit"], NoRender(renderer))
    assert resumed["reused"] is True
    snapshot = tree(s["cache"])
    assert stage(s, s["fit"], NoRender(renderer))["reused"] is True
    assert tree(s["cache"]) == snapshot
    assert pc.reconcile(s["cache"])["states"]["complete"] == [s["fit"]]


def test_partial_write_is_never_completed_or_silently_rerendered(setup, renderer, monkeypatch):
    s = setup
    intent(s)
    real = pc._write_exclusive

    def crash(path, payload):
        if path.name.endswith(".partial"):
            path.write_bytes(payload[: len(payload) // 2])
            raise OSError("crash mid write")
        real(path, payload)

    monkeypatch.setattr(pc, "_write_exclusive", crash)
    with pytest.raises(OSError):
        stage(s, s["fit"], renderer)
    monkeypatch.setattr(pc, "_write_exclusive", real)
    assert pc.reconcile(s["cache"])["states"]["stale_partial"] == [s["fit"]]
    with pytest.raises(pc.ProducerError, match="reconcile"):
        stage(s, s["fit"], NoRender(renderer))
    pc.release_stale_partial(s["cache"], s["fit"])
    assert not stage(s, s["fit"], renderer)["reused"]
    with pytest.raises(pc.ProducerError, match="stale partial"):
        pc.release_stale_partial(s["cache"], s["fit"])


def test_receipted_corrupt_bytes_and_tampered_members_refused(setup, renderer):
    s = setup
    intent(s)
    stage(s, s["fit"], renderer)
    stage(s, s["held"], renderer)
    member = s["cache"] / pi.CACHE_DIRECTORY / pc.MEMBERS / f"{s['fit']}.npz"
    data = bytearray(member.read_bytes())
    data[-10] ^= 1
    os.chmod(member, 0o600)
    member.write_bytes(bytes(data))
    assert pc.reconcile(s["cache"])["states"]["corrupt"] == [s["fit"]]
    with pytest.raises(pc.ProducerError):
        stage(s, s["fit"], NoRender(renderer))
    with pytest.raises(pc.ProducerError):
        pc.finalize(s["cache"], policy=POLICY)
    assert not (s["cache"] / pi.CACHE_DIRECTORY / pc.INDEX).exists()


def test_caps_refuse_before_any_write(setup, renderer, monkeypatch):
    s = setup
    with pytest.raises(pc.ProducerError, match="caps"):
        intent(s, caps=dict(CAPS, max_total_bytes=2 * pc.member_bytes() - 1))
    with pytest.raises(pc.ProducerError, match="caps"):
        intent(s, caps=dict(CAPS, extra=1))
    intent(s)
    before = tree(s["cache"])
    usage = pc.shutil.disk_usage(s["cache"])
    monkeypatch.setattr(pc.shutil, "disk_usage", lambda _: usage._replace(free=pc.member_bytes()))
    with pytest.raises(pc.ProducerError, match="reserve"):
        stage(s, s["fit"], NoRender(renderer))
    assert tree(s["cache"]) == before


def test_funded_bytes_include_interrupted_staging(setup, renderer):
    s = setup
    intent(s, caps=dict(CAPS, max_total_bytes=2 * pc.member_bytes()))
    stage(s, s["fit"], renderer)
    (s["cache"] / pi.CACHE_DIRECTORY / pc.STAGING / "foreign").write_bytes(b"x")
    with pytest.raises(pc.ProducerError, match="funded"):
        stage(s, s["held"], NoRender(renderer))


@pytest.mark.parametrize("native", ["float_member", "extra_member", "foreign_receipt"])
def test_native_or_foreign_cache_substitution_refused(setup, renderer, native):
    s = setup
    intent(s)
    stage(s, s["fit"], renderer)
    stage(s, s["held"], renderer)
    root = s["cache"] / pi.CACHE_DIRECTORY
    if native == "float_member":
        member = root / pc.MEMBERS / f"{s['held']}.npz"
        with np.load(member, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        arrays["images"] = arrays["images"].astype(np.float32) / 255
        member.unlink()
        np.savez(member, **arrays)
    elif native == "extra_member":
        np.savez(root / pc.MEMBERS / "native384.npz", images=np.zeros((1,), np.float16))
    else:
        (root / pc.RECEIPTS / "other.json").write_text("{}")
    with pytest.raises(pc.ProducerError):
        pc.finalize(s["cache"], policy=POLICY)
    assert not (root / pc.INDEX).exists()


def test_finalized_index_refuses_policy_and_roster_mismatch(setup, renderer):
    s = setup
    intent(s)
    stage(s, s["fit"], renderer)
    stage(s, s["held"], renderer)
    with pytest.raises(ValueError):
        pc.finalize(s["cache"], policy=dict(POLICY, adapter_sha256="0" * 64))
    pc.finalize(s["cache"], policy=POLICY)
    index, _ = pi.load_public_index(s["cache"], POLICY, "synthetic")
    with pytest.raises(ValueError):
        pi.verify_public_cohort(
            s["cache"], POLICY, frame([s["held"]]), frame([s["fit"]]), "synthetic"
        )
    summary = pi.verify_public_cohort(
        s["cache"], POLICY, frame([s["fit"]]), frame([s["held"]]), "synthetic"
    )
    assert summary["cohort_sha256"] == index["cohort_sha256"]
    with pytest.raises(ValueError):
        pi.load_public_index(s["cache"], POLICY, "other-revision")


def test_member_bytes_are_exact_and_content_independent():
    zero = {n: np.zeros(shape, dtype) for n, (dtype, shape) in pi.MEMBER_ARRAYS.items()}
    full = {n: np.ones(shape, dtype) for n, (dtype, shape) in pi.MEMBER_ARRAYS.items()}
    assert len(pc._serialize(zero)) == len(pc._serialize(full)) == pc.member_bytes()
    assert pc.member_bytes() >= 6 * 12 * 336 * 336
