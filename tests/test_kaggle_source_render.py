"""Kaggle CPU source-render placement; synthetic DICOM/arrays, engineering only (no parity claim)."""

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pydicom
import pytest
import torch

from rsna.competitive import kaggle_source_render as ksr
from rsna.competitive import public_cache_producer as pc
from rsna.competitive import public_input as pi
from tools.research.public_dinov2_input_reference import synthetic

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
CAPS = dict(max_total_bytes=4 * pc.member_bytes(), min_free_bytes=1)


@pytest.fixture(scope="module")
def renderer():
    return pc.Renderer(POLICY, copy.deepcopy(PIXEL_GROUP))


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


def write_study(root, name):
    study = synthetic.write_case(root / pc.SPLIT, name)
    records = []
    for path in sorted(study.rglob("*.dcm")):
        sop = str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)
        target = path.with_name(f"{sop}.dcm")
        path.rename(target)
        records.append(record(root, target))
    return study.name, records, synthetic.plane_map(study, synthetic.CASES[name])


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "source"
    fit = write_study(source, "complete_right")
    held = write_study(source, "absent_slots")
    records = {fit[0]: fit[1], held[0]: held[1]}
    return dict(
        tmp=tmp_path,
        source=source,
        fit=fit[0],
        held=held[0],
        records=records,
        plane_map=fit[2] | held[2],
        studies={u: pc.study_source_sha256(u, r) for u, r in records.items()},
    )


def make_intent(s, cache):
    cache.mkdir()
    return pc.create_intent(
        cache,
        policy=POLICY,
        pixel_group=copy.deepcopy(PIXEL_GROUP),
        source_revision="synthetic",
        fitting_uids=[s["fit"]],
        evaluation_uids=[s["held"]],
        source_studies=s["studies"],
        caps=CAPS,
        cohort_sizes=(1, 1),
    )


def cohort_of(s, per_shard=1):
    return ksr.cohort_manifest(
        fitting_uids=[s["fit"]],
        evaluation_uids=[s["held"]],
        source_studies=s["studies"],
        source_revision="synthetic",
        binding={"code_sha256": "0" * 64},
        studies_per_shard=per_shard,
    )


def render(s, uid, renderer):
    return pc.render_member(
        uid,
        s["source"],
        s["records"][uid],
        s["plane_map"],
        renderer=renderer,
        source_sha256=s["studies"][uid],
        source_revision="synthetic",
        role="fitting" if uid == s["fit"] else "evaluation",
    )


def build_shards(s, renderer, cohort, root):
    """Shard every study; returns {shard_dir: manifest_sha}."""
    shards = {}
    for index, uids in enumerate(ksr.assign_shards(s["studies"], cohort["studies_per_shard"])):
        directory = root / f"shard-{index}"
        rows = []
        for uid in uids:
            payload, receipt, inputs = render(s, uid, renderer)
            order = {"study_uid": uid, "series_order": [r["series_uid"] for r in s["records"][uid]]}
            rows.append(ksr.write_shard_member(directory, uid, payload, receipt, order, inputs))
        shards[directory] = ksr.seal_shard(directory, cohort, index, rows, {"runtime": "synthetic"})
    return shards


def test_shard_merge_equals_direct_producer_staging_and_finalizes(setup, renderer):
    s = setup
    direct = s["tmp"] / "direct"
    make_intent(s, direct)
    for uid in s["studies"]:
        pc.stage_member(direct, uid, s["source"], s["records"][uid], s["plane_map"], renderer=renderer)
    cohort = cohort_of(s)
    shards = build_shards(s, renderer, cohort, s["tmp"] / "kaggle")
    merged = s["tmp"] / "merged"
    make_intent(s, merged)
    result = ksr.merge_shards(merged, shards, cohort, renderer=renderer)
    assert result["cohort_complete"] and result["outcome"] == {"imported": 2, "replayed": 2}
    for uid in s["studies"]:
        a = pc._load_arrays((direct / pi.CACHE_DIRECTORY / f"members/{uid}.npz").read_bytes())
        b = pc._load_arrays((merged / pi.CACHE_DIRECTORY / f"members/{uid}.npz").read_bytes())
        assert all(np.array_equal(a[n], b[n]) for n in pi.MEMBER_ARRAYS)
    again = ksr.merge_shards(merged, shards, cohort)
    assert again["outcome"] == {"already_complete": 2}
    pc.finalize(merged, policy=POLICY)
    with pytest.raises(pc.ProducerError):
        ksr.merge_shards(merged, shards, cohort)


def test_merge_refuses_tampered_foreign_overlapping_or_probe_shards(setup, renderer):
    s = setup
    cohort = cohort_of(s, per_shard=2)
    shards = build_shards(s, renderer, cohort, s["tmp"] / "kaggle")
    (directory,) = shards
    cache = s["tmp"] / "cache"
    make_intent(s, cache)
    with pytest.raises(ksr.PlacementError):
        ksr.merge_shards(cache, {directory: "f" * 64}, cohort)
    with pytest.raises(ksr.PlacementError):
        ksr.merge_shards(cache, shards, cohort_of(s, per_shard=1))
    (directory / "extra.json").write_text("{}")
    with pytest.raises(ksr.PlacementError):
        ksr.merge_shards(cache, shards, cohort)
    (directory / "extra.json").unlink()
    blob = directory / ksr.MEMBERS / f"{s['fit']}.npz.gz"
    original = blob.read_bytes()
    os.chmod(blob, 0o600)
    blob.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ksr.PlacementError):
        ksr.merge_shards(cache, shards, cohort)
    blob.write_bytes(original)
    probe = s["tmp"] / "probe"
    payload, receipt, inputs = render(s, s["fit"], renderer)
    row = ksr.write_shard_member(probe, s["fit"], payload, receipt, {"study_uid": s["fit"]}, inputs)
    sha = ksr.seal_shard(probe, cohort, None, [row], {"runtime": "synthetic"})
    with pytest.raises(ksr.PlacementError):
        ksr.merge_shards(cache, {probe: sha}, cohort)
    assert pc.reconcile(cache)["counts"] == {"pending": 2}
    assert ksr.merge_shards(cache, shards, cohort)["cohort_complete"]


def test_import_member_refuses_foreign_receipts_and_bytes(setup, renderer):
    s = setup
    cache = s["tmp"] / "cache"
    make_intent(s, cache)
    payload, receipt, _ = render(s, s["fit"], renderer)
    for bad in (
        receipt | {"role": "evaluation"},
        receipt | {"intent_sha256": "0" * 64},
        receipt | {"source": receipt["source"] | {"records_sha256": "0" * 64}},
        receipt | {"source": receipt["source"] | {"source_revision": "other"}},
        receipt | {"labels_stored": True},
        receipt | {"occupied_slots": receipt["occupied_slots"] + 1},
    ):
        with pytest.raises(pc.ProducerError):
            pc.import_member(cache, s["fit"], payload, bad)
    with pytest.raises(pc.ProducerError):
        pc.import_member(cache, s["fit"], payload[:-1] + b"\0", receipt)
    with pytest.raises(pc.ProducerError):
        pc.import_member(cache, s["held"], payload, receipt)
    assert pc.reconcile(cache)["counts"] == {"pending": 2}
    pc.import_member(cache, s["fit"], payload, receipt)
    with pytest.raises(pc.ProducerError):
        pc.import_member(cache, s["fit"], payload, receipt)


def test_render_member_refuses_source_drift_before_render(setup, renderer):
    s = setup
    with pytest.raises(pc.ProducerError):
        pc.render_member(
            s["fit"],
            s["source"],
            s["records"][s["fit"]],
            s["plane_map"],
            renderer=renderer,
            source_sha256="0" * 64,
            source_revision="synthetic",
            role="fitting",
        )
    path = s["source"] / s["records"][s["fit"]][0]["path"]
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(pc.ProducerError):
        render(s, s["fit"], renderer)


def test_encoding_is_deterministic_lossless_and_bounded():
    arrays = {
        n: (np.arange(np.prod(shape)) % 7).astype(dtype).reshape(shape)
        for n, (dtype, shape) in pi.MEMBER_ARRAYS.items()
    }
    payload = pc._serialize(arrays)
    sha = hashlib.sha256(payload).hexdigest()
    blob = ksr.encode_member(payload)
    assert blob == ksr.encode_member(payload)
    assert ksr.decode_member(blob, size=len(payload), sha256=sha) == payload
    for bad in (blob + b"x", blob[:-8], gzip.compress(payload + b"x", mtime=0)):
        with pytest.raises((ksr.PlacementError, EOFError, OSError, ValueError)):
            ksr.decode_member(bad, size=len(payload), sha256=sha)
    with pytest.raises(ksr.PlacementError):
        ksr.decode_member(blob, size=len(payload), sha256="0" * 64)
    sizes = ksr.measured_encodings(payload)
    assert sizes["exact"] == pc.member_bytes() and set(sizes) == {"exact", "gzip", "npz_deflate"}


def test_parity_hash_matches_accepted_worker_layout():
    inputs = {
        "images": torch.rand(6, 10, 3, 4, 4),
        "features": torch.rand(6, 5),
        "positions": torch.rand(6, 10, 2),
        "series_mask": torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool),
        "slice_mask": torch.ones(6, 10, dtype=torch.bool),
    }
    labels = np.array([1, 0.5, 0, 1], dtype=np.float32)
    mask = np.array([1, 0, 1, 0], dtype=np.float32)
    y, m = torch.from_numpy(labels), torch.from_numpy(mask)
    row = inputs | {"labels": torch.where(m.bool(), y, torch.zeros_like(y)), "label_mask": m}
    expected = hashlib.sha256()
    for key in ("images", "features", "positions", "series_mask", "slice_mask", "labels", "label_mask"):
        expected.update(row[key].numpy().tobytes())
    assert ksr.parity_digest([ksr.parity_study_bytes(inputs, labels, mask)]) == expected.hexdigest()


def test_parity_roster_is_the_salted_rank():
    uids = [f"1.2.{i}" for i in range(40)]
    roster = ksr.parity_roster(reversed(uids))
    rank = sorted(uids, key=lambda u: hashlib.sha256(f"{ksr.PARITY_SALT}:{u}".encode()).hexdigest())
    assert roster == rank[:8]


def test_mount_and_corrections_bound_to_real_after_bytes(tmp_path):
    with pytest.raises(ksr.PlacementError):
        ksr.resolve_mount(tmp_path)
    mount = tmp_path / "competitions" / ksr.COMPETITION
    (mount / "train_series/s/x").mkdir(parents=True)
    (mount / "train_series.csv").write_text("a\n")
    assert ksr.resolve_mount(tmp_path) == mount
    alias = tmp_path / ksr.COMPETITION
    alias.symlink_to(mount)
    with pytest.raises(ksr.PlacementError):
        ksr.resolve_mount(tmp_path)
    alias.unlink()
    data = b"corrected"
    relative = "train_series/s/x/y.dcm"
    (mount / relative).write_bytes(data)
    change = {"after": {"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
    assert ksr.verify_corrections(mount, {relative: change})["corrected_objects_verified"] == 1
    (mount / relative).write_bytes(data + b"!")
    with pytest.raises(ksr.PlacementError):
        ksr.verify_corrections(mount, {relative: change})
    with pytest.raises(ksr.PlacementError):
        ksr.verify_corrections(mount, {"other": change})


def test_shard_plan_uses_measured_costs_and_refuses_unfit_limits():
    plan = ksr.plan_shards(
        cohort_studies=3403,
        seconds_per_study=10.0,
        encoded_bytes_per_study=1000,
        scratch_bytes_per_study=100,
        max_wall_seconds=4000,
        max_output_bytes=10**9,
        scratch_free_bytes=10**6,
        min_free_bytes=10,
    )
    assert plan["studies_per_shard"] == 200 and plan["shards"] == 18 and plan["bound"] == "time"
    with pytest.raises(ksr.PlacementError):
        ksr.plan_shards(
            cohort_studies=10,
            seconds_per_study=10.0,
            encoded_bytes_per_study=1000,
            scratch_bytes_per_study=10**6,
            max_wall_seconds=4000,
            max_output_bytes=10**9,
            scratch_free_bytes=10**6,
            min_free_bytes=10,
        )
    assert ksr.assign_shards(["c", "a", "b"], 2) == [["a", "b"], ["c"]]


def test_budget_binds_code_runtime_and_cpu_only(tmp_path):
    runtime = {"placement": ksr.PLACEMENT}
    budget = {
        "schema_version": 1,
        "placement": ksr.PLACEMENT,
        "competition": ksr.COMPETITION,
        "cpu_only": True,
        "gpus": 0,
        "internet": False,
        "max_studies": 8,
        "max_study_source_bytes": 10**9,
        "min_free_bytes": 10**9,
        "max_output_bytes": 10**9,
        "max_wall_seconds": 3600,
        "code_sha256": "c" * 64,
        "runtime_sha256": None,
    }

    def load(value):
        path = tmp_path / "budget.json"
        path.write_text(json.dumps(value))
        return ksr.load_budget(path, hashlib.sha256(path.read_bytes()).hexdigest(), "c" * 64, runtime)

    assert load(budget)["max_studies"] == 8
    assert load(budget | {"runtime_sha256": ksr.canonical_sha256(runtime)})
    for bad in (
        {"gpus": 1},
        {"internet": True},
        {"cpu_only": False},
        {"code_sha256": "d" * 64},
        {"runtime_sha256": "e" * 64},
        {"max_studies": 0},
        {"extra": 1},
    ):
        with pytest.raises(ksr.PlacementError):
            load(budget | bad)


def test_cpu_runtime_refuses_visible_cuda():
    with pytest.raises(ksr.PlacementError):
        ksr.check_cpu_runtime({"CUDA_VISIBLE_DEVICES": "0"})
