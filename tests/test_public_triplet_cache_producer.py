"""Streaming source-bound public-triplet cohort cache producer; engineering only."""

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from rsna.competitive import public_input as pi
from rsna.competitive import public_input_cache as pc
from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "initiative/audits/public-dinov2-real-inputs-20261013-v2"
PIXEL_GROUP = json.loads((COMPARISON / "actual-input-comparison.json").read_text())["provenance"][
    "member_pixel_group"
]
ADAPTER_SHA = hashlib.sha256(pi.ADAPTER_PATH.read_bytes()).hexdigest()
BINDING = dict(
    renderer=pi.RENDERER,
    source_sha256=pi.SOURCE_SHA256,
    node_text_sha256=pi.NODE_TEXT_SHA256,
    pixel_group_sha256=pi.canonical_sha256(PIXEL_GROUP),
    adapter_sha256=ADAPTER_SHA,
    plane_map_sha256="0" * 64,
)
POLICY = {k: BINDING[k] for k in pi.POLICY_KEYS - {"kind"}} | {"kind": pi.KIND}
CAPS = pc.Caps(
    max_study_objects=8,
    max_study_source_bytes=1 << 16,
    max_total_source_bytes=1 << 20,
    max_member_bytes=9_000_000,
    max_cache_bytes=200_000_000,
    reserve_bytes=1,
)
REVISION = "a" * 40


def uid(i):
    return f"1.2.840.{i + 1}"


def row(seed):
    generator = np.random.default_rng(seed)
    mask = np.array([True, True, True, False, True, False])
    images = generator.integers(0, 256, (6, 12, 336, 336), dtype=np.uint8)
    features = generator.random((6, 5), dtype=np.float32)
    positions = np.zeros((6, 12, 2), dtype=np.float32)
    positions[:, :, 0] = np.arange(12, dtype=np.float32) * 0.03 - 0.1
    positions[:, :, 1] = 1
    positions[1, 4:7] = 0
    images[~mask], features[~mask], positions[~mask] = 0, 0, 0
    return dict(images=images, slot_mask=mask, features=features, positions=positions)


class Store:
    """Owning-node original objects, served one object at a time."""

    def __init__(self, root, count=6, objects=2):
        self.root, self.objects, self.calls = Path(root), {}, []
        for i in range(count):
            study, records = uid(i), []
            for k in range(objects):
                series, sop = f"{study}.1", f"{study}.1.{k + 1}"
                payload = hashlib.sha256(f"{i}:{k}".encode()).digest() * (8 + k)
                path = f"train_series/{study}/{series}/{sop}.dcm"
                (self.root / path).parent.mkdir(parents=True, exist_ok=True)
                (self.root / path).write_bytes(payload)
                records.append(
                    dict(
                        split="train",
                        study_uid=study,
                        series_uid=series,
                        sop_uid=sop,
                        path=path,
                        sha256=hashlib.sha256(payload).hexdigest(),
                        bytes=len(payload),
                        storage="lfs",
                    )
                )
            self.objects[study] = records
        self.fail: dict = {}

    def fetch(self, record, target):
        self.calls.append(record["study_uid"])
        if self.fail.get(record["study_uid"]) == "transport":
            raise pc.TransportFailure("transport")
        payload = (self.root / record["path"]).read_bytes()
        if self.fail.get(record["study_uid"]) == "corrupt":
            payload = payload[:-1] + bytes([payload[-1] ^ 1])
        with target.open("xb") as stream:
            stream.write(payload)


class Render:
    def __init__(self, scratch, cache):
        self.scratch, self.cache, self.calls, self.fail = Path(scratch), Path(cache), [], set()
        self.resident = []

    def __call__(self, study):
        study = Path(study)
        self.calls.append(study.name)
        self.resident.append(sorted(p.name for p in self.scratch.iterdir()))
        if study.name in self.fail:
            raise RuntimeError("synthetic render failure " + study.name)
        files = sorted(study.rglob("*.dcm"))
        seed = int.from_bytes(hashlib.sha256(b"".join(f.read_bytes() for f in files)).digest()[:4])
        return row(seed)


def rosters(store, fitting=4):
    uids = sorted(store.objects)
    return uids[:fitting], uids[fitting:]


def run(tmp_path, store, render=None, scratch="scratch", **changes):
    fitting, evaluation = rosters(store)
    render = render or Render(tmp_path / scratch, tmp_path / "cache")
    values = (
        dict(
            render=render,
            renderer_binding=BINDING,
            transport=store,
            objects=store.objects,
            fitting_uids=fitting,
            evaluation_uids=evaluation,
            expected_counts=(len(fitting), len(evaluation)),
            pixel_group=copy.deepcopy(PIXEL_GROUP),
            source_revision="synthetic",
            raw_source_revision=REVISION,
            scratch=tmp_path / scratch,
            caps=CAPS,
            header_check=lambda path, record: None,
        )
        | changes
    )
    return pc.produce(tmp_path / "cache", **values), render


def frame(store):
    fitting, evaluation = rosters(store)
    rows = []
    for i, study in enumerate(fitting + evaluation):
        values = dict(study_uid=study, leakage_group=f"g{i}", split="train")
        for t in TARGETS:
            values[t], values[t + "__mask"] = i % 2, float((i + len(t)) % 3 != 0)
        rows.append(values)
    studies = pd.DataFrame(rows)
    return (
        studies,
        studies[studies.study_uid.isin(fitting)],
        studies[~studies.study_uid.isin(fitting)],
    )


def dataset(cache, studies):
    return pi.PublicTripletStudyDataset(
        studies,
        pd.DataFrame(),
        cache,
        public_input=POLICY,
        source_revision="synthetic",
        slices=10,
        side=336,
        max_series=6,
    )


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes())
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_streams_one_study_at_a_time_into_an_exact_loadable_cohort(tmp_path):
    store = Store(tmp_path / "store")
    summary, render = run(tmp_path, store)
    root = tmp_path / "cache" / pi.CACHE_DIRECTORY
    cohort = sorted(store.objects)
    assert render.calls == cohort
    # Only the current study's originals were resident and every earlier one was released.
    assert render.resident == [[study] for study in cohort]
    assert not (tmp_path / "scratch").exists()
    assert summary["members"] == 6 and summary["newly_produced_members"] == 6
    assert summary["peak_resident_studies"] == 1 and not summary["labels_or_masks_stored"]
    assert summary["source_objects"] == 12
    studies, train, outer = frame(store)
    receipt = pi.verify_public_cohort(tmp_path / "cache", POLICY, train, outer, "synthetic")
    assert receipt["index_sha256"] == summary["index_sha256"]
    data = dataset(tmp_path / "cache", studies)
    for index, study in enumerate(studies.study_uid):
        files = sorted((store.root / "train_series" / study).rglob("*.dcm"))
        seed = int.from_bytes(hashlib.sha256(b"".join(f.read_bytes() for f in files)).digest()[:4])
        expected = row(seed)
        with np.load(root / f"members/{study}.npz", allow_pickle=False) as member:
            assert set(member.files) == set(pi.MEMBER_ARRAYS)
            assert member["images"].dtype == np.uint8
            for name, value in expected.items():
                assert np.array_equal(member[name], value)
        item = data[index]
        windows = np.stack([expected["images"][:, s : s + 3] for s in range(10)], axis=1)
        assert torch.equal(item["images"], torch.from_numpy(windows).float().div(255))
        assert torch.equal(item["positions"], torch.from_numpy(expected["positions"][:, 1:11]))
        assert torch.equal(item["series_mask"], torch.from_numpy(expected["slot_mask"]))
        source = studies.iloc[index]
        assert item["label_mask"].tolist() == [source[t + "__mask"] for t in TARGETS]
    # Members are stored once; nothing float or per-view is retained.
    assert sorted(p.name for p in (root / "members").iterdir()) == [f"{u}.npz" for u in cohort]
    assert sorted(p.name for p in root.iterdir()) == [
        "index.json",
        "intent.json",
        "members",
        "members.jsonl",
    ]


def test_exported_summary_is_sanitized(tmp_path):
    store = Store(tmp_path / "store")
    summary, _ = run(tmp_path, store)
    text = json.dumps(summary)
    assert not any(study in text for study in store.objects)
    assert "train_series" not in text and str(tmp_path) not in text


@pytest.mark.parametrize(
    "tamper",
    ["overlap", "duplicate", "count", "unsafe_uid", "missing_object", "extra_object"],
)
def test_rosters_and_inventory_validated_before_any_intent(tmp_path, tamper):
    store = Store(tmp_path / "store")
    fitting, evaluation = rosters(store)
    changes = {}
    if tamper == "overlap":
        changes = dict(fitting_uids=fitting, evaluation_uids=evaluation[:-1] + fitting[:1])
    elif tamper == "duplicate":
        changes = dict(fitting_uids=fitting[:-1] + fitting[:1])
    elif tamper == "count":
        changes = dict(expected_counts=(2247, 1156))
    elif tamper == "unsafe_uid":
        changes = dict(fitting_uids=fitting[:-1] + ["../x"])
    elif tamper == "missing_object":
        changes = dict(objects={k: v for k, v in store.objects.items() if k != fitting[0]})
    elif tamper == "extra_object":
        changes = dict(objects=store.objects | {"1.2.9": store.objects[fitting[0]]})
    with pytest.raises(ValueError):
        run(tmp_path, store, **changes)
    assert not (tmp_path / "cache").exists() and not store.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "train_series/1.2.840.1/1.2.840.1.1/other.dcm"),
        ("split", "test"),
        ("study_uid", "1.2.840.2"),
        ("sha256", "z" * 64),
        ("bytes", 0),
        ("storage", "remote"),
    ],
)
def test_original_object_binding_tampering_refused_before_intent(tmp_path, field, value):
    store = Store(tmp_path / "store")
    objects = copy.deepcopy(store.objects)
    objects[uid(0)][0][field] = value
    with pytest.raises(ValueError):
        run(tmp_path, store, objects=objects)
    assert not (tmp_path / "cache").exists()


def test_duplicate_original_object_refused(tmp_path):
    store = Store(tmp_path / "store")
    objects = copy.deepcopy(store.objects)
    objects[uid(0)].append(copy.deepcopy(objects[uid(0)][0]))
    with pytest.raises(ValueError):
        run(tmp_path, store, objects=objects)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "cap", ["max_study_objects", "max_study_source_bytes", "max_total_source_bytes"]
)
def test_source_caps_refused_before_intent(tmp_path, cap):
    store = Store(tmp_path / "store")
    limit = {"max_study_objects": 1, "max_study_source_bytes": 300, "max_total_source_bytes": 1000}
    with pytest.raises(ValueError):
        run(tmp_path, store, caps=pc.Caps(**(CAPS.__dict__ | {cap: limit[cap]})))
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("mode", ["transport", "corrupt", "render", "header"])
def test_failure_leaves_identifiable_partial_cache_and_resume_reuses_completed(tmp_path, mode):
    store = Store(tmp_path / "store")
    cohort = sorted(store.objects)
    render = Render(tmp_path / "scratch", tmp_path / "cache")
    header = lambda path, record: None  # noqa: E731
    if mode in ("transport", "corrupt"):
        store.fail[cohort[3]] = mode
    elif mode == "render":
        render.fail.add(cohort[3])
    else:

        def header(path, record):
            if record["study_uid"] == cohort[3]:
                raise ValueError("header identity")

    with pytest.raises(pc.ProducerFailure) as failure:
        run(tmp_path, store, render=render, header_check=header)
    expected_stage = {"transport": "transport", "corrupt": "transport", "header": "transport"}
    assert failure.value.stage == expected_stage.get(mode, "render")
    assert failure.value.completed == 3
    assert cohort[3] not in str(failure.value) and "synthetic" not in str(failure.value)
    root = tmp_path / "cache" / pi.CACHE_DIRECTORY
    assert not (root / "index.json").exists()
    assert sorted(p.stem for p in (root / "members").iterdir()) == cohort[:3]
    assert not list((tmp_path / "scratch").iterdir())
    intent = json.loads((root / "intent.json").read_bytes())
    state = pc.reconcile(tmp_path / "cache", intent)
    assert state == {"state": "partial", "completed_members": 3}
    with pytest.raises(FileNotFoundError):
        pi.load_public_index(tmp_path / "cache", POLICY, "synthetic")
    before = snapshot(root / "members")
    # Without an explicit resume the existing cache is never touched.
    with pytest.raises(pc.ProducerFailure):
        run(tmp_path, store, scratch="scratch2")
    assert snapshot(root / "members") == before
    store.fail.clear()
    resumed, second = run(tmp_path, store, scratch="scratch2", resume=True)
    assert second.calls == cohort[3:]
    assert resumed["resumed_verified_members"] == 3 and resumed["newly_produced_members"] == 3
    after = snapshot(root / "members")
    assert all(after[name] == value for name, value in before.items())
    fresh_path = tmp_path / "fresh"
    fresh_path.mkdir()
    fresh, _ = run(fresh_path, Store(fresh_path / "store"))
    assert fresh["index_sha256"] == resumed["index_sha256"]


def test_complete_cache_is_reconciled_without_rendering(tmp_path):
    store = Store(tmp_path / "store")
    first, _ = run(tmp_path, store)
    root = tmp_path / "cache" / pi.CACHE_DIRECTORY
    before = snapshot(root)
    second, render = run(tmp_path, store, scratch="scratch2", resume=True)
    assert render.calls == [] and second["index_sha256"] == first["index_sha256"]
    assert snapshot(root) == before and not (tmp_path / "scratch2").exists()


@pytest.mark.parametrize("tamper", ["member_bytes", "unjournaled", "journal_tail", "intent"])
def test_interrupted_cache_reconciliation_is_read_only_and_fails_closed(tmp_path, tamper):
    store = Store(tmp_path / "store")
    cohort = sorted(store.objects)
    render = Render(tmp_path / "scratch", tmp_path / "cache")
    render.fail.add(cohort[3])
    with pytest.raises(pc.ProducerFailure):
        run(tmp_path, store, render=render)
    root = tmp_path / "cache" / pi.CACHE_DIRECTORY
    changes = {}
    if tamper == "member_bytes":
        path = root / f"members/{cohort[0]}.npz"
        payload = bytearray(path.read_bytes())
        payload[-100] ^= 1
        path.write_bytes(bytes(payload))
    elif tamper == "unjournaled":
        (root / f"members/{cohort[3]}.npz").write_bytes(b"partial")
    elif tamper == "journal_tail":
        with (root / "members.jsonl").open("ab") as stream:
            stream.write(b'{"uid":')
    else:
        changes = dict(caps=pc.Caps(**(CAPS.__dict__ | {"reserve_bytes": 2})))
    before = snapshot(root)
    store.calls.clear()
    with pytest.raises((ValueError, pc.ProducerFailure)):
        run(tmp_path, store, scratch="scratch2", resume=True, **changes)
    assert snapshot(root) == before and not store.calls
    assert not (root / "index.json").exists()


def test_member_and_cache_storage_caps_fail_closed(tmp_path):
    store = Store(tmp_path / "store")
    with pytest.raises(pc.ProducerFailure) as failure:
        run(tmp_path, store, caps=pc.Caps(**(CAPS.__dict__ | {"max_member_bytes": 1000})))
    assert failure.value.stage == "render" and failure.value.completed == 0
    other = tmp_path / "second"
    other.mkdir()
    store = Store(other / "store")
    with pytest.raises(pc.ProducerFailure) as failure:
        run(other, store, caps=pc.Caps(**(CAPS.__dict__ | {"max_cache_bytes": 30_000_000})))
    assert failure.value.stage == "storage" and failure.value.completed == 3
    assert not (other / "cache" / pi.CACHE_DIRECTORY / "index.json").exists()


def test_free_space_reserve_checked_before_each_study(tmp_path):
    store = Store(tmp_path / "store")
    usage = SimpleNamespace(free=CAPS.max_member_bytes)
    with pytest.raises(pc.ProducerFailure) as failure:
        run(tmp_path, store, disk_usage=lambda path: usage)
    assert failure.value.stage == "storage" and not store.calls


@pytest.mark.parametrize(
    "change",
    ["float_images", "extra_array", "nonzero_absent", "fabricated_geometry", "empty_mask"],
)
def test_malformed_rendered_rows_never_become_members(tmp_path, change):
    store = Store(tmp_path / "store")

    def render(study):
        value = row(1)
        if change == "float_images":
            value["images"] = value["images"].astype(np.float32) / 255
        elif change == "extra_array":
            value["labels"] = np.ones(12, dtype=np.float32)
        elif change == "nonzero_absent":
            value["images"][3, 0, 0, 0] = 1
        elif change == "fabricated_geometry":
            value["positions"][1, 5, 0] = 0.5
        else:
            value["slot_mask"][:] = False
        return value

    with pytest.raises(pc.ProducerFailure) as failure:
        run(tmp_path, store, render=render)
    assert failure.value.stage == "render"
    assert not list((tmp_path / "cache" / pi.CACHE_DIRECTORY / "members").iterdir())


def test_renderer_binding_must_match_member_pixel_group(tmp_path):
    store = Store(tmp_path / "store")
    with pytest.raises(ValueError):
        run(tmp_path, store, renderer_binding=BINDING | {"pixel_group_sha256": "0" * 64})
    assert not (tmp_path / "cache").exists()


class Response:
    def __init__(self, status, blocks, error=None):
        self.status_code, self.blocks, self.error = status, blocks, error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, size):
        for block in self.blocks:
            assert len(block) <= size
            yield block
        if self.error:
            raise self.error


class Session:
    def __init__(self, response):
        self.response, self.requests = response, []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.parametrize("mode", ["ok", "status", "connect", "stream", "oversize", "short", "sha"])
def test_bounded_resolve_transport_streams_exact_bytes_or_removes_partial(tmp_path, mode):
    payload = b"original-dicom" * 100
    record = dict(
        path="train_series/1.2/1.2.3/1.2.3.4.dcm",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    blocks = [payload[i : i + 64] for i in range(0, len(payload), 64)]
    response = {
        "ok": Response(200, blocks),
        "status": Response(404, []),
        "connect": OSError("secret-token https://host"),
        "stream": Response(200, blocks[:2], OSError("secret-token")),
        "oversize": Response(200, blocks + [b"x"]),
        "short": Response(200, blocks[:-1]),
        "sha": Response(200, [bytes([payload[0] ^ 1]) + payload[1:64]] + blocks[1:]),
    }[mode]
    session = Session(response)
    transport = pc.ResolveTransport(session, "secret-token", "owner/data", REVISION, block=64)
    target = tmp_path / "object.dcm"
    if mode == "ok":
        transport.fetch(record, target)
        assert target.read_bytes() == payload and transport.bytes == len(payload)
        url, kwargs = session.requests[0]
        assert f"/resolve/{REVISION}/train_series/1.2/1.2.3/1.2.3.4.dcm" in url
        assert kwargs["stream"] and kwargs["timeout"] == (10, 60)
        return
    with pytest.raises(pc.TransportFailure) as failure:
        transport.fetch(record, target)
    assert "secret" not in str(failure.value) and "host" not in str(failure.value)
    assert failure.value.__cause__ is None or "secret" not in str(failure.value.__cause__)
    assert not target.exists() and len(session.requests) == 1


def test_transport_never_overwrites_an_existing_object(tmp_path):
    target = tmp_path / "object.dcm"
    target.write_bytes(b"prior")
    transport = pc.ResolveTransport(Session(Response(200, [b"x"])), "t", "o/d", REVISION)
    with pytest.raises(FileExistsError):
        transport.fetch(dict(path="a.dcm", bytes=1, sha256="0" * 64), target)
    assert target.read_bytes() == b"prior"


def test_transport_requires_pinned_revision():
    with pytest.raises(ValueError):
        pc.ResolveTransport(Session(None), "t", "o/d", "main")


def test_bound_renderer_streams_original_render_exactly(tmp_path):
    import pydicom

    from tools.research.public_dinov2_input_reference import reference, synthetic
    from tools.research.public_dinov2_input_reference.dino_adapter import render_inputs

    namespace, receipt = reference.load()
    reference.configure(namespace, PIXEL_GROUP)
    source = tmp_path / "synthetic"
    store_root = tmp_path / "store"
    objects, plane_map, studies = {}, {}, []
    for side in ("left", "right"):
        study = synthetic.write_case(source, side, synthetic.MIRROR)
        plane_map |= synthetic.plane_map(study, synthetic.MIRROR[side])
        studies.append(study)
        records = []
        for path in sorted(study.rglob("*.dcm")):
            header = pydicom.dcmread(path, stop_before_pixels=True)
            series, sop = str(header.SeriesInstanceUID), str(header.SOPInstanceUID)
            relative = f"train_series/{study.name}/{series}/{sop}.dcm"
            (store_root / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, store_root / relative)
            size, sha = pc.file_sha256(path)
            records.append(
                dict(
                    split="train",
                    study_uid=study.name,
                    series_uid=series,
                    sop_uid=sop,
                    path=relative,
                    sha256=sha,
                    bytes=size,
                    storage="lfs",
                )
            )
        objects[study.name] = records
    renderer = pc.BoundRenderer(namespace, receipt, PIXEL_GROUP, plane_map)
    store = SimpleNamespace(
        fetch=lambda record, target: shutil.copyfile(store_root / record["path"], target)
    )
    left, right = (study.name for study in studies)
    caps = pc.Caps(**(CAPS.__dict__ | {"max_study_objects": 64, "max_study_source_bytes": 1 << 24}))
    caps = pc.Caps(**(caps.__dict__ | {"max_total_source_bytes": 1 << 25}))
    pc.produce(
        tmp_path / "cache",
        render=renderer,
        renderer_binding=renderer.binding,
        transport=store,
        objects=objects,
        fitting_uids=[left],
        evaluation_uids=[right],
        expected_counts=(1, 1),
        pixel_group=copy.deepcopy(PIXEL_GROUP),
        source_revision="synthetic",
        raw_source_revision=REVISION,
        scratch=tmp_path / "scratch",
        caps=caps,
    )
    frame = pd.DataFrame(
        [
            {"study_uid": s, **{t: 1.0 for t in TARGETS}, **{t + "__mask": 1.0 for t in TARGETS}}
            for s in (left, right)
        ]
    )
    policy = {k: renderer.binding[k] for k in pi.POLICY_KEYS - {"kind"}} | {"kind": pi.KIND}
    data = pi.PublicTripletStudyDataset(
        frame,
        pd.DataFrame(),
        tmp_path / "cache",
        public_input=policy,
        source_revision="synthetic",
        slices=10,
        side=336,
        max_series=6,
    )
    for index, study in enumerate(studies):
        public = reference.render_study(namespace, study, plane_map)
        expected = render_inputs(namespace, study, plane_map, torch.ones(12), torch.ones(12))
        item = data[index]
        assert torch.equal(item["images"], expected["images"])
        assert torch.equal(item["series_mask"], public["slot_mask"])
        assert torch.equal(item["positions"], expected["positions"])
        assert torch.equal(item["features"], expected["features"])
        assert torch.equal(item["labels"], torch.ones(12))
    assert renderer.adapter.study_inputs is not None


def test_bound_renderer_refuses_unreviewed_source_or_pixel_group():
    from tools.research.public_dinov2_input_reference import reference

    namespace, receipt = reference.load()
    reference.configure(namespace, PIXEL_GROUP)
    with pytest.raises(ValueError):
        pc.BoundRenderer(namespace, receipt | {"source_sha256": "0" * 64}, PIXEL_GROUP, {})
    with pytest.raises(ValueError):
        pc.BoundRenderer(namespace, receipt, PIXEL_GROUP | {"crop_mm": 161.25}, {})
