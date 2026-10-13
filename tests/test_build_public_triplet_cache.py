"""CPU-only public triplet cache wrapper: placement/funding, corrected sources, source order."""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pydicom
import pytest
from test_public_cache_producer import PIXEL_GROUP, POLICY, record
from test_public_source_table import build

from rsna.competitive import public_cache_producer as pc
from rsna.competitive import public_input as pi
from rsna.competitive import public_source_table as pst
from tools.research.public_dinov2_input_reference import reference, synthetic

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_public_triplet_cache", ROOT / "scripts/newton/build_public_triplet_cache.py"
)
assert SPEC and SPEC.loader
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)
LIMITS = dict(
    max_total_bytes=10**9, min_free_bytes=1, max_study_source_bytes=10**8, max_studies_per_run=2
)
TIE_CASES = {
    "tied": [
        synthetic.SeriesSpec("Sagittal", "PD FS", slices=20, seed=1),
        synthetic.SeriesSpec("Sagittal", "PD FS", slices=20, seed=2),
        synthetic.SeriesSpec("Sagittal", "PD FS", slices=26, seed=3),  # unaccepted, most slices
        synthetic.SeriesSpec("Coronal", "PD FS", slices=18, seed=4),
    ]
}
UNACCEPTED = 3


def node(monkeypatch, name="evc41", cuda=""):
    monkeypatch.setattr(wrapper.os, "uname", lambda: SimpleNamespace(nodename=name))
    for variable in wrapper.DEVICE_LIST_VARIABLES + wrapper.DEVICE_COUNT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    if cuda is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)


def funding_for(runtime, **changes):
    return (
        dict(
            schema_version=2,
            node=runtime["node"],
            placement=runtime["placement"],
            cpu_only=True,
            step_gpus=0,
            **wrapper.code_hashes(),
            runtime_sha256=pi.canonical_sha256(runtime),
            **LIMITS,
        )
        | changes
    )


def test_runtime_is_funded_evc41_or_alternate_evc31_cpu_only(monkeypatch):
    node(monkeypatch)
    runtime = wrapper.check_runtime()
    assert runtime["node"] == "evc41" and runtime["placement"] == "primary"
    assert runtime["step_gpus"] == 0 and runtime["cuda_visible_devices"] == ""
    node(monkeypatch, "evc31")
    assert wrapper.check_runtime()["placement"] == "independently_funded_alternate"
    node(monkeypatch)
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "0")
    wrapper.check_runtime()
    for name, cuda in (("evc34", ""), ("evc41", None), ("evc41", "0"), ("evc41", "GPU-1")):
        node(monkeypatch, name, cuda)
        with pytest.raises(ValueError):
            wrapper.check_runtime()
    for variable, value in (
        ("SLURM_JOB_GPUS", "0"),
        ("SLURM_STEP_GPUS", "1"),
        ("SLURM_GPUS_ON_NODE", "1"),
        ("SLURM_GPUS", "h100:1"),
    ):
        node(monkeypatch)
        monkeypatch.setenv(variable, value)
        with pytest.raises(ValueError, match="zero job/step GPUs"):
            wrapper.check_runtime()
    node(monkeypatch)
    monkeypatch.setattr(wrapper.sys.modules["torch"].cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="CUDA device"):
        wrapper.check_runtime()


def test_funding_binds_node_placement_code_and_runtime(tmp_path, monkeypatch):
    node(monkeypatch)
    runtime, hashes = wrapper.check_runtime(), wrapper.code_hashes()
    path = tmp_path / "funding.json"

    def load(value, runtime=runtime):
        path.write_text(json.dumps(value))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return wrapper.load_funding(path, sha, runtime, hashes)

    assert load(funding_for(runtime))["node"] == "evc41"
    path.write_text(json.dumps(funding_for(runtime)))
    with pytest.raises(ValueError, match="SHA256"):
        wrapper.load_funding(path, "0" * 64, runtime, hashes)
    for change in (
        dict(schema_version=1),
        dict(cpu_only=False),
        dict(step_gpus=1),
        dict(step_gpus=False),
        dict(max_total_bytes=0),
        dict(min_free_bytes=1.5),
        dict(extra=1),
    ):
        with pytest.raises(ValueError, match="fields"):
            load(funding_for(runtime, **change))
    legacy = {k: v for k, v in funding_for(runtime).items() if k in wrapper.LIMIT_KEYS}
    with pytest.raises(ValueError, match="fields"):
        load(legacy | dict(schema_version=1, node="evc31"))
    for change in (
        dict(node="evc31", placement="independently_funded_alternate"),
        dict(placement="independently_funded_alternate"),
        dict(node="evc34"),
    ):
        with pytest.raises(ValueError, match="node/placement"):
            load(funding_for(runtime, **change))
    alternate = runtime | dict(node="evc31", placement="independently_funded_alternate")
    with pytest.raises(ValueError, match="node/placement"):
        load(funding_for(alternate, placement="primary"), alternate)
    assert load(funding_for(alternate), alternate)["node"] == "evc31"
    for key in wrapper.HASH_KEYS:
        with pytest.raises(ValueError, match="hash differs"):
            load(funding_for(runtime, **{key: "0" * 64}))
    with pytest.raises(ValueError, match="hash differs"):
        load(funding_for(runtime), runtime | dict(torch="9.9"))


def test_private_output_does_not_overlap_git_packet_or_sources(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    out = wrapper.private_output(tmp_path / "private", [source])
    assert out.stat().st_mode & 0o777 == 0o700
    for bad in (
        Path("relative"),
        ROOT / "artifacts" / "cache",
        wrapper.RETAINED_PACKET / "x",
        source / "cache",
    ):
        with pytest.raises(ValueError):
            wrapper.private_output(bad, [source])
    with pytest.raises(ValueError, match="overlap"):
        wrapper.private_output(tmp_path, [source])
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(ValueError, match="owner-only"):
        wrapper.private_output(shared)
    link = tmp_path / "link"
    link.symlink_to(out)
    with pytest.raises(ValueError):
        wrapper.private_output(link)


def bound_case(tmp_path, **overrides):
    case, changes = build(**overrides)
    series = tmp_path / "series.csv"
    case["accepted_series"].assign(Anatomical_Plane="Sagittal").to_csv(series, index=False)
    plan, receipt = tmp_path / "plan.json", tmp_path / "receipt.json"
    plan.write_bytes(case["plan_bytes"])
    receipt.write_text(json.dumps(case["receipt"]))
    release = dict(
        series_sha256=hashlib.sha256(series.read_bytes()).hexdigest(), manifest_sha256="r" * 64
    )
    options = {k: case[k] for k in ("expected", "plan_file_sha256", "plan_sha256")}
    args = (case["inventory"], series, release, case["fitting_uids"], case["evaluation_uids"])
    return args, (plan, receipt), options, changes


def test_sources_bind_corrected_accepted_objects_then_six_field_records(tmp_path):
    args, files, options, changes = bound_case(tmp_path)
    objects, summary, hashes, corrected = wrapper.bind_sources(*args, *files, **options)
    assert summary["source_revision"] == pst.CORRECTED_SOURCE_REVISION
    assert corrected == {c["series_uid"] for c in changes}
    assert hashes["series_csv_sha256"] == args[2]["series_sha256"]
    records = wrapper.producer_records(objects)
    flat = {r["path"]: r for rows in records.values() for r in rows}
    assert all(set(r) == pc.RECORD_KEYS for r in flat.values())
    assert not any(r["series_uid"].endswith(".9") for r in flat.values())  # unaccepted excluded
    assert set(records) == {"1.1", "1.2", "1.3"}
    for change in changes:  # after-correction identities only
        assert flat[change["path"]]["sha256"] == change["after"]["sha256"]
        assert flat[change["path"]]["bytes"] == change["after"]["bytes"]
        assert flat[change["path"]]["sha256"] != change["before"]["sha256"]


def test_sources_refuse_series_csv_unfiltered_or_precorrected_inventory(tmp_path):
    args, files, options, changes = bound_case(tmp_path)
    release = dict(args[2], series_sha256="0" * 64)
    with pytest.raises(ValueError, match="series CSV"):
        wrapper.bind_sources(args[0], args[1], release, *args[3:], *files, **options)
    accepting_all = options | {"expected": dict(options["expected"], excluded_source_series=0)}
    with pytest.raises(ValueError, match="Unaccepted"):
        wrapper.bind_sources(*args, *files, **accepting_all)
    inventory = args[0].copy()
    path = changes[0]["path"]
    inventory.loc[inventory.path.eq(path), ["sha256", "bytes"]] = [
        changes[0]["after"]["sha256"],
        changes[0]["after"]["bytes"],
    ]
    with pytest.raises(ValueError, match="baseline"):
        wrapper.bind_sources(inventory, *args[1:], *files, **options)
    with pytest.raises(ValueError):
        wrapper.bind_sources(args[0].drop(index=0), *args[1:], *files, **options)


def test_producer_records_require_bound_identity_layout():
    good = dict(
        split=pc.SPLIT,
        study_uid="1.1",
        series_uid="1.1.1",
        sop_uid="1.1.1.1",
        path=f"{pc.SPLIT}/1.1/1.1.1/1.1.1.1.dcm",
        sha256="a" * 64,
        bytes=1,
        storage="lfs",
    )
    assert wrapper.producer_records({"1.1": [good]})["1.1"][0].keys() == pc.RECORD_KEYS
    for change in (
        dict(path=f"{pc.SPLIT}/1.1/1.1.2/1.1.1.1.dcm"),
        dict(path=f"{pc.SPLIT}/1.1/../1.1.1/1.1.1.1.dcm"),
        dict(split="test_series"),
        dict(study_uid="1.2"),
    ):
        with pytest.raises(ValueError):
            wrapper.producer_records({"1.1": [good | change]})
    with pytest.raises(ValueError):
        wrapper.producer_records({"1.1": [{k: v for k, v in good.items() if k != "storage"}]})


def write_study(root, name, cases=None):
    study = synthetic.write_case(root / pc.SPLIT, name, cases)
    records = []
    for path in sorted(study.rglob("*.dcm")):
        sop = str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)
        target = path.with_name(f"{sop}.dcm")
        path.rename(target)
        records.append(record(root, target))
    specs = (synthetic.CASES | TIE_CASES)[name]
    return study, records, synthetic.plane_map(study, specs)


def accepted_only(study, records):
    excluded = f"{study.name}.{UNACCEPTED}"
    return [r for r in records if r["series_uid"] != excluded]


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): (
            p.lstat().st_mode,
            p.lstat().st_mtime_ns,
            p.lstat().st_ino,
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
        )
        for p in sorted(root.rglob("*"))
    }


@pytest.fixture
def tied(tmp_path):
    source = tmp_path / "source"
    study, records, plane_map = write_study(source, "tied", TIE_CASES)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return SimpleNamespace(
        source=source,
        study=study,
        all=records,
        records=accepted_only(study, records),
        plane_map=plane_map,
        scratch=scratch,
        tmp=tmp_path,
    )


def test_transport_filters_unaccepted_series_before_any_read(tied):
    hidden = tied.study / f"{tied.study.name}.{UNACCEPTED}"
    for path in hidden.iterdir():
        path.chmod(0o000)  # any open/probe of the unaccepted series would fail
    hidden.chmod(0o000)
    try:
        before = snapshot(tied.source / pc.SPLIT / tied.study.name / f"{tied.study.name}.1")
        copied, order = wrapper.transport_study(
            tied.study, tied.scratch, tied.study.name, tied.records, LIMITS, origin="corrected"
        )
    finally:
        hidden.chmod(0o700)
    expected = [e.name for e in os.scandir(tied.study) if not e.name.endswith(f".{UNACCEPTED}")]
    assert order["series_order"] == expected == [e.name for e in os.scandir(copied)]
    assert order["excluded_series_present"] == 1 and order["transport_order_equal"]
    assert order["source_revision"] == pst.CORRECTED_SOURCE_REVISION
    assert sorted(p.name for p in copied.iterdir()) == sorted(expected)
    assert order["n_slices_tie_groups"] == [
        [s for s in expected if not s.endswith(".4")]
    ]  # the two 20-slice sagittal series, in scandir order
    assert snapshot(tied.source / pc.SPLIT / tied.study.name / f"{tied.study.name}.1") == before
    assert all(p.stat().st_mode & 0o777 == 0o400 for p in copied.rglob("*.dcm"))


def test_transport_refuses_unaccepted_inclusion_and_drift(tied, monkeypatch):
    def attempt(records=None, study=None, limits=LIMITS):
        wrapper.discard_unfinished_transport(tied.scratch)
        return wrapper.transport_study(
            study or tied.study,
            tied.scratch,
            tied.study.name,
            tied.records if records is None else records,
            limits,
            origin="corrected",
        )

    extra = tied.study / f"{tied.study.name}.1" / "stray.dcm"
    extra.write_bytes(b"x")
    with pytest.raises(ValueError, match="files differ"):
        attempt()
    extra.unlink()
    absent = f"{tied.study.name}.9"
    ghost = dict(tied.records[0], series_uid=absent, sop_uid=f"{absent}.1")
    ghost["path"] = f"{pc.SPLIT}/{tied.study.name}/{absent}/{absent}.1.dcm"
    with pytest.raises(ValueError, match="missing"):
        attempt(tied.records + [ghost])
    with pytest.raises(ValueError, match="layout"):
        attempt([dict(tied.records[0], path=f"{pc.SPLIT}/{tied.study.name}/../x/y.dcm")])
    with pytest.raises(ValueError, match="transport bound"):
        attempt(limits=LIMITS | dict(max_study_source_bytes=1))
    with pytest.raises(ValueError, match="reserve"):
        attempt(limits=LIMITS | dict(min_free_bytes=10**18))
    assert not any(tied.scratch.iterdir())
    drifted = [dict(r) for r in tied.records]
    drifted[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        attempt(drifted)
    attempt()
    with pytest.raises(ValueError, match="not empty"):
        wrapper.transport_study(
            tied.study, tied.scratch, tied.study.name, tied.records, LIMITS, origin="corrected"
        )


def test_transport_refuses_symlinks_and_traversal(tied):
    series = tied.study / f"{tied.study.name}.1"
    target = sorted(series.iterdir())[0]
    moved = tied.tmp / "elsewhere.dcm"
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(ValueError, match="regular files"):
        wrapper.transport_study(
            tied.study, tied.scratch, tied.study.name, tied.records, LIMITS, origin="corrected"
        )
    target.unlink()
    moved.rename(target)
    linked_root = tied.tmp / "linked"
    linked_root.symlink_to(tied.source)
    with pytest.raises(ValueError, match="symlinks"):
        wrapper.transport_study(
            linked_root / pc.SPLIT / tied.study.name,
            tied.scratch,
            tied.study.name,
            tied.records,
            LIMITS,
            origin="corrected",
        )
    other = tied.tmp / "series-elsewhere"
    series.rename(other)
    series.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="series directories"):
        wrapper.transport_study(
            tied.study, tied.scratch, tied.study.name, tied.records, LIMITS, origin="corrected"
        )


def test_transport_records_changed_series_order_for_explicit_replay(tied, monkeypatch):
    real = os.scandir

    def reversed_copy(path):
        entries = list(real(path))
        return iter(entries[::-1] if str(tied.scratch) in str(path) else entries)

    monkeypatch.setattr(wrapper.os, "scandir", reversed_copy)
    _, order = wrapper.transport_study(
        tied.study, tied.scratch, tied.study.name, tied.records, LIMITS, origin="corrected"
    )
    assert not order["transport_order_equal"]
    assert order["copied_series_order"] == order["series_order"][::-1]
    assert order["renderer_replays_original_series_order"]


def test_original_pick_slots_breaks_ties_by_first_scandir_row(tied, renderer):
    namespace = renderer.namespace
    copied, order = wrapper.transport_study(
        tied.study, tied.scratch, tied.study.name, tied.records, LIMITS, origin="corrected"
    )
    frame = namespace["annotate"](reference.header_frame(namespace, [copied], pc.SPLIT))
    assert list(frame.SeriesInstanceUID) == order["series_order"]
    first, second = order["n_slices_tie_groups"][0]
    slot = next(n for n, plane, _, fs in namespace["SLOTS"] if plane == "Sagittal" and fs)
    chosen = namespace["pick_slots"](frame, tied.plane_map)[tied.study.name][slot]
    assert chosen.SeriesInstanceUID == first
    reverse = frame.iloc[::-1].reset_index(drop=True)
    flipped = namespace["pick_slots"](reverse, tied.plane_map)[tied.study.name][slot]
    assert flipped.SeriesInstanceUID == second  # order is load-bearing; never reordered
    full = namespace["annotate"](reference.header_frame(namespace, [tied.study], pc.SPLIT))
    unfiltered = namespace["pick_slots"](full, tied.plane_map)[tied.study.name][slot]
    assert unfiltered.SeriesInstanceUID.endswith(f".{UNACCEPTED}")  # why filtering precedes probe


@pytest.fixture(scope="module")
def renderer():
    return pc.Renderer(POLICY, copy.deepcopy(PIXEL_GROUP))


def make_packet(tmp_path, study_name, records, revision=pst.HISTORICAL_SOURCE_REVISION):
    packet = tmp_path / "packet"
    packet.mkdir(parents=True)
    files = packet / wrapper.PACKET_FILES
    source = tmp_path / "source"
    for r in records:
        target = files / r["path"]
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_bytes((source / r["path"]).read_bytes())
    roster = json.dumps({"source_revision": revision, "files": records}).encode()
    (packet / wrapper.PACKET_ROSTER).write_bytes(roster)
    proof = {"private_inventory_sha256": hashlib.sha256(roster).hexdigest()}
    (packet / wrapper.PACKET_PROOF).write_text(json.dumps(proof))
    return packet


def test_retained_packet_reuse_requires_unchanged_uncorrected_objects(tied):
    uid = tied.study.name
    packet = make_packet(tied.tmp, uid, tied.all)
    records = {uid: tied.records}
    reusable, summary = wrapper.retained_studies(packet, records, set())
    assert reusable == {uid: packet / wrapper.PACKET_FILES / pc.SPLIT / uid}
    assert summary["retained_files"] == len(tied.records)
    overlap = {tied.records[0]["series_uid"]}
    reusable, summary = wrapper.retained_studies(packet, records, overlap)
    assert not reusable and summary["retained_refused"] == {"corrected_series_overlap": 1}
    corrected = [dict(r) for r in tied.records]
    corrected[0]["sha256"] = "f" * 64  # corrected object differs from the historical bytes
    reusable, summary = wrapper.retained_studies(packet, {uid: corrected}, set())
    assert not reusable and summary["retained_refused"] == {"changed_or_missing_object": 1}
    assert not wrapper.retained_studies(packet, {}, set())[0]
    (packet / wrapper.PACKET_PROOF).write_text(json.dumps({"private_inventory_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="admitted plan"):
        wrapper.retained_studies(packet, records, set())
    corrected_packet = make_packet(tied.tmp / "c", uid, [], pst.CORRECTED_SOURCE_REVISION)
    with pytest.raises(ValueError, match="historical"):
        wrapper.retained_studies(corrected_packet, records, set())


def cohort_intent(cache, records):
    return pc.create_intent(
        cache,
        policy=POLICY,
        pixel_group=copy.deepcopy(PIXEL_GROUP),
        source_revision=pst.CORRECTED_SOURCE_REVISION,
        fitting_uids=[sorted(records)[0]],
        evaluation_uids=[sorted(records)[1]],
        source_studies={u: pc.study_source_sha256(u, r) for u, r in records.items()},
        caps=dict(max_total_bytes=4 * pc.member_bytes(), min_free_bytes=1),
        cohort_sizes=(1, 1),
    )


def test_stage_from_retained_packet_read_only_with_bound_order(tied, renderer, monkeypatch):
    held_study, held, held_map = write_study(tied.source, "absent_slots")
    uid = tied.study.name
    packet = make_packet(tied.tmp, uid, tied.all)
    packet_before = snapshot(packet)
    records = {uid: tied.records, held_study.name: held}
    reusable, _ = wrapper.retained_studies(packet, records, set())
    out = tied.tmp / "out"
    cache = out / "cache"
    cache.mkdir(parents=True)
    intent_sha = cohort_intent(cache, records)
    plane_map = tied.plane_map | held_map
    real_scandir = os.scandir

    class Scan:
        def __init__(self, entries):
            self.entries = entries

        def __iter__(self):
            return iter(self.entries)

        def __enter__(self):
            return iter(self.entries)

        def __exit__(self, *_):
            return False

    def reversed_transport(path):
        with real_scandir(path) as scan:
            entries = list(scan)
        return Scan(entries[::-1] if str(tied.scratch) in str(path) else entries)

    monkeypatch.setattr(wrapper.os, "scandir", reversed_transport)
    receipt, order = wrapper.stage_study(
        cache,
        out,
        tied.scratch,
        uid,
        reusable[uid],
        records[uid],
        plane_map,
        LIMITS,
        renderer,
        origin="retained",
    )
    assert not receipt["reused"] and order["intent_sha256"] == intent_sha
    assert order["origin"] == "retained"
    assert order["source_revision"] == pst.HISTORICAL_SOURCE_REVISION
    first = order["n_slices_tie_groups"][0][0]
    assert first in receipt["slot_series"]
    assert not any(s and s.endswith(f".{UNACCEPTED}") for s in receipt["slot_series"])
    assert snapshot(packet) == packet_before  # retained originals untouched
    assert not any(tied.scratch.iterdir())  # temporary transport removed
    stored = (out / wrapper.ORDER_DIRECTORY / f"{uid}.json").read_bytes()
    assert json.loads(stored) == order
    again, _ = wrapper.stage_study(
        cache,
        out,
        tied.scratch,
        uid,
        reusable[uid],
        records[uid],
        plane_map,
        LIMITS,
        renderer,
        origin="retained",
    )
    assert again["reused"]
    with pytest.raises(ValueError, match="source-order receipt"):
        wrapper.bind_order_receipt(out, uid, dict(order, series_order=[]), intent_sha)
    report = json.dumps(wrapper.aggregate(cache, pc.reconcile(cache), out))
    assert uid not in report and "1.2.826" not in report
    assert json.loads(report)["source_order"]["studies_with_n_slices_ties"] == 1


def test_stage_failure_still_removes_transport(tied, renderer):
    held_study, held, _ = write_study(tied.source, "absent_slots")
    uid = tied.study.name
    records = {uid: tied.records, held_study.name: held}
    out = tied.tmp / "out"
    cache = out / "cache"
    cache.mkdir(parents=True)
    cohort_intent(cache, records)
    with pytest.raises(ValueError, match="acquired-slot"):
        wrapper.stage_study(
            cache,
            out,
            tied.scratch,
            uid,
            tied.study,
            records[uid],
            {},
            LIMITS,
            renderer,
            origin="corrected",
        )
    assert not any(tied.scratch.iterdir())


def test_unfinished_transport_and_reconciliation_are_explicit(tmp_path, tied):
    leftover = tied.scratch / pc.SPLIT / "1.2" / "1.2.1"
    leftover.mkdir(parents=True)
    (leftover / "a.dcm").write_bytes(b"x")
    (leftover / "a.dcm").chmod(0o400)
    assert wrapper.discard_unfinished_transport(tied.scratch) == 1
    assert not any(tied.scratch.iterdir())
    (tied.scratch / "link").symlink_to(tied.source)
    with pytest.raises(ValueError, match="symlink"):
        wrapper.discard_unfinished_transport(tied.scratch)
    assert tied.source.exists()
    base = dict(intent_sha256="0" * 64, unexpected=[], finalized=False, committed_bytes=0)
    assert wrapper.require_clean(base | dict(states={"pending": ["1"], "complete": ["2"]}))
    for change in (
        dict(states={"stale_partial": ["1"]}),
        dict(states={"receipt_unlinked": ["1"]}),
        dict(states={"corrupt": ["1"]}),
        dict(states={}, unexpected=["x"]),
        dict(states={}, finalized=True),
    ):
        with pytest.raises(ValueError, match="reconcile only"):
            wrapper.require_clean(base | change)


def test_aggregate_receipt_has_no_identifiers(tmp_path):
    cache = tmp_path / "cache"
    root = cache / pi.CACHE_DIRECTORY / pc.RECEIPTS
    root.mkdir(parents=True)
    secret = "1.2.826.0.1.3680043.8.498.999"
    receipt = dict(
        occupied_slots=5,
        slot_series=[secret + ".1"] * 5 + [None],
        laterality="R",
        role="fitting",
    )
    (root / f"{secret}.json").write_text(json.dumps(receipt))
    status = dict(
        intent_sha256="0" * 64,
        states={"complete": [secret]},
        counts={"complete": 1},
        unexpected=[],
        finalized=False,
        committed_bytes=1,
    )
    report = json.dumps(wrapper.aggregate(cache, status))
    assert secret not in report and "1.2.826" not in report
    assert json.loads(report)["occupied_slots_histogram"] == {"5": 1}


def test_real_inventory_parts_are_the_complete_plan_baseline():
    inventory = wrapper.read_inventory(wrapper.INVENTORY_DIR)
    plan = json.loads(wrapper.CORRECTION_PLAN.read_text())
    from rsna.audit import inventory_fingerprint

    assert list(inventory.columns) == pst.COLUMNS
    assert inventory_fingerprint(inventory) == plan["baseline_inventory_fingerprint_sha256"]
    assert hashlib.sha256(wrapper.CORRECTION_PLAN.read_bytes()).hexdigest() == (
        pst.CORRECTION_PLAN_FILE_SHA256
    )
    with pytest.raises(ValueError, match="Complete"):
        wrapper.read_inventory(wrapper.INVENTORY_DIR, parts=8)
    assert isinstance(inventory, pd.DataFrame)


@pytest.mark.parametrize(
    "argv",
    [
        ["--mode", "stage", "--private-out", "/x", "--funding", "f", "--funding-sha256", "0"],
        ["--mode", "stage", "--private-out", "/x", "--funding", "f", "--funding-sha256", "0"]
        + [f"--{k}=v" for k in ("source", "member-manifest", "studies", "artifact-root")]
        + ["--folds=v", "--series=v", "--dicom-root=/d"],
    ],
)
def test_stage_mode_requires_every_binding(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    with pytest.raises(SystemExit):
        wrapper.main()


def test_wrapper_never_dispatches_downloads_or_deletes_originals():
    text = (ROOT / "scripts/newton/build_public_triplet_cache.py").read_text()
    for forbidden in (
        "sbatch",
        "subprocess",
        "requests",
        "urllib",
        "huggingface_hub",
        "hf_hub",
        'evc31"\n',
    ):
        assert forbidden not in text
    assert text.count("shutil.rmtree") == 2  # temporary transport copies only


def test_readonly_private_output_does_not_create_missing_directory(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="Read-only reconciliation"):
        wrapper.private_output(missing, create=False)
    assert not missing.exists()


def test_reconcile_mode_never_creates_transport_directory(tmp_path, monkeypatch):
    out = tmp_path / "private"
    out.mkdir(mode=0o700)
    (out / "cache").mkdir()
    monkeypatch.setattr(wrapper, "check_runtime", lambda: {"node": "evc41"})
    monkeypatch.setattr(wrapper, "code_hashes", lambda: {})
    monkeypatch.setattr(wrapper, "load_funding", lambda *args: {})
    monkeypatch.setattr(
        pc,
        "reconcile",
        lambda cache: {
            "intent_sha256": "a" * 64,
            "states": {"complete": []},
            "counts": {},
            "unexpected": [],
            "finalized": False,
            "committed_bytes": 0,
        },
    )
    args = SimpleNamespace(
        mode="reconcile",
        private_out=out,
        dicom_root=None,
        retained_packet=None,
        funding="unused",
        funding_sha256="unused",
    )
    result = wrapper.run(args)
    assert result["mode"] == "reconcile" and not (out / "transport").exists()


def test_direct_source_and_transported_tied_series_match_bitwise(tied, renderer):
    # Exclude the deliberately foreign fixture series exactly as production does.
    accepted = {r["series_uid"] for r in tied.records}
    original = tied.tmp / "accepted-source" / pc.SPLIT / tied.study.name
    original.mkdir(parents=True)
    for entry in tied.study.iterdir():
        if entry.name in accepted:
            shutil.copytree(entry, original / entry.name)
    copied, order = wrapper.transport_study(
        original, tied.scratch, tied.study.name, tied.records, LIMITS, origin="corrected"
    )
    with reference.preserve_series_order(tied.study.name, order["series_order"]):
        direct, _ = renderer.render(original, tied.plane_map)
        transported, _ = renderer.render(copied, tied.plane_map)
    assert set(direct) == set(transported)
    for name in direct:
        assert np.array_equal(direct[name], transported[name])
