"""Read-only prefix/terminal guards for the prepared full-cache successor."""

import ast
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "initiative/research/predictive-full-cache-continuation-20261020-v1"
SPEC = importlib.util.spec_from_file_location(
    "predictive_cache_reconcile", CONTROL / "reconcile.py"
)
reconcile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reconcile)


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path):
    original = tmp_path / "original"
    original.mkdir(mode=0o700)
    for name in ("cache", "scratch", "proofs"):
        (original / name).mkdir(mode=0o700)
    records = []
    for i in range(3):
        uid, series, sop = f"1.2.{i}", f"1.3.{i}", f"1.4.{i}"
        body = f"original-{i}".encode()
        rows = [
            {
                "split": "train_series",
                "study_uid": uid,
                "series_uid": series,
                "sop_uid": sop,
                "path": f"train_series/{uid}/{series}/{sop}.dcm",
                "bytes": len(body),
                "sha256": reconcile.digest(body),
            }
        ]
        records.append(
            {
                "uid": uid,
                "source_files": rows,
                "series_csv": (
                    "StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n"
                    f"{uid},{series},1,0,sagittal\n"
                ),
            }
        )
    prior_path = tmp_path / "old16-cache.npz"
    prior_path.write_bytes(b"already independently admitted cache")
    prior = [
        {
            "uid": records[0]["uid"],
            "path": str(prior_path),
            "proof": {"stat": reconcile.signature(prior_path)},
        }
    ]
    for i in (1, 2):
        row = records[i]
        stem = f"case-{i:04d}"
        write_json(
            original / "proofs" / (stem + ".intent.json"),
            {
                "source_ordinal": i,
                "source_bytes": sum(x["bytes"] for x in row["source_files"]),
                "original_source_table_sha256": reconcile.digest(
                    reconcile.json_bytes(row["source_files"])
                ),
            },
        )
    cached = original / "cache/case-0001.private.npz"
    cached.write_bytes(b"second independently admitted cache")
    proof = {
        "files": [
            {"source_path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
            for row in records[1]["source_files"]
        ],
        "source_bytes": records[1]["source_files"][0]["bytes"],
        "study_uid_sha256": reconcile.digest(records[1]["uid"].encode()),
        "series_csv_sha256": reconcile.digest(records[1]["series_csv"].encode()),
    }
    compressed = gzip.compress(json.dumps(proof).encode())
    (original / "proofs/case-0001.source-proof.private.json.gz").write_bytes(compressed)
    write_json(
        original / "proofs/case-0001.verified.private.json",
        {
            "uid": records[1]["uid"],
            "path": str(cached),
            "proof": {"stat": reconcile.signature(cached)},
            "source_proof_sha256": reconcile.digest(compressed),
            "original16_reused": False,
        },
    )
    owned = original / "scratch/fitting-study-retained"
    data = owned / "data"
    owned.mkdir(mode=0o700)
    data.mkdir(mode=0o700)
    (data / "train.csv").write_text("StudyInstanceUID\n" + records[2]["uid"] + "\n")
    (data / "train_series.csv").write_text(records[2]["series_csv"])
    pixel = data / records[2]["source_files"][0]["path"]
    pixel.parent.mkdir(mode=0o700, parents=True)
    pixel.write_bytes(b"ori")
    orphan = original / "cache/case-0002.private.npz"
    orphan.write_bytes(b"PK partial original archive")
    return original, records, prior, data


def test_inventory_is_read_only_and_maps_original_partial_without_filename_guessing(case):
    original, records, prior, data = case
    files = [p for p in original.rglob("*") if p.is_file()]
    before = {str(p): (p.read_bytes(), p.stat()) for p in files}
    result = reconcile.inventory(original, records, prior)
    assert set(result["complete"]) == {0, 1}
    assert result["pending"] == [2]
    assert result["buffers"] == {2: data}
    assert result["orphan_caches"] == {2: original / "cache/case-0002.private.npz"}
    assert before == {str(p): (p.read_bytes(), p.stat()) for p in files}


@pytest.mark.parametrize("kind", ["cache_stat", "source_proof", "intent", "buffer_uid", "metadata"])
def test_changed_original_bindings_are_refused(case, kind):
    original, records, prior, data = case
    if kind == "cache_stat":
        (original / "cache/case-0001.private.npz").write_bytes(b"changed")
    elif kind == "source_proof":
        (original / "proofs/case-0001.source-proof.private.json.gz").write_bytes(b"changed")
    elif kind == "intent":
        path = original / "proofs/case-0002.intent.json"
        value = json.loads(path.read_text())
        value["source_ordinal"] = 1
        write_json(path, value)
    elif kind == "buffer_uid":
        (data / "train.csv").write_text("StudyInstanceUID\n1.999\n")
    else:
        (data / "train_series.csv").write_text("different")
    with pytest.raises(ValueError):
        reconcile.inventory(original, records, prior)


def test_orphan_cache_without_original_source_buffer_is_not_rebuilt_blindly(case):
    original, records, prior, data = case
    # Simulate a malformed, unmappable retained buffer, not a cleanup operation.
    (data / "train.csv").write_text("StudyInstanceUID\n")
    with pytest.raises(ValueError, match="mapped unambiguously"):
        reconcile.inventory(original, records, prior)


def test_interrupted_verified_record_remains_unverified_and_keeps_source_recovery(case):
    original, records, prior, data = case
    (original / "proofs/case-0002.verified.private.json").write_text('{"uid":')
    result = reconcile.inventory(original, records, prior)
    assert result["pending"] == [2]
    assert result["buffers"][2] == data
    assert 2 in result["orphan_caches"] and 2 in result["partial_verified_records"]


def test_interrupted_whole_source_record_is_unverified_and_requires_file_readmission(case):
    original, records, prior, data = case
    path = data.parent / "source-proof.private.json"
    path.write_text('{"files":')
    result = reconcile.inventory(original, records, prior)
    assert result["buffers"][2] == data and result["partial_source_records"] == {2: path}
    assert 2 not in result["complete"]
    assert path.read_text() == '{"files":'


def test_terminal_gate_requires_exact_original_and_no_live_queue():
    receipt = reconcile.terminal_receipt("978318|TIMEOUT|0:15\n", "", job_id="978318")
    assert receipt["state"] == "TIMEOUT" and len(receipt["sha256"]) == 64
    for accounting, queue in (
        ("978318|RUNNING|0:0\n", ""),
        ("978318|COMPLETED|0:0\n", ""),
        ("978319|TIMEOUT|0:15\n", ""),
        ("978318|TIMEOUT|0:15\n", "978318|RUNNING\n"),
        ("978318|TIMEOUT|0:15\n978318|TIMEOUT|0:15\n", ""),
        ("978318|TIMEOUT|garbled\n", ""),
    ):
        with pytest.raises(ValueError):
            reconcile.terminal_receipt(accounting, queue, job_id="978318")


def test_physical_accounting_counts_retained_hardlinks_once(tmp_path):
    import os

    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"accepted original")
    os.link(first, second)
    assert reconcile.physical_bytes([first, second, tmp_path]) == first.stat().st_size


def test_original_case_function_is_reused_with_only_output_path_change():
    original = (
        ROOT / "initiative/research/predictive-full-raw-cache-20261020-v1/program.py.txt"
    ).read_text()
    node = next(
        n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == "one"
    )
    source = ast.get_source_segment(original, node)
    path = 'p = D / "cache" / ("case-" + str(i).zfill(4) + ".private.npz")'
    assert source.count(path) == 1
    updated = source.replace(path, "p = output_path(i)")
    compile(updated, "distinct_path_original_numerics", "exec")
    # This check binds the prepared wrapper to its still-existing source anchor;
    # it is not an actual cluster continuation, cache admission or clinical pass.
    compile((CONTROL / "program.py.txt").read_text(), "prepared_terminal_successor", "exec")
    compile((CONTROL / "context.py.txt").read_text(), "prepared_context_successor", "exec")


def test_context_waits_for_a_confirmed_live_partial_record_then_accepts_bound_source(
    case, tmp_path
):
    import types

    original, records, prior, _ = case
    new = tmp_path / "successor"
    (new / "proofs").mkdir(mode=0o700, parents=True)
    row = records[2]
    pixel_proof = {
        "files": [
            {"source_path": r["path"], "bytes": r["bytes"], "sha256": r["sha256"]}
            for r in row["source_files"]
        ],
        "source_bytes": sum(r["bytes"] for r in row["source_files"]),
        "study_uid_sha256": reconcile.digest(row["uid"].encode()),
        "series_csv_sha256": reconcile.digest(row["series_csv"].encode()),
    }
    compressed = gzip.compress(json.dumps(pixel_proof).encode())
    (new / "proofs/case-0002.source-proof.private.json.gz").write_bytes(compressed)
    cache = original / "cache/case-0002.private.npz"
    record = {
        "uid": row["uid"],
        "path": str(cache),
        "proof": {"stat": reconcile.signature(cache)},
        "source_proof_sha256": reconcile.digest(compressed),
    }
    output = new / "proofs/case-0002.verified.private.json"
    output.write_text('{"uid":')
    tree = ast.parse((CONTROL / "context.py.txt").read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("raw_record", "require_source_record")
    ]
    calls = []

    def finish_original_write(seconds):
        calls.append(seconds)
        output.write_text(json.dumps(record))

    namespace = {
        "recon": reconcile,
        "original": reconcile.inventory(original, records, prior),
        "new_raw": new,
        "old_raw": original,
        "records": records,
        "Path": Path,
        "hashlib": __import__("hashlib"),
        "gzip": gzip,
        "json": json,
        "producer_state": lambda: ("RUNNING", "0:0"),
        "time": types.SimpleNamespace(sleep=finish_original_write),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "original_context_wait", "exec"), namespace
    )
    assert namespace["raw_record"](2) == record and calls == [1]
    output.unlink()
    namespace["producer_state"] = lambda: ("TIMEOUT", "0:15")
    with pytest.raises(RuntimeError, match="preserve all context prefixes"):
        namespace["raw_record"](2)


@pytest.mark.parametrize("intact_orphan", [False, True])
def test_original_case_recovers_only_missing_dicom_bytes_and_preserves_originals(
    tmp_path, monkeypatch, intact_orphan
):
    """Execute the actual original one() decoder/parity body, not a mock renderer."""
    import contextlib
    import hashlib
    import io
    import shutil
    import threading
    import types
    import zipfile

    import numpy as np
    import torch

    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

    from rsna.competitive.predictive import cache, inputs, source_resume
    from rsna.competitive.predictive.cases import RawGroup, assemble

    torch.set_num_threads(1)
    uid, series = "2.25.202610201234", "2.25.202610201234.1"
    revision = "52f995f4568e36bfa5fbbaf69a89af41226b4f5d"
    original, successor = tmp_path / "original", tmp_path / "successor"
    for root in (original, successor):
        root.mkdir(mode=0o700)
        for name in ("cache", "scratch", "proofs"):
            (root / name).mkdir(mode=0o700)
    owned = original / "scratch/fitting-study-synthetic"
    data = owned / "data"
    data.mkdir(mode=0o700, parents=True)
    owned.chmod(0o700)
    source_dir = data / "train_series" / uid / series
    source_dir.mkdir(mode=0o700, parents=True)
    rows, bodies = [], {}
    for i in range(3):
        sop = series + "." + str(i + 1)
        path = source_dir / (sop + ".dcm")
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = sop
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = uid, series, sop
        ds.Rows, ds.Columns = 80, 80
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
        ds.PixelRepresentation, ds.SamplesPerPixel = 0, 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.InstanceNumber = i + 1
        ds.ImageOrientationPatient, ds.ImagePositionPatient = [1, 0, 0, 0, 1, 0], [0, 0, i]
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = ((np.arange(6400).reshape(80, 80) + i * 17) % 4096).astype("<u2").tobytes()
        ds.save_as(path, enforce_file_format=True)
        body = path.read_bytes()
        relative = path.relative_to(data).as_posix()
        bodies[relative] = body
        rows.append(
            {
                "split": "train_series",
                "study_uid": uid,
                "series_uid": series,
                "sop_uid": sop,
                "path": relative,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    series_csv = (
        "StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n"
        f"{uid},{series},1,0,sagittal\n"
    )
    (data / "train.csv").write_text("StudyInstanceUID\n" + uid + "\n")
    (data / "train_series.csv").write_text(series_csv)
    record = {"uid": uid, "source_files": rows, "series_csv": series_csv}
    write_json(
        original / "proofs/case-0000.intent.json",
        {
            "source_ordinal": 0,
            "source_bytes": sum(r["bytes"] for r in rows),
            "original_source_table_sha256": reconcile.digest(reconcile.json_bytes(rows)),
        },
    )
    raw = (
        ROOT / "initiative/research/predictive-raw-input-pilot-20261020-v1/program.py.txt"
    ).read_text()
    decoder = raw[raw.index("groups = {}") : raw.index("roster = 0")].replace(
        "assert file_bytes == 25131360", 'assert file_bytes == sum(r["bytes"] for r in rows)'
    )
    conversion = (
        ROOT / "initiative/research/predictive-raw16-cache-20261020-v1/group_conversion.py.txt"
    ).read_text()
    namespace = {
        "uid": uid,
        "rows": rows,
        "ram": owned,
        "np": np,
        "torch": torch,
        "pydicom": pydicom,
        "hashlib": hashlib,
        "m": inputs,
    }
    exec(compile(decoder, "actual_original_decoder", "exec"), namespace)
    exec(compile(conversion, "actual_original_conversion", "exec"), namespace)
    orphan = original / "cache/case-0000.private.npz"
    cache.write(
        orphan, namespace["sources"], reconcile.source_provenance(record, revision), cap_bytes=2**20
    )
    if not intact_orphan:
        orphan.write_bytes(orphan.read_bytes()[:50])
    partial = data / rows[1]["path"]
    prefix = len(bodies[rows[1]["path"]]) // 2
    partial.write_bytes(bodies[rows[1]["path"]][:prefix])
    (data / rows[2]["path"]).unlink()
    snapshots = {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in original.rglob("*") if p.is_file()
    }
    inv = reconcile.inventory(original, [record], [])
    plan = source_resume.reconcile(
        data, uid=uid, rows=rows, series_csv=series_csv, revision=revision, cap=2**30
    )
    requests = []

    @contextlib.contextmanager
    def opener(row, offset):
        requests.append((row["path"], offset))
        body = bodies[row["path"]]
        headers = {"Content-Length": str(len(body) - offset)}
        if offset:
            headers["Content-Range"] = f"bytes {offset}-{len(body) - 1}/{len(body)}"
        yield source_resume.SourceResponse(
            io.BytesIO(body[offset:]), 206 if offset else 200, headers, revision, row["path"]
        )

    terminal = reconcile.terminal_receipt("978318|TIMEOUT|0:15\n", "", job_id="978318")
    namespace.update(
        {
            "D": successor,
            "records": [record],
            "inv": inv,
            "recon": reconcile,
            "revision": revision,
            "source_resume": source_resume,
            "range_opener": opener,
            "opener": None,
            "funding_gate": lambda: None,
            "uid_to_index": {uid: 0},
            "plans": {0: plan},
            "terminal": terminal,
            "old_helper": {},
            "orphan_proofs": {},
            "VERSION": cache.VERSION,
            "signature": cache.signature,
            "normalized": cache.normalized,
            "load": cache.load,
            "original_write": cache.write,
            "RawGroup": RawGroup,
            "assemble": assemble,
            "Path": Path,
            "zipfile": zipfile,
            "json": json,
            "gzip": gzip,
            "shutil": shutil,
            "stop": threading.Event(),
            "proof_lock": threading.Lock(),
            "control_committed": [0],
            "paid_raw": orphan.stat().st_size,
            "decoder": decoder,
            "v": {"conversion": conversion},
        }
    )
    wrapper = ast.parse((CONTROL / "program.py.txt").read_text())
    nodes = [
        n
        for n in wrapper.body
        if isinstance(n, ast.FunctionDef) and n.name in ("acquire", "output_path", "write")
    ]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "actual_successor_shims", "exec"),
        namespace,
    )
    namespace["helper"] = {"acquire": namespace["acquire"]}
    original_text = (
        ROOT / "initiative/research/predictive-full-raw-cache-20261020-v1/program.py.txt"
    ).read_text()
    one = next(
        n
        for n in ast.parse(original_text).body
        if isinstance(n, ast.FunctionDef) and n.name == "one"
    )
    one_text = ast.get_source_segment(original_text, one).replace(
        'p = D / "cache" / ("case-" + str(i).zfill(4) + ".private.npz")', "p = output_path(i)"
    )
    exec(compile(one_text, "actual_original_one_case", "exec"), namespace)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: types.SimpleNamespace(free=2**40))
    index, accepted = namespace["one"]((0, record))
    assert index == 0
    assert Path(accepted["path"]) == (
        orphan if intact_orphan else successor / "cache/case-0000.private.npz"
    )
    assert sorted(requests) == sorted([(rows[1]["path"], prefix), (rows[2]["path"], 0)])
    assert snapshots == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in snapshots}
    assert not list((successor / "scratch").iterdir())
    proof_path = successor / "proofs/case-0000.source-proof.private.json.gz"
    proof = json.loads(gzip.decompress(proof_path.read_bytes()))
    reconcile.source_proof_matches(proof, record)
    assert proof["fetched_bytes"] == rows[1]["bytes"] - prefix + rows[2]["bytes"]
