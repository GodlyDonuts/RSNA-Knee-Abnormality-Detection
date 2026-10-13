"""CPU-only owning-node public triplet cache wrapper: bindings and sanitized output only."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsna.competitive import public_cache_producer as pc
from rsna.competitive import public_input as pi

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_public_triplet_cache", ROOT / "scripts/newton/build_public_triplet_cache.py"
)
assert SPEC and SPEC.loader
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)
UID = "1.2.3"
FUNDING = dict(
    schema_version=1,
    node="evc31",
    max_total_bytes=10**9,
    min_free_bytes=1,
    max_study_source_bytes=100,
    max_studies_per_run=2,
)


def owning(monkeypatch, node="evc31", cuda=""):
    monkeypatch.setattr(wrapper.os, "uname", lambda: SimpleNamespace(nodename=node))
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    if cuda is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda)


def test_runtime_requires_owning_node_and_hidden_gpu(monkeypatch):
    owning(monkeypatch)
    runtime = wrapper.check_runtime()
    assert runtime["node"] == "evc31" and runtime["threads"] == 1
    for node, cuda in (("evc34", ""), ("evc31", None), ("evc31", "0")):
        owning(monkeypatch, node, cuda)
        with pytest.raises(ValueError):
            wrapper.check_runtime()
    owning(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    with pytest.raises(ValueError):
        wrapper.check_runtime()


def test_private_output_is_owner_only_outside_git_and_retained_packet(tmp_path, monkeypatch):
    out = wrapper.private_output(tmp_path / "private")
    assert out.stat().st_mode & 0o777 == 0o700
    for bad in (Path("relative"), ROOT / "artifacts" / "cache", wrapper.RETAINED_PACKET / "x"):
        with pytest.raises(ValueError):
            wrapper.private_output(bad)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(ValueError, match="owner-only"):
        wrapper.private_output(shared)
    link = tmp_path / "link"
    link.symlink_to(out)
    with pytest.raises(ValueError):
        wrapper.private_output(link)


def test_funding_is_bound_by_exact_bytes_and_fields(tmp_path):
    path = tmp_path / "funding.json"
    path.write_text(json.dumps(FUNDING))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert wrapper.load_funding(path, sha) == FUNDING
    with pytest.raises(ValueError, match="SHA256"):
        wrapper.load_funding(path, "0" * 64)
    for change in (
        dict(node="evc34"),
        dict(max_total_bytes=0),
        dict(extra=1),
        dict(min_free_bytes=1.5),
    ):
        path.write_text(json.dumps(FUNDING | change))
        with pytest.raises(ValueError, match="fields"):
            wrapper.load_funding(path, hashlib.sha256(path.read_bytes()).hexdigest())


def source_tree(tmp_path, payload=b"dicom-bytes"):
    source = tmp_path / "source"
    path = source / pc.SPLIT / UID / f"{UID}.1" / f"{UID}.1.1.dcm"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    record = {
        "path": path.relative_to(source).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "study_uid": UID,
        "series_uid": f"{UID}.1",
        "sop_uid": f"{UID}.1.1",
    }
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return source, scratch, path, record


def test_transport_copies_exact_bounded_original_bytes(tmp_path):
    source, scratch, path, record = source_tree(tmp_path)
    copied = wrapper.transport_study(source, scratch, UID, [record], FUNDING)
    assert (copied / f"{UID}.1" / f"{UID}.1.1.dcm").read_bytes() == path.read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        wrapper.transport_study(source, scratch, UID, [record], FUNDING)


@pytest.mark.parametrize("problem", ["over_cap", "grown", "hash", "symlink", "reserve"])
def test_transport_refuses_unbounded_or_drifted_source(tmp_path, monkeypatch, problem):
    source, scratch, path, record = source_tree(tmp_path)
    funding = dict(FUNDING)
    if problem == "over_cap":
        funding["max_study_source_bytes"] = record["bytes"] - 1
    elif problem == "grown":
        path.write_bytes(path.read_bytes() + b"x")
    elif problem == "hash":
        path.write_bytes(b"X" * record["bytes"])
    elif problem == "symlink":
        other = tmp_path / "elsewhere.dcm"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    else:
        funding["min_free_bytes"] = 10**18
    with pytest.raises(ValueError):
        wrapper.transport_study(source, scratch, UID, [record], funding)
    if problem in {"over_cap", "reserve"}:
        assert not any(scratch.iterdir())


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


def test_stage_mode_requires_every_binding(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--mode", "stage", "--private-out", "/x", "--funding", "f", "--funding-sha256", "0"],
    )
    with pytest.raises(SystemExit):
        wrapper.main()


def test_wrapper_never_dispatches_or_downloads():
    text = (ROOT / "scripts/newton/build_public_triplet_cache.py").read_text()
    for forbidden in ("sbatch", "subprocess", "requests", "urllib", "huggingface_hub", "hf_hub"):
        assert forbidden not in text
