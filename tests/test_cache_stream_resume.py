"""Exercise byte-boundary continuation and failure preservation on actual files."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / "scripts/newton" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sender = load("resume_cache_archive")
receiver = load("cache_resume_remote")


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr(receiver, "require_no_open_handles", lambda _: None)
    cache = tmp_path / "source"
    (cache / "train/study").mkdir(parents=True)
    contents = {
        "train/study/series.npz": bytes(range(256)) * 123,
        "train/study/series.json": b'{"v":3}',
    }
    approved = {}
    for name, data in contents.items():
        (cache / name).write_bytes(data)
        approved[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    # Recreate the original stream independently, including record padding.
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output, mode="w|", format=tarfile.USTAR_FORMAT, bufsize=1024**2, copybufsize=1024**2
    ) as tar:
        for name in ["cache", "cache/train", "cache/train/study"]:
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, 0o700
            tar.addfile(info)
        for name, data in sorted(contents.items()):
            info = tarfile.TarInfo("cache/" + name)
            info.size, info.mode = len(data), 0o600
            tar.addfile(info, io.BytesIO(data))
    return cache, approved, output.getvalue()


def make_plan(tmp_path, data, offset, approved):
    target = tmp_path / "full.tar"
    partial = Path(str(target) + ".partial")
    partial.write_bytes(data[:offset])
    actual = receiver.inspect(target, True)
    return (
        target,
        partial,
        {
            "operation_id": "synthetic-one-shot",
            "partial_state": actual["state"],
            "prefix_sha256": actual["sha256"],
            "archive_sha256": hashlib.sha256(data).hexdigest(),
            "archive_bytes": len(data),
            "file_members": len(approved),
            "unpacked_bytes": sum(r["bytes"] for r in approved.values()),
        },
    )


@pytest.mark.parametrize("offset", [0, 1, 511, 512, 513, 17001, 40959, 40960])
def test_exact_suffix_and_no_truncation(archive, tmp_path, offset):
    cache, approved, data = archive
    assert offset <= len(data)
    target, partial, plan = make_plan(tmp_path, data, offset, approved)
    suffix = io.BytesIO()
    result = sender.replay(
        cache, approved, sender.ReplaySink(offset, plan["prefix_sha256"], suffix)
    )
    assert result["archive_sha256"] == plan["archive_sha256"]
    assert suffix.getvalue() == data[offset:]
    inode = partial.stat().st_ino
    receipt = receiver.receive(target, plan, io.BytesIO(suffix.getvalue()))
    assert receipt["resumed_from_bytes"] == offset
    assert partial.read_bytes() == data
    assert partial.stat().st_ino == inode
    receiver.commit(target, plan)
    assert target.read_bytes() == data
    assert target.stat().st_ino == inode
    assert not partial.exists()
    with pytest.raises(ValueError, match="Existing final"):
        receiver.commit(target, plan)


def test_prefix_mismatch_forwards_nothing(archive):
    cache, approved, _ = archive
    output = io.BytesIO()
    with pytest.raises(ValueError, match="prefix differs"):
        sender.replay(cache, approved, sender.ReplaySink(513, "0" * 64, output))
    assert output.getvalue() == b""


def test_corrupted_accepted_source_rejected(archive):
    cache, approved, _ = archive
    p = cache / "train/study/series.json"
    p.write_bytes(b'{"v":4}')
    with pytest.raises(ValueError, match="accepted lineage"):
        sender.replay(cache, approved, sender.ReplaySink(0))


@pytest.mark.parametrize("failure", ["identity", "prefix", "short", "extra", "digest", "inventory"])
def test_failed_receive_preserves_partial_and_claim(archive, tmp_path, failure):
    _, approved, data = archive
    target, partial, plan = make_plan(tmp_path, data, 513, approved)
    suffix = data[513:]
    if failure == "identity":
        plan["partial_state"]["st_ino"] += 1
    elif failure == "prefix":
        plan["prefix_sha256"] = "0" * 64
    elif failure == "short":
        suffix = suffix[:-1]
    elif failure == "extra":
        suffix += b"x"
    elif failure == "digest":
        plan["archive_sha256"] = "0" * 64
    elif failure == "inventory":
        plan["file_members"] += 1
    with pytest.raises(ValueError):
        receiver.receive(target, plan, io.BytesIO(suffix))
    assert partial.read_bytes().startswith(data[:513])
    assert not target.exists()
    assert Path(str(partial) + ".resume-lock/intent.json").is_file()
    with pytest.raises(ValueError, match="lock"):
        receiver.inspect(target)


def test_verified_partial_mutation_cannot_publish(archive, tmp_path):
    _, approved, data = archive
    target, partial, plan = make_plan(tmp_path, data, 513, approved)
    receiver.receive(target, plan, io.BytesIO(data[513:]))
    with partial.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="changed"):
        receiver.commit(target, plan)
    assert not target.exists()


def test_symlink_partial_cannot_be_opened(archive, tmp_path):
    _, approved, data = archive
    target, partial, _ = make_plan(tmp_path, data, 513, approved)
    real = tmp_path / "saved"
    partial.rename(real)
    partial.symlink_to(real)
    with pytest.raises(OSError):
        receiver.inspect(target)
    assert real.read_bytes() == data[:513]


def test_lineage_uses_only_relative_identities(tmp_path):
    path = tmp_path / "lineage.jsonl"
    row = {
        "split": "train",
        "study_uid": "1",
        "series_uid": "2",
        "cache_bytes": 5,
        "cache_sha256": "a" * 64,
        "new_metadata": {"bytes": 7, "sha256": "b" * 64, "path": "/private/never-copy-this"},
    }
    path.write_text(json.dumps(row))
    assert set(sender.approved_files(path)) == {"train/1/2.npz", "train/1/2.json"}
    row["study_uid"] = ".."
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Unsafe"):
        sender.approved_files(path)


@pytest.mark.parametrize(
    "name,allowed", [("sshd", True), ("(sd-pam)", True), ("python3.12", False)]
)
def test_only_privileged_session_helpers_excluded(tmp_path, monkeypatch, name, allowed):
    proc = tmp_path / "proc"
    process = proc / "12345"
    (process / "fd").mkdir(parents=True)
    (process / "comm").write_text(name + "\n")
    original = Path.stat

    def fake_stat(path, *args, **kwargs):
        if path == process:
            return SimpleNamespace(st_uid=123)
        if path == process / "fd":
            return SimpleNamespace(st_uid=0)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    monkeypatch.setattr(receiver.os, "getuid", lambda: 123)
    if allowed:
        assert receiver.require_no_open_handles(tmp_path / "partial", proc) == [
            {"pid": 12345, "name": name}
        ]
    else:
        with pytest.raises(ValueError, match="uninspectable"):
            receiver.require_no_open_handles(tmp_path / "partial", proc)


def test_another_open_receiver_blocks(tmp_path):
    proc = tmp_path / "proc"
    fd = proc / "12345/fd"
    fd.mkdir(parents=True)
    partial = tmp_path / "partial"
    partial.write_bytes(b"preserve")
    (fd / "4").symlink_to(partial)
    with pytest.raises(ValueError, match="still has the partial open"):
        receiver.require_no_open_handles(partial, proc)
