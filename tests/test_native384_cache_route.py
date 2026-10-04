"""Bounded synthetic transport cases, not a full native-MRI copy proof."""

import copy
import hashlib
import io
import tarfile

import pytest

from scripts.newton.route_native384_cache import extract_exact, validate_cpu_fields


def payload(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content, kind in entries:
            item = tarfile.TarInfo(name)
            item.size = len(content)
            item.type = kind
            archive.addfile(item, io.BytesIO(content))
    stream.seek(0)
    return stream


def expected():
    return {"cache/a.npz": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}


def test_exact_private_copy(tmp_path):
    root = tmp_path / "owned"
    result = extract_exact(payload([("cache/a.npz", b"abc", tarfile.REGTYPE)]), root, expected())
    assert result == {"members": 1, "bytes": 3}
    assert (root / "cache/a.npz").read_bytes() == b"abc"
    assert (root / "cache/a.npz").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "kind",
    [
        "traversal",
        "absolute",
        "extra",
        "duplicate",
        "missing",
        "size",
        "hash",
        "symlink",
        "directory",
        "existing",
        "appended",
    ],
)
def test_reject_unaccepted_copy(tmp_path, kind):
    entries = [("cache/a.npz", b"abc", tarfile.REGTYPE)]
    root = tmp_path / "owned"
    if kind == "traversal":
        entries[0] = ("cache/../a.npz", b"abc", tarfile.REGTYPE)
    elif kind == "absolute":
        entries[0] = ("/cache/a.npz", b"abc", tarfile.REGTYPE)
    elif kind == "extra":
        entries.append(("cache/b.npz", b"abc", tarfile.REGTYPE))
    elif kind == "duplicate":
        entries *= 2
    elif kind == "missing":
        entries = []
    elif kind == "size":
        entries[0] = ("cache/a.npz", b"abcd", tarfile.REGTYPE)
    elif kind == "hash":
        entries[0] = ("cache/a.npz", b"abd", tarfile.REGTYPE)
    elif kind == "symlink":
        entries[0] = ("cache/a.npz", b"", tarfile.SYMTYPE)
    elif kind == "directory":
        entries[0] = ("cache/a.npz", b"", tarfile.DIRTYPE)
    elif kind == "existing":
        root.mkdir()
    stream = payload(entries)
    if kind == "appended":
        stream.seek(0, io.SEEK_END)
        stream.write(b"bad")
        stream.seek(0)
    with pytest.raises((ValueError, FileExistsError)):
        extract_exact(stream, root, expected())


@pytest.mark.parametrize("allocated", ["2", "4"])
def test_exact_cpu_request_accepts_scheduler_hardware_threads(allocated):
    validate_cpu_fields(
        dict(
            NumCPUs=allocated,
            NumTasks="2",
            NumNodes="2",
            **{
                "CPUs/Task": "1",
                "ReqTRES": "cpu=2,mem=4G,node=2",
                "AllocTRES": "cpu=" + allocated + ",mem=4G,node=2",
            },
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("NumCPUs", "8"),
        ("NumTasks", "4"),
        ("NumNodes", "1"),
        ("CPUs/Task", "2"),
        ("ReqTRES", "cpu=4,mem=4G,node=2"),
        ("AllocTRES", "cpu=4,gres/gpu=1"),
    ],
)
def test_reject_changed_cpu_or_gpu_allocation(field, value):
    fields = dict(
        NumCPUs="4",
        NumTasks="2",
        NumNodes="2",
        **{"CPUs/Task": "1", "ReqTRES": "cpu=2,mem=4G,node=2", "AllocTRES": "cpu=4,mem=4G,node=2"},
    )
    fields = copy.deepcopy(fields)
    fields[field] = value
    with pytest.raises(ValueError):
        validate_cpu_fields(fields)
