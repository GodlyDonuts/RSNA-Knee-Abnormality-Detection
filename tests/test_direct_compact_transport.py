import hashlib
import io
import socket
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.newton.direct_compact_transport import (
    confined,
    receive_file,
    receive_json,
    send_json,
    serve_file,
)


def transfer(source, target, record):
    a, b = socket.socketpair()

    def sender():
        with a, a.makefile("rb") as r, a.makefile("wb") as w:
            serve_file(r, w, source, record)

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(sender)
        with b, b.makefile("rb") as r, b.makefile("wb") as w:
            result = receive_file(r, w, target, record)
        future.result(timeout=3)
    return result


@pytest.mark.parametrize("offset", [0, 1, 999, 4096])
def test_exact_copy_or_prefix_reuse(tmp_path, offset):
    raw = bytes(range(256)) * 16
    source = tmp_path / "source"
    source.write_bytes(raw)
    target = tmp_path / "target"
    if offset:
        target.write_bytes(raw[:offset])
    result = transfer(
        source, target, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    )
    assert target.read_bytes() == raw
    assert result == {"copied_bytes": len(raw) - offset, "reused_bytes": offset}


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../b", "./file", "a//b", ""])
def test_reject_escape_or_alias(tmp_path, name):
    with pytest.raises(ValueError):
        confined(tmp_path, name)


def test_reject_symlink(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        confined(tmp_path, "link")


def test_bounded_json():
    stream = io.BytesIO()
    send_json(stream, {"count": 12})
    stream.seek(0)
    assert receive_json(stream) == {"count": 12}
    with pytest.raises(ValueError):
        receive_json(io.BytesIO(b"\xff" * 8))


def test_bad_existing_prefix_is_never_overwritten(tmp_path):
    raw = b"abcdef"
    source = tmp_path / "source"
    source.write_bytes(raw)
    target = tmp_path / "target"
    target.write_bytes(b"WRONG")
    with pytest.raises(ValueError):
        transfer(source, target, {"bytes": 6, "sha256": hashlib.sha256(raw).hexdigest()})
    assert target.read_bytes() == b"WRONG"


def test_empty_runtime_file_is_created_exclusively(tmp_path):
    source = tmp_path / "source"
    source.touch()
    target = tmp_path / "target"
    result = transfer(source, target, {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()})
    assert target.is_file() and result == {"copied_bytes": 0, "reused_bytes": 0}
