import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("contrast_transfer_test", Path(__file__).with_name("transfer.py"))
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def test_binary_framing_has_exact_bounds_and_rejects_truncation():
    stream = io.BytesIO()
    header = {"bytes": 5, "kind": "test"}
    transfer.write_header(stream, header)
    stream.write(b"abcde")
    stream.seek(0)
    assert transfer.read_header(stream) == header
    assert transfer.read_exact(stream, 5) == b"abcde"
    assert stream.read(1) == b""
    with pytest.raises(AssertionError):
        transfer.read_exact(io.BytesIO(b"abc"), 5)
    with pytest.raises(AssertionError):
        transfer.read_exact(io.BytesIO(), transfer.CONTROL_CAP + 1)
    with pytest.raises(AssertionError):
        transfer.read_header(io.BytesIO(b"x" * 8192))


def test_every_old_optional_scope_is_kept_with_additive_cache_and_checkpoint_budget():
    quota = "/home/sa305415 1026233008 1048576000 1059061760 - 994621 1000000 1010000 -"
    out = transfer.capacity(800 * 2**30, quota)
    assert out["total_required_bytes"] == 394815078400 + 7 * 2**30 + 4 * 2**20
    assert out["no_prior_scope_reassigned"] and out["shared_files_after_old_liabilities"] == 5003
    with pytest.raises(AssertionError):
        transfer.capacity(out["total_required_bytes"], quota)


def test_original_index_needs_both_ten_coordinate_roles_and_unique_members():
    rows = [{"uid": str(i), "variants": {role: {"provenance": {"role": role, "metadata_width": 10}}
                                         for role in ("control", "normalized")}} for i in range(16)]
    assert len(list(transfer.roles(rows))) == 32
    rows[1]["uid"] = rows[0]["uid"]
    with pytest.raises(AssertionError):
        list(transfer.roles(rows))
