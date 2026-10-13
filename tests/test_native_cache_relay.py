import hashlib
import io
import json

import pytest

from scripts.newton import native_cache_relay as relay


def fixture(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    members = {}
    for i, value in enumerate((b"original pixels", b"original metadata")):
        name = f"cache/train_series/study/series{i}.npz"
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        members[name] = dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
    proof = dict(
        registry_file_sha256="a" * 64, signatures={n: relay.signature(source / n) for n in members}
    )
    return source, target, members, proof


def stream(source, members, proof):
    b = io.BytesIO()
    relay.source_stream(b, source, members, proof, "a" * 64)
    return b.getvalue()


def test_full_stream_checks_original_bytes_and_never_overwrites(tmp_path):
    source, target, members, proof = fixture(tmp_path)
    raw = stream(source, members, proof)
    result = relay.target_stream(
        io.BytesIO(raw), target, members, "a" * 64, byte_cap=1000, remaining_funding=1
    )
    assert result["all_original_member_size_SHA_verified_during_transfer"]
    assert all((source / n).read_bytes() == (target / n).read_bytes() for n in members)
    with pytest.raises(ValueError, match="existing transfer"):
        relay.target_stream(
            io.BytesIO(raw), target, members, "a" * 64, byte_cap=1000, remaining_funding=1
        )


@pytest.mark.parametrize(
    "edit", [lambda b: b[:-2], lambda b: b + b"extra", lambda b: b[:59] + b"X" + b[60:]]
)
def test_bad_stream_refused_without_verification(tmp_path, edit):
    source, target, members, proof = fixture(tmp_path)
    with pytest.raises(ValueError):
        relay.target_stream(
            io.BytesIO(edit(stream(source, members, proof))),
            target,
            members,
            "a" * 64,
            byte_cap=1000,
            remaining_funding=1,
        )


def test_changed_source_refused_before_any_pixel_stream(tmp_path):
    source, target, members, proof = fixture(tmp_path)
    (source / next(iter(members))).write_bytes(b"changed")
    b = io.BytesIO()
    with pytest.raises(ValueError, match="signature"):
        relay.source_stream(b, source, members, proof, "a" * 64)
    assert not b.getvalue()


def test_byte_cap_refused_before_target_files(tmp_path):
    source, target, members, proof = fixture(tmp_path)
    with pytest.raises(ValueError, match="cap"):
        relay.target_stream(
            io.BytesIO(stream(source, members, proof)),
            target,
            members,
            "a" * 64,
            byte_cap=1,
            remaining_funding=1,
        )
    assert not list(target.iterdir())


def test_registry_refuses_unsafe_paths_and_wrong_original_hash(tmp_path):
    p = tmp_path / "registry.json"
    p.write_text(
        json.dumps(
            {"shards": [{"members": {"cache/../escaped": {"bytes": 1, "sha256": "a" * 64}}}]}
        )
    )
    with pytest.raises(ValueError, match="Invalid"):
        relay.members_from_registry(p, relay.sha(p), 1)
    with pytest.raises(ValueError, match="registry bytes"):
        relay.members_from_registry(p, "b" * 64, 1)
