import pytest
from test_native_cache_relay import fixture

from scripts.newton import native_cache_relay as relay
from scripts.newton import resume_native_cache_relay as recovery


def prefix_fixture(tmp_path):
    source, target, members, proof = fixture(tmp_path)
    first = next(iter(members))
    path = target / first
    path.parent.mkdir(parents=True)
    path.write_bytes((source / first).read_bytes())
    return source, target, members, relay.reconcile_prefix(target, members)


def test_missing_suffix_preserves_complete_original_prefix(tmp_path):
    source, target, members, receipt = prefix_fixture(tmp_path)
    missing = recovery.suffix(members, receipt, 1)
    assert len(missing) == 1 and next(iter(members)) not in missing
    staging = tmp_path / "suffix"
    staging.mkdir()
    for name in missing:
        path = staging / name
        path.parent.mkdir(parents=True)
        path.write_bytes((source / name).read_bytes())
    prefix = target / next(iter(members))
    before = relay.signature(prefix)
    recovery.merge_suffix(target, staging, missing, receipt)
    assert relay.signature(prefix) == before
    assert all((source / n).read_bytes() == (target / n).read_bytes() for n in members)
    for name in missing:
        assert (target / name).stat().st_ino == (staging / name).stat().st_ino


def test_unrelated_partial_path_refused(tmp_path):
    source, target, members, receipt = prefix_fixture(tmp_path)
    receipt["unverified_last_member"] = "../outside"
    with pytest.raises(ValueError):
        recovery.suffix(members, receipt, 1)


def test_changed_prefix_refused_without_linking_new_members(tmp_path):
    source, target, members, receipt = prefix_fixture(tmp_path)
    (target / next(iter(members))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="prefix changed"):
        recovery.merge_suffix(
            target, tmp_path / "none", recovery.suffix(members, receipt, 1), receipt
        )
