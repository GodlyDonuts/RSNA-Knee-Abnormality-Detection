"""Optional node-private release admission cache over the synthetic release fixture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from test_release_adapter import prepared  # noqa: F401

from rsna import release
from rsna import release_cache as rc


@pytest.fixture
def cached(prepared, tmp_path_factory):  # noqa: F811
    f, _, audit, studies, out = prepared
    binding = release.write_verified_release(out, studies, audit, f.root)
    cache = tmp_path_factory.mktemp("node-private") / "admission"
    summary = rc.produce(out / "studies.parquet", f.root, cache)
    return f.root, out / "studies.parquet", cache, binding, summary


def _manifest(cache: Path) -> dict:
    return json.loads((cache / rc.MANIFEST_NAME).read_text())


def _rewrite(cache: Path, record: dict) -> None:
    path = cache / rc.MANIFEST_NAME
    path.write_text(json.dumps(record, sort_keys=True))
    path.chmod(0o600)


def _assert_identical(first: pd.DataFrame, second: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert list(first.columns) == list(second.columns)
    assert list(first.dtypes) == list(second.dtypes)
    assert first.index.equals(second.index) and first.equals(second)


def test_cached_admission_is_exactly_the_original(cached) -> None:
    root, studies, cache, binding, summary = cached
    original, original_binding = release.load_verified_release(studies, root)
    frame, cached_binding = rc.consume(studies, root, cache)
    _assert_identical(original, frame)
    assert cached_binding == original_binding == binding
    record = _manifest(cache)
    assert record["status"] == "engineering_admission_receipt_not_scientific_evidence"
    assert {"release/studies", "release/receipt"} <= set(record["members"])
    assert summary["members"] == len(record["members"]) and summary["listings"] >= 1
    # Every file the original replay bound through Bindings is a cache member.
    assert "root/" + str(release.SOURCE_PATH) in record["members"]
    assert "root/" + str(release.PROVENANCE_PATH) in record["members"]
    assert (
        record["member_bytes"]
        == sum(
            (root / name[5:]).stat().st_size if name.startswith("root/") else 0
            for name in record["members"]
        )
        + studies.stat().st_size
        + (studies.parent / release.RECEIPT_NAME).stat().st_size
    )
    assert os.stat(cache).st_mode & 0o777 == 0o700
    assert {p.name: p.stat().st_mode & 0o777 for p in cache.iterdir()} == {
        rc.MANIFEST_NAME: 0o600,
        rc.FRAME_NAME: 0o600,
    }


def test_cold_import_and_admission_do_not_touch_global_rng(cached) -> None:
    root, studies, cache, _, _ = cached
    script = (
        "import random, sys\n"
        "import numpy as np\n"
        "random.seed(7); np.random.seed(7)\n"
        "before = (random.getstate(), np.random.get_state()[1].tobytes())\n"
        "from pathlib import Path\n"
        "from rsna import release_cache\n"
        "release_cache.consume(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))\n"
        "after = (random.getstate(), np.random.get_state()[1].tobytes())\n"
        "assert before == after, 'rng changed'\n"
        "print(random.random(), np.random.random())\n"
    )
    package = str(Path(rc.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": package}
    outputs = {
        subprocess.run(
            [sys.executable, "-c", script, str(studies), str(root), str(cache)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for _ in range(2)
    }
    expected = subprocess.run(
        [
            sys.executable,
            "-c",
            "import random; import numpy as np; random.seed(7); np.random.seed(7);"
            " print(random.random(), np.random.random())",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert outputs == {expected}


def _tamper_content(root: Path, studies: Path, cache: Path) -> None:
    path = root / release.SOURCE_PATH
    stat = path.stat()
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(bytes(data))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    # Even with a matching fresh stat signature, the original content hash refuses.
    record = _manifest(cache)
    fresh = path.stat()
    record["members"]["root/" + str(release.SOURCE_PATH)]["stat"] = [
        fresh.st_dev,
        fresh.st_ino,
        fresh.st_mode,
        fresh.st_size,
        fresh.st_mtime_ns,
        fresh.st_ctime_ns,
    ]
    _rewrite(cache, record)


def _tamper_stat(root: Path, studies: Path, cache: Path) -> None:
    path = root / release.PROVENANCE_PATH
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))


def _tamper_provenance(root: Path, studies: Path, cache: Path) -> None:
    path = root / release.PROVENANCE_PATH
    path.write_text(path.read_text().replace("Synthetic", "Synthetik", 1))


def _missing_member(root: Path, studies: Path, cache: Path) -> None:
    (root / release.SOURCE_PATH).unlink()


def _extra_member(root: Path, studies: Path, cache: Path) -> None:
    record = _manifest(cache)
    listed = next(iter(record["listings"]))
    (root / listed[5:] / "batch-999-extra.json").write_text("[]")


def _receipt_changed(root: Path, studies: Path, cache: Path) -> None:
    receipt = studies.parent / release.RECEIPT_NAME
    receipt.write_bytes(receipt.read_bytes() + b"\n")


def _frame_changed(root: Path, studies: Path, cache: Path) -> None:
    frame = pd.read_parquet(cache / rc.FRAME_NAME)
    frame.loc[0, "study_uid"] = "z"
    frame.to_parquet(cache / rc.FRAME_NAME, index=False)


def _frame_dtype_changed(root: Path, studies: Path, cache: Path) -> None:
    record = _manifest(cache)
    record["frame"]["dtypes"][-1] = "float16"
    _rewrite(cache, record)


def _roster_changed(root: Path, studies: Path, cache: Path) -> None:
    record = _manifest(cache)
    record["frame"]["roster_sha256"] = "0" * 64
    _rewrite(cache, record)


def _manifest_member_dropped(root: Path, studies: Path, cache: Path) -> None:
    record = _manifest(cache)
    record["members"].pop("root/" + str(release.SOURCE_PATH))
    _rewrite(cache, record)


def _manifest_member_added(root: Path, studies: Path, cache: Path) -> None:
    record = _manifest(cache)
    record["members"]["root/../outside"] = {"sha256": "0" * 64, "stat": [0] * 6}
    _rewrite(cache, record)


def _binding_changed(root: Path, studies: Path, cache: Path) -> None:
    record = _manifest(cache)
    record["binding"]["source_revision"] = "other"
    _rewrite(cache, record)


def _stale_code(root: Path, studies: Path, cache: Path) -> None:
    record = _manifest(cache)
    record["code"]["rsna/release.py"] = "0" * 64
    _rewrite(cache, record)


def _stale_runtime(root: Path, studies: Path, cache: Path) -> None:
    record = _manifest(cache)
    record["runtime"]["pandas"] = "0.0"
    _rewrite(cache, record)


def _stale_schema(root: Path, studies: Path, cache: Path) -> None:
    record = _manifest(cache)
    record["schema"] = "release_admission_cache_v0"
    _rewrite(cache, record)


def _unsafe_member_link(root: Path, studies: Path, cache: Path) -> None:
    path = root / release.SOURCE_PATH
    moved = path.with_name("moved.csv")
    path.rename(moved)
    path.symlink_to(moved)


def _unsafe_directory_link(root: Path, studies: Path, cache: Path) -> None:
    path = (root / release.PROVENANCE_PATH).parent
    moved = path.with_name(path.name + "-moved")
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=True)


def _unsafe_cache_link(root: Path, studies: Path, cache: Path) -> None:
    frame = cache / rc.FRAME_NAME
    moved = cache.parent / "frame-elsewhere"
    frame.rename(moved)
    frame.symlink_to(moved)


def _cache_dir_mode(root: Path, studies: Path, cache: Path) -> None:
    cache.chmod(0o755)


def _cache_file_mode(root: Path, studies: Path, cache: Path) -> None:
    (cache / rc.FRAME_NAME).chmod(0o644)


def _cache_extra_file(root: Path, studies: Path, cache: Path) -> None:
    extra = cache / "notes.txt"
    extra.write_text("x")
    extra.chmod(0o600)


def _cache_byte_cap(root: Path, studies: Path, cache: Path) -> None:
    rc.MAX_FRAME_BYTES = 16


def _source_byte_cap(root: Path, studies: Path, cache: Path) -> None:
    rc.MAX_MEMBER_BYTES = 64


TAMPERS = [
    _tamper_content,
    _tamper_stat,
    _tamper_provenance,
    _missing_member,
    _extra_member,
    _receipt_changed,
    _frame_changed,
    _frame_dtype_changed,
    _roster_changed,
    _manifest_member_dropped,
    _manifest_member_added,
    _binding_changed,
    _stale_code,
    _stale_runtime,
    _stale_schema,
    _unsafe_member_link,
    _unsafe_directory_link,
    _unsafe_cache_link,
    _cache_dir_mode,
    _cache_file_mode,
    _cache_extra_file,
    _cache_byte_cap,
    _source_byte_cap,
]


@pytest.mark.parametrize("tamper", TAMPERS, ids=lambda f: f.__name__.strip("_"))
def test_consumer_refuses_and_never_silently_falls_back(cached, monkeypatch, tamper) -> None:
    root, studies, cache, _, _ = cached
    monkeypatch.setattr(rc, "MAX_FRAME_BYTES", rc.MAX_FRAME_BYTES)
    monkeypatch.setattr(rc, "MAX_MEMBER_BYTES", rc.MAX_MEMBER_BYTES)
    calls = []
    original = release.load_verified_release
    monkeypatch.setattr(
        release, "load_verified_release", lambda *a: calls.append(a) or original(*a)
    )
    tamper(root, studies, cache)
    with pytest.raises(rc.AdmissionCacheRefused):
        rc.consume(studies, root, cache)
    with pytest.raises(rc.AdmissionCacheRefused):
        rc.admit_release(studies, root, cache)
    assert calls == []


def test_explicit_fallback_announces_and_runs_full_original_admission(cached, capsys) -> None:
    root, studies, cache, binding, _ = cached
    _stale_code(root, studies, cache)
    frame, fallback_binding = rc.admit_release(studies, root, cache, fallback=True)
    assert "cache_refused_running_full_original_admission" in capsys.readouterr().err
    original, _ = release.load_verified_release(studies, root)
    _assert_identical(original, frame)
    assert fallback_binding == binding


def test_fallback_does_not_hide_a_failed_original_admission(cached) -> None:
    root, studies, cache, _, _ = cached
    _tamper_provenance(root, studies, cache)
    with pytest.raises(ValueError) as error:
        rc.admit_release(studies, root, cache, fallback=True)
    assert not isinstance(error.value, rc.AdmissionCacheRefused)


def test_consumer_refuses_a_different_requested_release_tree(cached, tmp_path_factory) -> None:
    root, studies, cache, _, _ = cached
    other = tmp_path_factory.mktemp("other-root")
    with pytest.raises(rc.AdmissionCacheRefused, match="release/source tree differs"):
        rc.consume(studies, other, cache)


def test_producer_refuses_unsafe_or_shared_locations(cached, tmp_path_factory) -> None:
    root, studies, cache, _, _ = cached
    for location in (cache, root / "cache", studies.parent / "cache"):
        with pytest.raises(rc.AdmissionCacheRefused):
            rc.produce(studies, root, location)
    link = tmp_path_factory.mktemp("links") / "studies.parquet"
    link.symlink_to(studies)
    with pytest.raises(rc.AdmissionCacheRefused):
        rc.produce(link, root, tmp_path_factory.mktemp("c") / "admission")


def test_producer_enforces_member_bounds(
    prepared,  # noqa: F811
    tmp_path_factory,
    monkeypatch,
) -> None:
    f, _, audit, studies, out = prepared
    release.write_verified_release(out, studies, audit, f.root)
    monkeypatch.setattr(rc, "MAX_MEMBERS", 3)
    target = tmp_path_factory.mktemp("bounded") / "admission"
    with pytest.raises(rc.AdmissionCacheRefused, match="member bound"):
        rc.produce(out / "studies.parquet", f.root, target)
    assert not target.exists()
