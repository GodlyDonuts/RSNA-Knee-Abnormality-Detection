"""Read-only coordinator regressions; synthetic local files, no API calls."""

import hashlib
import importlib._bootstrap_external
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "scripts/kaggle" / filename
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


p = load("publication", "publication.py")
sys.modules["publication"] = p
c = load("epoch_coordinator_independent", "coordinator.py")


@pytest.fixture
def source(tmp_path):
    projection = tmp_path / "projection"
    path = projection / "scripts/kaggle/workflow.py"
    path.parent.mkdir(parents=True)
    path.write_text("VALUE = 'expected'\n")
    manifest = tmp_path / "projection.json"
    raw = path.read_bytes()
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    "scripts/kaggle/workflow.py": {
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                }
            }
        )
    )
    return projection, path, manifest, p.digest(manifest)


def test_pinned_clean_module_runs_twice_without_generated_code(source):
    root, path, manifest, digest = source
    for _ in range(2):
        module, _ = c.load_workflow(root, manifest, digest)
        assert module.VALUE == "expected"
    assert not Path(importlib.util.cache_from_source(str(path))).exists()


def test_actual_matching_timestamp_bytecode_attack_blocked(source):
    root, path, manifest, digest = source
    cache = Path(importlib.util.cache_from_source(str(path)))
    cache.parent.mkdir()
    code = compile("VALUE = 'unbound_cached_code_executed'\n", str(path), "exec")
    cache.write_bytes(
        importlib._bootstrap_external._code_to_timestamp_pyc(
            code, int(path.stat().st_mtime), path.stat().st_size
        )
    )
    with pytest.raises(ValueError, match="extra/missing files"):
        c.load_workflow(root, manifest, digest)


@pytest.mark.parametrize("name", ["workflow.so", "unbound.txt"])
def test_any_unbound_file_fails(source, name):
    root, path, manifest, digest = source
    path.with_name(name).write_bytes(b"unbound")
    with pytest.raises(ValueError, match="extra/missing files"):
        c.load_workflow(root, manifest, digest)


def test_linked_parent_with_exact_source_bytes_fails(source, tmp_path):
    root, path, manifest, digest = source
    original = root / "scripts"
    moved = tmp_path / "elsewhere"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="links"):
        c.load_workflow(root, manifest, digest)


def test_changed_manifest_fails_before_import(source):
    root, path, manifest, digest = source
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest differs"):
        c.load_workflow(root, manifest, digest)


def test_changed_source_fails_before_import(source):
    root, path, manifest, digest = source
    path.write_text("VALUE = 'changed!'\n")
    with pytest.raises(ValueError, match="projection changed"):
        c.load_workflow(root, manifest, digest)


def test_second_coordinator_cannot_remove_or_reuse_first_claim(tmp_path):
    with c.single_coordinator(tmp_path):
        claim = tmp_path / "coordinator-claim"
        before = (claim / "owner.json").read_bytes()
        with pytest.raises(FileExistsError):
            with c.single_coordinator(tmp_path):
                pytest.fail("Second coordinator entered protected scope")
        assert (claim / "owner.json").read_bytes() == before
    assert not claim.exists()


def test_corrected_publication_uses_distinct_verified_journal(tmp_path):
    sha = "a" * 64
    original = tmp_path / "dataset-publications" / sha
    original.mkdir(parents=True)
    (original / "verified.json").write_text(
        json.dumps({"status": "all_payload_bytes_verified", "slug": "user/old"})
    )
    meta = {"checkpoint_sha256": sha}
    operation = {"manifest_sha256": "b" * 64, "dataset_slug": "user/corrected"}
    assert c.publication_journal(tmp_path, meta, operation, False) == original
    assert c.publication_journal(tmp_path, meta, operation, True) == original.with_name(
        sha + "-" + "b" * 12
    )
    operation["dataset_slug"] = "user/old"
    with pytest.raises(ValueError, match="distinct Dataset"):
        c.publication_journal(tmp_path, meta, operation, True)


def test_corrected_push_preserves_unsubmitted_notebook_record():
    original = {
        "state": "pushed",
        "operation": {"kernel_slug": "user/old", "dataset_slug": "user/old-data"},
        "operation_sha256": "a" * 64,
        "kernel_version": 1,
        "dataset": {"id": 7},
        "kernel_absence_proof": {"count": 3},
    }
    saved = []
    workflow = SimpleNamespace(
        verify_account=lambda *_: None,
        read_entry=lambda *_: dict(original),
        check_kernel=lambda *_: {"status": "COMPLETE"},
        save_entry=lambda _db, entry, event: saved.append((dict(entry), event)),
    )
    db = SimpleNamespace(execute=lambda *_: None)
    operation = {
        "checkpoint_sha256": "c" * 64,
        "kernel_slug": "user/corrected",
        "dataset_slug": "user/corrected-data",
    }
    c.supersede_unsubmitted_kernel(workflow, None, db, operation, "a" * 64)
    entry, event = saved.pop()
    assert entry["state"] == "queued"
    assert "kernel_version" not in entry and "operation" not in entry
    assert entry["superseded_operations"] == [
        {
            "operation": original["operation"],
            "operation_sha256": "a" * 64,
            "kernel_version": 1,
            "dataset": {"id": 7},
            "state": "pushed_unsubmitted",
        }
    ]
    assert "corrected push" in event
    with pytest.raises(ValueError, match="exact unsubmitted"):
        c.supersede_unsubmitted_kernel(workflow, None, db, operation, "b" * 64)
    original["submission_id"] = 9
    with pytest.raises(ValueError, match="exact unsubmitted"):
        c.supersede_unsubmitted_kernel(workflow, None, db, operation, "a" * 64)


@pytest.fixture
def clean_cli_projection(tmp_path):
    import os
    import shutil

    root = tmp_path / "exact-projection"
    scripts = root / "scripts/kaggle"
    scripts.mkdir(parents=True)
    current = Path(__file__).resolve().parents[1] / "scripts/kaggle"
    for name in ("coordinator.py", "publication.py"):
        shutil.copyfile(current / name, scripts / name)
    (scripts / "workflow.py").write_text("VALUE = 'accepted'\n")
    manifest = tmp_path / "projection.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    str(path.relative_to(root)): {
                        "sha256": p.digest(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in scripts.iterdir()
                }
            }
        )
    )
    env = dict(os.environ)
    for name in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX", "PYTHONPATH"):
        env.pop(name, None)
    return root, scripts, manifest, p.digest(manifest), env


def clean_projection_process(fixture, mode):
    import subprocess

    root, scripts, manifest, digest, env = fixture
    if mode == "cli":
        args = [sys.executable, str(scripts / "coordinator.py"), "--help"]
    else:
        source = """
import json, runpy, sys
from pathlib import Path
assert sys.dont_write_bytecode is False
root, manifest, digest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(root / 'scripts/kaggle'))
coordinator = runpy.run_path(str(root / 'scripts/kaggle/coordinator.py'))
workflow, _ = coordinator['load_workflow'](root, manifest, digest)
assert workflow.VALUE == 'accepted'
print(json.dumps({'status': 'clean_projection_loaded'}))
"""
        args = [sys.executable, "-c", source, str(root), str(manifest), digest]
    return subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("mode", ["cli", "load_workflow"])
def test_actual_clean_subprocess_creates_no_import_cache(clean_cli_projection, mode):
    root, _, _, _, _ = clean_cli_projection
    result = clean_projection_process(clean_cli_projection, mode)
    assert result.returncode == 0, result.stderr
    assert not list(root.rglob("*.pyc"))
    assert not list(root.rglob("__pycache__"))


def test_clean_subprocess_still_rejects_existing_cache(clean_cli_projection):
    _, scripts, _, _, _ = clean_cli_projection
    cache = scripts / "__pycache__/unlisted.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"pre-existing unbound cache")
    result = clean_projection_process(clean_cli_projection, "load_workflow")
    assert result.returncode != 0
    assert "extra/missing files" in result.stderr
    assert cache.read_bytes() == b"pre-existing unbound cache"


def test_clean_subprocess_rejects_cached_helper_without_using_it(clean_cli_projection):
    _, scripts, _, _, _ = clean_cli_projection
    helper = scripts / "publication.py"
    cache = Path(importlib.util.cache_from_source(str(helper)))
    cache.parent.mkdir()
    # Harmless alternate helper code has a valid timestamp/size header. A normal
    # import would use it and fail before the intended exact-membership guard.
    alternate = compile("raise RuntimeError('cached_helper_was_used')\n", str(helper), "exec")
    cache.write_bytes(
        importlib._bootstrap_external._code_to_timestamp_pyc(
            alternate, int(helper.stat().st_mtime), helper.stat().st_size
        )
    )
    result = clean_projection_process(clean_cli_projection, "load_workflow")
    assert result.returncode != 0
    assert "extra/missing files" in result.stderr
    assert "cached_helper_was_used" not in result.stderr
