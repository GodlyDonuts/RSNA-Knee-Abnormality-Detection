"""Engineering tests for the private render capsule packager (no Kaggle/HF/Slurm, no pixels)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/kaggle/package_render_capsule.py"
SPEC = importlib.util.spec_from_file_location("package_render_capsule", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cap = importlib.util.module_from_spec(SPEC)
sys.modules["package_render_capsule"] = cap
SPEC.loader.exec_module(cap)

LIMITS = {"max_bytes": 1 << 20, "max_files": 10, "min_free_bytes": 1}
FILES = {"a/x.txt": b"x" * 10, "b.bin": bytes(range(256)) * 4, "a/c/empty": b""}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def synthetic(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "source"
    members = []
    for name, data in sorted(FILES.items()):
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(data)
        members.append(
            {"archive_path": f"code/{name}", "tree": "code", "path": name}
            | {"bytes": len(data), "sha256": sha(data), "git_blob": None}
        )
    members.sort(key=lambda m: m["archive_path"])
    manifest = {
        "kind": cap.KIND,
        "code_commit": "0" * 40,
        "members": members,
        "files": len(members),
        "bytes": sum(m["bytes"] for m in members),
        "render_arguments": {"script": "code/a/x.txt"},
        "verifier_sha256": sha(SCRIPT.read_bytes()),
    }
    return source, manifest


def packed(tmp_path: Path, name: str = "capsule") -> tuple[Path, dict, dict]:
    source, manifest = synthetic(tmp_path)
    out = tmp_path / name
    result = cap.pack(manifest, {"code": source}, out, LIMITS)
    return out, manifest, result


def sealed_for(out: Path, result: dict) -> dict:
    return cap.load_sealed(out / cap.MANIFEST, result["manifest_sha256"])


def test_pack_is_deterministic_and_exact(tmp_path):
    first, manifest, a = packed(tmp_path, "one")
    second, _, b = packed(tmp_path, "two")
    assert a == b
    assert (first / cap.ARCHIVE).read_bytes() == (second / cap.ARCHIVE).read_bytes()
    assert (first / cap.MANIFEST).read_bytes() == (second / cap.MANIFEST).read_bytes()
    assert a["bytes"] <= cap.estimate_archive_bytes(manifest)
    with tarfile.open(first / cap.ARCHIVE) as archive:
        infos = archive.getmembers()
    assert [i.name for i in infos] == [m["archive_path"] for m in manifest["members"]]
    assert all(i.isreg() and i.mtime == 0 and i.uid == 0 and i.mode == 0o444 for i in infos)
    assert oct(os.stat(first).st_mode & 0o777) == "0o700"


def test_verify_extracts_into_new_directory_and_rehashes(tmp_path):
    out, manifest, result = packed(tmp_path)
    sealed = sealed_for(out, result)
    target = tmp_path / "extract"
    report = cap.verify(out / cap.ARCHIVE, sealed, target, LIMITS, SCRIPT)
    assert report["status"] == "capsule_verified_before_any_read_or_render"
    assert report["render_arguments"]["script"] == str(target / "code/a/x.txt")
    for name, data in FILES.items():
        assert (target / "code" / name).read_bytes() == data
        assert not os.access(target / "code" / name, os.W_OK) or os.geteuid() == 0
    with pytest.raises(cap.CapsuleError, match="new"):
        cap.verify(out / cap.ARCHIVE, sealed, target, LIMITS, SCRIPT)
    assert cap.verify_tree(target, sealed)["files"] == len(FILES)


def test_verify_tree_rejects_extra_changed_or_linked_files(tmp_path):
    out, _, result = packed(tmp_path)
    sealed = sealed_for(out, result)
    target = tmp_path / "extract"
    cap.verify(out / cap.ARCHIVE, sealed, target, LIMITS, SCRIPT)
    (target / "code/extra").write_bytes(b"")
    with pytest.raises(cap.CapsuleError, match="differs"):
        cap.verify_tree(target, sealed)
    (target / "code/extra").unlink()
    (target / "code/a/link").symlink_to(target / "code/b.bin")
    with pytest.raises(cap.CapsuleError, match="link"):
        cap.verify_tree(target, sealed)
    (target / "code/a/link").unlink()
    path = target / "code/b.bin"
    path.chmod(0o600)
    path.write_bytes(b"y" * len(FILES["b.bin"]))
    with pytest.raises(cap.CapsuleError, match="Extracted differs"):
        cap.verify_tree(target, sealed)


def test_verify_rejects_tampered_archive_manifest_and_verifier(tmp_path):
    out, _, result = packed(tmp_path)
    with pytest.raises(cap.CapsuleError, match="Manifest SHA"):
        cap.load_sealed(out / cap.MANIFEST, "0" * 64)
    sealed = sealed_for(out, result)
    other = tmp_path / "other.py"
    other.write_bytes(b"pass\n")
    with pytest.raises(cap.CapsuleError, match="Verifier"):
        cap.verify(out / cap.ARCHIVE, sealed, tmp_path / "x", LIMITS, other)
    archive = out / cap.ARCHIVE
    data = bytearray(archive.read_bytes())
    data[600] ^= 1
    archive.chmod(0o600)
    archive.write_bytes(bytes(data))
    with pytest.raises(cap.CapsuleError, match="Archive SHA"):
        cap.verify(archive, sealed, tmp_path / "x", LIMITS, SCRIPT)
    assert not (tmp_path / "x").exists()


def hostile_archive(tmp_path: Path, info: tarfile.TarInfo, sealed: dict) -> tuple[Path, dict]:
    path = tmp_path / "hostile.tar"
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(info, io.BytesIO(b"") if info.isreg() else None)
    data = path.read_bytes()
    return path, sealed | {"archive": {"bytes": len(data), "sha256": sha(data)}}


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "escape", "foreign", "absolute"])
def test_archive_rejects_links_escape_and_foreign_members(tmp_path, kind):
    out, _, result = packed(tmp_path)
    sealed = sealed_for(out, result)
    info = tarfile.TarInfo("code/a/x.txt")
    if kind == "symlink":
        info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
    elif kind == "hardlink":
        info.type, info.linkname = tarfile.LNKTYPE, "code/b.bin"
    elif kind == "escape":
        info.name = "code/../../escape"
    elif kind == "foreign":
        info.name = "other/x.txt"
    else:
        info.name = "/code/x.txt"
    path, hostile = hostile_archive(tmp_path, info, sealed)
    with pytest.raises(cap.CapsuleError):
        cap.verify(path, hostile, tmp_path / "x", LIMITS, SCRIPT)
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize(
    "names",
    [
        ["code/a", "code/a"],
        ["code/A", "code/a"],
        ["code/a", "code/a/b"],
        ["code/../x"],
        ["code//x"],
        ["/code/x"],
        ["code/./x"],
        ["code\\x"],
        ["pixels/x"],
        [""],
    ],
)
def test_member_names_reject_duplicates_escapes_and_foreign_trees(names):
    with pytest.raises(cap.CapsuleError):
        cap.check_member_names(names)


def test_regular_file_rejects_links_and_escapes(tmp_path):
    root = tmp_path / "root"
    (root / "d").mkdir(parents=True)
    (root / "d/f").write_bytes(b"f")
    (tmp_path / "outside").write_bytes(b"o")
    (root / "file-link").symlink_to(root / "d/f")
    (root / "dir-link").symlink_to(root / "d")
    (root / "escape").symlink_to(tmp_path / "outside")
    assert cap.regular_file(root, "d/f") == root / "d/f"
    for name in ("file-link", "dir-link/f", "escape", "../outside", "d"):
        with pytest.raises((cap.CapsuleError, FileNotFoundError)):
            cap.regular_file(root, name)
    with pytest.raises(cap.CapsuleError, match="real directory"):
        cap.regular_file(root / "dir-link", "f")


@pytest.mark.parametrize(
    "values", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1), (True, 1, 1), (1.5, 1, 1)]
)
def test_bounds_must_be_explicit_positive_integers(values):
    with pytest.raises(cap.CapsuleError, match="bounds"):
        cap.bounds(*values)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"max_files": 2}, "file count"),
        ({"max_bytes": 4096}, "bytes"),
        ({"min_free_bytes": 1 << 62}, "free-space"),
    ],
)
def test_pack_enforces_root_bounds_before_writing(tmp_path, change, message):
    source, manifest = synthetic(tmp_path)
    with pytest.raises(cap.CapsuleError, match=message):
        cap.pack(manifest, {"code": source}, tmp_path / "out", LIMITS | change)
    assert not (tmp_path / "out").exists()


def test_verify_enforces_root_bounds(tmp_path):
    out, _, result = packed(tmp_path)
    sealed = sealed_for(out, result)
    for change in ({"max_files": 2}, {"max_bytes": 4096}, {"min_free_bytes": 1 << 62}):
        with pytest.raises(cap.CapsuleError):
            cap.verify(out / cap.ARCHIVE, sealed, tmp_path / "x", LIMITS | change, SCRIPT)
        assert not (tmp_path / "x").exists()


def test_pack_refuses_existing_output_inside_source_and_changed_source(tmp_path):
    source, manifest = synthetic(tmp_path)
    (tmp_path / "exists").mkdir()
    with pytest.raises(cap.CapsuleError, match="new absolute"):
        cap.pack(manifest, {"code": source}, tmp_path / "exists", LIMITS)
    with pytest.raises(cap.CapsuleError, match="inside a source"):
        cap.pack(manifest, {"code": source}, source / "out", LIMITS)
    (source / "b.bin").write_bytes(b"z" * len(FILES["b.bin"]))
    with pytest.raises(cap.CapsuleError, match="changed while packing"):
        cap.pack(manifest, {"code": source}, tmp_path / "out", LIMITS)


def test_reconcile_is_read_only_and_exact(tmp_path):
    out, manifest, result = packed(tmp_path)
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.iterdir()}
    report = cap.reconcile(manifest, out, result["manifest_sha256"])
    assert report["status"] == "reconciled_read_only" and report["sha256"] == result["sha256"]
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.iterdir()} == before
    with pytest.raises(cap.CapsuleError, match="fresh plan"):
        cap.reconcile(manifest | {"code_commit": "1" * 40}, out, result["manifest_sha256"])
    with pytest.raises(cap.CapsuleError, match="Manifest SHA"):
        cap.reconcile(manifest, out, "0" * 64)
    archive = out / cap.ARCHIVE
    archive.chmod(0o600)
    archive.write_bytes(archive.read_bytes()[:-1] + b"\1")
    with pytest.raises(cap.CapsuleError, match="Archive SHA"):
        cap.reconcile(manifest, out, result["manifest_sha256"])


def tracked(root: Path) -> set[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True)
    return set(out.stdout.decode().split("\0"))


def test_code_closure_covers_renderer_imports_and_is_tracked():
    names = cap.code_files(ROOT)
    for required in (
        cap.RENDERER,
        cap.WRAPPER,
        cap.SELF,
        "rsna/release.py",
        "rsna/competitive/folds.py",
        "rsna/competitive/train.py",
        "rsna/competitive/public_cache_producer.py",
        "tools/research/public_dinov2_input_reference/provenance.py",
    ):
        assert required in names
    assert all(name.endswith(".py") for name in names)
    assert set(names) <= tracked(ROOT)
    assert set(cap.CODE_DATA) <= tracked(ROOT)
    assert len(cap.inventory_files(ROOT)) == cap.INVENTORY_PARTS


def test_receipt_paths_are_the_original_trainer_inputs():
    relative = cap.receipt_relative(ROOT)
    assert relative == {
        "studies": "initiative/releases/report-supervision-v1/studies.parquet",
        "folds": "initiative/experiments/competitive-v1/folds",
        "series": "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv",
    }
    assert relative["series"] != "initiative/source/train_series.csv"
    with pytest.raises(cap.CapsuleError, match="receipt"):
        cap.trace(ROOT, ROOT, ROOT / relative["studies"], ROOT / relative["folds"], ROOT)


def test_trace_rejects_inputs_that_are_not_the_receipt(tmp_path):
    with pytest.raises(cap.CapsuleError, match="receipt"):
        cap.trace(ROOT, tmp_path, tmp_path / "s", tmp_path / "f", tmp_path / "c")


def test_plan_rejects_artifacts_that_are_not_the_declared_git_blob(tmp_path):
    if not set(cap.code_files(ROOT)) | set(cap.CODE_DATA) <= tracked(ROOT):
        pytest.skip("working tree has untracked capsule code")
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD"]).returncode:
        pytest.skip("working tree differs from HEAD; the plan binds exact committed blobs")
    relative = cap.receipt_relative(ROOT)
    foreign = tmp_path / "artifact" / relative["studies"]
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"not the release")
    closure = {
        "receipt_relative": relative,
        "files": [{"path": relative["studies"], "bytes": 15, "sha256": sha(b"not the release")}],
    }
    with pytest.raises(cap.CapsuleError, match="Git blob"):
        cap.plan(ROOT, tmp_path / "artifact", closure, "HEAD", tmp_path / "member.json")


def test_real_closure_pack_verify_and_replay_from_capsule_only(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD"]).returncode or not (
        set(cap.code_files(ROOT)) <= tracked(ROOT)
    ):
        pytest.skip("working tree differs from HEAD; the plan binds exact committed blobs")
    relative = cap.receipt_relative(ROOT)
    local = {"artifact-root": str(ROOT)} | {
        "studies": str(ROOT / relative["studies"]),
        "folds": str(ROOT / relative["folds"]),
        "series": str(ROOT / relative["series"]),
    }
    monkeypatch.setattr(cap, "receipt_inputs", lambda code_root: local)
    closure = cap.trace(
        ROOT, ROOT, Path(local["studies"]), Path(local["folds"]), Path(local["series"])
    )
    paths = [row["path"] for row in closure["files"]]
    assert paths == sorted(paths) and not any(p.endswith(".py") for p in paths)
    assert relative["series"] in paths and "initiative/source/train_series.csv" in paths
    assert not any(p.startswith(("artifacts/", "initiative/source/inventory/")) for p in paths)
    member = tmp_path / "inputs" / "member-manifest.json"
    member.parent.mkdir()
    member.write_bytes(b'{"synthetic": true}\n')
    monkeypatch.setattr(cap, "admitted_member_manifest_sha256", lambda root: sha(member.read_bytes()))
    manifest = cap.plan(ROOT, ROOT, closure, "HEAD", member)
    manifest["verifier_sha256"] = sha(SCRIPT.read_bytes())
    assert manifest["bindings"]["accepted_series_sha256"] == closure["release_series_sha256"]
    assert manifest["bytes"] < 512 * 1024 * 1024
    limits = cap.bounds(512 * 1024 * 1024, 4096, 1)
    roots = {cap.CODE: ROOT, cap.ARTIFACT: ROOT, cap.INPUTS: member.parent}
    result = cap.pack(manifest, roots, tmp_path / "capsule", limits)
    assert cap.reconcile(manifest, tmp_path / "capsule", result["manifest_sha256"])
    sealed = sealed_for(tmp_path / "capsule", result)
    target = tmp_path / "extract"
    report = cap.verify(tmp_path / "capsule" / cap.ARCHIVE, sealed, target, limits, SCRIPT)
    arguments = report["render_arguments"]
    replay = (
        "import json, sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]);"
        "from rsna.release import load_verified_release;"
        "from rsna.competitive.folds import load_folds;"
        "f, b = load_verified_release(Path(sys.argv[2]), Path(sys.argv[3]));"
        "o = load_folds(Path(sys.argv[4]), f, b);"
        "print(json.dumps([b['binding_sha256'], len(f), len(o)]))"
    )
    code = str(target / cap.CODE)
    out = subprocess.run(
        [sys.executable, "-c", replay, code, arguments["--studies"]]
        + [arguments["--artifact-root"], arguments["--folds"]],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )
    assert json.loads(out.stdout) == [
        closure["release_binding_sha256"],
        closure["release_studies"],
        closure["fold_rows"],
    ]
