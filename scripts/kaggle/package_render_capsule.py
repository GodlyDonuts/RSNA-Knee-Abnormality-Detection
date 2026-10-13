"""Private input capsule for the canonical Kaggle CPU public renderer (prepared, NOT run).

Root alone runs this, funds storage and dispatches. ``trace`` replays the existing
``load_verified_release``/``load_folds`` on the retained artifact root at the exact paths of
the original input receipt and records every artifact-root file they read (needs the repo
runtime, pandas/pyarrow; no torch). ``pack`` (stdlib) writes one deterministic uncompressed
PAX tar plus a manifest of path/size/SHA256/Git blob: the renderer's import closure and its
data inputs from the current Git commit (config, overlay, nine inventory parts, correction
plan/receipt, review, Pilkwang v15 source, input receipts), the traced release/fold/series
closure bound to a declared Git commit, and the root-held member manifest. ``reconcile``
recomputes the plan read-only and compares it with an existing capsule. ``verify`` (stdlib,
the Kaggle notebook's first step) checks the manifest/archive SHA, refuses links, path
escape, duplicates and foreign members, extracts into a new directory within root-supplied
byte/file/free-space bounds and re-hashes every file before anything is read or rendered.
No Kaggle/HF/Slurm call, no download, no pixel/cache/weight/credential is packaged.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
KIND = "rsna_public_render_capsule_v1"
CLOSURE_KIND = "rsna_public_render_release_closure_v1"
ARCHIVE, MANIFEST = "capsule.tar", "capsule-manifest.json"
CODE, ARTIFACT, INPUTS = "code", "artifact", "inputs"
MEMBER_MANIFEST = INPUTS + "/member-manifest.json"
RENDERER = "scripts/kaggle/render_public_source.py"
WRAPPER = "scripts/newton/build_public_triplet_cache.py"
SELF = "scripts/kaggle/package_render_capsule.py"
IMPORT_PACKAGES = ("rsna", "tools", "scripts")
INPUT_RECEIPT = "initiative/audits/public-label-source-inputs-v1/actual-CPU-accepted.json"
PLAN_ADMITTED = "initiative/audits/public-dinov2-real-inputs-20261013-v2/actual-plan-admitted.json"
OVERLAY = "initiative/experiments/competitive-v1/public-label-source-v1/fold0-overlay.json"
SOURCE_NOTEBOOK = "initiative/audits/public-dinov2-source-handoff-20261013-v1/source.notebook.json"
INVENTORY_DIR, INVENTORY_PARTS = "initiative/source/inventory", 9
CODE_DATA = (
    "initiative/experiments/competitive-v1/public-input-recipe-v1/public-triplet-config.json",
    OVERLAY,
    "initiative/audits/source-pixel-representation-v1/source_correction_plan.json",
    "initiative/audits/source-pixel-representation-v1/hf-source-correction-receipt.json",
    "tools/research/public_dinov2_input_reference/review.json",
    SOURCE_NOTEBOOK,
    INPUT_RECEIPT,
    PLAN_ADMITTED,
)
RECEIPT_KEYS = ("artifact-root", "studies", "folds", "series")
EXCLUDED = ("pixels", "caches", "model_weights", "credentials", "git_history", "other_datasets")


class CapsuleError(ValueError):
    pass


def _require(condition: object, message: str) -> None:
    if not condition:
        raise CapsuleError(message)


def canonical_sha256(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def safe_relative(name: str) -> PurePosixPath:
    """Plain relative POSIX path: no absolute, empty, dot, dot-dot, backslash or NUL part."""
    _require(isinstance(name, str) and name and "\\" not in name and "\0" not in name, name)
    path = PurePosixPath(name)
    _require(not path.is_absolute() and str(path) == name, f"Non-canonical path: {name!r}")
    _require(all(part not in {"", ".", ".."} for part in path.parts), f"Path escape: {name!r}")
    return path


def regular_file(root: Path, name: str) -> Path:
    """A regular file under ``root`` whose every component below ``root`` is a real entry."""
    relative = safe_relative(name)
    current = Path(root)
    _require(not current.is_symlink() and current.is_dir(), "Source root must be a real directory")
    for part in relative.parts:
        current = current / part
        _require(not current.is_symlink(), f"Source path traverses a link: {name}")
    _require(stat.S_ISREG(current.lstat().st_mode), f"Not a regular file: {name}")
    _require(current.resolve().is_relative_to(Path(root).resolve()), f"Path escape: {name}")
    return current


def file_digests(path: Path) -> tuple[int, str, str]:
    """Bytes, SHA256 and Git blob SHA1 of one file, read once."""
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return len(data), hashlib.sha256(data).hexdigest(), blob


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True
    ).stdout


def git_commit(root: Path, revision: str = "HEAD") -> str:
    return git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()


def git_tree(root: Path, commit: str) -> dict[str, str]:
    """Regular-file blobs of a commit (symlinks/submodules are never admissible members)."""
    blobs = {}
    for entry in git(root, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if entry:
            meta, _, name = entry.partition(b"\t")
            mode, kind, sha = meta.decode().split()
            if kind == "blob" and mode in {"100644", "100755"}:
                blobs[name.decode()] = sha
    return blobs


def code_files(root: Path = ROOT) -> list[str]:
    """Renderer CODE_FILES, the wrapper, this verifier and their static repo import closure."""
    tree = ast.parse((root / RENDERER).read_text())
    listed: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CODE_FILES" for t in node.targets
        ):
            listed = list(ast.literal_eval(node.value))
    _require(RENDERER in listed and WRAPPER in listed, "Renderer CODE_FILES were not found")
    seen: set[str] = set()
    todo = [*listed, SELF]

    def module_file(module: str) -> str | None:
        base = PurePosixPath(*module.split("."))
        for candidate in (f"{base}.py", f"{base}/__init__.py"):
            if (root / candidate).is_file():
                return candidate
        return None

    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        if not name.endswith(".py"):
            continue
        package = list(PurePosixPath(name).parent.parts)
        for node in ast.walk(ast.parse((root / name).read_text())):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package[: len(package) - node.level + 1] if node.level else []
                module = ".".join(base + ([node.module] if node.module else []))
                modules = [module] + [f"{module}.{alias.name}" for alias in node.names]
            for module in modules:
                parts = module.split(".")
                if parts[0] not in IMPORT_PACKAGES:
                    continue
                for end in range(1, len(parts) + 1):
                    found = module_file(".".join(parts[:end]))
                    if found and found not in seen:
                        todo.append(found)
    return sorted(seen)


def inventory_files(root: Path) -> list[str]:
    names = sorted(
        f"{INVENTORY_DIR}/{p.name}" for p in (root / INVENTORY_DIR).iterdir() if p.is_file()
    )
    _require(
        len(names) == INVENTORY_PARTS and all(n.endswith(".parquet") for n in names),
        "Exactly the nine original inventory parquet parts are required",
    )
    return names


def receipt_inputs(code_root: Path) -> dict[str, str]:
    receipt = json.loads(regular_file(code_root, INPUT_RECEIPT).read_bytes())
    inputs = receipt.get("trainer_inputs", {})
    _require(all(isinstance(inputs.get(k), str) for k in RECEIPT_KEYS), "Input receipt fields")
    return {k: inputs[k] for k in RECEIPT_KEYS}


def receipt_relative(code_root: Path) -> dict[str, str]:
    """Release/fold/series locations relative to the receipt's original artifact root."""
    inputs = receipt_inputs(code_root)
    root = PurePosixPath(inputs["artifact-root"])
    relative = {}
    for key in ("studies", "folds", "series"):
        path = PurePosixPath(inputs[key])
        _require(path.is_relative_to(root), f"Receipt {key} is outside its artifact root")
        relative[key] = str(safe_relative(str(path.relative_to(root))))
    return relative


# ---------------------------------------------------------------- trace (repo runtime)

_RECORDER: dict | None = None
_HOOKED: list[bool] = []


def _audit(event: str, args: tuple) -> None:
    if _RECORDER is None or event != "open" or isinstance(args[0], int):
        return
    try:
        path = Path(os.fsdecode(args[0])).resolve()
    except (TypeError, ValueError, OSError):
        return
    _RECORDER["reads"].add(path)


def trace(code_root: Path, artifact_root: Path, studies: Path, folds: Path, series: Path) -> dict:
    """Every artifact-root file read by the unchanged release/fold validators at receipt paths."""
    global _RECORDER
    expected = receipt_inputs(code_root)
    given = {"artifact-root": artifact_root, "studies": studies, "folds": folds, "series": series}
    _require(
        all(str(given[k]) == expected[k] for k in RECEIPT_KEYS),
        "Trace paths must equal the original input receipt's trainer inputs exactly",
    )
    from rsna.competitive.folds import load_folds
    from rsna.release import load_verified_release

    root = artifact_root.resolve()
    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED.append(True)
    _RECORDER = {"reads": set()}
    try:
        frame, binding = load_verified_release(studies, artifact_root)
        overlay = load_folds(folds, frame, binding)
        reads = set(_RECORDER["reads"])
    finally:
        _RECORDER = None
    names = set()
    code = code_root.resolve()
    for path in reads:
        if path.is_relative_to(root):
            relative = path.relative_to(root)
            if "__pycache__" not in relative.parts and path.is_file():
                names.add(relative.as_posix())
        elif path.is_relative_to(code) and path.suffix not in {".py", ".pyc"}:
            raise CapsuleError("Validators read repository data outside the artifact root")
    relative = receipt_relative(code_root)
    names |= {relative["studies"], relative["series"]}
    names |= {f"{relative['folds']}/{n}" for n in ("fold.csv", "folds.json")}
    files = []
    for name in sorted(names):
        size, sha, blob = file_digests(regular_file(artifact_root, name))
        files.append({"path": name, "bytes": size, "sha256": sha, "git_blob": blob})
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CLOSURE_KIND,
        "receipt_relative": relative,
        "release_binding_sha256": binding["binding_sha256"],
        "release_manifest_sha256": binding["manifest_sha256"],
        "release_series_sha256": binding["series_sha256"],
        "release_studies": len(frame),
        "fold_rows": len(overlay),
        "files": files,
        "labels_analyzed": False,
    }


# ---------------------------------------------------------------- plan / pack


def load_closure(path: Path, expected_sha256: str) -> dict:
    payload = Path(path).read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == expected_sha256, "Closure differs from SHA")
    closure = json.loads(payload)
    _require(
        closure.get("schema_version") == SCHEMA_VERSION and closure.get("kind") == CLOSURE_KIND,
        "Closure schema differs",
    )
    return closure


def plan(
    code_root: Path,
    artifact_root: Path,
    closure: dict,
    artifact_commit: str,
    member_manifest: Path,
) -> dict:
    """Exact deterministic member list and bindings; reads only, writes nothing."""
    code_commit = git_commit(code_root)
    code_tree = git_tree(code_root, code_commit)
    artifact_commit = git_commit(code_root, artifact_commit)
    artifact_tree = git_tree(code_root, artifact_commit)
    members: list[dict] = []

    def add(tree: str, root: Path, name: str, blobs: dict[str, str] | None) -> dict:
        size, sha, blob = file_digests(regular_file(root, name))
        if blobs is not None:
            _require(blobs.get(name) == blob, f"{tree} file is not its exact Git blob: {name}")
        row = {"archive_path": f"{tree}/{name}", "tree": tree, "path": name}
        row |= {"bytes": size, "sha256": sha, "git_blob": blob if blobs is not None else None}
        members.append(row)
        return row

    for name in sorted({*code_files(code_root), *CODE_DATA, *inventory_files(code_root)}):
        add(CODE, code_root, name, code_tree)
    relative = closure["receipt_relative"]
    _require(relative == receipt_relative(code_root), "Closure receipt paths differ")
    for row in closure["files"]:
        added = add(ARTIFACT, artifact_root, row["path"], artifact_tree)
        _require(
            (added["bytes"], added["sha256"]) == (row["bytes"], row["sha256"]),
            f"Artifact file changed since its trace: {row['path']}",
        )
    member_manifest = Path(member_manifest)
    size, sha, _ = file_digests(regular_file(member_manifest.parent, member_manifest.name))
    _require(
        sha == admitted_member_manifest_sha256(code_root),
        "Member manifest differs from the original admitted manifest",
    )
    members.append(
        {"archive_path": MEMBER_MANIFEST, "tree": INPUTS, "path": member_manifest.name}
        | {"bytes": size, "sha256": sha, "git_blob": None}
    )
    members.sort(key=lambda m: m["archive_path"])
    check_member_names([m["archive_path"] for m in members])
    by_path = {m["archive_path"]: m for m in members}

    def sha_of(tree: str, name: str) -> str:
        return by_path[f"{tree}/{name}"]["sha256"]

    overlay = json.loads((code_root / OVERLAY).read_bytes())
    release = json.loads(
        (artifact_root / PurePosixPath(relative["studies"]).parent / "verified_release.json")
        .read_bytes()
    )
    folds = relative["folds"]
    bindings = {
        "release_sha256": sha_of(ARTIFACT, relative["studies"]),
        "folds_sha256": sha_of(ARTIFACT, f"{folds}/fold.csv"),
        "accepted_series_sha256": sha_of(ARTIFACT, relative["series"]),
        "source_notebook_sha256": sha_of(CODE, SOURCE_NOTEBOOK),
        "member_manifest_sha256": sha,
        "overlay_sha256": sha_of(CODE, OVERLAY),
        "input_receipt_sha256": sha_of(CODE, INPUT_RECEIPT),
        "mounted_series_csv": "bound_independently_at_render_time",
    }
    _require(
        bindings["release_sha256"] == overlay["release_sha256"] == release["studies_sha256"]
        and bindings["release_sha256"] == closure["release_manifest_sha256"],
        "Release studies differ from the overlay/verified release",
    )
    _require(bindings["folds_sha256"] == overlay["folds_sha256"], "Folds differ from overlay")
    _require(
        bindings["accepted_series_sha256"]
        == release["series_sha256"]
        == closure["release_series_sha256"],
        "Accepted series CSV differs from the verified release series_sha256",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "prepared_not_funded_not_dispatched",
        "private": True,
        "code_commit": code_commit,
        "artifact_commit": artifact_commit,
        "closure_sha256": canonical_sha256(closure),
        "release_binding_sha256": closure["release_binding_sha256"],
        "receipt_relative": relative,
        "bindings": bindings,
        "members": members,
        "files": len(members),
        "bytes": sum(m["bytes"] for m in members),
        "render_arguments": render_arguments(relative),
        "excluded": list(EXCLUDED),
        "labels_analyzed": False,
        "source_downloaded": False,
    }


def admitted_member_manifest_sha256(code_root: Path) -> str:
    admitted = json.loads(regular_file(code_root, PLAN_ADMITTED).read_bytes())
    return str(admitted["original_member_manifest_sha256"])


def render_arguments(relative: dict[str, str]) -> dict[str, str]:
    """Renderer paths relative to the extraction root (the mount is bound separately)."""
    return {
        "script": f"{CODE}/{RENDERER}",
        "--source": f"{CODE}/{SOURCE_NOTEBOOK}",
        "--member-manifest": MEMBER_MANIFEST,
        "--studies": f"{ARTIFACT}/{relative['studies']}",
        "--artifact-root": ARTIFACT,
        "--folds": f"{ARTIFACT}/{relative['folds']}",
        "--accepted-series": f"{ARTIFACT}/{relative['series']}",
    }


def check_member_names(names: list[str]) -> None:
    """Unique, case-unique, safe paths where no member is also another member's directory."""
    for name in names:
        safe_relative(name)
        _require(name.split("/", 1)[0] in {CODE, ARTIFACT, INPUTS}, f"Foreign member: {name}")
    _require(len(set(names)) == len(names), "Duplicate capsule member")
    _require(len({n.casefold() for n in names}) == len(names), "Case-colliding capsule member")
    directories = {str(p) for n in names for p in PurePosixPath(n).parents}
    _require(not directories & set(names), "A member path is also a directory")


def _tarinfo(member: dict) -> tarfile.TarInfo:
    info = tarfile.TarInfo(member["archive_path"])
    info.size, info.mtime, info.mode, info.type = member["bytes"], 0, 0o444, tarfile.REGTYPE
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


class _Hashing(io.RawIOBase):
    def __init__(self, handle, limit: int) -> None:
        self.handle, self.limit, self.size, self.digest = handle, limit, 0, hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        block = self.handle.read(size)
        self.size += len(block)
        _require(self.size <= self.limit, "Source grew while packing")
        self.digest.update(block)
        return block


def estimate_archive_bytes(manifest: dict) -> int:
    """Upper bound: header + PAX header per member, padded data, end blocks, record padding."""
    total = 1024 + sum(3 * 512 + -(-m["bytes"] // 512) * 512 for m in manifest["members"])
    return -(-total // tarfile.RECORDSIZE) * tarfile.RECORDSIZE


def bounds(max_bytes: int, max_files: int, min_free_bytes: int) -> dict:
    _require(
        all(type(v) is int and v > 0 for v in (max_bytes, max_files, min_free_bytes)),
        "Root must supply positive byte/file/free-space bounds",
    )
    return {"max_bytes": max_bytes, "max_files": max_files, "min_free_bytes": min_free_bytes}


def pack(manifest: dict, roots: dict[str, Path], out: Path, limits: dict) -> dict:
    """Write the archive and manifest once into a new owner-only directory."""
    _require(manifest["files"] <= limits["max_files"], "Capsule exceeds the funded file count")
    estimate = estimate_archive_bytes(manifest)
    _require(estimate <= limits["max_bytes"], "Capsule exceeds the funded bytes")
    out = Path(out)
    _require(out.is_absolute() and not out.exists(), "Output must be a new absolute directory")
    for root in roots.values():
        resolved = Path(root).resolve()
        _require(not out.resolve().is_relative_to(resolved), "Output may not be inside a source")
    _require(
        shutil.disk_usage(out.parent).free - estimate >= limits["min_free_bytes"],
        "Packing would breach the free-space reserve",
    )
    out.mkdir(mode=0o700)
    descriptor = os.open(out / ARCHIVE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as raw:
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for member in manifest["members"]:
                source = regular_file(roots[member["tree"]], member["path"])
                with source.open("rb") as handle:
                    reader = _Hashing(handle, member["bytes"])
                    archive.addfile(_tarinfo(member), reader)  # type: ignore[arg-type]
                _require(
                    (reader.size, reader.digest.hexdigest()) == (member["bytes"], member["sha256"]),
                    f"Source changed while packing: {member['archive_path']}",
                )
    size, sha, _ = file_digests(out / ARCHIVE)
    _require(size <= limits["max_bytes"], "Written capsule exceeds the funded bytes")
    sealed = manifest | {
        "bounds": limits,
        "archive": {"name": ARCHIVE, "bytes": size, "sha256": sha, "format": "pax_tar"},
    }
    payload = json_bytes(sealed)
    write_exclusive(out / MANIFEST, payload)
    return {"manifest_sha256": hashlib.sha256(payload).hexdigest(), **sealed["archive"]}


def reconcile(manifest: dict, out: Path, manifest_sha256: str) -> dict:
    """Read-only: an existing capsule must equal a freshly computed plan, byte for byte."""
    payload = (Path(out) / MANIFEST).read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == manifest_sha256, "Manifest SHA differs")
    sealed = json.loads(payload)
    keys = set(manifest)
    _require({k: sealed[k] for k in keys} == manifest, "Capsule differs from the fresh plan")
    names = check_archive(Path(out) / ARCHIVE, sealed)
    return {"status": "reconciled_read_only", "files": len(names), **sealed["archive"]}


# ---------------------------------------------------------------- verify (Kaggle side)


def load_sealed(path: Path, expected_sha256: str) -> dict:
    payload = regular_file(Path(path).parent, Path(path).name).read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == expected_sha256, "Manifest SHA differs")
    sealed = json.loads(payload)
    _require(sealed.get("kind") == KIND and "archive" in sealed, "Not a sealed capsule manifest")
    check_member_names([m["archive_path"] for m in sealed["members"]])
    return sealed


def check_archive(path: Path, sealed: dict) -> list[str]:
    """Archive bytes/SHA plus every header: regular files only, exact names/sizes/order."""
    size, sha, _ = file_digests(regular_file(Path(path).parent, Path(path).name))
    _require(
        (size, sha) == (sealed["archive"]["bytes"], sealed["archive"]["sha256"]), "Archive SHA"
    )
    expected = [(m["archive_path"], m["bytes"]) for m in sealed["members"]]
    with tarfile.open(path, mode="r:") as archive:
        observed = []
        for info in archive:
            _require(info.isreg() and not (info.islnk() or info.issym()), "Non-regular member")
            safe_relative(info.name)
            observed.append((info.name, info.size))
    _require(observed == expected, "Archive members differ from the manifest")
    return [name for name, _ in observed]


def verify(
    archive_path: Path, sealed: dict, extract_to: Path, limits: dict, verifier: Path
) -> dict:
    """Validate, then extract into a new directory and re-hash; nothing is rendered here."""
    _require(file_digests(verifier)[1] == sealed["verifier_sha256"], "Verifier SHA differs")
    _require(sealed["files"] <= limits["max_files"], "Capsule exceeds the funded file count")
    _require(sealed["archive"]["bytes"] <= limits["max_bytes"], "Capsule exceeds funded bytes")
    check_archive(archive_path, sealed)
    extract_to = Path(extract_to)
    _require(extract_to.is_absolute() and not extract_to.exists(), "Extraction must be new")
    _require(
        shutil.disk_usage(extract_to.parent).free - sealed["bytes"] >= limits["min_free_bytes"],
        "Extraction would breach the free-space reserve",
    )
    extract_to.mkdir(mode=0o700)
    members = {m["archive_path"]: m for m in sealed["members"]}
    with tarfile.open(archive_path, mode="r:") as archive:
        for info in archive:
            member = members[info.name]
            target = extract_to / info.name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source = archive.extractfile(info)
            _require(source is not None, "Unreadable member")
            reader = _Hashing(source, member["bytes"])
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "wb") as handle:
                shutil.copyfileobj(reader, handle)  # type: ignore[misc]
            target.chmod(0o400)
            _require(reader.digest.hexdigest() == member["sha256"], "Member SHA differs")
    return verify_tree(extract_to, sealed)


def verify_tree(root: Path, sealed: dict) -> dict:
    """Read-only check of a materialized tree: exact regular-file set, sizes and SHA256."""
    found = []
    for directory, dirs, files in os.walk(root):
        for name in dirs + files:
            path = Path(directory) / name
            _require(not path.is_symlink(), "Extracted tree holds a link")
        found += [(Path(directory) / f).relative_to(root).as_posix() for f in files]
    members = {m["archive_path"]: m for m in sealed["members"]}
    _require(sorted(found) == sorted(members), "Extracted tree differs from the manifest")
    for name, member in members.items():
        size, sha, _ = file_digests(regular_file(root, name))
        _require((size, sha) == (member["bytes"], member["sha256"]), f"Extracted differs: {name}")
    arguments = {k: str(Path(root) / v) for k, v in sealed["render_arguments"].items()}
    return {
        "status": "capsule_verified_before_any_read_or_render",
        "files": len(members),
        "bytes": sealed["bytes"],
        "archive_sha256": sealed["archive"]["sha256"],
        "code_commit": sealed["code_commit"],
        "render_arguments": arguments,
    }


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--out", type=Path, required=True)
    for name in ("artifact-root", "studies", "folds", "accepted-series"):
        t.add_argument("--" + name, type=Path, required=True)
    for mode in ("pack", "reconcile"):
        p = sub.add_parser(mode)
        p.add_argument("--code-root", type=Path, default=ROOT)
        p.add_argument("--artifact-root", type=Path, required=True)
        p.add_argument("--artifact-commit", required=True)
        p.add_argument("--closure", type=Path, required=True)
        p.add_argument("--closure-sha256", required=True)
        p.add_argument("--member-manifest", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if mode == "reconcile":
            p.add_argument("--manifest-sha256", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--manifest-sha256", required=True)
    group = v.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path)
    group.add_argument("--tree", type=Path, help="Read-only check of an already materialized tree")
    v.add_argument("--extract-to", type=Path)
    for sp in (sub.choices["pack"], v):
        for name in ("max-bytes", "max-files", "min-free-bytes"):
            sp.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode == "trace":
        closure = trace(
            ROOT, args.artifact_root, args.studies, args.folds, args.accepted_series
        )
        payload = json_bytes(closure)
        write_exclusive(args.out, payload)
        result = {k: v for k, v in closure.items() if k != "files"}
        result |= {"files": len(closure["files"]), "closure_sha256": hashlib.sha256(payload)
                   .hexdigest()}
    elif args.mode in ("pack", "reconcile"):
        closure = load_closure(args.closure, args.closure_sha256)
        manifest = plan(
            args.code_root, args.artifact_root, closure, args.artifact_commit, args.member_manifest
        )
        manifest["verifier_sha256"] = file_digests(args.code_root / SELF)[1]
        if args.mode == "pack":
            limits = bounds(args.max_bytes, args.max_files, args.min_free_bytes)
            roots = {CODE: args.code_root, ARTIFACT: args.artifact_root}
            roots[INPUTS] = args.member_manifest.parent
            result = pack(manifest, roots, args.out, limits)
        else:
            result = reconcile(manifest, args.out, args.manifest_sha256)
    else:
        sealed = load_sealed(args.manifest, args.manifest_sha256)
        if args.tree is not None:
            _require(args.extract_to is None, "--tree is read-only; no extraction target")
            limits = bounds(args.max_bytes, args.max_files, args.min_free_bytes)
            _require(file_digests(Path(__file__))[1] == sealed["verifier_sha256"], "Verifier")
            _require(sealed["files"] <= limits["max_files"], "Capsule exceeds funded files")
            _require(sealed["bytes"] <= limits["max_bytes"], "Capsule exceeds funded bytes")
            result = verify_tree(args.tree, sealed)
        else:
            _require(args.extract_to is not None, "--archive requires --extract-to")
            limits = bounds(args.max_bytes, args.max_files, args.min_free_bytes)
            result = verify(args.archive, sealed, args.extract_to, limits, Path(__file__))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
