"""Build a small exact-Git projection without touching the training checkout."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{40}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def selected(path):
    return (
        path == "pyproject.toml"
        or path.startswith("rsna/")
        and path.endswith(".py")
        or path.startswith("scripts/kaggle/")
        and path.endswith((".py", ".sh"))
    )


def files_at(repo, commit):
    if not SHA.fullmatch(commit):
        raise ValueError("Exact full Git commit required")
    if git(repo, "rev-parse", "--verify", commit + "^{commit}").decode().strip() != commit:
        raise ValueError("Commit did not resolve exactly")
    result = {}
    for item in git(
        repo, "ls-tree", "-rz", commit, "--", "rsna", "scripts/kaggle", "pyproject.toml"
    ).split(b"\0"):
        if not item:
            continue
        header, raw_path = item.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        path = raw_path.decode()
        if not selected(path):
            continue
        if kind != "blob" or mode not in ("100644", "100755") or ".." in Path(path).parts:
            raise ValueError("Only regular confined source files can enter projection")
        payload = git(repo, "cat-file", "blob", oid)
        if len(payload) > 5 * 1024**2 or payload.startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            raise ValueError("Large or unresolved source object rejected")
        result[path] = (payload, int(mode[-3:], 8))
    if not {
        "pyproject.toml",
        "rsna/__init__.py",
        "scripts/kaggle/workflow.py",
        "scripts/kaggle/notebook_entry.py",
    } <= set(result):
        raise ValueError("Complete runtime/coordinator sources are required")
    return result


def build(repo, package_commit, training_commit, output, preserve_training_code=False):
    package = files_at(repo, package_commit)
    training = files_at(repo, training_commit)
    train_code = {p: digest(b) for p, (b, _) in training.items() if p.startswith("rsna/")}
    package_code = {p: digest(b) for p, (b, _) in package.items() if p.startswith("rsna/")}
    if train_code != package_code:
        if not preserve_training_code:
            raise ValueError("Packaging projection changes the pinned training runtime")
        # Deployment wrappers may advance after training. Use the exact training
        # revision for every model/runtime module, including files since deleted
        # or added in the newer package commit.
        package = {p: value for p, value in package.items() if not p.startswith("rsna/")}
        package.update({p: value for p, value in training.items() if p.startswith("rsna/")})
    if {p: digest(b) for p, (b, _) in package.items() if p.startswith("rsna/")} != train_code:
        raise ValueError("Pinned training runtime could not be preserved")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "purpose": "minimal_exact_code_projection",
        "package_git_commit": package_commit,
        "training_git_commit": training_commit,
        "package_git_tree": git(repo, "rev-parse", package_commit + "^{tree}").decode().strip(),
        "training_code_sha256": train_code,
        "training_code_origin": training_commit,
        "packaging_code_origin": package_commit,
        "files": {
            p: {"sha256": digest(b), "bytes": len(b), "mode": mode}
            for p, (b, mode) in sorted(package.items())
        },
        "excluded": "Git metadata, reports, labels, image/cache arrays, weights, credentials and unrelated repository files.",
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (output / "projection.json").write_bytes(manifest_bytes)
    archive = output / "projection.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as handle:
        for path, (payload, mode) in sorted(package.items()):
            info = tarfile.TarInfo(path)
            info.size = len(payload)
            info.mode = mode
            info.mtime = 0
            handle.addfile(info, io.BytesIO(payload))
    receipt = {
        "schema_version": 1,
        "projection_sha256": digest(manifest_bytes),
        "archive_sha256": digest(archive.read_bytes()),
        "archive_bytes": archive.stat().st_size,
        "files": len(package),
        "unpacked_bytes": sum(len(b) for b, _ in package.values()),
        "package_git_commit": package_commit,
        "training_git_commit": training_commit,
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "training_checkout_mutated": False,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--package-commit", required=True)
    p.add_argument("--training-commit", required=True)
    p.add_argument("--preserve-training-code", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            build(a.repo, a.package_commit, a.training_commit, a.output, a.preserve_training_code),
            sort_keys=True,
        )
    )
