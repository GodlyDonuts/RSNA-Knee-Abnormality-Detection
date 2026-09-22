"""Unpack a pinned environment on private node-local disk; never install or download."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shutil
import signal
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from runtime_archive import HashingReader, local_disk


def cleanup(root: Path, job_id: str) -> None:
    if not re.fullmatch(r"[1-9][0-9]*", job_id) or root.is_symlink() or not root.is_dir():
        raise ValueError("owned_environment_root_required")
    if (
        not root.name.startswith("rsna-environment-" + job_id + "-")
        or root.stat().st_uid != os.getuid()
    ):
        raise ValueError("environment_owner_or_job_mismatch")
    marker = root / ".rsna-environment.json"
    if marker.is_symlink():
        raise ValueError("linked_environment_marker")
    record = json.loads(marker.read_text())
    if record != {"kind": "rsna-newton-environment-v1", "job_id": job_id, "uid": os.getuid()}:
        raise ValueError("environment_marker_mismatch")
    shutil.rmtree(root)


def safe_link(name: PurePosixPath, target: str, shared_python: Path) -> None:
    link = PurePosixPath(target)
    if not target or any(ord(char) < 32 for char in target):
        raise ValueError("unsafe_environment_link")
    if link.is_absolute():
        if name.parent != PurePosixPath("environment/bin") or not re.fullmatch(
            r"python(?:3(?:\.12)?)?", name.name
        ):
            raise ValueError("only_python_may_link_to_shared_base")
        if ".." in link.parts or Path(target) != shared_python:
            raise ValueError("external_shared_python_mismatch")
    else:
        normalized = PurePosixPath(posixpath.normpath(str(name.parent / link)))
        if not normalized.parts or normalized.parts[0] != "environment" or ".." in normalized.parts:
            raise ValueError("environment_link_escapes_root")


def extract(
    archive: Path,
    sha256: str,
    expected_files: int,
    expected_bytes: int,
    scratch_base: Path,
    min_free_gib: int,
    shared_python: Path,
    job_id: str,
) -> dict:
    if not re.fullmatch(r"[1-9][0-9]*", job_id) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("real_job_and_archive_hash_required")
    if min(expected_files, expected_bytes, min_free_gib) <= 0:
        raise ValueError("positive_environment_bounds_required")
    if (
        not shared_python.is_absolute()
        or ".." in shared_python.parts
        or not shared_python.is_file()
        or not os.access(shared_python, os.X_OK)
    ):
        raise ValueError("real_shared_shared_python_required")
    if (
        archive.is_symlink()
        or not archive.is_file()
        or not scratch_base.is_dir()
        or not local_disk(scratch_base)
    ):
        raise ValueError("regular_archive_and_local_disk_required")
    if shutil.disk_usage(scratch_base).free < expected_bytes + min_free_gib * 2**30:
        raise ValueError("insufficient_environment_disk")
    root = Path(tempfile.mkdtemp(prefix="rsna-environment-" + job_id + "-", dir=scratch_base))
    (root / ".rsna-environment.json").write_text(
        json.dumps({"kind": "rsna-newton-environment-v1", "job_id": job_id, "uid": os.getuid()})
    )
    count, size, seen, links = 0, 0, set(), []
    previous_handlers = {}

    def interrupted(_signum, _frame):
        raise InterruptedError("environment_extraction_interrupted")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        with archive.open("rb") as raw:
            verified = HashingReader(raw)
            with tarfile.open(fileobj=verified, mode="r|*") as source:
                for member in source:
                    name = PurePosixPath(member.name)
                    if (
                        name.is_absolute()
                        or ".." in name.parts
                        or not name.parts
                        or name.parts[0] != "environment"
                        or str(name) != member.name.rstrip("/")
                        or str(name) in seen
                    ):
                        raise ValueError("unsafe_or_duplicate_environment_member")
                    seen.add(str(name))
                    if len(seen) > expected_files * 8 + 20:
                        raise ValueError("environment_member_count_exceeds_bound")
                    destination = root.joinpath(*name.parts)
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
                    elif member.isfile():
                        count += 1
                        size += member.size
                        if count > expected_files or size > expected_bytes:
                            raise ValueError("environment_exceeds_declared_bounds")
                        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        stream = source.extractfile(member)
                        if stream is None:
                            raise ValueError("missing_environment_file_stream")
                        with stream, destination.open("xb") as output:
                            shutil.copyfileobj(stream, output, length=4 * 1024 * 1024)
                        destination.chmod(0o600 | (member.mode & 0o111))
                    elif member.issym():
                        safe_link(name, member.linkname, shared_python)
                        links.append((destination, member.linkname))
                    else:
                        raise ValueError("unsupported_environment_member_type")
            while verified.read(4 * 1024 * 1024):
                pass
            if verified.digest.hexdigest() != sha256:
                raise ValueError("environment_archive_sha256_mismatch")
        if count != expected_files or size != expected_bytes:
            raise ValueError("environment_inventory_mismatch")
        # Links are installed last, so no archive write can follow an archive-created link.
        for destination, target in links:
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            destination.symlink_to(target)
        interpreter = root / "environment/bin/python"
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise ValueError("relocated_environment_interpreter_missing")
        if (
            not interpreter.resolve().is_relative_to(root)
            and interpreter.resolve() != shared_python.resolve()
        ):
            raise ValueError("interpreter_outside_verified_roots")
        if not (root / "environment/pyvenv.cfg").is_file():
            raise ValueError("relocatable_venv_configuration_missing")
        return {
            "runtime_root": str(root),
            "python": str(interpreter),
            "archive_sha256": sha256,
            "file_members": count,
            "unpacked_file_bytes": size,
        }
    except BaseException:
        cleanup(root, job_id)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    unpack = commands.add_parser("extract")
    for name in ("archive", "scratch-base", "shared-python"):
        unpack.add_argument("--" + name, type=Path, required=True)
    unpack.add_argument("--sha256", required=True)
    unpack.add_argument("--expected-files", type=int, required=True)
    unpack.add_argument("--expected-bytes", type=int, required=True)
    unpack.add_argument("--min-free-gib", type=int, default=20)
    remove = commands.add_parser("cleanup")
    remove.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "cleanup":
            cleanup(args.root, os.environ.get("SLURM_JOB_ID", ""))
        else:
            fields = vars(args).copy()
            fields.pop("command")
            print(json.dumps(extract(**fields, job_id=os.environ.get("SLURM_JOB_ID", ""))))
    except (OSError, ValueError, KeyError, tarfile.TarError):
        print(json.dumps({"status": "blocked", "reason": "job_local_environment_failed"}))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
