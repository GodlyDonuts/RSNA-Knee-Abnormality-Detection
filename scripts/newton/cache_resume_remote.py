"""Receiver for a reviewed deterministic cache continuation; never truncate a partial."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import sys
import tarfile
from pathlib import Path


def state(fd):
    s = os.fstat(fd)
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid():
        raise ValueError("Partial must be an owned ordinary file")
    return {k: getattr(s, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def digest(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    h = hashlib.sha256()
    while block := os.read(fd, 4 * 1024**2):
        h.update(block)
    return h


def durable(path, value):
    with path.open("x") as f:
        os.chmod(path, 0o600)
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    d = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(d)
    finally:
        os.close(d)


def locations(target):
    target = Path(target)
    return target, Path(str(target) + ".partial"), Path(str(target) + ".partial.resume-lock")


def require_unpublished(target, partial):
    for p in (target, Path(str(target) + ".receipt.json"), Path(str(partial) + ".verified.json")):
        if os.path.lexists(p):
            raise ValueError("Existing publication state requires read-only reconciliation")


def inspect(target, hash_bytes=False):
    target, partial, lock = locations(target)
    require_unpublished(target, partial)
    if os.path.lexists(lock):
        raise ValueError("Existing continuation lock requires reconciliation")
    fd = os.open(partial, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = state(fd)
        result = {"state": before}
        if hash_bytes:
            result["sha256"] = digest(fd).hexdigest()
        if state(fd) != before:
            raise ValueError("Partial changed during inspection")
        return result
    finally:
        os.close(fd)


def require_no_open_handles(partial, proc_root=Path("/proc")):
    """Inspect only this account's processes; the old nonlocking receiver must be gone."""
    resolved = str(partial.resolve())
    excluded_services = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            # Linux marks these privileged session helpers nondumpable: their
            # proc directory can be user-owned while fd remains root-owned.
            # Never waive an unreadable Python receiver or arbitrary process.
            if (process / "fd").stat().st_uid == 0 and os.getuid() != 0:
                name = (process / "comm").read_text().strip()
                if name not in {"sshd", "(sd-pam)"}:
                    raise ValueError("Unexpected uninspectable account process")
                excluded_services.append({"pid": int(process.name), "name": name})
                continue
            for handle in (process / "fd").iterdir():
                try:
                    if os.readlink(handle) == resolved:
                        raise ValueError("Another process still has the partial open")
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            continue
    return excluded_services


def receive(target, plan, source):
    target, partial, lock = locations(target)
    require_unpublished(target, partial)
    excluded_services = require_no_open_handles(partial)
    lock.mkdir(mode=0o700)
    durable(lock / "intent.json", plan)
    fd = os.open(partial, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = state(fd)
        if before != plan["partial_state"]:
            raise ValueError("Partial identity changed before continuation")
        full = digest(fd)
        if full.hexdigest() != plan["prefix_sha256"] or state(fd) != before:
            raise ValueError("Existing prefix differs from the reviewed source")
        offset = before["st_size"]
        if not 0 <= offset <= plan["archive_bytes"]:
            raise ValueError("Invalid continuation offset")
        remaining = plan["archive_bytes"] - offset
        with os.fdopen(os.dup(fd), "wb", buffering=0) as output:
            while remaining:
                block = source.read(min(4 * 1024**2, remaining))
                if not block:
                    raise ValueError("Interrupted suffix; partial and lock retained")
                view = memoryview(block)
                while view:
                    written = output.write(view)
                    if written is None or written <= 0:
                        raise OSError("Incomplete append")
                    view = view[written:]
                full.update(block)
                remaining -= len(block)
            if source.read(1):
                raise ValueError("Unexpected suffix bytes")
            os.fsync(output.fileno())
        if full.hexdigest() != plan["archive_sha256"]:
            raise ValueError("Completed stream digest mismatch")
        completed = state(fd)
        if completed["st_size"] != plan["archive_bytes"]:
            raise ValueError("Completed stream length mismatch")
        # Independently reread all persisted bytes and validate the actual TAR.
        if digest(fd).hexdigest() != plan["archive_sha256"]:
            raise ValueError("Persisted archive digest mismatch")
        os.lseek(fd, 0, os.SEEK_SET)
        names, count, unpacked = set(), 0, 0
        with os.fdopen(os.dup(fd), "rb") as raw, tarfile.open(fileobj=raw, mode="r|") as tar:
            for member in tar:
                parts = member.name.split("/")
                if (
                    member.name.startswith("/")
                    or ".." in parts
                    or parts[0] != "cache"
                    or member.name in names
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("Unsafe or duplicate archive member")
                names.add(member.name)
                if member.isfile():
                    count += 1
                    unpacked += member.size
        if (
            count != plan["file_members"]
            or unpacked != plan["unpacked_bytes"]
            or state(fd) != completed
        ):
            raise ValueError("Persisted archive membership or identity differs")
        receipt = {
            "status": "verified_resumed_partial",
            "operation_id": plan["operation_id"],
            "sha256": full.hexdigest(),
            "bytes": completed["st_size"],
            "file_members": count,
            "unpacked_file_bytes": unpacked,
            "state": completed,
            "resumed_from_bytes": offset,
            "privileged_session_helpers_excluded_from_fd_scan": excluded_services,
        }
        durable(Path(str(partial) + ".verified.json"), receipt)
        return receipt
    finally:
        os.close(fd)


def commit(target, plan):
    target, partial, lock = locations(target)
    intent = json.loads((lock / "intent.json").read_text())
    receipt = json.loads(Path(str(partial) + ".verified.json").read_text())
    if intent != plan or receipt["operation_id"] != plan["operation_id"]:
        raise ValueError("Continuation operation identity differs")
    if receipt["sha256"] != plan["archive_sha256"] or receipt["bytes"] != plan["archive_bytes"]:
        raise ValueError("Continuation verification differs")
    if os.path.lexists(target) or os.path.lexists(Path(str(target) + ".receipt.json")):
        raise ValueError("Existing final state requires read-only reconciliation")
    fd = os.open(partial, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state(fd) != receipt["state"]:
            raise ValueError("Verified partial changed")
        os.link(partial, target, follow_symlinks=False)
        final = dict(receipt, status="published_verified_archive")
        durable(Path(str(target) + ".receipt.json"), final)
        os.unlink(partial)
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        durable(lock / "completed.json", final)
        return final
    finally:
        os.close(fd)


if __name__ == "__main__":
    os.umask(0o077)
    action, target = sys.argv[1:3]
    if action in {"inspect", "hash"}:
        result = inspect(target, action == "hash")
    else:
        plan = json.loads(sys.argv[3])
        if action == "receive":
            result = receive(target, plan, sys.stdin.buffer)
        elif action == "commit":
            result = commit(target, plan)
        else:
            raise ValueError("Unsupported phase")
    print(json.dumps(result, sort_keys=True), flush=True)
