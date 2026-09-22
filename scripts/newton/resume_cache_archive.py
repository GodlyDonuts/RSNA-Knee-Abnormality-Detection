"""Review then resume a deterministic cache stream against accepted per-file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import stat
import subprocess
import tarfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def approved_files(lineage):
    result = {}
    for line in lineage.read_text().splitlines():
        row = json.loads(line)
        parts = [row[k] for k in ("split", "study_uid", "series_uid")]
        if any(not p or p in {".", ".."} or "/" in p or "\\" in p for p in parts):
            raise ValueError("Unsafe relative lineage identity")
        base = "/".join(parts)
        for suffix, size, sha in (
            (".npz", row["cache_bytes"], row["cache_sha256"]),
            (".json", row["new_metadata"]["bytes"], row["new_metadata"]["sha256"]),
        ):
            if base + suffix in result:
                raise ValueError("Duplicate lineage identity")
            if not isinstance(size, int) or size < 0 or len(sha) != 64:
                raise ValueError("Invalid approved content identity")
            int(sha, 16)
            result[base + suffix] = {"bytes": size, "sha256": sha}
    if not result:
        raise ValueError("Empty approved inventory")
    return result


class HashReader:
    def __init__(self, source):
        self.source, self.sha, self.bytes = source, hashlib.sha256(), 0

    def read(self, size):
        block = self.source.read(size)
        self.sha.update(block)
        self.bytes += len(block)
        return block


class ReplaySink:
    """Hash all bytes, but forward only the exact suffix after a verified prefix."""

    def __init__(self, offset, expected_prefix=None, output=None, progress=None):
        self.offset, self.expected_prefix, self.output = offset, expected_prefix, output
        self.progress = progress
        self.sha, self.prefix, self.bytes = hashlib.sha256(), hashlib.sha256(), 0
        self.prefix_verified = False

    def write(self, block):
        prefix_size = max(0, min(len(block), self.offset - self.bytes))
        self.prefix.update(block[:prefix_size])
        if self.bytes + len(block) >= self.offset and not self.prefix_verified:
            if self.expected_prefix is not None and self.prefix.hexdigest() != self.expected_prefix:
                raise ValueError("Replay prefix differs before forwarding suffix")
            self.prefix_verified = True
        if self.output is not None:
            suffix = memoryview(block)[prefix_size:]
            while suffix:
                written = self.output.write(suffix)
                if written is None or written <= 0:
                    raise OSError("Incomplete suffix write")
                suffix = suffix[written:]
        self.sha.update(block)
        self.bytes += len(block)
        if self.progress is not None:
            self.progress(self.bytes)
        return len(block)


def replay(cache, approved, sink):
    """Match the original USTAR emitter byte-for-byte and verify every source file."""
    if cache.is_symlink() or not cache.is_dir():
        raise ValueError("Cache root must be an ordinary directory")
    files, directories = {}, {cache}
    for parent, children, names in os.walk(cache, followlinks=False):
        for name in children:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError("Symlink cache directory")
            directories.add(path)
        for name in names:
            path = Path(parent) / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError("Nonregular cache member")
            files[path.relative_to(cache).as_posix()] = path
    if files.keys() != approved.keys():
        raise ValueError("Actual source membership differs from accepted lineage")
    with tarfile.open(
        fileobj=sink,
        mode="w|",
        format=tarfile.USTAR_FORMAT,
        bufsize=1024**2,
        copybufsize=1024**2,
    ) as archive:
        for path in sorted(directories):
            info = tarfile.TarInfo(
                "cache" if path == cache else "cache/" + path.relative_to(cache).as_posix()
            )
            info.type, info.mode = tarfile.DIRTYPE, 0o700
            archive.addfile(info)
        for relative, path in sorted(files.items()):
            expected = approved[relative]
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != expected["bytes"]:
                    raise ValueError("Source file identity differs")
                info = tarfile.TarInfo("cache/" + relative)
                info.size, info.mode = before.st_size, 0o600
                reader = HashReader(source)
                archive.addfile(info, reader)
                after = os.fstat(source.fileno())
                if (
                    (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                    or reader.bytes != expected["bytes"]
                    or reader.sha.hexdigest() != expected["sha256"]
                ):
                    raise ValueError("Source bytes differ from accepted lineage")
    if not sink.prefix_verified:
        raise ValueError("Partial is larger than complete source archive")
    return {
        "archive_bytes": sink.bytes,
        "archive_sha256": sink.sha.hexdigest(),
        "prefix_sha256": sink.prefix.hexdigest(),
        "file_members": len(files),
        "unpacked_bytes": sum(v["bytes"] for v in approved.values()),
    }


def command(connection, remote_source, action, target, plan=None):
    remote = ["/usr/bin/python3.12", "-c", remote_source, action, target]
    if plan is not None:
        remote.append(json.dumps(plan, sort_keys=True))
    return [
        "/usr/bin/ssh",
        "-S",
        connection["control_socket"],
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ControlMaster=no",
        connection["user"] + "@" + connection["host"],
        shlex.join(remote),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["plan", "execute"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--connection", type=Path, required=True)
    parser.add_argument("--remote-target", required=True)
    parser.add_argument("--operation", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    approved = approved_files(args.lineage)
    connection = json.loads(args.connection.read_text())
    remote_source = Path(__file__).with_name("cache_resume_remote.py").read_text()
    pins = {
        "lineage_sha256": hashlib.sha256(args.lineage.read_bytes()).hexdigest(),
        "sender_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "receiver_sha256": hashlib.sha256(remote_source.encode()).hexdigest(),
        "remote_target": args.remote_target,
        "remote_host": connection["host"],
        "remote_user": connection["user"],
    }

    def remote(action, plan=None):
        p = subprocess.run(
            command(connection, remote_source, action, args.remote_target, plan),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if p.returncode:
            # No command/source/credential-bearing traceback is copied to public receipts.
            (args.operation / (action + ".stderr.txt")).write_text(p.stderr)
            raise RuntimeError("Remote phase failed; inspect private diagnostics")
        return json.loads(p.stdout)

    if args.phase == "plan":
        args.operation.mkdir(parents=True, exist_ok=False)
        save(args.operation / "planning-intent.json", dict(pins, pid=os.getpid()))
        initial = remote("inspect")
        save(args.operation / "partial-inspection.json", initial)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(remote, "hash")
            local = replay(args.cache, approved, ReplaySink(initial["state"]["st_size"]))
            actual = future.result()
        if actual["state"] != initial["state"] or actual["sha256"] != local["prefix_sha256"]:
            raise ValueError("Stable remote prefix does not match approved deterministic source")
        plan = dict(pins, **local, partial_state=actual["state"], operation_id=str(uuid.uuid4()))
        save(args.operation / "plan.json", plan)
        print(json.dumps(dict(status="reviewed_prefix_and_all_source_hashes", **local)), flush=True)
        return

    plan = json.loads((args.operation / "plan.json").read_text())
    if any(plan[k] != v for k, v in pins.items()):
        raise ValueError("Reviewed code, data or destination changed")
    if remote("inspect")["state"] != plan["partial_state"]:
        raise ValueError("Partial changed since review")
    save(
        args.operation / "execute-intent.json",
        {"operation_id": plan["operation_id"], "pid": os.getpid()},
    )
    started, updated = time.monotonic(), 0.0

    def progress(size):
        nonlocal updated
        if time.monotonic() - updated >= 10:
            updated = time.monotonic()
            value = {
                "replayed_bytes": size,
                "suffix_bytes": max(0, size - plan["partial_state"]["st_size"]),
                "elapsed_seconds": updated - started,
                "pid": os.getpid(),
            }
            temp = args.operation / "progress.updating"
            temp.write_text(json.dumps(value))
            temp.replace(args.operation / "progress.json")

    with (args.operation / "receive.stderr.txt").open("x") as errors:
        child = subprocess.Popen(
            command(connection, remote_source, "receive", args.remote_target, plan),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        try:
            sink = ReplaySink(
                plan["partial_state"]["st_size"], plan["prefix_sha256"], child.stdin, progress
            )
            local = replay(args.cache, approved, sink)
            if any(plan[k] != v for k, v in local.items()):
                raise ValueError("Second replay differs from reviewed archive")
            child.stdin.close()
            child.stdin = None
            stdout, _ = child.communicate(timeout=3600)
            if child.returncode:
                raise RuntimeError("Receive failed; preserve partial and reconcile")
            receipt = json.loads(stdout)
            save(args.operation / "verified-partial.json", receipt)
            if (
                receipt["sha256"] != plan["archive_sha256"]
                or receipt["bytes"] != plan["archive_bytes"]
            ):
                raise ValueError("Sender and receiver disagree")
            save(args.operation / "commit-intent.json", {"operation_id": plan["operation_id"]})
            final = remote("commit", plan)
            save(args.operation / "completed.json", final)
            print(json.dumps(final), flush=True)
        except BaseException as error:
            save(
                args.operation / "failure.json",
                {"error_type": type(error).__name__, "reconcile_before_retry": True},
            )
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise


if __name__ == "__main__":
    main()
