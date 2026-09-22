"""Stream an existing cache through verified SSH; no local archive and no overwrite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

RECEIVE = r"""
import hashlib,json,os,sys,tarfile
target,expected=sys.argv[1],int(sys.argv[2]);partial=target+'.partial'
os.umask(0o077)
if os.path.lexists(target):raise SystemExit('final_archive_already_exists')
received=hashlib.sha256();size=0
with open(partial,'xb') as output:
    while True:
        block=sys.stdin.buffer.read(4*1024*1024)
        if not block:break
        output.write(block);received.update(block);size+=len(block)
    output.flush();os.fsync(output.fileno())
class Reader:
    def __init__(self,source):self.source=source;self.sha=hashlib.sha256();self.size=0
    def read(self,n=-1):
        block=self.source.read(n);self.sha.update(block);self.size+=len(block);return block
count=0;members=set();unpacked=0
with open(partial,'rb') as source:
    reader=Reader(source)
    with tarfile.open(fileobj=reader,mode='r|') as archive:
        for member in archive:
            parts=member.name.split('/')
            if member.name.startswith('/') or '..' in parts or parts[0]!='cache':raise ValueError('unsafe_member')
            if member.name in members or not(member.isdir() or member.isfile()):raise ValueError('duplicate_or_nonregular_member')
            members.add(member.name)
            if member.isfile():count+=1;unpacked+=member.size
    while reader.read(4*1024*1024):pass
if count!=expected or reader.sha.hexdigest()!=received.hexdigest() or reader.size!=size:raise ValueError('archive_verification_failed')
state=os.stat(partial)
receipt={'status':'verified_partial','sha256':received.hexdigest(),'bytes':size,'file_members':count,'unpacked_file_bytes':unpacked,'inode':state.st_ino,'mtime_ns':state.st_mtime_ns}
with open(partial+'.verified.json','x') as handle:json.dump(receipt,handle);handle.flush();os.fsync(handle.fileno())
print(json.dumps(receipt))
"""

COMMIT = r"""
import json,os,sys
target,sha,size=sys.argv[1:];partial=target+'.partial'
receipt=json.load(open(partial+'.verified.json'));state=os.stat(partial,follow_symlinks=False)
if receipt['sha256']!=sha or receipt['bytes']!=int(size) or state.st_ino!=receipt['inode'] or state.st_mtime_ns!=receipt['mtime_ns'] or state.st_size!=int(size):raise ValueError('verified_partial_changed')
os.link(partial,target);os.unlink(partial)
receipt['status']='published_verified_archive'
with open(target+'.receipt.json','x') as handle:json.dump(receipt,handle);handle.flush();os.fsync(handle.fileno())
print(json.dumps(receipt))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--connection", type=Path, required=True)
    parser.add_argument("--remote-target", required=True)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.connection.read_text())
    files, directories = [], {args.cache}
    for parent, children, names in os.walk(args.cache, followlinks=False):
        for name in children:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError("symlink_cache_directory")
            directories.add(path)
        for name in names:
            path = Path(parent) / name
            if path.is_symlink() or not path.is_file() or path.suffix not in {".npz", ".json"}:
                raise ValueError("unexpected_cache_member")
            files.append(path)
    if len(files) != args.expected_files or args.cache.is_symlink():
        raise ValueError("cache_file_count_mismatch")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "pid": os.getpid(),
        "status": "starting",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "remote_target": args.remote_target,
        "expected_file_members": len(files),
        "source_file_bytes": sum(path.stat().st_size for path in files),
        "local_archive_created": False,
        "no_overwrite": True,
    }
    with args.receipt.open("x") as handle:
        json.dump(record, handle, indent=2)

    def save(**updates):
        record.update(updates)
        temporary = args.receipt.with_suffix(".updating")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(temporary, args.receipt)

    ssh = [
        "/usr/bin/ssh",
        "-S",
        state["control_socket"],
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ControlMaster=no",
        state["user"] + "@" + state["host"],
    ]
    receive = shlex.join(["python3", "-c", RECEIVE, args.remote_target, str(len(files))])
    child = subprocess.Popen(
        ssh + [receive], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    started = time.monotonic()

    class Writer:
        def __init__(self):
            self.sha, self.bytes, self.updated = hashlib.sha256(), 0, 0.0

        def write(self, data):
            child.stdin.write(data)
            self.sha.update(data)
            self.bytes += len(data)
            if time.monotonic() - self.updated > 10:
                self.updated = time.monotonic()
                save(
                    status="streaming",
                    stream_bytes_sent=self.bytes,
                    elapsed_seconds=time.monotonic() - started,
                )
            return len(data)

    writer = Writer()
    try:
        with tarfile.open(
            fileobj=writer,
            mode="w|",
            format=tarfile.USTAR_FORMAT,
            bufsize=1024 * 1024,
            copybufsize=1024 * 1024,
        ) as archive:
            for path in sorted(directories):
                info = tarfile.TarInfo(
                    "cache"
                    if path == args.cache
                    else "cache/" + path.relative_to(args.cache).as_posix()
                )
                info.type, info.mode = tarfile.DIRTYPE, 0o700
                archive.addfile(info)
            for path in sorted(files):
                info = tarfile.TarInfo("cache/" + path.relative_to(args.cache).as_posix())
                before = path.stat()
                info.size, info.mode = before.st_size, 0o600
                with path.open("rb") as source:
                    archive.addfile(info, source)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("source_changed_while_streaming")
        child.stdin.close()
        child.stdin = None
        save(
            status="remote_verification",
            stream_bytes_sent=writer.bytes,
            sender_sha256=writer.sha.hexdigest(),
            elapsed_seconds=time.monotonic() - started,
        )
        stdout, _ = child.communicate()
        if child.returncode != 0:
            raise ValueError("remote_receive_or_verification_failed")
        remote = json.loads(stdout)
        if remote["sha256"] != writer.sha.hexdigest() or remote["bytes"] != writer.bytes:
            raise ValueError("sender_receiver_digest_mismatch")
        command = shlex.join(
            ["python3", "-c", COMMIT, args.remote_target, writer.sha.hexdigest(), str(writer.bytes)]
        )
        result = subprocess.run(ssh + [command], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise ValueError("archive_publication_requires_read_only_reconciliation")
        save(
            status="published_verified_archive",
            remote=json.loads(result.stdout),
            elapsed_seconds=time.monotonic() - started,
        )
        print(json.dumps(record, indent=2), flush=True)
    except BaseException as error:
        save(
            status="failed_or_interrupted_partial_preserved",
            error_type=type(error).__name__,
            stream_bytes_sent=writer.bytes,
            elapsed_seconds=time.monotonic() - started,
        )
        if child.stdin is not None:
            child.stdin.close()
        if child.poll() is None:
            child.terminate()
        raise


if __name__ == "__main__":
    main()
