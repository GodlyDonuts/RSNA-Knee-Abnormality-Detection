"""Forward checkpoint requests to trainer workers, never to torchrun's elastic agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def checkpoint_stop_verified(checkpoint: Path, requested_wall: float) -> bool:
    """A periodic save alone is insufficient: require the trainer's exact stop receipt."""
    receipt_path, run_path = checkpoint.parent / "invocation.json", checkpoint.parent / "run.json"
    try:
        for path in (receipt_path, run_path, checkpoint):
            if path.is_symlink() or not path.is_file():
                return False
        if receipt_path.stat().st_mtime < requested_wall:
            return False
        if max(receipt_path.stat().st_size, run_path.stat().st_size) > 50_000_000:
            return False
        receipt, run = json.loads(receipt_path.read_text()), json.loads(run_path.read_text())
        if (
            receipt.get("status")
            not in {
                "checkpointed_for_resume",
                "final_inference_interrupted_resume_final_checkpoint",
            }
            or receipt.get("exit_code") != 75
        ):
            return False
        binding_hash = hashlib.sha256(
            json.dumps(
                run["binding"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        if receipt.get("binding_sha256") != binding_hash:
            return False
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(2**20), b""):
                digest.update(chunk)
        return receipt.get("checkpoint_sha256") == digest.hexdigest()
    except (OSError, ValueError, KeyError, TypeError):
        return False


def worker_pids(parent: int) -> list[int]:
    """Discover only direct trainer children; exclude data-loader descendants."""
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, check=True, timeout=10
    )
    children = [
        int(row[0])
        for line in result.stdout.splitlines()
        if len(row := line.split()) == 2 and int(row[1]) == parent
    ]
    workers = []
    for pid in children:
        command = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True, timeout=10
        )
        if command.returncode == 0 and "rsna.competitive.train" in command.stdout.split():
            workers.append(pid)
    return workers


def supervise(command: list[str], checkpoint: Path, grace: int) -> int:
    requested = 0.0
    requested_wall = 0.0

    def request(_signum: int, _frame: object) -> None:
        nonlocal requested, requested_wall
        if not requested:
            requested, requested_wall = time.monotonic(), time.time()

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request)
    child = subprocess.Popen(command, start_new_session=True)
    signaled: set[int] = set()
    timed_out = False
    try:
        while child.poll() is None:
            if requested:
                for pid in worker_pids(child.pid):
                    if pid not in signaled:
                        try:
                            os.kill(pid, signal.SIGUSR1)
                            signaled.add(pid)
                        except ProcessLookupError:
                            pass
                if time.monotonic() - requested >= grace:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    break
            time.sleep(0.2)
        code = child.wait()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    if requested:
        stop_verified = checkpoint_stop_verified(checkpoint, requested_wall)
        success = bool(signaled) and stop_verified and not timed_out
        print(
            json.dumps(
                {
                    "stage": "checkpoint_stop",
                    "checkpoint_stop_verified": stop_verified,
                    "workers_signaled": len(signaled),
                    "grace_expired": timed_out,
                }
            ),
            flush=True,
        )
        return 0 if code == 0 and not timed_out else 75 if success else 2
    return code if code >= 0 else 128 - code


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grace-seconds", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.grace_seconds <= 0 or not command:
        p.error("positive grace and torchrun command required")
    try:
        raise SystemExit(supervise(command, args.checkpoint, args.grace_seconds))
    except (OSError, subprocess.SubprocessError):
        print(json.dumps({"status": "blocked", "reason": "supervisor_process_failure"}))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
