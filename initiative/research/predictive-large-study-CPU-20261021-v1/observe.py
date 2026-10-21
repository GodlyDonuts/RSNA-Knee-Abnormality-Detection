"""Finite read-only follow-up of the one already-pushed private CPU notebook."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
JOURNAL = REPO / "artifacts" / HERE.name
PIN = "4222cda6cfbd28737e758ae149fd48909334dad7"


def write(name, value):
    with (JOURNAL / name).open("x") as f:
        json.dump(value, f, sort_keys=True)
        f.write("\n")


def main():
    os.umask(0o077)
    deadline = time.time() + 3600
    write(
        "observer-intent.json",
        {
            "PID": os.getpid(),
            "kernel_id": 140052135,
            "source_git": PIN,
            "read_only": True,
            "deadline": deadline,
        },
    )
    try:
        while time.time() < deadline:
            result = subprocess.run(
                [sys.executable, "-B", str(HERE / "run_once.py"), "status", "--source-git", PIN],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=620,
            )
            if result.returncode:
                write(
                    "observer-stopped.json",
                    {"reason": "readonly_transport_failed", "returncode": result.returncode},
                )
                return
            value = json.loads(result.stdout)
            if value.get("status") == "COMPLETE":
                write("observer-complete.json", value)
                return
            if value.get("status") in ("ERROR", "CANCELED", "CANCELLED", "FAILED"):
                write("observer-stopped.json", value)
                return
            time.sleep(min(120, max(0, deadline - time.time())))
        write("observer-stopped.json", {"reason": "finite_observation_window_ended"})
    except BaseException as error:
        if not (JOURNAL / "observer-stopped.json").exists():
            write(
                "observer-stopped.json", {"reason": "observer_error", "type": type(error).__name__}
            )
        raise


if __name__ == "__main__":
    main()
