"""Fail closed while any known account owner/transport is active; no remote use."""

import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

ACCOUNT_MARKERS = (
    "pointwise-final-deployment-20261020-v1",
    "teacher-rank-final-deployment-20261020-v1",
    "teacher-rank-delta-deployment-20261020-v1",
    "source-zero-final-serving-20261020-",
    "predictive-final-deployment-20261020-",
    "S6-Kaggle-production-route-20261019-v1",
    "scripts/kaggle/",
    "coordinator-claim",
    "rsna-pointwise-serving-",
    "predictive-c0-portable-placement-20261020-v1/deployment.py",
    "rsna-predictive-final-",
)


def account_process(command):
    """Recognize dual-mode recovery launchers without blocking fitting-only work."""
    if any(marker in command for marker in ACCOUNT_MARKERS):
        return True
    launchers = (
        "predictive-v4-owner-handoff-20261020-v1/launch.py",
        "predictive-http-owner-handoff-20261020-v1/launch.py",
        "predictive-source-correction-owner-handoff-20261020-v1/launch.py",
        "predictive-http499-owner-handoff-20261020-v1/launch.py",
        "predictive-evc27-deadline-placement-20261020-v1/owner.py",
    )
    if not any(launcher in command for launcher in launchers):
        return False
    try:
        words = shlex.split(command)
    except ValueError:
        return True  # Unreadable matching owner command requires reconciliation.
    for i, word in enumerate(words):
        if any(word.endswith(launcher) for launcher in launchers):
            # Mode is the first positional CLI argument in the frozen launcher.
            # Unknown/missing mode is not evidence of account quiescence.
            return i + 1 >= len(words) or words[i + 1] != "fitting"
    return True


def check(root, processes=None, own_pid=None):
    root = Path(root)
    journal = root / "artifacts/pointwise-final-deployment-20261020-v1"
    intent_path = journal / "continuation-intent.json"
    intent = json.loads(intent_path.read_text())
    assert intent["original_source_git"] == "01cbe205080487b3fb39fbf7c0d28e112fcd7d77"
    assert type(intent["PID"]) is int and intent["PID"] > 0
    terminals = [
        p for p in (journal / "complete.json", journal / "continuation-stopped.json") if p.is_file()
    ]
    assert len(terminals) == 1, "Wait for the actual pointwise continuation terminal"
    body = terminals[0].read_bytes()
    value = json.loads(body)
    if terminals[0].name == "complete.json":
        assert value["outcome"] in ("scored", "failed")
    else:
        assert value["automatic_retry"] is False and isinstance(value["error_type"], str)
    if processes is None:
        processes = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    own_pid = os.getpid() if own_pid is None else own_pid
    active = []
    for row in processes.splitlines():
        parts = row.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, command = int(parts[0]), parts[1]
        assert pid != intent["PID"], "Original continuation PID still exists; reconcile it first"
        if pid != own_pid and account_process(command):
            active.append(pid)
    assert not active, "Other account owner/transport active; preserve its shared-file envelope"
    return {
        "status": "actual_local_account_owners_quiescent",
        "checked_at_unix": time.time(),
        "pointwise_continuation_PID": intent["PID"],
        "pointwise_terminal_name": terminals[0].name,
        "pointwise_terminal_sha256": hashlib.sha256(body).hexdigest(),
        "process_inventory_sha256": hashlib.sha256(processes.encode()).hexdigest(),
        "active_known_account_owners_or_transports": 0,
        "no_remote_mutation": True,
    }
