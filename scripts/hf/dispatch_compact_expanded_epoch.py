"""One CPU-only publication of a completed compact epoch, never an automatic retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

from scripts.hf.dispatch_native384_epoch import credential, save_new, ssh

ROOT = Path(__file__).resolve().parents[2]
REMOTE_CONTROL = "/tmp/rsna-compact-expanded2592-engine-20261016-v1/production-control-v1"
REMOTE_PYTHON = "/home/sa305415/rsna-knee/hf-epoch-publisher-v1/env-cp312/bin/python"
TRAINING = "/tmp/rsna-compact-expanded2592-production-20261016-v1"
REPO = "GodlyDonuts/RSNA-Data"
PREFIX = "experiments/compact-expanded2592/fold0-a1ca2385-seed20260922"
HOST = "sa305415@newton.ist.ucf.edu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epoch", type=int)
    parser.add_argument("--ssh-control", type=Path, required=True)
    args = parser.parse_args()
    if args.epoch not in range(1, 17) or not args.ssh_control.is_socket():
        raise ValueError("Completed epoch1–16 and live SSH control socket required")
    folder = ROOT / "artifacts/compact-native-n3-launch-20261016-v1/expanded2592-publication"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent, result = (
        folder / f"epoch-{args.epoch:03d}-{name}.json" for name in ("intent", "result")
    )
    if intent.exists() or result.exists():
        raise ValueError("Existing local publication attempt; reconcile read-only")
    publisher = ROOT / "scripts/hf/publish_compact_expanded_epoch.py"
    digest = hashlib.sha256(publisher.read_bytes()).hexdigest()
    # Separate ordinary CPU allocation: no GPU or training process interruption.
    precheck = r"""
import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);expected=sys.argv[2];epoch=int(sys.argv[3]);root=Path(sys.argv[4])
assert os.uname().nodename=='evc33'
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
assert not (root/f'publication/epoch-{epoch:03d}').exists()
side=root/f'production/epoch-checkpoints/epoch-{epoch:03d}.json'
value=json.loads(side.read_text());b=value['binding']
assert value['epoch']==epoch and value['optimizer_steps']==324*epoch
assert b['diagnostic_only'] is False and b['fitting_count']==2592 and b['total_steps']==5184
assert b==json.loads((root/'production/run.json').read_text())['binding']
print(json.dumps({'epoch':epoch,'checkpoint_sha256':value['checkpoint_sha256'],'checkpoint_bytes':value['checkpoint_bytes']}))
"""
    remote = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--nodelist=evc33",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=2G",
        "--time=00:02:00",
        "--job-name=rsna-expanded2592-epoch-precheck",
        REMOTE_PYTHON,
        "-B",
        "-c",
        precheck,
        f"{REMOTE_CONTROL}/publish_compact_expanded_epoch.py",
        digest,
        str(args.epoch),
        TRAINING,
    ]
    snapshot = json.loads(ssh(args.ssh_control, remote, timeout=150))
    token = credential()
    api = HfApi(token=token)
    if api.repo_info(REPO, repo_type="dataset").private is not True:
        raise ValueError("Private repository required")
    prefix = f"{PREFIX}/epoch-{args.epoch:03d}"
    paths = [f"{prefix}/{name}" for name in ("checkpoint.pt", "manifest.json", "snapshot.json")]
    if list(api.get_paths_info(REPO, paths=paths, repo_type="dataset")):
        raise ValueError("Existing private epoch path; reconcile read-only")
    save_new(
        intent,
        {
            "epoch": args.epoch,
            "publisher_sha256": digest,
            "snapshot": snapshot,
            "CPU": 1,
            "GPU": 0,
            "prefix": prefix,
            "automatic_retry": False,
        },
    )
    command = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--nodelist=evc33",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=2G",
        "--time=00:30:00",
        f"--job-name=rsna-expanded2592-publish-{args.epoch}",
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONNOUSERSITE=1",
        "HF_HOME=/tmp/rsna-compact-expanded2592-engine-20261016-v1/production-control-v1/hf-cache",
        REMOTE_PYTHON,
        "-B",
        f"{REMOTE_CONTROL}/publish_compact_expanded_epoch.py",
        "publish",
        str(args.epoch),
    ]
    completed = subprocess.run(
        ["ssh", "-S", str(args.ssh_control), "-o", "BatchMode=yes", HOST, shlex.join(command)],
        input=json.dumps({"token": token}),
        capture_output=True,
        text=True,
        timeout=1900,
    )
    record = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    save_new(result, record)
    print(json.dumps(record), flush=True)
    if completed.returncode:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
