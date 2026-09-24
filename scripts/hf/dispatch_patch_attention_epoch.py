"""Stream one local HF credential to a CPU-only Newton epoch publisher.

An intent makes an uncertain dispatch terminal until its remote state is
reconciled read-only. This command never retries an epoch upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/home/sa305415/rsna-knee"
HOST = "sa305415@newton.ist.ucf.edu"
CONTROL = f"{REMOTE}/production-launch-203f419-patchattention-v1"
RUN = f"{REMOTE}/runs/dinov3-l16-patchattention-fold0-203f419-production-v1"
CODE = f"{REMOTE}/code-patch-attention-203f419"
SCRIPT = f"{CONTROL}/publish_patch_attention_epoch.py"
PYTHON = f"{REMOTE}/hf-epoch-publisher-v1/env-cp312/bin/python"
LICENSE = f"{REMOTE}/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md"
JOURNAL = f"{CONTROL}/epoch-publication"
PREFIX = "experiments/dinov3-v1/training-epochs/patchattention-fold0-203f419-seed20260922"
REPO = "GodlyDonuts/RSNA-Data"
PUBLISHER_SHA = "f040c22f09c34c4cdc1d7fdd96219f1a99a863b866889f3f8b53905f8b3601a3"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def credential() -> str:
    env = ROOT / ".env"
    if env.stat().st_mode & 0o077:
        raise ValueError("Local credential file must be private")
    values = {}
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    token = values.get("HF_TOKEN", "")
    if not token.startswith("hf_"):
        raise ValueError("Hugging Face token unavailable")
    return token


def ssh(socket: Path, remote_args: list[str], *, timeout: int = 30) -> str:
    command = [
        "ssh",
        "-S",
        str(socket),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        HOST,
        shlex.join(remote_args),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise ValueError(f"Read-only Newton precheck failed: {result.returncode}")
    return result.stdout.strip()


def inspect_snapshot(socket: Path, epoch: int) -> dict:
    code = """
import hashlib,json,sys
from pathlib import Path
sidecar=Path(sys.argv[1]); run=Path(sys.argv[2])
record=json.loads(sidecar.read_text()); binding=json.loads((run/'run.json').read_text())['binding']
checkpoint=sidecar.parent/record['checkpoint_file']
assert record['status']=='scheduled_epoch_snapshot'
assert record['epoch']==int(sys.argv[3]) and record['optimizer_steps']==281*int(sys.argv[3])
assert record['fold']==0 and record['diagnostic_only'] is False
assert checkpoint.is_file() and not checkpoint.is_symlink()
assert checkpoint.stat().st_size==record['checkpoint_bytes']
assert binding['config']['pooling']=='patch_attention'
assert binding['config']['submission_every_epochs']==1
assert binding['config']['diagnostic_study_limit'] is None
print(json.dumps({'sidecar_sha256':hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                  'checkpoint_sha256':record['checkpoint_sha256'],
                  'checkpoint_bytes':record['checkpoint_bytes'],
                  'epoch':record['epoch']}))
"""
    sidecar = f"{RUN}/epoch-checkpoints/epoch-{epoch:03d}.json"
    output = ssh(socket, ["python3", "-c", code, sidecar, RUN, str(epoch)])
    return json.loads(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epoch", type=int)
    parser.add_argument("--ssh-control", type=Path, required=True)
    args = parser.parse_args()
    if args.epoch not in range(1, 11) or not args.ssh_control.is_socket():
        raise ValueError("Completed epoch 1-10 and active SSH control socket required")
    if sha(ROOT / "scripts/hf/publish_patch_attention_epoch.py") != PUBLISHER_SHA:
        raise ValueError("Reviewed publisher source changed")
    if ssh(args.ssh_control, ["sha256sum", SCRIPT]).split()[0] != PUBLISHER_SHA:
        raise ValueError("Staged Newton publisher changed")
    if ssh(args.ssh_control, ["git", "-C", CODE, "rev-parse", "HEAD"]) != (
        "203f419ffac058496cdc3fbce0976bcedbb63286"
    ):
        raise ValueError("Training source changed")
    epoch = args.epoch
    snapshot = inspect_snapshot(args.ssh_control, epoch)
    folder = ROOT / "artifacts/hf-patch-attention-epochs-v1"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    intent = folder / f"epoch-{epoch:03d}-dispatch-intent.json"
    result_path = folder / f"epoch-{epoch:03d}-dispatch-result.json"
    if intent.exists() or result_path.exists():
        raise ValueError("Epoch dispatch already attempted; reconcile read-only")
    journal_exists = ssh(
        args.ssh_control,
        [
            "python3",
            "-c",
            "from pathlib import Path; import sys; print(int(Path(sys.argv[1]).exists()))",
            f"{JOURNAL}/epoch-{epoch:03d}",
        ],
        timeout=20,
    )
    if journal_exists == "1":
        raise ValueError("Remote publication journal already exists")
    token = credential()
    api = HfApi(token=token)
    if api.repo_info(REPO, repo_type="dataset").private is not True:
        raise ValueError("Hugging Face dataset is not private")
    prefix = f"{PREFIX}/epoch-{epoch:03d}"
    paths = [
        f"{prefix}/{name}"
        for name in ("checkpoint.pt", "snapshot.json", "DINOv3-LICENSE.md", "manifest.json")
    ]
    if list(api.get_paths_info(REPO, paths=paths, repo_type="dataset")):
        raise ValueError("Private epoch path already exists")
    save_new(
        intent,
        {
            "schema_version": 1,
            "status": "single_cpu_publication_dispatch_intent",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "publisher_sha256": PUBLISHER_SHA,
            "snapshot": snapshot,
            "repo_id": REPO,
            "prefix": prefix,
            "cpus": 1,
            "gpus": 0,
            "credential_transport": "local_ignored_env_to_ssh_stdin_to_srun_stdin_only",
        },
    )
    publisher = [
        PYTHON,
        SCRIPT,
        "publish",
        "--fold",
        "0",
        "--snapshot",
        f"{RUN}/epoch-checkpoints/epoch-{epoch:03d}.json",
        "--run",
        RUN,
        "--license-file",
        LICENSE,
        "--journal",
        JOURNAL,
        "--code-root",
        CODE,
    ]
    shell = (
        "set -euo pipefail; cache=$(mktemp -d /tmp/rsna-patch-epoch.XXXXXXXX); "
        'trap \'rm -rf "$cache"\' EXIT; export HF_HOME="$cache" '
        "HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONDONTWRITEBYTECODE=1; " + shlex.join(publisher)
    )
    remote = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=8G",
        "--time=01:30:00",
        "/bin/bash",
        "-c",
        shell,
    ]
    command = [
        "ssh",
        "-S",
        str(args.ssh_control),
        "-o",
        "BatchMode=yes",
        HOST,
        shlex.join(remote),
    ]
    outcome = subprocess.run(
        command, input=json.dumps({"token": token}), capture_output=True, text=True, timeout=6000
    )
    secret_pattern = re.compile(r"hf_[A-Za-z0-9]{10,}")
    public = {
        "schema_version": 1,
        "status": "cpu_publication_returned"
        if outcome.returncode == 0
        else "publication_uncertain_or_failed",
        "epoch": epoch,
        "remote_exit": outcome.returncode,
        "stdout_tail": secret_pattern.sub("[redacted]", outcome.stdout[-2500:]),
        "stderr_tail": secret_pattern.sub("[redacted]", outcome.stderr[-1200:]),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_new(result_path, public)
    print(json.dumps(public, sort_keys=True))
    if outcome.returncode:
        raise SystemExit(outcome.returncode)


if __name__ == "__main__":
    main()
