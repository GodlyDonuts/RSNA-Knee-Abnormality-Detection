"""Restore preserved checkpoint bytes on CPU storage, without republishing weights."""

import hashlib
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path("/home/sa305415/rsna-knee")
RUN = ROOT / "runs/dinov3-l16-fold2-b74060b-1gpu-production-v1/epoch-checkpoints"
CONTROL = ROOT / "kaggle/launches/checkpoint-comparison-inputguard-v1"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("CPU-only allocation required")
    secret = json.load(sys.stdin)
    if set(secret) != {"token"} or not secret["token"].startswith("hf_"):
        raise ValueError("One private Hugging Face token required on stdin")
    for epoch, source in (
        (8, ROOT / "kaggle/bundles/fold2-epoch008-eb280864bd1c-metadata-v1/dataset/checkpoint.pt"),
        (
            6,
            ROOT
            / "kaggle/bundles/fold2-epoch006-532331e0fd06-e89f604-pixelfix-v1/dataset/checkpoint.pt",
        ),
    ):
        target = RUN / f"epoch-{epoch:03d}.pt"
        meta = json.loads(target.with_suffix(".json").read_text())
        if target.exists() or target.is_symlink():
            raise ValueError("Existing restore needs read-only reconciliation")
        if (
            source.is_symlink()
            or source.stat().st_size != meta["checkpoint_bytes"]
            or digest(source) != meta["checkpoint_sha256"]
        ):
            raise ValueError("Preserved source bytes differ")
        os.link(source, target)
        result = {
            "epoch": epoch,
            "status": "verified_hardlink_restored",
            "sha256": meta["checkpoint_sha256"],
            "bytes": meta["checkpoint_bytes"],
        }
        (CONTROL / f"restore-epoch-{epoch:03d}.json").open("x").write(
            json.dumps(result, sort_keys=True) + "\n"
        )
        print(json.dumps(result), flush=True)
    for record in json.loads((CONTROL / "restore-sources.json").read_text()):
        epoch = record["epoch"]
        target = RUN / f"epoch-{epoch:03d}.pt"
        partial = target.with_suffix(".restore-partial")
        meta = json.loads(target.with_suffix(".json").read_text())
        if target.exists() or target.is_symlink() or partial.exists():
            raise ValueError("Existing restore or partial needs reconciliation")
        if (meta["checkpoint_sha256"], meta["checkpoint_bytes"]) != (
            record["sha256"],
            record["bytes"],
        ):
            raise ValueError("Pinned private publication differs from accepted sidecar")
        url = f"https://huggingface.co/datasets/{record['repo_id']}/resolve/{record['revision']}/{record['path']}"
        h = hashlib.sha256()
        count = 0
        try:
            with requests.get(
                url,
                headers={"Authorization": "Bearer " + secret["token"]},
                stream=True,
                timeout=(20, 120),
            ) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Private checkpoint download HTTP {response.status_code}")
                with partial.open("xb") as handle:
                    for block in response.iter_content(4 * 1024**2):
                        if block:
                            handle.write(block)
                            h.update(block)
                            count += len(block)
                            if count > record["bytes"]:
                                raise ValueError("Checkpoint exceeded its published size")
                    handle.flush()
                    os.fsync(handle.fileno())
        except requests.RequestException:
            raise RuntimeError(
                "Private transfer interrupted; reconcile partial bytes before retry"
            ) from None
        if h.hexdigest() != record["sha256"] or count != record["bytes"]:
            raise ValueError("Private checkpoint transfer failed size/hash verification")
        partial.replace(target)
        result = {
            "epoch": epoch,
            "status": "verified_private_download_restored",
            "sha256": h.hexdigest(),
            "bytes": count,
            "revision": record["revision"],
        }
        (CONTROL / f"restore-epoch-{epoch:03d}.json").open("x").write(
            json.dumps(result, sort_keys=True) + "\n"
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
