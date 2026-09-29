"""Evict a finite list of privately preserved redundant snapshots, with receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, record: dict) -> None:
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "Run file hashing on a CPU allocation"
    assert not args.receipt.exists() and not args.receipt.with_suffix(".intent.json").exists()
    plan = json.loads(args.plan.read_text())
    assert plan["private"] is True and plan["repo_id"] == "GodlyDonuts/RSNA-Data"
    assert plan["retain_all_final_checkpoints"] is True
    verified = []
    for record in plan["records"]:
        path = Path(record["local_path"])
        assert path.parent.name == "epoch-checkpoints" and path.name != "epoch-010.pt"
        assert path.is_relative_to("/home/sa305415/rsna-knee/runs")
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == record["bytes"]
        assert file_sha256(path) == record["sha256"]
        assert (path.parent.parent / "checkpoint.pt").is_file()
        verified.append(record)
    assert len({r["local_path"] for r in verified}) == len(verified)
    intent = {
        "status": "verified_before_eviction",
        "job_id": os.environ["SLURM_JOB_ID"],
        "plan_sha256": file_sha256(args.plan),
        "private_revision": plan["fresh_revision"],
        "records": verified,
        "bytes": sum(r["bytes"] for r in verified),
    }
    atomic_json(args.receipt.with_suffix(".intent.json"), intent)
    for record in verified:
        Path(record["local_path"]).unlink()
    atomic_json(args.receipt, dict(intent, status="redundant_snapshots_evicted"))
    print(
        json.dumps(
            {
                "status": "redundant_snapshots_evicted",
                "files": len(verified),
                "bytes": intent["bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
