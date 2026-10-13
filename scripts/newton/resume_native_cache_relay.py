"""Explicit missing-suffix recovery after terminal original native-cache relay."""

import argparse
import json
import os
import sys
from pathlib import Path

if __package__:
    from scripts.newton.native_cache_relay import (
        members_from_registry,
        sha,
        signature,
        source_stream,
        target_stream,
    )
else:
    from native_cache_relay import (
        members_from_registry,
        sha,
        signature,
        source_stream,
        target_stream,
    )


def suffix(members, receipt, count):
    names = list(members)
    if (
        receipt["verified_prefix_members"] != count
        or not 0 <= count < len(names)
        or receipt["verified_prefix_bytes"] != sum(members[n]["bytes"] for n in names[:count])
        or set(receipt["verified_signatures"]) != set(names[:count])
        or receipt["read_only"] is not True
        or receipt["unverified_last_member"] not in (None, names[count])
    ):
        raise ValueError("Exact original verified prefix required")
    return {name: members[name] for name in names[count:]}


def merge_suffix(root, staging, members, receipt):
    for name, prior in receipt["verified_signatures"].items():
        if (root / name).is_symlink() or signature(root / name) != prior:
            raise ValueError("Verified original prefix changed")
    partial = receipt["unverified_last_member"]
    if partial is not None:
        path = root / partial
        quarantine = root / "original-unverified-copy"
        quarantine.mkdir(mode=0o700)
        path.rename(quarantine / path.name)
    for name in members:
        source, target = staging / name, root / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.link(source, target)  # Exclusive: never replace a prefix or existing target.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("source", "target"))
    for name in ("registry", "root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--registry-sha", required=True)
    p.add_argument("--skip-prefix", type=int, required=True)
    p.add_argument("--proof", type=Path)
    p.add_argument("--proof-sha")
    p.add_argument("--prefix-receipt", type=Path)
    p.add_argument("--prefix-receipt-sha")
    p.add_argument("--node", required=True)
    args = p.parse_args()
    if os.uname().nodename != args.node or os.environ.get("SLURM_STEP_GPUS"):
        raise ValueError("Exact CPU-only endpoint required")
    all_members = members_from_registry(args.registry, args.registry_sha, 37812)
    names = list(all_members)
    if not 0 <= args.skip_prefix < len(names):
        raise ValueError("Only an incomplete original prefix can be recovered")
    missing = {name: all_members[name] for name in names[args.skip_prefix :]}
    if args.mode == "source":
        if sha(args.proof) != args.proof_sha:
            raise ValueError("Original full byte proof differs")
        proof = json.loads(args.proof.read_text())
        proof["signatures"] = {n: proof["signatures"][n] for n in missing}
        source_stream(sys.stdout.buffer, args.root, missing, proof, args.registry_sha)
    else:
        if sha(args.prefix_receipt) != args.prefix_receipt_sha:
            raise ValueError("Original private prefix receipt differs")
        receipt = json.loads(args.prefix_receipt.read_text())
        missing = suffix(all_members, receipt, args.skip_prefix)
        staging = args.root / "missing-suffix-recovery-v1"
        staging.mkdir(mode=0o700)
        with (staging / "intent.json").open("x") as f:
            json.dump(
                {
                    "prefix_receipt_sha": args.prefix_receipt_sha,
                    "original_registry_sha": args.registry_sha,
                    "missing_members": len(missing),
                    "automatic_retry": False,
                },
                f,
            )
        result = target_stream(
            sys.stdin.buffer,
            staging,
            missing,
            args.registry_sha,
            byte_cap=160 * 2**30,
            remaining_funding=190 * 2**30 + 128 * 2**20 - receipt["verified_prefix_bytes"],
        )
        merge_suffix(args.root, staging, missing, receipt)
        actual = {
            str(p.relative_to(args.root)) for p in (args.root / "cache").rglob("*") if p.is_file()
        }
        if actual != set(all_members):
            raise ValueError("Merged original member inventory differs")
        full = dict(
            result,
            status="original_complete_native_cache_private_suffix_recovery_verified",
            members=len(all_members),
            bytes=sum(r["bytes"] for r in all_members.values()),
            original_prefix_verified_bytes=receipt["verified_prefix_bytes"],
            only_missing_pixel_bytes_transferred=result["bytes"],
            node=args.node,
            CPU_job=os.environ["SLURM_JOB_ID"],
            no_Mac_or_shared_pixel_storage=True,
        )
        with (args.root / "full-cache-transfer-verified.json").open("x") as f:
            json.dump(full, f, indent=2)
        print(json.dumps(full))


if __name__ == "__main__":
    main()
