"""Private CPU-only advisory image signatures from original admitted volumes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

from rsna.pixel_overlap import volume_signature


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def volumes(source):
    root = Path(source["pixel_root"])
    proof = Path(source["inventory_path"])
    if sha(proof) != source["inventory_sha256"]:
        raise ValueError("Original admitted source inventory changed")
    if source["role"] == "stanford":
        records = [json.loads(line) for line in proof.read_text().splitlines()]
        if len(records) != 3750:
            raise ValueError("Complete original Stanford pixel inventory required")
        for row in sorted(records, key=lambda r: (r["split"], r["name"])):
            if row["split"] not in ("train", "valid") or not re.fullmatch(
                r"(axial|coronal|sagittal)/[0-9]{4}\.npy", row["name"]
            ):
                raise ValueError("Original three-plane pixel paths required")
            yield {
                "study": Path(row["name"]).stem,
                "volume": row["split"] + "/" + row["name"],
                "plane": row["name"].split("/")[0],
                "split": row["split"],
                "path": root / row["split"] / row["name"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "shape": row["shape"],
            }
    else:
        registry = json.loads(proof.read_text())
        if (
            registry["series_count"] != 18906
            or registry["study_count"] != 3403
            or registry["cache_side"] != 384
        ):
            raise ValueError("Original eligible native384 development pool required")
        eligible = set(registry["training_uids"]) | set(registry["outer_uids"])
        seen = set()
        for shard in registry["shards"]:
            for row in shard["series"]:
                relative = Path(row["cache_path"])
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or relative.suffix != ".npz"
                    or row["study_uid"] not in eligible
                    or row["cache_path"] in seen
                ):
                    raise ValueError("Original unique eligible pixel paths required")
                seen.add(row["cache_path"])
                original = shard["members"]["cache/" + row["cache_path"]]
                yield {
                    "study": row["study_uid"],
                    "volume": row["series_uid"],
                    "plane": row["plane"],
                    "path": root / relative,
                    "bytes": original["bytes"],
                    "sha256": original["sha256"],
                    "shape": [row["n_slices"], 384, 384],
                }
        if len(seen) != 18906:
            raise ValueError("Incomplete eligible series enumeration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.uname().nodename != source["node"]
        or source["role"] not in ("stanford", "rsna")
    ):
        raise ValueError("One CPU, zero GPU on the original owning node required")
    if shutil.disk_usage(args.output.parent).free < 20 * 1024**3 + 64 * 1024**2:
        raise ValueError("Private retained index must preserve local disk reserve")
    started = time.monotonic()
    count = size = slices = sampled_count = 0
    studies = set()
    with args.output.open("x") as stream:
        for row in volumes(source):
            path = row.pop("path")
            if path.is_symlink() or path.stat().st_size != row["bytes"]:
                raise ValueError("Changed original pixel asset")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Original pixel hash changed during necessary decoding")
            if source["role"] == "stanford":
                volume = np.load(io.BytesIO(raw), allow_pickle=False)
                expected_dtype = np.uint8
            else:
                with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                    volume = archive["volume"]
                expected_dtype = np.uint16
            if list(volume.shape) != row["shape"] or volume.dtype != expected_dtype:
                raise ValueError("Original primitive stack envelope changed")
            samples = volume_signature(volume)
            stream.write(json.dumps({**row, "samples": samples}, sort_keys=True) + "\n")
            count += 1
            studies.add(row["study"])
            size += len(raw)
            slices += len(volume)
            sampled_count += len(samples)
            if count % 250 == 0:
                stream.flush()
                print(
                    json.dumps(
                        {"completed_volumes": count, "elapsed_s": time.monotonic() - started}
                    ),
                    flush=True,
                )
            if args.output.stat().st_size > 64 * 1024**2:
                raise ValueError("Private pixel index exceeds its separate funded cap")
    expected = 3750 if source["role"] == "stanford" else 18906
    if count != expected or len(studies) != (1250 if source["role"] == "stanford" else 3403):
        raise ValueError("Complete original pixel-only source coverage required")
    receipt = {
        "status": "original_pixel_signature_index_complete_not_overlap_conclusion",
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": os.uname().nodename,
        "source_git": source["source_git"],
        "role": source["role"],
        "inventory_sha256": source["inventory_sha256"],
        "studies": len(studies),
        "volumes": count,
        "original_bytes_decoded": size,
        "original_slices": slices,
        "sampled_slices": sampled_count,
        "index_path": str(args.output),
        "index_sha256": sha(args.output),
        "index_bytes": args.output.stat().st_size,
        "elapsed_s": time.monotonic() - started,
        "clinical_labels_read": False,
        "protected_RSNA_cohorts_read": False,
        "source_pixels_published": False,
        "independence_established": False,
    }
    with args.receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
