"""Bounded private source intake, with no clinical records in published output."""

import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import requests

REPO = "YiHui0124/KneeCoT"
REVISION = "e5944fd466f4b05a3141e67eae78a22a871fd062"
MAX_BYTES = 1024**3


def validate_manifest(value):
    if (
        value["repo"] != REPO
        or value["revision"] != REVISION
        or value["split"] != "train_data"
        or value["owner_agreement_confirmed"] is not True
    ):
        raise ValueError("Pinned approved private research source required")
    seen, groups = set(), {}
    for row in value["files"]:
        name = row["path"]
        match = re.fullmatch(r"train_data/(GJB[0-9]+T)/\1(?:\.json|_[0-9]+\.nii)", name)
        if not match or name in seen or type(row["bytes"]) is not int or row["bytes"] <= 0:
            raise ValueError("Unsafe, repeated or empty canonical source member")
        if not re.fullmatch(r"[0-9a-f]{40}", row["blob_id"]):
            raise ValueError("Pinned Git object identity required")
        if row["sha256"] is not None and not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid LFS identity")
        seen.add(name)
        groups.setdefault(match[1], []).append(name)
    if len(groups) != 2 or value["exams"] != 2:
        raise ValueError("Exactly two complete engineering-source exams required")
    for names in groups.values():
        if sum(n.endswith(".json") for n in names) != 1 or not any(
            n.endswith(".nii") for n in names
        ):
            raise ValueError("Complete report plus MRI volumes required")
    total = sum(r["bytes"] for r in value["files"])
    if total != value["source_bytes"] or total > MAX_BYTES:
        raise ValueError("Source cap or declared byte count differs")
    return total


def nifti_header(raw, file_bytes):
    if len(raw) < 348:
        raise ValueError("Incomplete NIfTI header")
    endian = next((e for e in ("<", ">") if struct.unpack(e + "i", raw[:4])[0] == 348), None)
    if endian is None or raw[344:348] != b"n+1\x00":
        raise ValueError("Only explicit single-file NIfTI-1 accepted in this intake")
    dim = struct.unpack_from(endian + "8h", raw, 40)
    rank = dim[0]
    if rank not in (3, 4) or any(d <= 0 or d > 4096 for d in dim[1 : rank + 1]):
        raise ValueError("Invalid or unsupported volume shape")
    datatype, bitpix = struct.unpack_from(endian + "2h", raw, 70)
    types = {2: 8, 4: 16, 8: 32, 16: 32, 64: 64, 128: 24, 256: 8, 512: 16, 768: 32}
    offset = struct.unpack_from(endian + "f", raw, 108)[0]
    if (
        datatype not in types
        or bitpix != types[datatype]
        or not math.isfinite(offset)
        or offset < 352
        or offset != int(offset)
        or int(offset) + math.prod(dim[1 : rank + 1]) * bitpix // 8 > file_bytes
    ):
        raise ValueError("Volume datatype or declared pixel envelope differs")
    # Do not expose descrip, aux_file, intent_name or any other free-text header.
    return {"shape": list(dim[1 : rank + 1]), "datatype_code": datatype, "bitpix": bitpix}


def main():
    payload = json.load(sys.stdin)
    manifest = payload["manifest"]
    total = validate_manifest(manifest)
    if (
        os.uname().nodename != "evc31"
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_JOB_GPUS")
    ):
        raise ValueError("Owning-node one-CPU zero-GPU intake required")
    root = Path("/tmp/rsna-kneecot-private-intake-20261006-v1")
    if root.exists() or shutil.disk_usage("/tmp").free < (20 + 50 + 3) * 2**30:
        raise ValueError("Original intent exists or private reserve is insufficient")
    root.mkdir(mode=0o700)
    (root / "intent.json").write_text(
        json.dumps(
            {"manifest": manifest, "CPU_job": os.environ["SLURM_JOB_ID"], "automatic_retry": False},
            indent=2,
        )
        + "\n"
    )
    rows, headers, reports = [], [], 0
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + payload["token"]
    for row in manifest["files"]:
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        sha = hashlib.sha256()
        git = hashlib.sha1(b"blob " + str(row["bytes"]).encode() + b"\x00")
        count = 0
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row['path']}"
        with session.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            if response.url.split(":", 1)[0] != "https":
                raise ValueError("HTTPS source required")
            with path.open("xb") as output:
                for block in response.iter_content(1024**2):
                    count += len(block)
                    if count > row["bytes"]:
                        raise ValueError("Source member exceeds exact declared bytes")
                    output.write(block)
                    sha.update(block)
                    git.update(block)
        if count != row["bytes"] or (
            sha.hexdigest() != row["sha256"] if row["sha256"] else git.hexdigest() != row["blob_id"]
        ):
            raise ValueError("Original pinned source bytes differ")
        if path.suffix == ".nii":
            with path.open("rb") as stream:
                headers.append(nifti_header(stream.read(352), count))
        else:
            if count > 4 * 1024**2 or not isinstance(json.loads(path.read_text()), dict):
                raise ValueError("Bounded original report object required")
            reports += 1
        rows.append({"path": row["path"], "bytes": count, "sha256": sha.hexdigest()})
    private = {"files": rows, "headers": headers, "clinical_reports": reports}
    (root / "private-verified.json").write_text(json.dumps(private, indent=2) + "\n")
    receipt = {
        "status": "private_KneeCoT_two_exam_original_bytes_and_NIfTI_headers_verified",
        "repo": REPO,
        "revision": REVISION,
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "exams": 2,
        "MRI_volumes": len(headers),
        "reports": reports,
        "verified_source_bytes": total,
        "private_receipt_sha256": hashlib.sha256(
            (root / "private-verified.json").read_bytes()
        ).hexdigest(),
        "owner_agreement_confirmed": True,
        "node_retained_envelope_bytes": 3 * 2**30,
        "new_shared_bytes": 0,
        "GPU_model_or_clinical_inference_executed": False,
        "clinical_records_exported": False,
        "anatomical_alignment_or_label_correctness_proven": False,
    }
    (root / "sanitized-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    os.umask(0o077)
    main()
