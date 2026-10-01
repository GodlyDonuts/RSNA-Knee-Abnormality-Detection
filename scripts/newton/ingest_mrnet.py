"""Verbatim private MRNet acquisition; source labels never become RSNA labels.

One CPU, zero GPUs. Read the read-only Redivis token from stdin. Preserve original
files on node-local storage, verify source MD5 and SHA256, and reject unsafe NPYs.
There is no automatic network retry, raw-data publication or model fitting here.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

TARGETS = ("acl", "meniscus", "abnormal")
PLANES = ("axial", "coronal", "sagittal")
SOURCE = "aimi.mrnet_knee_mri_s:4a2c:v1_0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def case_id(value):
    if not re.fullmatch(r"[0-9]{1,4}", value):
        raise ValueError("invalid_MRNet_case_id")
    return value.zfill(4)


def read_labels(path, target):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        # These are the actual Redivis v1.0 imports, including their omitted row.
        if reader.fieldnames not in (["_0000", "_0"], ["_0000", "_1"], ["_1130", "_0"]):
            raise ValueError("unaccepted_original_label_schema")
        left, right = reader.fieldnames
        result = {}
        for row in reader:
            uid = case_id(row[left])
            if uid in result or row[right] not in ("0", "1") or None in row:
                raise ValueError("duplicate_or_nonbinary_source_label")
            result[uid] = int(row[right])
    return result


def read_index(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["file_id", "file_name", "size", "added_at", "md5_hash"]:
            raise ValueError("unaccepted_original_file_schema")
        rows, seen_names, seen_ids = [], set(), set()
        for row in reader:
            name = row["file_name"]
            if not re.fullmatch(r"(axial|coronal|sagittal)/[0-9]{4}\.npy", name):
                raise ValueError("unsafe_or_unknown_MRNet_filename")
            if name in seen_names or row["file_id"] in seen_ids:
                raise ValueError("duplicate_MRNet_file")
            seen_names.add(name)
            seen_ids.add(row["file_id"])
            checksum = base64.b64decode(row["md5_hash"], validate=True)
            if len(checksum) != 16 or not row["size"].isdecimal():
                raise ValueError("invalid_source_size_or_checksum")
            size = int(row["size"])
            if not 65536 <= size <= 1024 * 256 * 256 + 65536:
                raise ValueError("unexpected_source_volume_size")
            rows.append(
                {"file_id": row["file_id"], "name": name, "bytes": size, "md5": checksum.hex()}
            )
    return sorted(rows, key=lambda row: row["name"])


def admit_sources(root):
    manifest = json.loads((root / "downloads.json").read_text())
    if {row["name"] for row in manifest} != {
        "train",
        "valid",
        *(f"{split}-{target}" for split in ("train", "valid") for target in TARGETS),
    } or len(manifest) != 8:
        raise ValueError("complete_eight_original_tables_required")
    for row in manifest:
        path = root / (row["name"] + ".csv")
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("original_table_bytes_changed")
        if not row["qualifiedReference"].startswith(SOURCE + "."):
            raise ValueError("wrong_MRNet_source_version")
    splits = {}
    for split, start, stop, missing in (("train", 0, 1130, "0000"), ("valid", 1130, 1250, "1130")):
        index = read_index(root / (split + ".csv"))
        cases = {f"{number:04d}" for number in range(start, stop)}
        if {row["name"] for row in index} != {
            f"{plane}/{uid}.npy" for uid in cases for plane in PLANES
        }:
            raise ValueError("exact_original_three_plane_case_membership_required")
        labels = {target: read_labels(root / f"{split}-{target}.csv", target) for target in TARGETS}
        if any(set(values) != cases - {missing} for values in labels.values()):
            raise ValueError("source_labels_do_not_match_observed_missing_case")
        splits[split] = {
            "files": index,
            "labels": labels,
            "unlabeled_cases": [missing],
            "supervised_cases": sorted(cases - {missing}),
            "images": len(index),
            "bytes": sum(row["bytes"] for row in index),
            "positive_counts": {target: sum(values.values()) for target, values in labels.items()},
            "training_permitted": split == "train",
        }
    return {
        "source": SOURCE,
        "target_order": list(TARGETS),
        "rsna_target_mapping": None,
        "generic_meniscus_is_not_medial_or_lateral": True,
        "geometry_unknown": True,
        "splits": splits,
    }


def validate_npy(path, expected_bytes):
    """Inspect the primitive NPY envelope without loading NumPy or unpickling."""
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("invalid_NPY_magic")
        version = stream.read(2)
        if version == b"\x01\x00":
            length = struct.unpack("<H", stream.read(2))[0]
        elif version == b"\x02\x00":
            length = struct.unpack("<I", stream.read(4))[0]
        else:
            raise ValueError("unaccepted_NPY_version")
        if length > 65536:
            raise ValueError("oversized_NPY_header")
        header = ast.literal_eval(stream.read(length).decode("latin1"))
        if set(header) != {"descr", "fortran_order", "shape"}:
            raise ValueError("unexpected_NPY_header")
        shape = header["shape"]
        if (
            header["descr"] not in ("|u1", "<u1", ">u1")
            or header["fortran_order"] is not False
            or type(shape) is not tuple
            or len(shape) != 3
            or any(type(value) is not int for value in shape)
            or not 1 <= shape[0] <= 1024
            or shape[1:] != (256, 256)
            or stream.tell() + shape[0] * 256 * 256 != expected_bytes
            or path.stat().st_size != expected_bytes
        ):
            raise ValueError("unsafe_or_unexpected_MRNet_uint8_volume")
    return list(shape)


class PrivateRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise ValueError("non_https_file_redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def acquire(row, split, scratch, token):
    path = scratch / split / row["name"]
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    md5, digest, size = hashlib.md5(), hashlib.sha256(), 0
    request = Request(
        "https://redivis.com/api/v1/rawFiles/" + quote(row["file_id"], safe=""),
        headers={"Authorization": "Bearer " + token},
    )
    partial = path.with_suffix(".partial")
    with build_opener(PrivateRedirect()).open(request, timeout=120) as response:
        if (
            response.status != 200
            or int(response.headers.get("Content-Length", -1)) != row["bytes"]
        ):
            raise ValueError("source_file_response_size_mismatch")
        with partial.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > row["bytes"]:
                    raise ValueError("source_file_exceeds_index_size")
                stream.write(chunk)
                digest.update(chunk)
                md5.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    if size != row["bytes"] or md5.hexdigest() != row["md5"]:
        raise ValueError("source_file_MD5_or_byte_mismatch")
    shape = validate_npy(partial, size)
    os.rename(partial, path)
    return {**row, "split": split, "sha256": digest.hexdigest(), "shape": shape}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--scratch-parent", type=Path, default=Path("/tmp"))
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    if sha(args.source_directory / "downloads.json") != args.manifest_sha256:
        raise ValueError("pinned_original_manifest_required")
    admitted = admit_sources(args.source_directory)
    if args.schema_only:
        print(json.dumps({"status": "original_MRNet_source_schema_accepted", "source": admitted}))
        return
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 1:
        raise ValueError("one_CPU_Slurm_allocation_required")
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS"):
        raise ValueError("no_GPU_acquisition_allowed")
    total = sum(split["bytes"] for split in admitted["splits"].values())
    if shutil.disk_usage(args.scratch_parent).free - total - 64 * 1024**2 < 20 * 1024**3:
        raise ValueError("funded_node_local_reserve_required")
    # Local intent prevents accidental concurrent requests; token is never saved.
    save_new(
        args.control / "ingestion-intent.json",
        {"job_id": os.environ["SLURM_JOB_ID"], "automatic_retry": False},
    )
    token = sys.stdin.readline().strip()
    if not token or len(token) > 8192:
        raise ValueError("read_only_token_required_on_stdin")
    scratch = Path(tempfile.mkdtemp(prefix="rsna-mrnet-original-", dir=args.scratch_parent))
    save_new(
        args.control / "scratch.json",
        {
            "path": str(scratch),
            "node": os.uname().nodename,
            "expected_bytes": total,
            "job_id": os.environ["SLURM_JOB_ID"],
        },
    )
    save_new(scratch / "source-admission.json", admitted)
    for source in args.source_directory.iterdir():
        if source.is_file():
            shutil.copyfile(source, scratch / source.name)
    rows = []
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(acquire, row, split, scratch, token)
                for split, spec in admitted["splits"].items()
                for row in spec["files"]
            ]
            with (scratch / "verified-files.jsonl").open("x") as journal:
                for future in as_completed(futures):
                    try:
                        row = future.result()
                    except Exception:
                        for pending in futures:
                            pending.cancel()
                        raise
                    rows.append(row)
                    journal.write(json.dumps(row, sort_keys=True) + "\n")
                    journal.flush()
                    if len(rows) % 100 == 0:
                        print(
                            json.dumps(
                                {
                                    "verified_files": len(rows),
                                    "verified_bytes": sum(row["bytes"] for row in rows),
                                }
                            ),
                            flush=True,
                        )
        token = ""
        if len(rows) != 3750 or sum(row["bytes"] for row in rows) != total:
            raise ValueError("incomplete_source_acquisition")
        receipt = {
            "status": "verbatim_private_MRNet_images_and_labels_verified",
            "job_id": os.environ["SLURM_JOB_ID"],
            "node": os.uname().nodename,
            "files": len(rows),
            "bytes": total,
            "source": SOURCE,
            "manifest_sha256": args.manifest_sha256,
            "file_journal_sha256": sha(scratch / "verified-files.jsonl"),
            "supervised_train_exams": 1129,
            "held_out_valid_exams": 119,
            "missing_labels_not_invented": True,
            "model_training_ran": False,
            "raw_data_published": False,
            "storage_scope": "private_node_local_noncommercial_research",
            "source_weight_publication_compatibility": "unresolved",
        }
        save_new(args.control / "accepted.json", receipt)
        print(json.dumps(receipt), flush=True)
    except Exception as error:
        # Never print authenticated URLs, HTTP response bodies or token strings.
        save_new(
            args.control / "failed.json",
            {
                "status": "acquisition_failed_reconcile_original_journal",
                "exception_type": type(error).__name__,
                "verified_files": len(rows),
                "automatic_retry": False,
            },
        )
        raise SystemExit("MRNet_acquisition_failed_reconcile_private_receipts") from None


if __name__ == "__main__":
    main()
