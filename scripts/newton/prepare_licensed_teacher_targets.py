"""One-time private fitting-target/teacher extraction from admitted feature bytes.

No new MRI/model/optimizer/network operations. Never publish the resulting UID,
source-label or teacher arrays in Git or small Hugging Face milestone mirrors.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np

INPUT_SHA = "a4c5b4910dc818293cdfb6ed59ba92bfa16faa5e0cb2c39c7a5e5b41df877ab0"
INDEX_SHA = "5a1a4c77be60a46973c9513b7115ba68edfc7bf76c6205a89de61217a75bf113"
UID_SHA = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_targets(path, uids, labels, masks, teacher):
    y = np.asarray(labels, dtype=np.float32)
    mask = np.asarray(masks)
    p = np.asarray(teacher)
    n = len(uids)
    if (
        not n
        or len(set(uids)) != n
        or y.shape != (n, 12)
        or mask.dtype != np.bool_
        or mask.shape != y.shape
        or p.dtype != np.float32
        or p.shape != (n, 2, 12)
        or not np.isfinite(y[mask]).all()
        or ((y[mask] < 0) | (y[mask] > 1)).any()
        or not np.isfinite(p).all()
    ):
        raise ValueError(
            "Unique aligned private FP32 fitting labels/teacher and bool masks required"
        )
    path = Path(path)
    with path.open("xb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        np.savez_compressed(handle, uids=np.asarray(uids), labels=y, masks=mask, parent_logits=p)
    if path.stat().st_size > 2 * 2**20:
        raise ValueError("Compact training-target file exceeds its2MiB bound")
    with np.load(path, allow_pickle=False) as arrays:
        if (
            set(arrays.files) != {"uids", "labels", "masks", "parent_logits"}
            or arrays["uids"].tolist() != uids
            or not np.array_equal(arrays["labels"], y, equal_nan=True)
            or not np.array_equal(arrays["masks"], mask)
            or not np.array_equal(arrays["parent_logits"], p)
        ):
            raise ValueError("Private target/teacher roundtrip differs")
    return {
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "studies": n,
        "source_arrays_sha256": hashlib.sha256(y.tobytes() + mask.tobytes()).hexdigest(),
        "teacher_array_sha256": hashlib.sha256(p.tobytes()).hexdigest(),
    }


def prepare(inputs, output, eligible_release):
    if digest(inputs) != INPUT_SHA or Path(output).exists():
        raise ValueError("Exact original source manifest and fresh output required")
    value = json.loads(Path(inputs).read_text())
    if (
        digest(value["index"]) != INDEX_SHA
        or value["uids_sha256"] != UID_SHA
        or value["clinical_evaluation_used"] is not False
        or value["source_rows_filtered_before_numeric_read"] is not True
        or value["independent_internal_validation"] is not False
    ):
        raise ValueError("Original fitting-only feature/label boundary required")
    if any(digest(name) != expected for name, expected in value["source_files"].items()):
        raise ValueError("Original frozen teacher numerical source changed")
    rows = json.loads(Path(value["index"]).read_text())["records"]
    uids = [r["study_uid"] for r in rows]
    if (
        len(uids) != 3887
        or len(set(uids)) != 3887
        or hashlib.sha256(json.dumps(uids, separators=(",", ":")).encode()).hexdigest() != UID_SHA
    ):
        raise ValueError("Exact original3887 fitting population required")
    if digest(eligible_release) != value["binding"]["eligible_release_sha256"]:
        raise ValueError("Original eligible release changed")
    import pyarrow.dataset as ds

    eligible = (
        ds.dataset(eligible_release, format="parquet")
        .to_table(
            columns=["study_uid"],
            filter=(ds.field("split") == "train") & ds.field("report_supervision_allowed"),
        )
        .column("study_uid")
        .to_pylist()
    )
    if set(eligible) != set(uids):
        raise ValueError("Training-eligible membership differs")
    teacher = []
    signatures: dict[Path, tuple] = {}
    members = {}
    for row in rows:
        root = Path(value["roots"][row.get("source_kernel", value.get("default_source_kernel"))])
        name = row["file"]
        if (
            name != hashlib.sha256(row["study_uid"].encode()).hexdigest() + ".npz"
            or not 0 < row["bytes"] <= 2**20
        ):
            raise ValueError("Exact bounded feature member required")
        archive = row.get("archive")
        if archive not in (None, "features.tar"):
            raise ValueError("Only original admitted streaming feature archives required")
        path = root / (archive or name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Original regular feature file required")
        s = path.stat()
        sig = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if path in signatures and signatures[path] != sig:
            raise ValueError("Original admitted feature file changed")
        signatures[path] = sig
        if archive and path not in members:
            with tarfile.open(path, "r:") as f:
                table = f.getmembers()
            members[path] = {m.name: m for m in table}
            if len(members[path]) != len(table) or any(
                not m.isreg() or Path(m.name).name != m.name or not 0 < m.size <= 2**20
                for m in table
            ):
                raise ValueError("Unsafe original feature archive member table")
        with path.open("rb") as f:
            if archive:
                member = members[path][name]
                if member.size != row["bytes"]:
                    raise ValueError("Original feature member size differs")
                f.seek(member.offset_data)
            raw = f.read(row["bytes"])
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Original teacher member bytes differ")
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            p = arrays["parent_logits"]
            if p.dtype != np.float32 or p.shape != (2, 12) or not np.isfinite(p).all():
                raise ValueError("Exact original two-view FP32 teacher logits required")
            teacher.append(p.copy())
    Path(output).mkdir(mode=0o700)
    proof = write_targets(
        Path(output) / "targets.private.npz",
        uids,
        value["labels"],
        np.asarray(value["masks"]),
        np.stack(teacher),
    )
    for path, expected in signatures.items():
        s = path.stat()
        if (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) != expected:
            raise ValueError("Original teacher file changed during necessary extraction")
    receipt = {
        "status": "all3887_original_source_targets_and_two_view_teacher_logits_compacted_exactly",
        "CPU_job": os.environ.get("SLURM_JOB_ID"),
        "original_input_sha256": INPUT_SHA,
        "original_index_sha256": INDEX_SHA,
        "original_UID_sha256": UID_SHA,
        "parent_weights_sha256": value["binding"]["parent_weights_sha256"],
        "parent_source_sha256": value["binding"]["parent_source_sha256"],
        "label_source_sha256": value["binding"]["label_source_sha256"],
        "eligible_release_sha256": value["binding"]["eligible_release_sha256"],
        "targets": proof,
        "only_original_fitting_members_used": True,
        "all_view_logits_and_stored_source_arrays_bit_equal": True,
        "new_MRI_or_encoder_or_optimizer_or_network": False,
        "clinical_or_competition_accuracy_claim": False,
        "private_target_payload_not_in_Git_or_small_HF_mirror": True,
    }
    (Path(output) / "accepted.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eligible-release", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.inputs, args.output, args.eligible_release)), flush=True)


if __name__ == "__main__":
    main()
