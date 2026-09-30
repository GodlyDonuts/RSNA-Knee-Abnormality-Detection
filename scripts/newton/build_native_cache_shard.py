"""Rebuild a bounded paired cache shard from pinned original DICOMs.

This worker does not publish artifacts or reserve GPUs. Production orchestration
must supply an owned node-local output and preserve/verify it before reclamation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi

import rsna.competitive.data
import rsna.data
import rsna.dicom
import rsna.preprocess
import rsna.schema
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.dicom import decode_slice, make_volume
from rsna.preprocess import Source
from rsna.schema import CACHE_VERSION

SCIENCE = "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
SOURCE_REVISION = "0b5e410288232f570c69ba4f44723356e6ae6ab7"
REPO = "GodlyDonuts/RSNA-Data"
INVENTORY = "initiative/audits/final-release-technical-inputs-v1/inventory_effective.parquet.gz"
INVENTORY_GZ_SHA = "33065e6a60b878e5eda90ff2ffe865b5778cdcfa668adab23f2fd65bccdde240"
INVENTORY_SHA = "68d202eb951babc9365eddd2f7a1219201cd7b0905f04afdace785ad0e079537"
MANIFEST_SHA = "44e6775c5413e42d8744ead0f7d7bb0de258893a10867123e47fbd770f84793f"
GIB = 1024**3
SIDES = (256, 384)


def digest(path: Path) -> str:
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


def safe_path(name: str, suffix: str | None = None) -> str:
    p = PurePosixPath(name)
    if (
        not name
        or not p.parts
        or p.is_absolute()
        or ".." in p.parts
        or "\\" in name
        or str(p) != name
        or (suffix is not None and p.suffix != suffix)
    ):
        raise ValueError("unsafe_or_unexpected_member_path")
    return name


def verify_bytes(data: bytes, record: dict) -> bytes:
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("source_or_reference_bytes_changed")
    return data


def load_plan(root: Path, manifest_path: Path, number: int) -> tuple[dict, pd.DataFrame]:
    if manifest_path.is_symlink() or digest(manifest_path) != MANIFEST_SHA:
        raise ValueError("full_cohort_manifest_binding_changed")
    manifest = json.loads(manifest_path.read_text())
    name = f"shard-{number:03d}.json"
    record = manifest["shards"][name]
    path = manifest_path.parent / name
    shard = json.loads(verify_bytes(path.read_bytes(), record))
    for key in (
        "scientific_source_git",
        "source_revision",
        "selection_sha256",
        "input_bindings",
        "cache_sides",
        "inventory_parquet_sha256",
        "training_uids",
        "outer_uids",
    ):
        if shard[key] != manifest[key]:
            raise ValueError("shard_global_binding_changed")
    if shard["scientific_source_git"] != SCIENCE or shard["source_revision"] != SOURCE_REVISION:
        raise ValueError("unaccepted_science_or_source")
    if shard["cache_sides"] != list(SIDES) or shard["shard"] != number:
        raise ValueError("unexpected_shard_or_cache_sides")
    for name, record in shard["input_bindings"].items():
        path = root / safe_path(name)
        if (
            path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or digest(path) != record["sha256"]
        ):
            raise ValueError("accepted_input_binding_changed")
    data_gz = (root / INVENTORY).read_bytes()
    if hashlib.sha256(data_gz).hexdigest() != INVENTORY_GZ_SHA:
        raise ValueError("corrected_inventory_required")
    data = gzip.decompress(data_gz)
    if hashlib.sha256(data).hexdigest() != INVENTORY_SHA:
        raise ValueError("corrected_inventory_unpacked_binding_changed")
    inventory = pd.read_parquet(io.BytesIO(data))
    if len(inventory) != 819635 or inventory.path.duplicated().any():
        raise ValueError("effective_inventory_membership_changed")
    selected = inventory.loc[inventory.series_uid.isin({r["series_uid"] for r in shard["series"]})]
    return shard, selected


def source_members(shard: dict, inventory: pd.DataFrame) -> dict[str, list[dict]]:
    rows = shard["series"]
    if (
        not rows
        or len({r["series_uid"] for r in rows}) != len(rows)
        or len({r["cache_path"] for r in rows}) != len(rows)
        or sorted({r["study_uid"] for r in rows}) != shard["study_uids"]
        or len(rows) != shard["series_count"]
        or sum(r["n_slices"] for r in rows) != shard["slices"]
    ):
        raise ValueError("shard_membership_changed")
    groups = {str(k): g for k, g in inventory.groupby("series_uid", sort=False)}
    if set(groups) != {r["series_uid"] for r in rows}:
        raise ValueError("inventory_series_membership_changed")
    output = {}
    for row in rows:
        safe_path(row["cache_path"], ".npz")
        if row["cache_path"] != f"train_series/{row['study_uid']}/{row['series_uid']}.npz":
            raise ValueError("cache_path_identity_changed")
        objects = groups[row["series_uid"]].sort_values("path").to_dict("records")
        if (
            len(objects) != row["n_slices"]
            or sum(o["bytes"] for o in objects) != row["source_bytes"]
            or len({o["path"] for o in objects}) != len(objects)
            or len({o["sop_uid"] for o in objects}) != len(objects)
            or any(
                o["study_uid"] != row["study_uid"] or o["split"] != "train_series" for o in objects
            )
        ):
            raise ValueError("source_series_membership_changed")
        for obj in objects:
            safe_path(obj["path"], ".dcm")
            if (
                obj["path"]
                != f"train_series/{row['study_uid']}/{row['series_uid']}/{obj['sop_uid']}.dcm"
            ):
                raise ValueError("source_filename_sop_binding_changed")
        output[row["series_uid"]] = objects
    if sum(r["source_bytes"] for r in rows) != shard["source_bytes"]:
        raise ValueError("shard_source_byte_count_changed")
    return output


class ReferenceArchive:
    """Index only; every selected reference object is fully hash checked on read."""

    def __init__(self, archive: Path, rows: list[dict]):
        if archive.is_symlink() or not archive.is_file():
            raise ValueError("regular_accepted_reference_archive_required")
        self.archive = archive
        self.records = {}
        for row in rows:
            stem = str(PurePosixPath(safe_path(row["cache_path"], ".npz")).with_suffix(""))
            self.records["cache/" + stem + ".npz"] = {
                "bytes": row["cache_bytes"],
                "sha256": row["cache_sha256"],
            }
            self.records["cache/" + stem + ".json"] = {
                "bytes": row["cache_metadata_bytes"],
                "sha256": row["cache_metadata_sha256"],
            }
        self.offsets = {}
        with tarfile.open(archive, "r:") as tar:
            for member in tar:
                if member.name in self.records:
                    if (
                        not member.isfile()
                        or member.size != self.records[member.name]["bytes"]
                        or member.name in self.offsets
                    ):
                        raise ValueError("invalid_or_duplicate_reference_member")
                    self.offsets[member.name] = (member.offset_data, member.size)
        if self.offsets.keys() != self.records.keys():
            raise ValueError("missing_reference_member")

    def read(self, name: str) -> bytes:
        offset, size = self.offsets[name]
        with self.archive.open("rb") as stream:
            stream.seek(offset)
            return verify_bytes(stream.read(size), self.records[name])


def archive_tree(root: Path, target: Path) -> dict:
    records = {}
    with tarfile.open(target, "x", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("unexpected_cache_symlink")
            if not path.is_file():
                continue
            if path.suffix not in (".npz", ".json"):
                raise ValueError("unexpected_cache_member")
            name = safe_path("cache/" + path.relative_to(root).as_posix())
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = path.stat().st_size, 0o600, 0
            with path.open("rb") as stream:
                tar.addfile(item, stream)
            records[name] = {"bytes": item.size, "sha256": digest(path)}
    return {
        "bytes": target.stat().st_size,
        "sha256": digest(target),
        "members": records,
        "regular_members": len(records),
        "unpacked_file_bytes": sum(r["bytes"] for r in records.values()),
    }


def verify_archive(archive: Path, expected: dict, destination: Path | None = None) -> None:
    """Full byte/member validation; optional fresh extraction without tar.extract."""
    if (
        archive.is_symlink()
        or archive.stat().st_size != expected["bytes"]
        or digest(archive) != expected["sha256"]
    ):
        raise ValueError("cache_archive_bytes_changed")
    members = expected["members"]
    if (
        len(members) != expected["regular_members"]
        or sum(r["bytes"] for r in members.values()) != expected["unpacked_file_bytes"]
    ):
        raise ValueError("cache_archive_manifest_changed")
    for name in members:
        if not safe_path(name).startswith("cache/") or PurePosixPath(name).suffix not in (
            ".npz",
            ".json",
        ):
            raise ValueError("unexpected_cache_member")
    with tarfile.open(archive, "r:") as tar:
        listed = tar.getmembers()
        if (
            len(listed) != len(members)
            or len({m.name for m in listed}) != len(listed)
            or any(
                not m.isfile() or m.name not in members or m.size != members[m.name]["bytes"]
                for m in listed
            )
        ):
            raise ValueError("cache_archive_membership_changed")
        if destination is not None:
            destination.mkdir(mode=0o700, parents=False)
        for member in listed:
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError("unreadable_cache_member")
            with stream:
                data = verify_bytes(stream.read(), members[member.name])
            if destination is not None:
                path = destination / member.name
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with path.open("xb") as output:
                    output.write(data)


def compare_metadata(info: dict, metadata: dict, *, native_side: int) -> None:
    for key, value in info.items():
        if not (key == "resized_shape" and native_side != 256) and metadata.get(key) != value:
            raise ValueError("source_geometry_pixel_or_intensity_binding_changed:" + key)


def compare_reference(volume: np.ndarray, info: dict, old: np.ndarray, metadata: dict) -> dict:
    compare_metadata(info, metadata, native_side=256)
    if old.dtype != np.uint16 or old.shape != volume.shape:
        raise ValueError("accepted_reference_volume_shape_or_dtype_changed")
    difference = np.abs(volume.astype(np.int32) - old.astype(np.int32))
    maximum = int(difference.max())
    if maximum > 1:
        raise ValueError("native256_reconstruction_exceeds_accepted_rounding_band")
    return {
        "max_absolute_uint16_difference": maximum,
        "differing_pixels": int(np.count_nonzero(difference)),
        "exact": maximum == 0,
        "cause_proven": False,
    }


def check_loaders(shard: dict, roots: dict[int, Path]) -> dict:
    table = pd.DataFrame(
        [
            {
                "StudyInstanceUID": r["study_uid"],
                "SeriesInstanceUID": r["series_uid"],
                "Anatomical_Plane": r["plane"],
                "Fluid_Sensitive": r["fluid_sensitive"],
                "Fat_Suppression": r["fat_suppression"],
                "Selection_Tier": r["selection_tier"],
            }
            for r in shard["series"]
        ]
    )
    studies = pd.DataFrame({"study_uid": shard["study_uids"]})
    report = {}
    for cache_side, input_side in ((256, 224), (384, 384)):
        dataset = CompetitiveStudyDataset(
            studies,
            table,
            roots[cache_side],
            slices=16,
            side=input_side,
            max_series=6,
            augment=False,
        )
        valid = 0
        for i in range(len(dataset)):
            item = dataset[i]
            image = item["images"]
            if (
                item["study_uid"] != studies.iloc[i].study_uid
                or tuple(image.shape) != (6, 16, 1, input_side, input_side)
                or not torch.isfinite(image).all()
                or image.min() < 0
                or image.max() > 1
                or item["label_mask"].any()
            ):
                raise ValueError("unmodified_mri_loader_acceptance_failed")
            valid += int(item["slice_mask"].sum())
        report[str(cache_side)] = {
            "studies": len(dataset),
            "valid_sampled_slices": valid,
            "source_cache_side": dataset.cache_side,
            "input_side": input_side,
            "original_source_revision": dataset.source_revision,
            "labels_read": False,
        }
    return report


def build_shard(
    shard: dict,
    inventory: pd.DataFrame,
    reader: Source,
    reference: ReferenceArchive,
    output: Path,
    execution_provenance: dict | None = None,
) -> dict:
    objects = source_members(shard, inventory)
    output.mkdir(mode=0o700, parents=False)
    save_new(
        output / "build-intent.json",
        {
            "selection_sha256": shard["selection_sha256"],
            "shard": shard["shard"],
            "source_revision": SOURCE_REVISION,
            "scientific_source_git": SCIENCE,
            "once_only": True,
            "target_values_or_reports_read": False,
        },
    )
    roots = {side: output / f"side-{side}" / "cache" for side in SIDES}
    records = []
    with (output / "series-proofs.jsonl").open("x") as journal:
        for number, row in enumerate(shard["series"], 1):
            items = objects[row["series_uid"]]
            names = [o["path"] for o in items]
            raw = reader.read_series(names)
            if len(raw) != len(items):
                raise ValueError("missing_downloaded_source_object")
            decoded = [
                decode_slice(
                    verify_bytes(data, obj), row["study_uid"], row["series_uid"], obj["sop_uid"]
                )
                for data, obj in zip(raw, items)
            ]
            del raw
            old_meta = json.loads(
                reference.read(
                    "cache/" + str(PurePosixPath(row["cache_path"]).with_suffix(".json"))
                )
            )
            for key, value in {
                "study_uid": row["study_uid"],
                "series_uid": row["series_uid"],
                "side": 256,
                "source_revision": SOURCE_REVISION,
                "cache_version": CACHE_VERSION,
                "n_slices": row["n_slices"],
                "cache_path": row["cache_path"],
                "cache_bytes": row["cache_bytes"],
                "cache_sha256": row["cache_sha256"],
                "split": "train_series",
                "path_fingerprint": hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest(),
            }.items():
                if old_meta.get(key) != value:
                    raise ValueError("accepted_reference_identity_changed:" + key)
            files = {}
            difference = None
            for side in SIDES:
                volume, info = make_volume(decoded, side)
                if volume.dtype != np.uint16 or volume.shape != (row["n_slices"], side, side):
                    raise ValueError("rebuilt_volume_shape_or_dtype_changed")
                if side == 256:
                    with np.load(
                        io.BytesIO(reference.read("cache/" + row["cache_path"])), allow_pickle=False
                    ) as old:
                        difference = compare_reference(volume, info, old["volume"], old_meta)
                else:
                    compare_metadata(info, old_meta, native_side=side)
                destination = roots[side] / row["cache_path"]
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    np.savez_compressed(stream, volume=volume)
                info.update(
                    {
                        "cache_version": CACHE_VERSION,
                        "source_revision": SOURCE_REVISION,
                        "path_fingerprint": old_meta["path_fingerprint"],
                        "study_uid": row["study_uid"],
                        "series_uid": row["series_uid"],
                        "split": "train_series",
                        "side": side,
                        "n_slices": row["n_slices"],
                        "cache_path": row["cache_path"],
                        "cache_sha256": digest(destination),
                        "cache_bytes": destination.stat().st_size,
                    }
                )
                save_new(destination.with_suffix(".json"), info)
                files[str(side)] = {
                    "npz": {"bytes": destination.stat().st_size, "sha256": info["cache_sha256"]},
                    "metadata": {
                        "bytes": destination.with_suffix(".json").stat().st_size,
                        "sha256": digest(destination.with_suffix(".json")),
                    },
                }
            entry = {
                "study_uid": row["study_uid"],
                "series_uid": row["series_uid"],
                "slices": row["n_slices"],
                "source_bytes": row["source_bytes"],
                "source_membership_sha256": hashlib.sha256(
                    json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "accepted256_difference": difference,
                "files": files,
            }
            journal.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
            records.append(entry)
            if (
                sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                > shard["owned_cache_archive_scratch_bound_bytes"]
            ):
                raise ValueError("owned_scratch_bound_exceeded")
            print(
                json.dumps(
                    {
                        "shard": shard["shard"],
                        "series_completed": number,
                        "series_total": len(shard["series"]),
                    }
                ),
                flush=True,
            )
    loaders = check_loaders(shard, roots)
    archives = {}
    for side in SIDES:
        path = output / f"cache-side-{side}.tar"
        record = archive_tree(roots[side], path)
        if (
            record["bytes"] > shard["archive_upper_bytes_by_side"][str(side)]
            or record["regular_members"] != shard["regular_members_per_side"]
        ):
            raise ValueError("archive_byte_or_member_bound_exceeded")
        verify_archive(path, record)
        archives[path.name] = record
    report = {
        "schema_version": 1,
        "status": "native_paired_shard_cache_and_loader_verified",
        "selection_sha256": shard["selection_sha256"],
        "shard": shard["shard"],
        "scientific_source_git": SCIENCE,
        "source_revision": SOURCE_REVISION,
        "studies": len(shard["study_uids"]),
        "series": len(records),
        "slices": shard["slices"],
        "source_bytes_verified": shard["source_bytes"],
        "archives": archives,
        "loaders": loaders,
        "source_proofs": {
            "bytes": (output / "series-proofs.jsonl").stat().st_size,
            "sha256": digest(output / "series-proofs.jsonl"),
        },
        "max_reference_uint16_difference": max(
            r["accepted256_difference"]["max_absolute_uint16_difference"] for r in records
        ),
        "rounding_cause_proven": False,
        "target_values_or_reports_read": False,
        "accepted_cache_labels_masks_folds_mutated": False,
        "full_cohort_built": False,
        "model_training_or_accuracy_claim": False,
        "execution_provenance": execution_provenance or {"scope": "local_fixture_or_library_call"},
    }
    if (
        sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        + len(json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode())
        + 1
        > shard["owned_cache_archive_scratch_bound_bytes"]
    ):
        raise ValueError("owned_scratch_bound_exceeded")
    save_new(output / "cache-verified.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--science-binding", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_GPUS")
        or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 1
    ):
        raise ValueError("one_cpu_zero_gpu_allocation_required")
    science = json.loads(args.science_binding.read_text())
    code = Path(science["checkout"])
    loaded = {
        "rsna/dicom.py": rsna.dicom,
        "rsna/preprocess.py": rsna.preprocess,
        "rsna/competitive/data.py": rsna.competitive.data,
        "rsna/data.py": rsna.data,
        "rsna/schema.py": rsna.schema,
    }
    if not set(loaded).issubset(science["files"]):
        raise ValueError("complete_scientific_module_binding_required")
    for name, module in loaded.items():
        if Path(module.__file__).resolve() != (code / name).resolve():
            raise ValueError("worker_imported_unbound_scientific_module")
    if (
        subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
        != SCIENCE
    ):
        raise ValueError("scientific_checkout_changed")
    for name, record in science["files"].items():
        path = code / safe_path(name)
        committed = subprocess.check_output(["git", "-C", str(code), "show", SCIENCE + ":" + name])
        if (
            path.is_symlink()
            or digest(path) != record["sha256"]
            or hashlib.sha256(committed).hexdigest() != record["sha256"]
        ):
            raise ValueError("scientific_module_changed")
    if (
        args.output.exists()
        or args.output.parent.stat().st_dev == Path("/home/sa305415/rsna-knee").stat().st_dev
    ):
        raise ValueError("fresh_owned_node_local_output_required")
    shard, inventory = load_plan(args.root, args.plan_manifest, args.shard)
    if (
        shutil.disk_usage(args.output.parent).free
        < shard["owned_cache_archive_scratch_bound_bytes"] + 20 * GIB
    ):
        raise ValueError("node_local_disk_reserve_required")
    if args.reference_archive.stat().st_size != 82834493440:
        raise ValueError("accepted_reference_archive_size_changed")
    payload = json.load(sys.stdin)
    token = payload.pop("token")
    if not token.startswith("hf_") or payload:
        raise ValueError("private_credential_stdin_required")
    api = HfApi(token=token)
    if api.repo_info(REPO, repo_type="dataset", revision=SOURCE_REVISION).private is not True:
        raise ValueError("private_original_source_required")
    groups = source_members(shard, inventory)
    objects = {o["path"]: o for group in groups.values() for o in group}
    paths = sorted(objects)
    for start in range(0, len(paths), 500):
        batch = paths[start : start + 500]
        info = list(
            api.get_paths_info(REPO, paths=batch, revision=SOURCE_REVISION, repo_type="dataset")
        )
        if (
            len(info) != len(batch)
            or {v.path for v in info} != set(batch)
            or any(
                v.size != objects[v.path]["bytes"]
                or not v.lfs
                or v.lfs.sha256 != objects[v.path]["sha256"]
                for v in info
            )
        ):
            raise ValueError("private_original_object_binding_changed")
    os.environ["HF_TOKEN"] = token
    reader = Source(None, objects, repo_id=REPO, revision=SOURCE_REVISION)
    reader.downloads.shutdown(wait=True)
    reader.downloads = ThreadPoolExecutor(max_workers=2)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        report = build_shard(
            shard,
            inventory,
            reader,
            ReferenceArchive(args.reference_archive, shard["series"]),
            args.output,
            {
                "job_id": os.environ["SLURM_JOB_ID"],
                "cpus": 1,
                "gpus": 0,
                "worker_sha256": digest(Path(__file__)),
                "science_binding_sha256": digest(args.science_binding),
                "plan_manifest_sha256": MANIFEST_SHA,
                "shard_file_sha256": digest(
                    args.plan_manifest.parent / f"shard-{args.shard:03d}.json"
                ),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "accepted_reference_scope": "full_size_and_all_selected_member_bytes_verified",
            },
        )
        print(
            json.dumps(
                {"status": report["status"], "shard": args.shard, "series": report["series"]}
            ),
            flush=True,
        )
    finally:
        reader.downloads.shutdown(wait=True)
        os.environ.pop("HF_TOKEN", None)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(
            json.dumps(
                {"status": "native_shard_worker_failed", "error_type": type(error).__name__}
            ),
            flush=True,
        )
        raise SystemExit(2) from None
