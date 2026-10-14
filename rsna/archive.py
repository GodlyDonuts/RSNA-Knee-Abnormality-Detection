from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rsna.inventory import validate_metadata
from rsna.labels import read_reviews
from rsna.schema import TARGETS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def import_artifacts(root: Path, imports: list[dict]) -> None:
    destination = root / "reviews"
    provenance_path = destination / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    for record in imports:
        batch, phase = record["batch"], record["phase"]
        if not re.fullmatch(r"batch-\d{3}", batch) or phase not in ("draft", "reviewed"):
            raise ValueError("Invalid batch or phase")
        for field, prefix in (("session_id", "devin-"), ("agent_id", "wfar-")):
            if not re.fullmatch(prefix + r"[a-f0-9]{32}", record[field]):
                raise ValueError(f"Invalid {field}")
        source = Path(record["path"])
        path = destination / ("drafts" if phase == "draft" else "reviewed")
        path = path / f"{batch}-{phase}.jsonl"
        metadata = {
            key: record[key]
            for key in ("session_id", "agent_id", "workflow_run_id", "n_studies", "notes")
        }
        for field in ("rubric_sha256", "source_reports_sha256", "parent_artifact_sha256"):
            if field in record:
                if not re.fullmatch(r"[a-f0-9]{64}", record[field]):
                    raise ValueError(f"Invalid {field}")
                metadata[field] = record[field]
        metadata["sha256"] = sha256(source)
        metadata["path"] = str(path.relative_to(root))
        existing = provenance.get(batch, {}).get(phase)
        if existing is not None and existing != metadata:
            raise ValueError(f"Cannot replace archived provenance: {batch}/{phase}")
        if path.exists() and sha256(path) != metadata["sha256"]:
            raise ValueError(f"Cannot replace archived output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copyfile(source, path)
        provenance.setdefault(batch, {})[phase] = metadata
        write_json(provenance_path, provenance)


def archive_inventory(source: Path, root: Path) -> None:
    inventory = pd.read_parquet(source)
    validate_metadata(inventory, root / "source")
    destination = root / "source/inventory"
    destination.mkdir(parents=True, exist_ok=True)
    if list(destination.iterdir()):
        raise ValueError("Inventory destination must be empty")
    for index, start in enumerate(range(0, len(inventory), 100_000)):
        inventory.iloc[start : start + 100_000].to_parquet(
            destination / f"part-{index:03d}.parquet", index=False
        )
    if not inventory.equals(pd.read_parquet(destination)):
        raise ValueError("Sharded inventory differs from the original")
    write_json(
        root / "source/inventory-provenance.json",
        {
            "original_sha256": sha256(source),
            "rows": len(inventory),
            "partition": "Contiguous original row ranges of at most 100000 rows",
            "round_trip_equal": True,
            "source_revision": json.loads((root / "source/dicom_inventory.json").read_text())[
                "revision"
            ],
        },
    )


def review_index(root: Path, create_inputs: bool = False) -> dict:
    train = pd.read_csv(root / "source/train.csv").sort_values("StudyInstanceUID")
    if train.StudyInstanceUID.duplicated().any() or train.Report.isna().any():
        raise ValueError("Invalid source report identities")
    records = [
        {"study_uid": uid, "report": report}
        for uid, report in zip(train.StudyInstanceUID, train.Report)
    ]
    provenance = json.loads((root / "reviews/provenance.json").read_text())
    batches: list[dict] = []
    known_paths = set()
    for start in range(0, len(records), 50):
        name = f"batch-{start // 50:03d}"
        expected = records[start : start + 50]
        reports = {row["study_uid"]: row["report"] for row in expected}
        input_path = root / f"reviews/inputs/{name}.json"
        if create_inputs and not input_path.exists():
            input_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(input_path, expected)
        if json.loads(input_path.read_text()) != expected:
            raise ValueError(f"Blinded input differs from original reports: {name}")
        batch = {
            "batch": name,
            "n_studies": len(expected),
            "input": str(input_path.relative_to(root)),
            "input_sha256": sha256(input_path),
        }
        for phase, folder in (("draft", "drafts"), ("reviewed", "reviewed")):
            path = root / f"reviews/{folder}/{name}-{phase}.jsonl"
            known_paths.add(path)
            metadata = provenance.get(name, {}).get(phase)
            if not path.exists() and metadata is None:
                batch[phase] = {"status": "not_archived"}
                continue
            if metadata is None or not path.exists():
                raise ValueError(f"Artifact/provenance pair missing: {path}")
            if metadata["sha256"] != sha256(path):
                raise ValueError(f"Artifact differs from recorded agent output: {path}")
            if metadata["path"] != str(path.relative_to(root)):
                raise ValueError(f"Provenance path differs from actual artifact: {path}")
            if metadata["n_studies"] != len(expected):
                raise ValueError(f"Agent count differs from batch: {path}")
            for field in ("session_id", "agent_id", "workflow_run_id", "notes"):
                if not isinstance(metadata.get(field), str) or not metadata[field].strip():
                    raise ValueError(f"Missing review identity/completion provenance: {path}")
            try:
                read_reviews(path.parent, reports, f"{name}-{phase}")
                batch[phase] = {"status": "valid", "path": str(path.relative_to(root))}
            except (ValueError, KeyError, TypeError) as error:
                batch[phase] = {
                    "status": "invalid",
                    "path": str(path.relative_to(root)),
                    "error": str(error),
                }
        phases = provenance.get(name, {})
        if set(phases) == {"draft", "reviewed"} and any(
            phases["draft"][field] == phases["reviewed"][field]
            for field in ("session_id", "agent_id")
        ):
            batch["reviewed"] = {
                "status": "invalid",
                "path": f"reviews/reviewed/{name}-reviewed.jsonl",
                "error": "Reviewer must have a separate agent and session identity",
            }
        batches.append(batch)
    actual_paths = set((root / "reviews/drafts").glob("*.jsonl")) | set(
        (root / "reviews/reviewed").glob("*.jsonl")
    )
    if actual_paths - known_paths or set(provenance) - {batch["batch"] for batch in batches}:
        raise ValueError("Unknown batch artifacts/provenance")
    if any(set(phases) - {"draft", "reviewed"} for phases in provenance.values()):
        raise ValueError("Unknown provenance phase")
    counts = {
        phase: {
            status: sum(batch[phase]["status"] == status for batch in batches)
            for status in ("valid", "invalid", "not_archived")
        }
        for phase in ("draft", "reviewed")
    }
    return {
        "studies": len(records),
        "targets": list(TARGETS),
        "batch_size": 50,
        "counts": counts,
        "complete": all(counts[phase]["valid"] == len(batches) for phase in counts),
        "scope": "Report/evidence validation only; no diagnostic image rereading",
        "batches": batches,
    }


def snapshot_cache(root: Path, cache: Path) -> None:
    inventory = json.loads((root / "source/dicom_inventory.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted(cache.glob("*/*/*.json"))]
    write_json(
        root / "audits/cache-progress.json",
        {
            "snapshot_utc": datetime.now(timezone.utc).isoformat(),
            "cache_path_on_originating_vm": str(cache),
            "source_revision": inventory["revision"],
            "expected_series": inventory["unique_series"],
            "expected_slices": inventory["files"],
            "metadata_records": len(records),
            "slices_recorded_by_writer": sum(row["n_slices"] for row in records),
            "cache_bytes_recorded_by_writer": sum(row["cache_bytes"] for row in records),
            "npz_files_observed": len(list(cache.glob("*/*/*.npz"))),
            "series_by_split": dict(Counter(row["split"] for row in records)),
            "series_by_transfer_syntax": dict(
                Counter(syntax for row in records for syntax in row["transfer_syntaxes"])
            ),
            "warning_series_counts": dict(
                Counter(warning for row in records for warning in row["warnings"])
            ),
            "full_audit_passed": False,
            "scope": "Live writer-metadata snapshot, not a full cache/content audit",
        },
    )


def file_manifest(root: Path) -> dict:
    manifest = {}
    for path in sorted(root.rglob("*")):
        # Python imports create local, version-specific bytecode. It is not an
        # original audit artifact and cannot be reproduced by a clean checkout.
        if "__pycache__" in path.relative_to(root).parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"Archive contains a symlink: {path}")
        if path.is_file() and path != root / "manifest.json":
            manifest[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive and verify the initiative handoff bundle")
    parser.add_argument("--root", type=Path, default=Path("initiative"))
    parser.add_argument("--imports", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--cache-progress", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if (args.imports or args.inventory or args.cache_progress) and not args.refresh:
        parser.error("Imports/inventory/progress require --refresh")
    if args.imports:
        import_artifacts(args.root, json.loads(args.imports.read_text()))
    if args.inventory:
        archive_inventory(args.inventory, args.root)
    if args.cache_progress:
        snapshot_cache(args.root, args.cache_progress)
    index = review_index(args.root, create_inputs=args.refresh)
    if args.refresh:
        write_json(args.root / "reviews/index.json", index)
        write_json(args.root / "manifest.json", file_manifest(args.root))
    else:
        if json.loads((args.root / "reviews/index.json").read_text()) != index:
            raise ValueError("Review index differs from current artifacts")
        if json.loads((args.root / "manifest.json").read_text()) != file_manifest(args.root):
            raise ValueError("Archive file checksums or membership differ")
    print(json.dumps({key: value for key, value in index.items() if key != "batches"}, indent=2))
    if args.require_complete and not index["complete"]:
        raise ValueError("Both report-review stages must be complete and valid")


if __name__ == "__main__":
    main()
