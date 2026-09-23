"""Extract only the eight deterministic update-23 studies and verify CPU inputs."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

SOURCE = Path("/home/sa305415/rsna-knee/code-stopfix-1f486765")
CONTROL = Path("/home/sa305415/rsna-knee/diagnostics/update23-v1")
REQUEST = CONTROL / "request.json"
RESULT = CONTROL / "result.json"
CACHE = CONTROL / "cache"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if RESULT.exists() or CACHE.exists() or CACHE.is_symlink():
        raise ValueError("Diagnostic output already exists")
    torch.set_num_threads(1)
    request = json.loads(REQUEST.read_text())
    assert request["schema_version"] == 1 and request["update_zero_based"] == 22
    assert len(request["studies"]) == 8 and request["rank_count"] == 2
    sources = {
        "studies.parquet": SOURCE / "initiative/releases/report-supervision-v1/studies.parquet",
        "fold.csv": SOURCE / "initiative/experiments/competitive-v1/folds/fold.csv",
        "series_sampling.csv": SOURCE
        / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv",
        "series_cache_index.parquet": SOURCE
        / "initiative/audits/full-image-recovery-v1/audit/series_cache_index.parquet",
    }
    for name, path in sources.items():
        assert digest(path) == request["source_sha256"][name], name
    tar_path = Path(request["tar_path"])
    assert tar_path.is_file() and tar_path.stat().st_size == request["tar_bytes"]
    expected = {row["name"]: row for row in request["members"]}
    assert len(expected) == len(request["members"]) == 98
    CACHE.mkdir(mode=0o700)
    found: dict[str, dict] = {}
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive:
            row = expected.get(member.name)
            if row is None:
                continue
            if (
                member.name in found
                or not member.isfile()
                or not member.name.startswith("cache/train_series/")
            ):
                raise ValueError("Duplicate or unsafe selected TAR member")
            relative = Path(member.name).relative_to("cache")
            if ".." in relative.parts or relative.is_absolute():
                raise ValueError("Unconfined selected TAR member")
            if "bytes" in row and member.size != row["bytes"]:
                raise ValueError("Selected cache volume size mismatch")
            output = CACHE / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            reader = archive.extractfile(member)
            if reader is None:
                raise ValueError("Selected TAR member is unreadable")
            h = hashlib.sha256()
            count = 0
            with reader, output.open("xb") as writer:
                while block := reader.read(1024 * 1024):
                    writer.write(block)
                    h.update(block)
                    count += len(block)
                writer.flush()
                os.fsync(writer.fileno())
            if count != member.size or ("sha256" in row and h.hexdigest() != row["sha256"]):
                raise ValueError("Selected cache volume checksum mismatch")
            found[member.name] = {"bytes": count, "sha256": h.hexdigest()}
    if set(found) != set(expected):
        raise ValueError("Selected cache members missing from full verified archive")

    sys.path.insert(0, str(SOURCE))
    from rsna.competitive.data import CompetitiveStudyDataset
    from rsna.competitive.train import select_development

    manifest = pd.read_parquet(sources["studies.parquet"])
    folds = pd.read_csv(sources["fold.csv"])
    series = pd.read_csv(sources["series_sampling.csv"])
    train, _ = select_development(manifest, folds, 0)
    assert len(train) == 2247
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(20260922)).tolist()
    selected = []
    for row in request["studies"]:
        index = order[22 * 8 + row["micro"] * 2 + row["rank"]]
        assert index == row["train_index"] and train.iloc[index].study_uid == row["study_uid"]
        selected.append(train.iloc[index])
    dataset = CompetitiveStudyDataset(
        pd.DataFrame(selected), series, CACHE, slices=16, side=224, max_series=6
    )
    validated = []
    for index, row in enumerate(request["studies"]):
        sample = dataset[index]
        image = sample["images"]
        assert image.shape == (6, 16, 1, 224, 224)
        assert torch.isfinite(image).all() and (image >= 0).all() and (image <= 1).all()
        assert sample["series_mask"].any() and torch.equal(
            sample["slice_mask"].any(dim=1), sample["series_mask"]
        )
        assert torch.isfinite(sample["positions"]).all()
        validated.append(
            {
                "study_uid": row["study_uid"],
                "rank": row["rank"],
                "micro": row["micro"],
                "valid_series": int(sample["series_mask"].sum()),
                "valid_slices": int(sample["slice_mask"].sum()),
                "observed_targets": int(sample["label_mask"].sum()),
                "image_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
            }
        )
    result = {
        "status": "eight_update23_studies_cpu_cache_and_inputs_verified",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_sha256": digest(REQUEST),
        "source_git": request["source_git"],
        "archive_bytes": tar_path.stat().st_size,
        "archive_expected_sha256_from_verified_receipt": request["tar_sha256"],
        "extracted_files": len(found),
        "extracted_bytes": sum(row["bytes"] for row in found.values()),
        "files": found,
        "studies": validated,
        "gpu_requested": False,
    }
    with RESULT.open("x") as writer:
        json.dump(result, writer, sort_keys=True, indent=2)
        writer.write("\n")
        writer.flush()
        os.fsync(writer.fileno())
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "extracted_files", "extracted_bytes", "gpu_requested")
            }
        )
    )


if __name__ == "__main__":
    main()
