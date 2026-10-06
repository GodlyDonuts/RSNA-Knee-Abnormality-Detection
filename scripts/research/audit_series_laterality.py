"""Census per-series laterality and orientation headers across the corpus.

Reads a 64 KiB range prefix of one slice per series (headers only) from the HF
mirror and records Laterality/ImageLaterality/FrameLaterality plus the
orientation fields needed for anatomical canonicalization
(ImageOrientationPatient, ImagePositionPatient, PatientPosition, PixelSpacing).

The Laterality study-level audit found ~50% empty; this census measures
per-series coverage and whether geometry-based fallback is viable. JSONL
output is append-only and resumable.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom
import requests

RESOLVE = "https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/resolve/main/"
FIELDS = (
    "Laterality",
    "ImageLaterality",
    "FrameLaterality",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PatientPosition",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "Rows",
    "Columns",
    "SeriesDescription",
)
PREFIX_BYTES = 64 * 1024


def read_one(path: str, token: str) -> dict:
    headers = {"Range": f"bytes=0-{PREFIX_BYTES - 1}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(3):
        try:
            response = requests.get(RESOLVE + path, headers=headers, timeout=120)
            response.raise_for_status()
            dataset = pydicom.dcmread(io.BytesIO(response.content), stop_before_pixels=True)
            return {field: str(dataset.get(field, "") or "") for field in FIELDS} | {
                "parse_ok": True
            }
        except requests.RequestException:
            if attempt == 2:
                return {"parse_ok": False, "error": "download_failed"}
        except Exception:
            return {"parse_ok": False, "error": "parse_failed"}
    raise RuntimeError("Unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    inventory = pd.concat(
        [pd.read_parquet(path) for path in sorted(args.inventory.glob("*.parquet"))]
    )
    first = (
        inventory.sort_values(["split", "study_uid", "series_uid", "sop_uid"])
        .groupby("series_uid", sort=True)
        .head(1)
    )
    plane = pd.read_csv(args.series)
    frame = first.merge(
        plane[["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]],
        left_on="series_uid",
        right_on="SeriesInstanceUID",
        how="left",
    )
    if args.limit:
        frame = frame.head(args.limit)
    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            done.add(json.loads(line)["series_uid"])
    pending = frame[~frame.series_uid.isin(done)]
    token = os.environ.get("HF_TOKEN", "")
    start = time.monotonic()

    def work(row):
        record = {
            "split": row.split,
            "study_uid": row.study_uid,
            "series_uid": row.series_uid,
            "sop_uid": row.sop_uid,
            "anatomical_plane": row.Anatomical_Plane,
            "fluid_sensitive": row.Fluid_Sensitive,
        }
        return record | read_one(row.path, token)

    with args.output.open("a", buffering=1) as out:
        with ThreadPoolExecutor(max_workers=32) as pool:
            for count, record in enumerate(
                pool.map(work, pending.itertuples(index=False)), start=1
            ):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                if count % 500 == 0:
                    rate = count / (time.monotonic() - start)
                    print(
                        f"{len(done) + count}/{len(done) + len(pending)} series ({rate:.1f}/s)",
                        flush=True,
                    )
    print(json.dumps({"done": len(done) + len(pending), "elapsed_s": time.monotonic() - start}))


if __name__ == "__main__":
    main()
