"""Build the SSL pretraining series manifest (knee-ssl-pretraining-v1).

Joins the source-object inventory (per-series slice counts, sha-bound) with
the series-headers census (plane, protocol flags, geometry, laterality) into
one pinned parquet the SSL job consumes. Read-only over archived inputs; no
target values, reports or labels are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "initiative/source/inventory"
CENSUS = ROOT / "initiative/audits/series-headers-census-v1/series-headers.jsonl.gz"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lateral(row: pd.Series) -> str | None:
    for col in ("Laterality", "ImageLaterality", "FrameLaterality"):
        value = row[col]
        if isinstance(value, str) and value.strip() in {"L", "R"}:
            return value.strip()
    return None


def _spacing(value: object) -> tuple[float | None, float | None]:
    if isinstance(value, str) and value.startswith("["):
        try:
            parts = [float(v) for v in value.strip("[]").split(",")]
            return (parts[0], parts[1]) if len(parts) == 2 else (None, None)
        except ValueError:
            return None, None
    return None, None


def build() -> pd.DataFrame:
    inventory = pd.concat([pd.read_parquet(p) for p in sorted(INVENTORY.glob("part-*.parquet"))])
    counts = (
        inventory.groupby(["split", "study_uid", "series_uid"])
        .size()
        .rename("n_slices")
        .reset_index()
    )
    census = pd.read_json(CENSUS, lines=True)
    merged = counts.merge(
        census, on=["split", "study_uid", "series_uid"], how="left", validate="one_to_one"
    )
    if len(merged) != len(counts):
        raise ValueError("Census merge changed series membership")
    merged["laterality"] = merged.apply(_lateral, axis=1)
    spacing = merged["PixelSpacing"].map(_spacing)
    merged["pixel_spacing_row"] = [v[0] for v in spacing]
    merged["pixel_spacing_col"] = [v[1] for v in spacing]
    merged["anatomical_plane"] = merged["anatomical_plane"].fillna("unknown")
    merged["header_available"] = merged["parse_ok"] == True  # noqa: E712
    columns = [
        "split",
        "study_uid",
        "series_uid",
        "n_slices",
        "anatomical_plane",
        "fluid_sensitive",
        "laterality",
        "pixel_spacing_row",
        "pixel_spacing_col",
        "Rows",
        "Columns",
        "SliceThickness",
        "SeriesDescription",
        "header_available",
    ]
    return merged[columns].sort_values(["split", "study_uid", "series_uid"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.output, index=False)
    provenance = {
        "schema_version": 1,
        "manifest_sha256": _sha(args.output),
        "series": len(manifest),
        "slices": int(manifest.n_slices.sum()),
        "studies": int(manifest.study_uid.nunique()),
        "inputs": {
            "inventory_parts": [
                {"path": p.name, "sha256": _sha(p)}
                for p in sorted(INVENTORY.glob("part-*.parquet"))
            ],
            "census_sha256": _sha(CENSUS),
            "script": str(Path(__file__).relative_to(ROOT)),
            "script_sha256": _sha(Path(__file__)),
        },
        "plane_counts": manifest.anatomical_plane.value_counts().to_dict(),
        "header_unavailable_series": int((~manifest.header_available).sum()),
    }
    (args.output.parent / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"{len(manifest)} series / {manifest.n_slices.sum()} slices / "
        f"{manifest.study_uid.nunique()} studies -> {args.output}"
    )


if __name__ == "__main__":
    sys.exit(main())
