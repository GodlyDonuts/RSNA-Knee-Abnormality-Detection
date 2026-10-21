"""CPU-only diagnostic of strict serving assumptions on accessible source DICOMs.

No hidden data, labels, model, raw-output pixels, clinical prediction or entry.
The complete actual public decoder is also run once on each public study.
"""

import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import time
from pathlib import Path

LIMIT = 900


def inspect_header(path):
    import pydicom

    from rsna.competitive.predictive.dicom import _integer, _position, _uid

    try:
        for x in (path.parent.parent.name, path.parent.name, path.stem):
            _uid(x)
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        actual = tuple(
            str(ds.get(n, "")) for n in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")
        )
        if actual != (path.parent.parent.name, path.parent.name, path.stem):
            return "identity_path_mismatch"
        bits, signed = _integer(ds, "BitsStored"), _integer(ds, "PixelRepresentation")
        if not 1 <= bits <= 16 or signed not in (0, 1):
            return "integer_encoding"
        if str(ds.get("PhotometricInterpretation", "")) not in ("MONOCHROME1", "MONOCHROME2"):
            return "photometric"
        if _integer(ds, "SamplesPerPixel") != 1 or (
            "NumberOfFrames" in ds and _integer(ds, "NumberOfFrames") != 1
        ):
            return "not_single_monochrome_frame"
        slope, intercept = float(ds.get("RescaleSlope", 1)), float(ds.get("RescaleIntercept", 0))
        if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
            return "rescale"
        instance = _integer(ds, "InstanceNumber")
        _position(ds, instance)
        _integer(ds, "Rows")
        _integer(ds, "Columns")
        return "accepted"
    except ValueError as error:
        # Only fixed source-defined error messages; never private headers/paths.
        message = str(error)
        known = {
            "Required DICOM integer metadata is missing or invalid": "required_integer_metadata",
            "DICOM geometry is incomplete or invalid": "geometry",
            "Valid DICOM UID required": "UID",
        }
        return known.get(message, "other_ValueError")
    except Exception as error:
        return "header_read_" + type(error).__name__


def main(binding):
    import _predictive_serving_mounts as mounts
    import torch

    from rsna.competitive.predictive.dicom import load_study

    torch.set_num_threads(1)
    assert torch.cuda.device_count() == 0
    begin = time.monotonic()
    roster = mounts.resolve_competition_roster("/kaggle/input")
    with roster.open() as f:
        ids = [r["StudyInstanceUID"] for r in csv.DictReader(f)]
    assert len(ids) == 3
    public = []
    for i, uid in enumerate(ids):
        try:
            groups, stats = load_study(roster.parent / "test_series" / uid, uid)
            public.append({"index": i, "status": "accepted", "stats": stats})
            del groups
        except Exception as error:
            public.append({"index": i, "status": "error", "type": type(error).__name__})
    train = roster.parent / "train_series"
    studies = (
        sorted(
            (p for p in train.iterdir() if p.is_dir()),
            key=lambda p: hashlib.sha256(p.name.encode()).hexdigest(),
        )
        if train.is_dir()
        else []
    )
    counts = collections.Counter()
    studied = 0
    inventory = 0
    start_headers = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for study in studies:
            if time.monotonic() - begin > 720:
                break
            files = sorted(study.glob("*/*.dcm"))
            inventory += len(files)
            for result in pool.map(inspect_header, files):
                counts[result] += 1
            studied += 1
            if studied % 32 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "source_metadata_progress",
                            "studies": studied,
                            "headers": sum(counts.values()),
                            "categories": dict(counts),
                        }
                    ),
                    flush=True,
                )
    result = {
        "status": "actual_accessible_DICOM_assumptions_audited",
        "binding": binding,
        "public": public,
        "training_directory_available": train.is_dir(),
        "training_studies_available": len(studies),
        "training_studies_checked": studied,
        "headers": sum(counts.values()),
        "header_categories": dict(counts),
        "complete_accessible_training_header_coverage": studied == len(studies) and bool(studies),
        "training_header_seconds": time.monotonic() - start_headers,
        "elapsed_seconds": time.monotonic() - begin,
        "GPU_count": 0,
        "model_or_label_reads": False,
        "clinical_score": None,
        "submission_csv_created": False,
        "hidden_error_cause_proven": False,
    }
    assert result["elapsed_seconds"] < LIMIT and sum(counts.values()) == inventory
    Path("/kaggle/working/predictive-hidden-input-audit.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
