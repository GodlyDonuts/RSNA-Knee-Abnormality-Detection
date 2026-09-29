"""Replay one pinned deployment gap; never executes full upstream notebooks."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

from rsna.competitive.data import CompetitiveStudyDataset

ROOT = Path.cwd()
AUDIT = ROOT / "artifacts/kaggle-successful-submission-review-v2"
projection = json.loads(
    (ROOT / "initiative/audits/kaggle-series-metadata-recovery-v1/projection.json").read_text()
)
paths = (
    "scripts/kaggle/chunked_infer.py",
    "rsna/competitive/data.py",
    "rsna/dicom.py",
    "rsna/preprocess.py",
    "rsna/data.py",
)
bindings = {}
for name in paths:
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert actual == projection["files"][name]["sha256"], name
    bindings[name] = actual
spec = importlib.util.spec_from_file_location("comparison_chunked", ROOT / paths[0])
chunked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunked)

upstream = ROOT / "artifacts/successful-submission-review-v1/rsna-knee-0942-restructured.py"
upstream_sha = hashlib.sha256(upstream.read_bytes()).hexdigest()
assert upstream_sha == "86fb30039835bac99f82bca9af8c35718dc33b79372597bb68a8401b320ffeb8"
notebook = json.loads(upstream.read_text())
functions = {}
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = cell["source"]
    source = "".join(source) if isinstance(source, list) else source
    if not any("def " + name + "(" in source for name in ("read_slot", "order_slices")):
        continue
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("read_slot", "order_slices"):
            functions[node.name] = node
assert set(functions) == {"read_slot", "order_slices"}
events = []
namespace = {
    "os": os,
    "pydicom": pydicom,
    "np": np,
    "torch": torch,
    "F": F,
    "GROUP": 3,
    "IMG": 16,
    "SLICE_BAND": (0.2, 0.8),
    "CROP_MM": 130.0,
    "RULES": {"order": "normal", "decode_fill": "nearest"},
    "DECODE_FAILED": [],
    "ORDER_TAGS": [(32, 50), (32, 55), (32, 19)],
    "rsna_event": lambda *args, **kwargs: events.append({"event": args[0], **kwargs}),
}
# Only the two manually reviewed, data-reading functions are compiled, never notebook cells.
for name in ("order_slices", "read_slot"):
    exec(
        compile(ast.Module(body=[functions[name]], type_ignores=[]), str(upstream), "exec"),
        namespace,
    )

with tempfile.TemporaryDirectory(prefix="rsna-geometry-comparison-") as directory:
    root = Path(directory) / "dicoms"
    cache = Path(directory) / "cache"
    study = "2.3"
    records = []
    for index, plane in enumerate(("Sagittal", "Sagittal", "Coronal", "Axial"), 1):
        sid = f"{study}.{index}"
        records.append(
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": sid,
                "Anatomical_Plane": plane,
                "Fluid_Sensitive": int(index == 1),
                "Fat_Suppression": 0,
            }
        )
        folder = root / "test_series" / study / sid
        folder.mkdir(parents=True)
        for instance, z in enumerate((0, 0, 2) if index == 1 else (0, 1, 2), 1):
            sop = f"{sid}.{instance}"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = sop
            ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
            ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = study, sid, sop
            ds.SOPClassUID = MRImageStorage
            ds.Rows, ds.Columns, ds.SamplesPerPixel = 16, 32, 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation = 15, 0
            ds.ImagePositionPatient = [0, 0, z]
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.PixelSpacing, ds.InstanceNumber = [2, 1], instance
            ds.RescaleSlope, ds.RescaleIntercept = 1, 0
            ds.PixelData = (np.arange(512, dtype=np.uint16) + instance).tobytes()
            ds.save_as(folder / f"{sop}.dcm", enforce_file_format=True)
    selected, preparation = chunked.prepare_usable_series(
        pd.DataFrame(records),
        [study],
        dicom_root=root,
        cache=cache,
        cache_side=16,
        max_series=3,
        source_revision="comparison-diagnostic",
        allow_unusable_studies=True,
    )
    bad = study + ".1"
    metadata = json.loads((cache / "test_series" / study / f"{bad}.json").read_text())
    assert "duplicate_slice_positions" in metadata["warnings"]
    assert bad in selected.SeriesInstanceUID.tolist()
    try:
        CompetitiveStudyDataset(
            pd.DataFrame({"study_uid": [study]}),
            selected,
            cache,
            slices=3,
            side=16,
            max_series=3,
            raw_split="test_series",
        )
    except ValueError as exc:
        ours_error = str(exc)
    else:
        raise AssertionError("Expected current pinned loader rejection")
    assert ours_error == "Patient-geometry cache slices must be strictly ordered"
    folder = root / "test_series" / study / bad
    rec = {
        "files": sorted(p.name for p in folder.glob("*.dcm")),
        "dir": str(folder),
        "px": 2.0,
        "SeriesInstanceUID": bad,
    }
    ordered, geometric = namespace["order_slices"](rec)
    assert geometric
    rec["ordered"] = ordered
    pixels = namespace["read_slot"](rec)
    assert pixels is not None and torch.isfinite(pixels).all()
    result = {
        "schema_version": 1,
        "status": "reproduced_deployment_difference",
        "hidden_cause_proven": False,
        "diagnostic_only": True,
        "fixture": "valid synthetic grayscale DICOM series with z positions [0,0,2], healthy same-plane alternate and two other planes",
        "source_sha256": bindings,
        "successful_notebook_source_sha256": upstream_sha,
        "upstream_execution_scope": "only manually reviewed order_slices/read_slot functions; no model/notebook execution",
        "our_preparation": preparation,
        "our_selected_series": selected.SeriesInstanceUID.tolist(),
        "our_cache_warnings": metadata["warnings"],
        "our_dataset_construction_error": ours_error,
        "our_neutral_fallback_triggered": study in preparation["unusable_study_ids"],
        "successful_notebook_reader": {
            "output_shape": list(pixels.shape),
            "dtype": str(pixels.dtype),
            "finite": bool(torch.isfinite(pixels).all()),
            "events": events,
        },
        "model_weights_modified": False,
        "kaggle_mutation": False,
    }
(AUDIT / "geometry-gap-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
