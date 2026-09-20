#!/usr/bin/env python3
"""Generate shuffled synthetic DICOM slices and verify physical ordering."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

try:
    from .constants import PLANE_TO_ID
    from .extract_features import _read_series, prepare_25d_images
    from .raw_mil import SeriesRecord, load_series_tensor
except ImportError:
    from constants import PLANE_TO_ID
    from extract_features import _read_series, prepare_25d_images
    from raw_mil import SeriesRecord, load_series_tensor


def write_slice(
    path: Path,
    position: float,
    value: int,
    instance: int,
    corrupt: bool = False,
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.Rows = 32
    ds.Columns = 40
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0, 0, position]
    ds.PixelSpacing = [1.0, 1.0]
    ds.InstanceNumber = instance
    ds.ImageLaterality = "L"
    yy, xx = np.mgrid[:32, :40]
    pixels = (value + yy + 2 * xx).astype(np.uint16)
    pixel_bytes = pixels.tobytes()
    ds.PixelData = pixel_bytes[: len(pixel_bytes) // 2] if corrupt else pixel_bytes
    ds.save_as(path, enforce_file_format=True)


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        # Filename order deliberately disagrees with physical z order.
        specifications = [
            ("slice_00.dcm", 12.0, 500, 30),
            ("slice_01.dcm", 0.0, 100, 10),
            ("slice_02.dcm", 8.0, 400, 25),
            ("slice_03.dcm", 4.0, 200, 20),
        ]
        for filename, z, value, instance in specifications:
            write_slice(root / filename, z, value, instance)
        # A malformed frame is retained by header parsing but must be skipped
        # during pixel decoding without discarding the four valid neighbors.
        write_slice(root / "slice_04_corrupt.dcm", 6.0, 300, 22, corrupt=True)
        paths = sorted(root.glob("*.dcm"))
        datasets, positions = _read_series(paths)
        expected_with_corrupt = np.asarray([0.0, 4.0, 6.0, 8.0, 12.0], dtype=np.float32)
        expected = np.asarray([0.0, 4.0, 8.0, 12.0], dtype=np.float32)
        if not np.array_equal(positions, expected_with_corrupt):
            raise AssertionError(f"physical order failed: {positions.tolist()}")
        images, selected = prepare_25d_images(
            datasets,
            positions,
            plane=PLANE_TO_ID["sagittal"],
            max_slices=5,
            crop_mm=24.0,
        )
        if len(images) != 4 or images[0].shape != (24, 24, 3):
            raise AssertionError(f"unexpected 2.5-D output shape: {images[0].shape}")
        if not np.array_equal(selected, expected):
            raise AssertionError("selected physical positions changed unexpectedly")
        raw_pixels, raw_positions = load_series_tensor(
            SeriesRecord(
                uid="synthetic",
                plane=PLANE_TO_ID["sagittal"],
                fluid=0,
                fatsat=0,
                directory=str(root),
            ),
            slices=5,
            crop_mm=24.0,
            image_size=32,
            training=False,
        )
        if raw_pixels.shape != (5, 3, 32, 32) or not raw_pixels.isfinite().all():
            raise AssertionError(f"raw loader did not recover from corrupt frame: {raw_pixels.shape}")
        if raw_positions.shape != (5,):
            raise AssertionError("raw loader returned invalid physical positions")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "filename_order_positions": [12.0, 0.0, 8.0, 4.0],
                    "recovered_positions_mm": positions.tolist(),
                    "image_shape": list(images[0].shape),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
