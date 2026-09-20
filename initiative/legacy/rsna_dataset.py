"""
Helper module for training with the RSNA knee abnormality dataset on Hugging Face.

Usage:
    from rsna_dataset import RSNAStreamingDataset, TARGETS

    ds = RSNAStreamingDataset(split="train", transform=your_transforms)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=0)
    for images, labels in loader:
        ...
"""

import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils.data import IterableDataset

# 12-class target labels in the order used by the RSNA competition labels.
TARGETS: List[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

DEFAULT_REPO: str = "GodlyDonuts/RSNA-Data"


def _get_xopen():
    """Lazy import so the module is importable even without datasets installed."""
    try:
        from datasets.download.streaming_download_manager import xopen
        return xopen
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "`datasets` is required for streaming from Hugging Face. "
            "Install with `pip install datasets`."
        ) from exc


def _as_list(value: Any):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def decode_dicom_bytes(dicom_bytes: bytes) -> Image.Image:
    """Decode raw DICOM bytes and return an 8-bit grayscale PIL Image."""
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))

    # Force pixel_array to use the installed pixel data handlers (pylibjpeg, etc.).
    arr = ds.pixel_array.astype(np.float32)

    # Apply modality rescale if present.
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    arr = arr * slope + intercept

    # Apply VOI windowing if a window is specified, otherwise min-max scale.
    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    if center is not None and width is not None:
        center = _as_list(center)
        width = _as_list(width)
        if center and width:
            c = float(center[0])
            w = float(width[0])
            if w < 1:
                w = 1
            win_min = c - w / 2 + 0.5
            win_max = c + w / 2 - 0.5
            arr = ((arr - win_min) / (win_max - win_min + 1e-8)) * 255
            arr = np.clip(arr, 0, 255)
        else:
            arr = _min_max_scale(arr)
    else:
        arr = _min_max_scale(arr)

    # MONOCHROME1 means pixel value 0 is white; invert so 0 is black.
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        arr = 255 - arr

    return Image.fromarray(arr.astype(np.uint8), mode="L")


def _min_max_scale(arr: np.ndarray) -> np.ndarray:
    amin = arr.min()
    amax = arr.max()
    if amax > amin:
        arr = (arr - amin) / (amax - amin + 1e-8) * 255
    else:
        arr = np.zeros_like(arr)
    return np.clip(arr, 0, 255)


def decode_dicom(example: Dict[str, Any], repo_id: str = DEFAULT_REPO) -> Dict[str, Any]:
    """datasets `.map` friendly wrapper that adds an `image` key to a metadata example."""
    path = example["path"]
    if not path.startswith("hf://"):
        path = f"hf://datasets/{repo_id}/{path}"
    xopen = _get_xopen()
    with xopen(path, "rb") as f:
        dicom_bytes = f.read()
    example["image"] = decode_dicom_bytes(dicom_bytes)
    return example


class RSNAStreamingDataset(IterableDataset):
    """
    PyTorch IterableDataset that streams metadata from Hugging Face and decodes DICOMs on-the-fly.

    Args:
        split: one of "train", "validation", "test", or "gold".
        repo_id: Hugging Face dataset repository id.
        streaming: passed through to `load_dataset`.
        transform: optional torchvision-style transform applied to each PIL image.
        label_mode: "hard" (default) or "soft". Soft returns the calibrated probabilities.
    """

    def __init__(
        self,
        split: str = "train",
        repo_id: str = DEFAULT_REPO,
        streaming: bool = True,
        transform: Optional[Any] = None,
        label_mode: str = "hard",
    ):
        super().__init__()
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise ImportError(
                "`datasets` is required. Install with `pip install datasets`."
            ) from exc
        self.split = split
        self.repo_id = repo_id
        self.streaming = streaming
        self.transform = transform
        self.label_mode = label_mode
        self.target_suffix = "" if label_mode == "hard" else "_soft"

        self._ds = load_dataset(repo_id, split=split, streaming=streaming)
        self._ds = self._ds.map(decode_dicom, fn_kwargs={"repo_id": repo_id})

    def __iter__(self):
        for example in self._ds:
            image = example["image"]
            if self.transform is not None:
                image = self.transform(image)
            labels = []
            for t in TARGETS:
                key = f"{t}{self.target_suffix}"
                # Fall back to hard labels when soft labels are not present (e.g., gold split).
                if key in example:
                    value = example[key]
                else:
                    value = example.get(t, 0.0)
                labels.append(float(value or 0.0))
            yield image, torch.as_tensor(labels, dtype=torch.float32)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Default collate for tensors returned by `RSNAStreamingDataset`."""
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)
