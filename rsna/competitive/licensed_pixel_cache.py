"""Lossless bounded storage for the pinned parent's selected uint8 MRI volume.

No rendering, labels, identity handling or inference. Original preprocessing
is supplied separately under its own exact source/permission binding.
"""

from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np

VERSION = "licensed_parent_uint8_volume_cache_v1"
VOLUME_SHAPE = (96, 384, 384)
MASK_SHAPE = (96,)
MEMBER_CAP = 16 * 2**20


def validate(volume, mask):
    if (
        not isinstance(volume, np.ndarray)
        or not isinstance(mask, np.ndarray)
        or volume.shape != VOLUME_SHAPE
        or mask.shape != MASK_SHAPE
        or volume.dtype != np.uint8
        or mask.dtype != np.uint8
        or not volume.flags.c_contiguous
        or not mask.flags.c_contiguous
        or not np.array_equal(mask, (volume.reshape(96, -1).sum(1) > 0).astype(np.uint8))
    ):
        raise ValueError("Exact original C-contiguous96x384uint8 pixels and presence mask required")


def encode(volume, mask, *, remaining_bytes):
    validate(volume, mask)
    if type(remaining_bytes) is not int or remaining_bytes <= 0:
        raise ValueError("Positive separately funded remaining-byte budget required")
    handle = io.BytesIO()
    np.savez_compressed(handle, volume=volume, mask=mask)
    raw = handle.getvalue()
    if len(raw) > min(MEMBER_CAP, remaining_bytes):
        raise ValueError("Compressed member exceeds pre-write byte budget")
    return raw


def decode(raw, expected_sha256):
    if (
        not isinstance(raw, bytes)
        or not 0 < len(raw) <= MEMBER_CAP
        or not isinstance(expected_sha256, str)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValueError("Exact bounded original encoded-member bytes required")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        expected = {"volume.npy": VOLUME_SHAPE, "mask.npy": MASK_SHAPE}
        if len(members) != 2 or {m.filename for m in members} != set(expected):
            raise ValueError("Only the two original pixel/mask NPY members are accepted")
        for member in members:
            if (
                member.flag_bits & 1
                or member.compress_type != zipfile.ZIP_DEFLATED
                or member.file_size > int(np.prod(expected[member.filename])) + 4096
            ):
                raise ValueError("Bounded original lossless array storage required")
            # Inspect declared shape/dtype before NumPy could allocate a malformed
            # array. The archive and every member are also whole-byte hash bound.
            with archive.open(member) as handle:
                if np.lib.format.read_magic(handle) != (1, 0):
                    raise ValueError("Original simple NPYv1 header required")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                if (
                    handle.tell() > 4096
                    or shape != expected[member.filename]
                    or fortran
                    or dtype != np.uint8
                ):
                    raise ValueError("Original uint8 shapes and storage order required")
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        volume, mask = arrays["volume"].copy(), arrays["mask"].copy()
    validate(volume, mask)
    return volume, mask
