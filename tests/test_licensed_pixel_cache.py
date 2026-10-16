import hashlib
import io
import zipfile

import numpy as np
import pytest

from rsna.competitive.licensed_pixel_cache import decode, encode, validate


def pixels():
    volume = np.zeros((96, 384, 384), np.uint8)
    volume[2:5, 80:300, 70:310] = np.random.default_rng(7).integers(
        0, 256, (3, 220, 240), dtype=np.uint8
    )
    mask = (volume.reshape(96, -1).sum(1) > 0).astype(np.uint8)
    return volume, mask


def test_lossless_sparse_volume_missing_slots_and_integer_extremes():
    volume, mask = pixels()
    raw = encode(volume, mask, remaining_bytes=16 * 2**20)
    restored, present = decode(raw, hashlib.sha256(raw).hexdigest())
    assert np.array_equal(volume, restored) and np.array_equal(mask, present)
    assert restored.dtype == present.dtype == np.uint8
    assert restored.min() == 0 and restored.max() == 255
    assert len(raw) < volume.nbytes / 4


@pytest.mark.parametrize("bad", ["mask", "dtype", "shape", "noncontiguous"])
def test_changed_input_contract_rejected(bad):
    volume, mask = pixels()
    if bad == "mask":
        mask[0] = 1
    elif bad == "dtype":
        volume = volume.astype(np.float32)
    elif bad == "shape":
        volume = volume[:95]
    else:
        volume = volume[:, :, ::-1]
    with pytest.raises(ValueError):
        validate(volume, mask)


def test_prewrite_budget_and_changed_archive_rejected():
    volume, mask = pixels()
    with pytest.raises(ValueError, match="budget"):
        encode(volume, mask, remaining_bytes=1)
    raw = encode(volume, mask, remaining_bytes=16 * 2**20)
    with pytest.raises(ValueError, match="bytes"):
        decode(raw + b"changed", hashlib.sha256(raw).hexdigest())


def test_extra_members_and_oversized_declared_shapes_rejected_before_numpy_allocation():
    volume, mask = pixels()
    handle = io.BytesIO()
    np.savez_compressed(handle, volume=volume, mask=mask, unexpected=np.array([0]))
    raw = handle.getvalue()
    with pytest.raises(ValueError, match="two original"):
        decode(raw, hashlib.sha256(raw).hexdigest())
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        declared = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            declared, {"descr": "|u1", "fortran_order": False, "shape": (2**50,)}
        )
        z.writestr("volume.npy", declared.getvalue())
        m = io.BytesIO()
        np.save(m, mask, allow_pickle=False)
        z.writestr("mask.npy", m.getvalue())
    raw = handle.getvalue()
    with pytest.raises(ValueError, match="shapes"):
        decode(raw, hashlib.sha256(raw).hexdigest())
