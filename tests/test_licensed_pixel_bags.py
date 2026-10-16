import hashlib
import io
import tarfile

import numpy as np
import pytest
import torch

from rsna.competitive.licensed_pixel_bags import PixelBags
from rsna.competitive.licensed_pixel_cache import encode


@pytest.fixture
def admitted(tmp_path):
    torch.set_num_threads(1)
    uid = "synthetic-engineering-only"
    volume = np.full((96, 384, 384), 17, dtype=np.uint8)
    mask = np.ones(96, dtype=np.uint8)
    raw = encode(volume, mask, remaining_bytes=16 * 2**20)
    name = hashlib.sha256(uid.encode()).hexdigest() + ".npz"
    archive = tmp_path / "original.tar"
    with tarfile.open(archive, "w:") as f:
        member = tarfile.TarInfo(name)
        member.size = len(raw)
        f.addfile(member, io.BytesIO(raw))

    def render(v, m, mirror):
        return torch.full((1, 3, 320, 320), int(mirror), dtype=torch.float32).expand(94, -1, -1, -1)

    row = {
        "study_uid": uid,
        "holder": "original",
        "file": name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "volume_sha256": hashlib.sha256(volume.tobytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
    }
    for mirror in (False, True):
        row["original_mirrored_input_sha256" if mirror else "original_plain_input_sha256"] = (
            hashlib.sha256(render(volume, mask, mirror).contiguous().numpy().tobytes()).hexdigest()
        )
    targets = {
        "uids": [uid],
        "labels": torch.full((1, 12), 0.5),
        "masks": torch.ones((1, 12), dtype=torch.bool),
        "parent_logits": torch.stack((torch.ones(12), torch.ones(12) * 2)).unsqueeze(0),
    }
    return [row], {"original": archive}, targets, render


def test_reads_original_archive_without_extracting_images_and_aligns_anatomical_teacher(admitted):
    data = PixelBags(*admitted, diagnostic_only=True)
    plain = data.case(0, [3, 1, 50], False)
    mirror = data.case(0, list(range(94)), True)
    assert plain["images"].shape == (3, 3, 320, 320) and not plain["images"].count_nonzero()
    assert mirror["images"].shape == (94, 3, 320, 320) and (mirror["images"] == 1).all()
    assert torch.equal(plain["parent_logits"], torch.ones(12))
    assert torch.equal(mirror["parent_logits"], torch.ones(12) * 2)
    assert len(data.verified_inputs) == 2
    admitted[2]["labels"].zero_()
    assert (data.labels == 0.5).all()


def test_engineering_fixture_cannot_claim_full_original_fitting_admission(admitted):
    with pytest.raises(ValueError, match="population"):
        PixelBags(*admitted)


def test_archive_changes_and_view_hash_changes_fail_before_model_forward(admitted):
    data = PixelBags(*admitted, diagnostic_only=True)
    data.records[0]["original_plain_input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rendered"):
        data.case(0, [1], False)
    path = admitted[1]["original"]
    with path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed after admission"):
        data.case(0, [1], True)


@pytest.mark.parametrize("positions", [[True], [94], [1, 1], []])
def test_window_selection_cannot_change_original94_index_contract(admitted, positions):
    data = PixelBags(*admitted, diagnostic_only=True)
    with pytest.raises(ValueError, match="indices"):
        data.case(0, positions, False)
