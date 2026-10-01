import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from rsna.competitive.adapters import ResidualLinear
from rsna.competitive.mrnet import (
    MRNET_PLANES,
    MRNET_TARGETS,
    MRNetAdapterNet,
    MRNetStudyDataset,
    transfer_encoder_adapters,
)


def private_source(root):
    splits, rows, original = {}, [], []
    for split, uid in (("train", "0001"), ("valid", "1131")):
        splits[split] = {
            "supervised_cases": [uid],
            "unlabeled_cases": ["0000" if split == "train" else "1130"],
            "training_permitted": split == "train",
            "labels": {target: {uid: i % 2} for i, target in enumerate(MRNET_TARGETS)},
        }
        for target, labels in splits[split]["labels"].items():
            (root / f"{split}-{target}.csv").write_text("case,label\n" + f"{uid},{labels[uid]}\n")
        (root / f"{split}.csv").write_text("engineering_fixture\n")
        for plane_index, plane in enumerate(MRNET_PLANES):
            path = root / split / plane / (uid + ".npy")
            path.parent.mkdir(parents=True)
            volume = np.full((plane_index + 2, 256, 256), (plane_index + 1) * 64, dtype=np.uint8)
            np.save(path, volume, allow_pickle=False)
            raw = path.read_bytes()
            rows.append(
                {
                    "split": split,
                    "name": f"{plane}/{uid}.npy",
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "shape": list(volume.shape),
                }
            )
        for name in (split, *(f"{split}-{t}" for t in MRNET_TARGETS)):
            path = root / (name + ".csv")
            raw = path.read_bytes()
            original.append(
                {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            )
    source = "aimi.mrnet_knee_mri_s:4a2c:v1_0"
    (root / "source-admission.json").write_text(
        json.dumps(
            {
                "source": source,
                "target_order": list(MRNET_TARGETS),
                "rsna_target_mapping": None,
                "splits": splits,
            }
        )
    )
    journal = root / "verified-files.jsonl"
    journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest = root / "downloads.json"
    manifest.write_text(json.dumps(original))
    receipt = root / "accepted.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "verbatim_private_MRNet_images_and_labels_verified",
                "source": source,
                "files": len(rows),
                "file_journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        )
    )
    return receipt


def test_source_exams_remain_three_planes_and_three_source_tasks(tmp_path):
    receipt = private_source(tmp_path)
    train = MRNetStudyDataset(tmp_path, receipt, "train", slices=8)
    valid = MRNetStudyDataset(tmp_path, receipt, "valid", slices=8)
    example = train[0]
    assert train.uids == ("0001",) and valid.uids == ("1131",)
    assert train.training_permitted and not valid.training_permitted
    assert example["images"].shape == (3, 8, 1, 256, 256)
    assert example["labels"].tolist() == [0, 1, 0]
    assert example["slice_mask"].sum(dim=1).tolist() == [2, 3, 4]
    assert example["images"][0, 0, 0, 0, 0] == pytest.approx(64 / 255)
    source_path = tmp_path / "source-admission.json"
    source = json.loads(source_path.read_text())
    source["splits"]["train"]["labels"]["acl"]["0001"] = 1
    source_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="original source values"):
        MRNetStudyDataset(tmp_path, receipt, "train")


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = ResidualLinear(nn.Linear(1, 8), rank=2, alpha=2, dropout=0)

    def forward(self, images):
        return self.layer(images.mean(dim=(2, 3)))


def test_exam_loss_reaches_adapter_and_padding_never_creates_nan_gradient():
    torch.manual_seed(31)
    encoder = TinyEncoder()
    net = MRNetAdapterNet(encoder, 8, hidden_dim=12, chunk=3)
    base = encoder.layer.weight.detach().clone()
    images = torch.rand(2, 3, 5, 1, 32, 32)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)
    mask[:, :, -1] = False
    positions = torch.linspace(0, 1, 5).expand(2, 3, 5).clone()
    images[~mask] = float("nan")
    positions[~mask] = float("nan")
    prediction = net(images, mask, positions)
    assert prediction.shape == (2, 3)
    nn.functional.binary_cross_entropy_with_logits(prediction, torch.ones(2, 3)).backward()
    assert encoder.layer.lora_B.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=0.01)
    optimizer.step()
    assert torch.equal(base, encoder.layer.weight)


def test_only_compatible_encoder_adapters_transfer_atomically():
    source, destination = TinyEncoder(), TinyEncoder()
    base = destination.layer.weight.detach().clone()
    with torch.no_grad():
        source.layer.lora_B.fill_(0.5)
    provenance = {
        "architecture": "dinov3_l16",
        "config_sha256": "a" * 64,
        "weights_sha256": "b" * 64,
        "source_id": "official",
        "source_revision": "revision",
        "input_contract": "gray",
        "adapter": {"rank": 2, "alpha": 2},
    }
    assert (
        transfer_encoder_adapters(
            source, destination, source_provenance=provenance, destination_provenance=provenance
        )
        == 2
    )
    assert torch.equal(destination.layer.lora_B, source.layer.lora_B)
    assert torch.equal(destination.layer.weight, base)
    before = copy.deepcopy(destination.state_dict())
    with pytest.raises(ValueError, match="provenance"):
        transfer_encoder_adapters(
            source,
            destination,
            source_provenance=provenance,
            destination_provenance={**provenance, "weights_sha256": "c" * 64},
        )
    assert all(torch.equal(value, destination.state_dict()[name]) for name, value in before.items())
