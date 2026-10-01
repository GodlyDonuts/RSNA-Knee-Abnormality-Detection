from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.request import Request

import numpy as np
import pytest

from scripts.newton.ingest_mrnet import (
    PLANES,
    SOURCE,
    TARGETS,
    PrivateRedirect,
    admit_sources,
    read_index,
    read_labels,
    sha,
    validate_npy,
)


def original_fixture(root: Path):
    manifest = []
    for split, start, stop in (("train", 0, 1130), ("valid", 1130, 1250)):
        path = root / (split + ".csv")
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file_id", "file_name", "size", "added_at", "md5_hash"])
            for uid in range(start, stop):
                for plane in PLANES:
                    writer.writerow(
                        [
                            f"{split}-{uid}-{plane}",
                            f"{plane}/{uid:04d}.npy",
                            65536 + 128,
                            "",
                            "AAAAAAAAAAAAAAAAAAAAAA==",
                        ]
                    )
        for target in TARGETS:
            label_path = root / f"{split}-{target}.csv"
            with label_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "_0000" if split == "train" else "_1130",
                        "_1" if target == "abnormal" and split == "train" else "_0",
                    ]
                )
                for uid in range(start + 1, stop):
                    writer.writerow([str(uid), uid % 2])
        for name in (split, *(f"{split}-{target}" for target in TARGETS)):
            source = root / (name + ".csv")
            manifest.append(
                {
                    "name": name,
                    "bytes": source.stat().st_size,
                    "sha256": sha(source),
                    "qualifiedReference": SOURCE + "." + name,
                }
            )
    (root / "downloads.json").write_text(json.dumps(manifest))


def test_missing_source_cases_never_become_supervision(tmp_path):
    original_fixture(tmp_path)
    admitted = admit_sources(tmp_path)
    assert admitted["splits"]["train"]["supervised_cases"][0] == "0001"
    assert len(admitted["splits"]["train"]["supervised_cases"]) == 1129
    assert admitted["splits"]["valid"]["supervised_cases"][0] == "1131"
    assert admitted["splits"]["valid"]["training_permitted"] is False
    assert admitted["target_order"] == ["acl", "meniscus", "abnormal"]
    assert admitted["rsna_target_mapping"] is None
    assert admitted["geometry_unknown"] is True
    path = tmp_path / "train-acl.csv"
    path.write_text(path.read_text() + "0000,1\n")
    with pytest.raises(ValueError, match="bytes_changed"):
        admit_sources(tmp_path)


def test_invalid_source_label_and_duplicate_are_rejected(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("_0000,_0\n0001,1\n1,0\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_labels(path, "acl")
    path.write_text("_0000,_0\n0001,2\n")
    with pytest.raises(ValueError, match="nonbinary"):
        read_labels(path, "acl")


@pytest.mark.parametrize(
    "name", ["../escape.npy", "axial/0001/../../escape.npy", "unknown/0001.npy"]
)
def test_source_paths_cannot_escape_or_invent_planes(tmp_path, name):
    path = tmp_path / "index.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file_id", "file_name", "size", "added_at", "md5_hash"])
        writer.writerow(["id", name, 65536 + 128, "", "AAAAAAAAAAAAAAAAAAAAAA=="])
    with pytest.raises(ValueError, match="filename"):
        read_index(path)


def test_npy_rejects_pickle_wrong_geometry_and_truncation(tmp_path):
    path = tmp_path / "volume.npy"
    np.save(path, np.zeros((3, 256, 256), dtype=np.uint8), allow_pickle=False)
    assert validate_npy(path, path.stat().st_size) == [3, 256, 256]
    with pytest.raises(ValueError, match="volume"):
        validate_npy(path, path.stat().st_size - 1)
    np.save(path, np.zeros((3, 128, 256), dtype=np.uint8), allow_pickle=False)
    with pytest.raises(ValueError, match="volume"):
        validate_npy(path, path.stat().st_size)
    np.save(path, np.array([object()], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="volume"):
        validate_npy(path, path.stat().st_size)


def test_cross_origin_redirect_does_not_forward_credential():
    request = Request(
        "https://redivis.com/api/v1/rawFiles/id", headers={"Authorization": "Bearer sentinel"}
    )
    redirect = PrivateRedirect().redirect_request(
        request, None, 302, "", {}, "https://cdn.example.org/file"
    )
    assert redirect.get_header("Authorization") is None
    with pytest.raises(ValueError, match="https"):
        PrivateRedirect().redirect_request(
            request, None, 302, "", {}, "http://cdn.example.org/file"
        )
