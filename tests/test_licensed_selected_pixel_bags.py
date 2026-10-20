"""Synthetic byte-level reader contracts; actual full-cohort admission is separate."""

import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from rsna.competitive import licensed_selected_pixel_bags as selected
from rsna.competitive.licensed_pixel_bags import PixelBags

spec = importlib.util.spec_from_file_location(
    "original_pixel_fixture", Path(__file__).with_name("test_licensed_pixel_bags.py")
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
admitted = fixture.admitted


@pytest.fixture
def original_manifest(admitted, tmp_path, monkeypatch):
    # Explicit synthetic one-study scope; production constants remain3887/fullcohort.
    monkeypatch.setattr(selected, "STUDIES", 1)
    data = PixelBags(*admitted, diagnostic_only=True)
    values = np.zeros((1, 2, 94, 32), dtype=np.uint8)
    for mirror in (False, True):
        full = data.case(0, list(range(94)), mirror)["images"]
        for j in range(94):
            values[0, int(mirror), j] = np.frombuffer(
                hashlib.sha256(full[j].numpy().tobytes()).digest(), dtype=np.uint8
            )
    path = tmp_path / "integrity.private.npy"
    np.save(path, values, allow_pickle=False)
    path.chmod(0o600)
    receipt = dict(
        status=selected.STATUS,
        studies=1,
        views=2,
        windows_per_view=94,
        cohort_sha256=selected.POPULATION_SHA,
        record_binding_sha256=selected.record_digest(data.records),
        original_full94_tensor_hash_checks=2,
        manifest_bytes=path.stat().st_size,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    calls = []

    def render(volume, mask, positions, mirror):
        calls.append((list(positions), mirror))
        return admitted[3](volume, mask, mirror)[positions].contiguous()

    return data, path, receipt, render, calls


def test_selected_windows_and_targets_equal_original_without_whole_bag_render(original_manifest):
    data, path, receipt, render, calls = original_manifest
    reader = selected.SelectedPixelBags(data, path, receipt, render)
    for mirror in (False, True):
        actual = reader.case(0, [93, 2, 50], mirror)
        expected = data.case(0, [93, 2, 50], mirror)
        assert all(torch.equal(actual[k], expected[k]) for k in actual)
    assert calls == [([93, 2, 50], False), ([93, 2, 50], True)]
    assert not reader.hashes.flags.writeable


def test_changed_renderer_cannot_pass_original_manifest(original_manifest):
    data, path, receipt, render, _ = original_manifest
    reader = selected.SelectedPixelBags(data, path, receipt, lambda *args: render(*args) + 0.1)
    with pytest.raises(ValueError, match="differs from verified"):
        reader.case(0, [2], False)


def test_other_view_is_not_interchangeable(original_manifest):
    data, path, receipt, render, _ = original_manifest
    reader = selected.SelectedPixelBags(
        data, path, receipt, lambda v, m, p, mirror: render(v, m, p, not mirror)
    )
    with pytest.raises(ValueError, match="differs from verified"):
        reader.case(0, [2], True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_sha256", "0" * 64),
        ("record_binding_sha256", "1" * 64),
        ("original_full94_tensor_hash_checks", 1),
        ("cohort_sha256", "2" * 64),
    ],
)
def test_false_or_incomplete_admission_is_rejected(original_manifest, field, value):
    data, path, receipt, render, _ = original_manifest
    receipt = {**receipt, field: value}
    with pytest.raises(ValueError):
        selected.SelectedPixelBags(data, path, receipt, render)


def test_manifest_mutation_and_public_permissions_rejected(original_manifest):
    data, path, receipt, render, _ = original_manifest
    reader = selected.SelectedPixelBags(data, path, receipt, render)
    with path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed after admission"):
        reader.case(0, [2], False)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="Owner-only"):
        selected.SelectedPixelBags(data, path, receipt, render)


def test_archive_or_record_mutation_rejected(original_manifest):
    data, path, receipt, render, _ = original_manifest
    reader = selected.SelectedPixelBags(data, path, receipt, render)
    old = copy.deepcopy(data.records[0])
    data.records[0]["volume_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="record binding"):
        reader.case(0, [2], False)
    data.records[0] = old
    with next(iter(data.roots.values())).open("ab") as f:
        f.write(b"mutated")
    with pytest.raises(ValueError, match="changed after admission"):
        reader.case(0, [2], False)


@pytest.mark.parametrize("positions", [[True], [94], [2, 2], []])
def test_original_unique_index_contract_preserved(original_manifest, positions):
    data, path, receipt, render, _ = original_manifest
    reader = selected.SelectedPixelBags(data, path, receipt, render)
    with pytest.raises(ValueError, match="unique window"):
        reader.case(0, positions, False)
