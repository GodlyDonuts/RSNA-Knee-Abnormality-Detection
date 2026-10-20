"""Concurrency regressions for checkpoint sidecars and independent publication."""

import importlib.machinery
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path("initiative/research/predictive-full-fitting-20261020-v1/worker.py.txt")
loader = importlib.machinery.SourceFileLoader("predictive_full_worker_io", str(PATH))
worker = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(worker)


def test_sidecar_is_complete_at_first_visibility_and_never_overwrites(tmp_path, monkeypatch):
    target = tmp_path / "epoch-001.json"
    original_link = worker.os.link
    observations = []

    def observe(source, destination):
        assert not Path(destination).exists()
        observations.append(json.loads(Path(source).read_text()))
        original_link(source, destination)
        assert json.loads(Path(destination).read_text()) == observations[-1]

    monkeypatch.setattr(worker.os, "link", observe)
    payload = {"checkpoint_sha256": "a" * 64, "completed_updates": 486}
    worker.write_once(target, payload)
    assert observations == [payload]
    monkeypatch.setattr(worker.os, "link", original_link)
    with pytest.raises(FileExistsError):
        worker.write_once(target, {"completed_updates": 999})
    assert json.loads(target.read_text()) == payload
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("disposable", [True, False])
def test_vanished_publication_download_does_not_mask_missing_original(
    tmp_path, monkeypatch, disposable
):
    root = tmp_path
    relative = (
        "publication/epoch-001/independent-download-example/model.pt"
        if disposable
        else "checkpoints/epoch-001.pt"
    )
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"file")
    original_stat = Path.stat
    original_is_file = Path.is_file

    def is_file(candidate):
        if candidate == path:
            return True
        return original_is_file(candidate)

    def stat(candidate, **kwargs):
        if candidate == path:
            raise FileNotFoundError(str(candidate))
        return original_stat(candidate, **kwargs)

    monkeypatch.setattr(Path, "is_file", is_file)
    monkeypatch.setattr(Path, "stat", stat)
    if disposable:
        assert worker.output_usage(root) == (0, 0)
    else:
        with pytest.raises(FileNotFoundError):
            worker.output_usage(root)
