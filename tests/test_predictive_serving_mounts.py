"""Bounded discovery must never traverse mounted private MRI trees."""

import importlib.util
from pathlib import Path

import pytest

D = Path("initiative/research/predictive-final-serving-20261020-v1")
spec = importlib.util.spec_from_file_location("predictive_mounts", D / "mounts.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic")
    return path


@pytest.mark.parametrize("namespaced", [False, True])
def test_exact_mounts_never_walk_unrelated_trees(tmp_path, monkeypatch, namespaced):
    def forbidden(*args, **kwargs):
        raise AssertionError("No directory traversal is permitted")

    for operation in ("rglob", "glob", "iterdir"):
        monkeypatch.setattr(Path, operation, forbidden)
    prefix = tmp_path / "datasets" / "owner" if namespaced else tmp_path
    expected = touch(prefix / "weights" / "model.pt")
    touch(tmp_path / "unrelated" / "model.pt")
    assert m.resolve_dataset_file(tmp_path, "owner/weights", "model.pt") == expected
    competition = tmp_path / "competitions" if namespaced else tmp_path
    roster = touch(competition / m.COMPETITION / "test.csv")
    (roster.parent / "test_series").mkdir()
    assert m.resolve_competition_roster(tmp_path) == roster


def test_ambiguity_missing_and_flat_alias_are_explicit(tmp_path):
    with pytest.raises(ValueError):
        m.resolve_dataset_file(tmp_path, "owner/weights", "model.pt")
    flat = touch(tmp_path / "weights" / "model.pt")
    other = touch(tmp_path / "datasets" / "owner" / "weights" / "model.pt")
    with pytest.raises(ValueError):
        m.resolve_dataset_file(tmp_path, "owner/weights", "model.pt")
    other.unlink()
    other.symlink_to(flat)
    assert m.resolve_dataset_file(tmp_path, "owner/weights", "model.pt").resolve() == flat
    for prefix in (tmp_path, tmp_path / "competitions"):
        roster = touch(prefix / m.COMPETITION / "test.csv")
        (roster.parent / "test_series").mkdir()
    with pytest.raises(ValueError):
        m.resolve_competition_roster(tmp_path)


@pytest.mark.parametrize(
    "reference,filename",
    [
        ("../weights", "model.pt"),
        ("owner/../weights", "model.pt"),
        ("owner", "model.pt"),
        (None, "model.pt"),
        ("owner/weights", "../model.pt"),
        ("owner/weights", "/model.pt"),
        ("owner/weights", "sub\\model.pt"),
        ("owner/weights", ""),
    ],
)
def test_malformed_paths_are_rejected(tmp_path, reference, filename):
    with pytest.raises(ValueError):
        m.resolve_dataset_file(tmp_path, reference, filename)


def test_worker_has_no_global_recursive_discovery():
    source = (D / "worker.py.txt").read_text()
    assert ".rglob(" not in source and ".glob(" not in source
    assert source.index("cache_descriptors_once(groups)") < source.index("assemble(groups,")
