import copy
import hashlib
import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import torch

from rsna.competitive.predictive import (
    cache,
    cases,
    context_cache,
    visible_contrast,
)
from rsna.competitive.predictive import (
    visible_contrast_cache as candidate,
)

RECIPE = dict(side=32, halo=8, context_cap=4, classes=2)


def same(left, right):
    for a, b in zip(left, right, strict=True):
        for key in context_cache.KEYS:
            assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key])
        assert a["supports"].region_count == b["supports"].region_count
        for key in ("indptr", "regions", "log_prior"):
            assert torch.equal(getattr(a["supports"], key), getattr(b["supports"], key))


@pytest.fixture
def original(tmp_path):
    generator = torch.Generator().manual_seed(51)
    raw = torch.randint(80, 600, (4, 96, 112), generator=generator, dtype=torch.int32)
    raw[1] = raw[0]
    group = cache.SourceGroup(raw, torch.full((4,), 12, dtype=torch.int64),
        torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool),
        torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 2, 2]), torch.tensor([0, 0, 4, 8]))
    raw_path = tmp_path / "original-raw.npz"
    proof = cache.write(raw_path, [group], {"synthetic": True}, cap_bytes=16 * 2**20)
    raw_row = dict(uid="synthetic-case", path=str(raw_path), proof=proof)
    groups, _ = cache.load(raw_path, proof)
    contexts = cases.assemble(groups, **RECIPE)
    provenance = {"original_raw_cache_sha256": proof["sha256"], **RECIPE}
    path = tmp_path / "original-context.npz"
    row = dict(uid=raw_row["uid"], path=str(path), provenance=provenance,
               proof=context_cache.write(path, contexts, provenance, cap_bytes=16 * 2**20))
    return raw_row, row, groups


def prepare(original, path, **kwargs):
    return candidate.prepare_one(*original[:2], path,
        source=candidate.source_pins("a" * 40), cap_bytes=16 * 2**20, **RECIPE, **kwargs)


def test_one_new_file_replaces_two_variant_cache_without_changing_either_input(original, tmp_path):
    raw, old, groups = original
    before = {r["path"]: hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() for r in (raw, old)}
    rng = torch.get_rng_state().clone()
    row = prepare(original, tmp_path / "only-new-normalized.npz")
    pair = candidate.load_pair(row)
    for role, normalize in (("control", False), ("normalized", True)):
        expected, _ = visible_contrast.assemble(groups, normalize_images=normalize, **RECIPE)
        same(pair[role], expected)
        same(candidate.load_role(row, role), expected)
    assert len(list(tmp_path.glob("*.npz"))) == 3  # Two existing originals, one new file.
    assert row["original_control"] == old
    assert row["proof"]["bytes"] == (tmp_path / "only-new-normalized.npz").stat().st_size
    assert torch.equal(rng, torch.get_rng_state())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in before.items())


def test_does_not_overwrite_or_repeat_original_preparation(original, tmp_path):
    path = tmp_path / "normalized.npz"
    row = prepare(original, path)
    with pytest.raises(FileExistsError):
        prepare(original, path)
    assert cache.signature(path) == row["proof"]["stat"]


@pytest.mark.parametrize("change,match", [
    ("uid", "same original study"),
    ("raw", "different raw source"),
    ("recipe", "rendering recipe"),
])
def test_rejects_wrong_original_binding_before_writing(original, tmp_path, change, match):
    raw, old, groups = copy.deepcopy(original)
    if change == "uid":
        old["uid"] = "different"
    elif change == "raw":
        old["provenance"]["original_raw_cache_sha256"] = "0" * 64
    else:
        old["provenance"]["side"] = 64
    path = tmp_path / "absent.npz"
    with pytest.raises(ValueError, match=match):
        prepare((raw, old, groups), path)
    assert not path.exists()


@pytest.mark.parametrize("field", ["targets", "observed", "metadata", "priors"])
def test_virtual_control_rejects_semantically_misaligned_contexts(original, field):
    _, old, groups = original
    normalized, _ = visible_contrast.assemble(groups, **RECIPE)
    reference = context_cache.load(old["path"], old["proof"], provenance=old["provenance"])
    if field == "targets":
        normalized[0][field][0, 0] += .1
    elif field == "observed":
        normalized[0][field][0] = ~normalized[0][field][0]
    elif field == "metadata":
        normalized[0]["token_metadata"][0, 0] += 1
    else:
        normalized[0]["supports"].log_prior[0] += 1
    with pytest.raises(ValueError):
        candidate.matched_control(reference, normalized)


def test_undersized_storage_cap_rejects_before_creating_any_output(original, tmp_path):
    path = tmp_path / "absent.npz"
    with pytest.raises(ValueError, match="cap"):
        candidate.prepare_one(*original[:2], path, source=candidate.source_pins("a" * 40),
                              cap_bytes=1024, **RECIPE)
    assert not path.exists()


def test_load_rejects_changed_original_and_wrong_source_reference(original, tmp_path):
    row = prepare(original, tmp_path / "normalized.npz")
    changed = copy.deepcopy(row)
    changed["original_control"]["uid"] = "different"
    with pytest.raises(ValueError, match="provenance binding"):
        candidate.load_pair(changed)
    with open(original[1]["path"], "ab") as out:
        out.write(b"changed")
    with pytest.raises(ValueError, match="immutable context"):
        candidate.load_pair(row)


def test_unbound_implementation_rejects_before_output(original, tmp_path):
    pins = candidate.source_pins("a" * 40)
    pins["candidate_sha256"] = "0" * 64
    path = tmp_path / "absent.npz"
    with pytest.raises(ValueError, match="Loaded contrast implementation"):
        candidate.prepare_one(*original[:2], path, source=pins, cap_bytes=16 * 2**20, **RECIPE)
    assert not path.exists()


def test_control_decompresses_only_metadata_from_normalized_archive(original, tmp_path, monkeypatch):
    row = prepare(original, tmp_path / "normalized.npz")
    expected = candidate.load_pair(row)["control"]
    allowed = {"manifest", "c0_token_metadata", "c1_token_metadata",
               "c0_region_metadata", "c1_region_metadata"}
    getitem = np.lib.npyio.NpzFile.__getitem__
    accessed = []

    def guarded(self, key):
        if str(self.zip.filename) == row["path"]:
            assert key in allowed, "Virtual control must not inflate the second image archive"
            accessed.append(key)
        return getitem(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded)
    same(candidate.load_role(row, "control"), expected)
    assert set(accessed) == allowed


def test_single_role_load_rejects_modified_metadata_and_wrong_role(original, tmp_path):
    row = prepare(original, tmp_path / "normalized.npz")
    with pytest.raises(ValueError, match="explicit matched contrast role"):
        candidate.load_role(row, "other")
    with open(row["path"], "ab") as out:
        out.write(b"changed")
    for role in ("control", "normalized"):
        with pytest.raises(ValueError, match="immutable"):
            candidate.load_role(row, role)


def paired_cohort(tmp_path):
    rows, refs = [], []
    for i in range(2):
        folder = tmp_path / f"source-{i}"
        folder.mkdir()
        raw, old, _ = original.__wrapped__(folder)
        raw["uid"] = old["uid"] = f"synthetic-{i}"
        rows.append(raw)
        refs.append(old)
    return rows, refs


def iterator(rows, refs, tmp_path):
    roster = hashlib.sha256(json.dumps([r["uid"] for r in rows], separators=(",", ":")).encode()).hexdigest()
    return candidate.iter_units(rows, refs, expected_count=len(rows), roster_sha256=roster,
        scratch_parent=tmp_path, source=candidate.source_pins("a" * 40),
        per_study_cap_bytes=16 * 2**20, **RECIPE)


def test_streaming_only_retains_one_temporary_unit_and_preserves_originals(tmp_path):
    rows, refs = paired_cohort(tmp_path)
    previous = None
    with closing(iterator(rows, refs, tmp_path)) as stream:
        for ordinal, row in stream:
            assert ordinal in (0, 1) and row["uid"] == rows[ordinal]["uid"]
            assert len(list(tmp_path.glob("contrast-stream-unit-*"))) == 1
            if previous is not None:
                assert not previous.exists()
            previous = Path(row["path"])
            assert len(candidate.load_pair(row)) == 2
    assert not previous.exists() and not list(tmp_path.glob("contrast-stream-unit-*"))
    assert all(Path(row["path"]).exists() for row in rows + refs)


def test_interrupted_sender_only_removes_its_temporary_not_original_inputs(tmp_path):
    rows, refs = paired_cohort(tmp_path)
    with pytest.raises(RuntimeError, match="transport"):
        with closing(iterator(rows, refs, tmp_path)) as stream:
            _, row = next(stream)
            assert Path(row["path"]).is_file()
            raise RuntimeError("transport failed; receiver preserves its own partial output")
    assert not list(tmp_path.glob("contrast-stream-unit-*"))
    assert all(Path(row["path"]).exists() for row in rows + refs)


def test_incomplete_misaligned_or_reused_cohort_rejected_before_scratch(tmp_path):
    rows, refs = paired_cohort(tmp_path)
    variants = [
        (rows, refs[:1]),
        (rows, list(reversed(refs))),
        (list(reversed(rows)), list(reversed(refs))),
        ([rows[0], {**rows[0], "uid": rows[1]["uid"]}], refs),
    ]
    for raw, old in variants:
        with pytest.raises(ValueError):
            next(iterator(raw, old, tmp_path))
    assert not list(tmp_path.glob("contrast-stream-unit-*"))
