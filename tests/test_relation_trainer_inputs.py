import copy
import hashlib
import io
import json
import tarfile

import pandas as pd
import pytest

from scripts.newton import launch_relation_gradient_probe as helper
from scripts.newton.admit_relation_trainer_inputs import fixture_request, verify_source
from scripts.newton.launch_relation_trainer_inputs import copy_fixture, verify_fixture_reference


def source_fixture(tmp_path):
    names = ["rsna/__init__.py", "scripts/newton/measure_relation_gradients.py"]
    extras = [
        "scripts/newton/" + name + ".py"
        for name in (
            "admit_relation_trainer_inputs",
            "launch_relation_trainer_inputs",
            "probe_relation_trainer_recovery",
        )
    ]
    files = {}
    for name in names + extras:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original immutable module\n")
        files[name] = {"bytes": path.stat().st_size, "sha256": helper.sha(path)}
    ref = {"source_git": "original", "files": {name: files[name] for name in names}}
    return ref, {"reference_source_git": "original", "files": files}


@pytest.mark.parametrize("change", [None, "core", "extra", "missing", "revision", "materialized"])
def test_expanded_source_preserves_the_whole_original_numerical_closure(tmp_path, change):
    reference, source = source_fixture(tmp_path)
    source = copy.deepcopy(source)
    if change == "core":
        source["files"]["rsna/__init__.py"]["sha256"] = "replacement"
    elif change == "extra":
        source["files"]["unapproved.py"] = {}
    elif change == "missing":
        del source["files"]["scripts/newton/probe_relation_trainer_recovery.py"]
    elif change == "revision":
        source["reference_source_git"] = "other"
    elif change == "materialized":
        (tmp_path / "rsna/__init__.py").write_text("modified")
    if change is None:
        verify_source(reference, source, tmp_path)
    else:
        with pytest.raises(ValueError):
            verify_source(reference, source, tmp_path)


def test_fixture_selection_is_training_outer_disjoint_and_bound_to_the_full_rosters():
    train = pd.DataFrame({"study_uid": [f"training-{i}" for i in range(20)]})
    outer = pd.DataFrame({"study_uid": [f"outer-{i}" for i in range(20)]})
    request = fixture_request(train, outer, {"accepted_release": "hash"})
    assert request["training_uids"] == train.head(16).study_uid.tolist()
    assert request["outer_uids"] == outer.head(16).study_uid.tolist()
    assert request["clinical_selection_allowed"] is False
    other = train.copy()
    other.loc[19, "study_uid"] = "different-full-training-roster"
    assert (
        fixture_request(other, outer, {})["full_training_uids_sha256"]
        != request["full_training_uids_sha256"]
    )
    with pytest.raises(ValueError):
        fixture_request(train, train, {})
    with pytest.raises(ValueError):
        fixture_request(train.head(15), outer, {})


@pytest.mark.parametrize("altered", [None, "training", "training_hash", "outer_hash", "release"])
def test_fixture_reference_rejects_substituted_original_development_membership(altered):
    train = [f"training-{i:02}" for i in range(20)]
    fixture = {
        "training_uids": train[:16],
        "full_training_uids_sha256": "t" * 64,
        "full_outer_uids_sha256": "o" * 64,
        "release_binding": {"accepted": "source"},
    }
    geometry = {"training_uids": train, "training_uids_sha256": "t" * 64}
    run = {
        "binding": {
            "training_uids_sha256": "t" * 64,
            "outer_uids_sha256": "o" * 64,
            "release": {"accepted": "source"},
        }
    }
    if altered == "training":
        fixture["training_uids"][0] = "different"
    elif altered == "training_hash":
        fixture["full_training_uids_sha256"] = "changed"
    elif altered == "outer_hash":
        fixture["full_outer_uids_sha256"] = "changed"
    elif altered == "release":
        fixture["release_binding"] = {"unaccepted": "source"}
    if altered is None:
        verify_fixture_reference(fixture, geometry, run)
    else:
        with pytest.raises(ValueError):
            verify_fixture_reference(fixture, geometry, run)


def cache_fixture(tmp_path, change):
    training = [f"training-{i}" for i in range(16)]
    outer = [f"outer-{i}" for i in range(16)]
    fixture = {
        "training_uids": training,
        "outer_uids": outer,
        "diagnostic_only": True,
        "clinical_selection_allowed": False,
    }
    table = tmp_path / "series.csv"
    rows = [{"StudyInstanceUID": uid, "SeriesInstanceUID": "series"} for uid in training + outer]
    if change == "csv_duplicate":
        rows.append(rows[0])
    if change == "path":
        rows[0]["SeriesInstanceUID"] = "../escape"
    pd.DataFrame(rows).to_csv(table, index=False)
    archive = tmp_path / "original.tar"
    with tarfile.open(archive, "w") as tar:
        for index, uid in enumerate(training + outer + ["unadmitted"]):
            for suffix in (".json", ".npz"):
                if change == "missing" and index == 0 and suffix == ".json":
                    continue
                payload = json.dumps({"original": uid}).encode()
                member = tarfile.TarInfo(f"cache/train_series/{uid}/series{suffix}")
                member.size = len(payload)
                if change == "link" and index == 0 and suffix == ".json":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/outside"
                    member.size = 0
                    tar.addfile(member)
                else:
                    tar.addfile(member, io.BytesIO(payload))
                    if change == "duplicate" and index == 0 and suffix == ".json":
                        tar.addfile(member, io.BytesIO(payload))
    proof = {"assets": {str(archive): {"stat": helper.signature(archive)}}}
    if change == "source_stat":
        proof["assets"][str(archive)]["stat"]["bytes"] += 1
    if change == "overlap":
        fixture["outer_uids"][0] = training[0]
    return archive, tmp_path / "cache", fixture, table, proof


@pytest.mark.parametrize(
    "change",
    [None, "missing", "duplicate", "link", "csv_duplicate", "path", "source_stat", "overlap"],
)
def test_original_cache_copy_admits_exact_32_studies_and_excludes_every_other_exam(
    tmp_path, change
):
    archive, cache, fixture, table, proof = cache_fixture(tmp_path, change)
    if change is not None:
        with pytest.raises(ValueError):
            copy_fixture(archive, cache, fixture, table, proof, helper)
        return
    result = copy_fixture(archive, cache, fixture, table, proof, helper)
    assert result["studies"] == 32 and len(result["members"]) == 64
    assert result["shared_pixel_bytes_written"] == 0
    assert not (cache / "train_series/unadmitted").exists()
    for name, row in result["members"].items():
        path = cache / name.removeprefix("cache/")
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    with pytest.raises(ValueError):
        copy_fixture(archive, cache, fixture, table, proof, helper)
