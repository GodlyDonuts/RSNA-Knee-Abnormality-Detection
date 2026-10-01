import json
import tarfile
from io import BytesIO

import pytest

from scripts.newton.audit_relation_geometry import coordinates, digest, signature, survey


def sample(tmp_path):
    archive = tmp_path / "cache.tar"
    metadata = {
        "study_uid": "a",
        "series_uid": "s",
        "source_revision": "rev",
        "side": 256,
        "n_slices": 3,
        "cache_path": "train_series/a/s.npz",
        "ordering": "patient_geometry",
        "orientation": [1, 0, 0, 0, 1, 0],
        "positions": [[0, 0, 0], [0, 0, 5], [0, 0, 12]],
    }
    with tarfile.open(archive, "w") as stream:
        for name, raw in [
            ("cache/train_series/a/s.json", json.dumps(metadata).encode()),
            ("cache/train_series/outer/x.json", b"not-valid-json"),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            stream.addfile(member, BytesIO(raw))
    request = {
        "training_uids": ["a"],
        "training_uids_sha256": digest(["a"]),
        "source_revision": "rev",
        "series_table_sha256": "series-hash",
        "series": [
            {
                "study_uid": "a",
                "series_uid": "s",
                "n_slices": 3,
                "plane": "Sagittal",
                "uniform_indices": [0, 1, 2],
            }
        ],
    }
    proof = {
        "assets": {
            str(archive): {"stat": signature(archive), "sha256": "accepted-whole-byte-hash"}
        },
        "data": {"release_binding": {"source_revision": "rev"}},
    }
    run = {
        "training_studies": 1,
        "binding": {
            "training_uids_sha256": digest(["a"]),
            "release": {"series_sha256": "series-hash"},
        },
    }
    return request, archive, proof, run


def test_survey_reads_exact_training_geometry_and_skips_invalid_outer_json(tmp_path):
    result = survey(*sample(tmp_path))
    assert result["studies_with_eligible_relations"] == 1
    assert result["counts"]["possible_uniform_pairs"] == 3
    assert result["unsigned_uniform_pair_mm"]["median"] == 7
    assert result["unsigned_uniform_pair_mm"]["maximum"] == 12
    assert result["outer_geometry_read"] is False


@pytest.mark.parametrize(
    "failure",
    ["foreign_uid", "duplicate_series", "missing_member", "archive_changed", "series_binding"],
)
def test_survey_refuses_membership_or_asset_changes(tmp_path, failure):
    request, archive, proof, run = sample(tmp_path)
    if failure == "foreign_uid":
        request["training_uids"] = ["outer"]
    elif failure == "duplicate_series":
        request["series"] *= 2
    elif failure == "missing_member":
        request["series"][0]["series_uid"] = "absent"
    elif failure == "archive_changed":
        proof["assets"][str(archive)]["stat"]["bytes"] += 1
    else:
        request["series_table_sha256"] = "wrong"
    with pytest.raises(ValueError):
        survey(request, archive, proof, run)


def test_unknown_and_reversed_geometry_are_not_invented():
    assert coordinates({"n_slices": 2, "ordering": "instance_number"}) is None
    with pytest.raises(ValueError):
        coordinates(
            {
                "n_slices": 2,
                "ordering": "patient_geometry",
                "orientation": [1, 0, 0, 0, 1, 0],
                "positions": [[0, 0, 2], [0, 0, 1]],
            }
        )
