import json
import sqlite3

import pytest

from scripts.kaggle.gpu_capacity import FAMILIES, admission


class Remote:
    username = "godlydonuts"

    def __init__(self, statuses):
        self.statuses = statuses
        self.calls = []

    def kernel(self, method, slug, version):
        self.calls.append(method)
        if self.statuses[slug] == "VERSION_DRIFT":
            raise ValueError("Version changed")
        return (
            {"metadata": {"enableGpu": True}}
            if method == "GetKernel"
            else {"status": self.statuses[slug]}
        )


def ledger(*values):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE meta(key TEXT,value TEXT)")
    for key, value in values:
        db.execute("INSERT INTO meta VALUES(?,?)", (key, json.dumps(value)))
    return db


def pushed(slug, main=False):
    return {
        "state": "whole94_full_fit_pushed" if main else "importance_full_fit_pushed",
        "kernel_id": 1 if main else 2,
        "slug": slug,
        "version": 1,
    }


def test_main_and_sampler_fill_both_actual_lanes_before_any_mutation():
    db = ledger(
        (FAMILIES[0] + "a", pushed("godlydonuts/main", True)),
        (FAMILIES[1] + "b", pushed("godlydonuts/sample")),
    )
    remote = Remote({"godlydonuts/main": "RUNNING", "godlydonuts/sample": "RUNNING"})
    out = admission(db, remote, caller="serving", now=100)
    assert not out["available"] and out["effective_occupied_lanes"] == 2
    assert out["new_GPU_mutations"] == 0 and "SaveKernel" not in remote.calls


def test_main_lane_survives_gap_between_original_epochs():
    db = ledger((FAMILIES[1] + "b", pushed("godlydonuts/sample")))
    assert not admission(db, Remote({"godlydonuts/sample": "RUNNING"}), caller="serving")[
        "available"
    ]


def test_verified_final_main_releases_reserved_lane():
    db = ledger(
        (
            FAMILIES[0] + "final",
            {
                "state": "whole94_full_fit_actual_complete",
                "segment": 3,
                "receipt": {"completed": 1458},
            },
        ),
        (FAMILIES[1] + "b", pushed("godlydonuts/sample")),
    )
    assert admission(db, Remote({"godlydonuts/sample": "RUNNING"}), caller="serving")["available"]


def test_live_serving_lease_gets_next_auxiliary_lane_but_expires():
    db = ledger(
        (
            FAMILIES[2] + "c",
            {"state": "full_parent_epoch_qualification_accepted", "GPU_wait_until": 200},
        )
    )
    assert not admission(db, Remote({}), caller="sampling", now=100)["available"]
    assert admission(db, Remote({}), caller="serving", now=100)["available"]
    assert admission(db, Remote({}), caller="sampling", now=201)["available"]


@pytest.mark.parametrize("state", [None, "QUEUED", "RUNNING", "VERSION_DRIFT"])
def test_unresolved_or_changed_execution_is_never_called_idle(state):
    db = ledger((FAMILIES[1] + "b", pushed("godlydonuts/sample")))
    remote = Remote({"godlydonuts/sample": state})
    if state == "VERSION_DRIFT":
        with pytest.raises(ValueError):
            admission(db, remote, caller="serving")
    else:
        assert not admission(db, remote, caller="serving")["available"]


def test_repeated_nested_identity_is_counted_once():
    row = pushed("godlydonuts/sample")
    row["same_execution"] = dict(row)
    row["binding"] = {"accepted_recovery": {"kernel_id": 123}}
    db = ledger((FAMILIES[1] + "b", row))
    assert (
        len(
            admission(db, Remote({"godlydonuts/sample": "RUNNING"}), caller="sampling")[
                "actual_active_registered_sessions"
            ]
        )
        == 1
    )
