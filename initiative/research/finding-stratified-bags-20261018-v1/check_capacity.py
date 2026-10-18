import json
import sqlite3

import pytest
from capacity import admission

from scripts.kaggle import gpu_capacity


@pytest.mark.parametrize("active,expected", [(0, True), (1, True), (2, False)])
def test_new_registered_live_actors_count_and_original_module_is_unchanged(active, expected):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE meta(key TEXT,value TEXT)")
    db.execute(
        "INSERT INTO meta VALUES(?,?)",
        (
            "licensed-parent-whole94-full-fit:finished",
            json.dumps(
                {
                    "state": "whole94_full_fit_actual_complete",
                    "segment": 3,
                    "receipt": {"completed": 1458},
                }
            ),
        ),
    )
    for i in range(active):
        key = ["family-influence-full:", "finding-strata-recovery:"][i]
        db.execute(
            "INSERT INTO meta VALUES(?,?)",
            (
                key + str(i),
                json.dumps(
                    {
                        "state": "current_pushed",
                        "slug": "godlydonuts/live-" + str(i),
                        "kernel_id": i + 1,
                        "version": 1,
                    }
                ),
            ),
        )

    class Remote:
        username = "godlydonuts"

        def kernel(self, method, slug, version):
            assert version == 1
            return (
                {"metadata": {"enableGpu": True}}
                if method == "GetKernel"
                else {"status": "RUNNING"}
            )

    before = gpu_capacity.FAMILIES
    out = admission(gpu_capacity, db, Remote())
    assert out["available"] is expected and out["effective_occupied_lanes"] == active
    assert gpu_capacity.FAMILIES == before
