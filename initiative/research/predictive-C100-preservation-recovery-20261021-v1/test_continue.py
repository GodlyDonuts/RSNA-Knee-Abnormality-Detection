import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("C100_recovery_tested", HERE / "continue.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def evidence():
    return (
        json.loads((HERE / "actual-original-failure.json").read_text()),
        json.loads((HERE / "actual-permissions-corrected.json").read_text()),
        {"phase": "publish", "epoch": 1, "source_git": m.SOURCE},
        {"returncode": 1, "stdout": ""},
    )


def test_actual_preupload_failure_admitted():
    m.admit(*evidence())


@pytest.mark.parametrize("change", [
    lambda a: a[0].update(publication_root_exists=True),
    lambda a: a[0].update(funding_exists=True),
    lambda a: a[0]["publisher_scheduler"].update(state="RUNNING"),
    lambda a: a[0]["publisher_scheduler"].update(job_id="different"),
    lambda a: a[1].update(training_restarted=True),
    lambda a: a[1]["files"][0].update(content_written=True),
    lambda a: a[2].update(epoch=2),
    lambda a: a[3].update(returncode=0),
])
def test_uncertain_or_unrelated_failure_refused(change):
    args = copy.deepcopy(evidence())
    change(args)
    with pytest.raises(ValueError):
        m.admit(*args)


def test_reconciled_first_epoch_then_new_epochs_only(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "publish-001-intent.json").touch()
    state = {"ready_epochs": [1, 2], "publications": {}}
    assert m.pending(state, old, new) == [1, 2]
    (new / "publish-001-intent.json").touch()
    with pytest.raises(ValueError):
        m.pending(state, old, new)
    state["publications"]["1"] = {
        "status": "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"}
    assert m.pending(state, old, new) == [2]
    (old / "publish-002-intent.json").touch()
    with pytest.raises(ValueError):
        m.pending(state, old, new)


def test_remote_uncertainty_refused(tmp_path):
    with pytest.raises(ValueError):
        m.pending({"ready_epochs": [1], "publications": {"1": {"status": "uncertain"}}},
                  tmp_path, tmp_path)
