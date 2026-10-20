"""Only unattempted E2/E3 may follow an actual preserved E1; no repeated publication."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

D = Path("initiative/research/S6-Kaggle-production-route-20261019-v1")
SOURCE = (D / "finish_remaining_epochs_after_publication_once.py.txt").read_text()


@pytest.mark.parametrize("failure", [None, "publish2"])
def test_reuses_preserved_E1_and_only_runs_remaining_epochs(tmp_path, monkeypatch, failure):
    d = tmp_path / D
    d.mkdir(parents=True)
    logs = tmp_path / "artifacts/S6-Kaggle-production-route-20261019-v1"
    logs.mkdir(parents=True)
    for n in (
        "prepare_once.py.txt",
        "prepare.py.txt",
        "run_phase.py.txt",
        "cloud.py.txt",
        "publish_once.py.txt",
        "publish_epoch.py.txt",
        "epoch_check.py.txt",
    ):
        (d / n).write_text("# immutable synthetic control\n")
    (d / "actual-push-epoch-1.json").write_text(json.dumps({"kernel_id": 139699006}))
    (d / "actual-owner-handoff.json").write_text(
        json.dumps({"old_idle_owners_retired": [97036, 97047, 97060]})
    )
    (d / "actual-private-verified-epoch-1.json").write_text(
        json.dumps(
            {
                "status": "actual_complete_target_twoT4_epoch_private_all_objects_download_size_SHA_verified",
                "epoch": 1,
                "model_sha256": "4a7a9f9b59493d45b2820a9619f40fe4c60aaa77a5f5fcb07699864cc5f789df",
                "revision": "verified",
            }
        )
    )
    (d / "actual-publication-pointer-epoch-1.json").write_text(
        json.dumps(
            {
                "status": "actual_private_epoch_pointer_delivered_to_original_trainer_owner",
                "revision": "verified",
            }
        )
    )
    events = []

    def run(args, **kwargs):
        if args[0] == "ps":
            return SimpleNamespace(returncode=1)
        name = Path(args[1]).name
        epoch = int(args[-1])
        tag = (
            "publish"
            if name == "publish_once.py.txt"
            else "compile"
            if name == "prepare_once.py.txt"
            else "push"
        ) + str(epoch)
        events.append(tag)
        if tag == failure:
            return SimpleNamespace(returncode=1, stdout="", stderr="synthetic")
        if name == "publish_once.py.txt":
            z = {
                "publication": {
                    "status": "actual_complete_target_twoT4_epoch_private_all_objects_download_size_SHA_verified",
                    "epoch": epoch,
                },
                "pointer": {
                    "status": "actual_private_epoch_pointer_delivered_to_original_trainer_owner"
                },
            }
        elif name == "prepare_once.py.txt":
            z = {"status": "actual_changed_role_fixed3_full3887_T4_source_compiled"}
        else:
            identity = {"kernel_id": 100 + epoch, "source_sha256": str(epoch) * 64, "version": 1}
            (d / ("actual-push-epoch-" + str(epoch) + ".json")).write_text(json.dumps(identity))
            (d / ("actual-terminal-epoch-" + str(epoch) + ".json")).write_text(
                json.dumps(
                    {**identity, "status": "COMPLETE", "receipt": {"completed": 486 * epoch}}
                )
            )
            z = {"status": "complete_teacher_cross_full_fit_pushed"}
        return SimpleNamespace(returncode=0, stdout=json.dumps(z), stderr="")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "a" * 40 + "\n")
    scope = {
        "__file__": str(d / "finish_remaining_epochs_after_publication_once.py.txt"),
        "__name__": "synthetic_only",
    }
    if failure:
        with pytest.raises(AssertionError):
            exec(compile(SOURCE, "reconciled_owner", "exec"), scope)
        assert events == ["compile2", "push2", "publish2"]
    else:
        exec(compile(SOURCE, "reconciled_owner", "exec"), scope)
        assert events == ["compile2", "push2", "publish2", "compile3", "push3", "publish3"]
        assert (
            len(json.loads((d / "actual-three-epoch-complete.json").read_text())["publications"])
            == 3
        )
