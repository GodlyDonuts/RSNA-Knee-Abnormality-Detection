"""Synthetic orchestration only: no network, training, publication or score."""

import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path("initiative/research/S6-Kaggle-production-route-20261019-v1")
SOURCE = (ROOT / "finish_three_epochs_once.py.txt").read_text()


def test_all_outer_and_embedded_pointer_programs_compile():
    for p in ROOT.glob("*.py.txt"):
        tree = ast.parse(p.read_text())
        compile(tree, str(p), "exec")
        for n in tree.body:
            if (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "program" for t in n.targets)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            ):
                compile(n.value.value, str(p) + ":embedded", "exec")


@pytest.mark.parametrize("failure", [None, "publish1", "pointer1", "push2"])
def test_each_epoch_is_preserved_before_continuation_and_failures_stop(
    tmp_path, monkeypatch, failure
):
    d = tmp_path / ROOT
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
        (d / n).write_text("# synthetic immutable control\n")

    def pushed(epoch):
        return {
            "kernel_id": 139699005 + epoch,
            "source_sha256": str(epoch) * 64,
            "version": 1,
            "epoch": epoch,
        }

    def terminal(epoch):
        return {**pushed(epoch), "status": "COMPLETE", "receipt": {"completed": 486 * epoch}}

    (d / "actual-push-epoch-1.json").write_text(json.dumps(pushed(1)))
    (d / "actual-terminal-epoch-1.json").write_text(json.dumps(terminal(1)))
    (d / "actual-owner-handoff.json").write_text(
        json.dumps({"old_idle_owners_retired": [97036, 97047, 97060]})
    )
    events = []

    def fake_run(args, **kwargs):
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
        if failure == tag:
            return SimpleNamespace(returncode=1, stdout="", stderr="synthetic phase failure")
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
            if failure == "pointer1" and epoch == 1:
                z["pointer"]["status"] = "not_delivered"
        elif name == "prepare_once.py.txt":
            z = {"status": "actual_changed_role_fixed3_full3887_T4_source_compiled"}
        else:
            z = {"status": "complete_teacher_cross_full_fit_pushed"}
            (d / ("actual-push-epoch-" + str(epoch) + ".json")).write_text(
                json.dumps(pushed(epoch))
            )
            (d / ("actual-terminal-epoch-" + str(epoch) + ".json")).write_text(
                json.dumps(terminal(epoch))
            )
        return SimpleNamespace(returncode=0, stdout=json.dumps(z), stderr="")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "a" * 40 + "\n")
    monkeypatch.setattr(subprocess, "run", fake_run)
    scope = {
        "__file__": str(d / "finish_three_epochs_once.py.txt"),
        "__name__": "synthetic_route_only",
    }
    if failure:
        with pytest.raises(AssertionError):
            exec(compile(SOURCE, "synthetic_finite_owner", "exec"), scope)
        assert not (d / "actual-three-epoch-complete.json").exists()
        assert (
            json.loads((logs / "full-route-stopped.json").read_text())["status"]
            == "stopped_without_retry_or_replacement"
        )
        assert events == (
            ["publish1"]
            if failure in ("publish1", "pointer1")
            else ["publish1", "compile2", "push2"]
        )
    else:
        exec(compile(SOURCE, "synthetic_finite_owner", "exec"), scope)
        assert events == [
            "publish1",
            "compile2",
            "push2",
            "publish2",
            "compile3",
            "push3",
            "publish3",
        ]
        z = json.loads((d / "actual-three-epoch-complete.json").read_text())
        assert (
            z["completed"] == 1458 and len(z["publications"]) == 3 and z["competition_entries"] == 0
        )
