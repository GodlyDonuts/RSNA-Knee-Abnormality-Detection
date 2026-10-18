"""Simulate orchestration side effects, never launch GPU/API/publication work."""

import contextlib
import io
import json
import os
import runpy
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).parent
program = (HERE / "finish_campaign_after_first_once.py.txt").read_text()


def scenario(damage=None):
    calls = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "artifacts/multisource-influence-20261018-v1"
        data.mkdir(parents=True)
        owner = root / "finish_campaign_after_first_once.py.txt"
        owner.write_text(program)
        for name in (
            "coordinate_full_once.py.txt",
            "full_coordinator_worker.py.txt",
            "publish_epoch_once.py.txt",
            "publish_epoch_worker.py.txt",
            "epoch_verify.py",
        ):
            (root / name).write_text("# simulated immutable source\n")
        (data / "first-full-owner.launched.json").write_text(
            json.dumps({"PID": 17423, "maximum_full_epoch_pushes": 1})
        )
        (data / "first-full-owner.result.json").write_text(
            json.dumps(
                {
                    "status": "failure"
                    if damage == "first_failure"
                    else "actual_family_influence_first_epoch_complete_requires_private_preservation",
                    "actual": {"kernel_id": 139338298},
                }
            )
        )

        def actual(epoch):
            return {
                "status": "COMPLETE",
                "epoch": epoch,
                "role": "influence",
                "kernel_id": 139338297 + epoch,
                "source_sha256": str(epoch) * 64,
            }

        (root / "epoch-1-actual-accepted.json").write_text(json.dumps(actual(1)))

        def run(args, **kwargs):
            name = Path(args[1]).name
            if name == "publish_epoch_once.py.txt":
                epoch = int(args[2])
                calls.append(("publish", epoch))
                if damage == "publish_failure":
                    return subprocess.CompletedProcess(args, 1, "", "synthetic refusal")
                a = actual(epoch)
                out = {
                    "status": "actual_family_influence_epoch_private_full_download_size_SHA_verified",
                    "epoch": epoch,
                    "completed": epoch * 486,
                    "original_Kaggle_kernel": a["kernel_id"],
                    "original_source_sha256": a["source_sha256"],
                    "revision": "simulated",
                }
                (root / f"epoch-{epoch}-private-verified.json").write_text(json.dumps(out))
            else:
                assert name == "coordinate_full_once.py.txt"
                phase, epoch = args[2], int(args[3])
                calls.append((phase, epoch))
                assert epoch in (2, 3), "Never repeat the original E1 push"
                assert (root / f"epoch-{epoch - 1}-private-verified.json").exists()
                if phase == "prepare":
                    out = {"status": "family_influence_epoch_prepared_no_GPU"}
                    (root / f"epoch-{epoch}-prepared.json").write_text(json.dumps(out))
                elif phase == "push":
                    out = {"status": "one_family_influence_full_epoch_pushed"}
                    (root / f"epoch-{epoch}-pushed.json").write_text(json.dumps(out))
                else:
                    out = (
                        {"status": "ERROR"}
                        if damage == "downstream_failure"
                        else {**actual(epoch), "receipt": {"synthetic": True}}
                    )
                    if out["status"] == "COMPLETE":
                        (root / f"epoch-{epoch}-actual-accepted.json").write_text(json.dumps(out))
            return subprocess.CompletedProcess(args, 0, json.dumps(out), "")

        old = Path.cwd()
        try:
            os.chdir(root)
            with (
                patch("subprocess.run", side_effect=run),
                patch("time.sleep"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                try:
                    runpy.run_path(str(owner))
                except (AssertionError, RuntimeError):
                    assert damage is not None
                else:
                    assert damage is None
        finally:
            os.chdir(old)
        result = json.loads((data / "continuation-owner.result.json").read_text())
        if damage is None:
            assert calls == [
                ("publish", 1),
                ("prepare", 2),
                ("push", 2),
                ("status", 2),
                ("publish", 2),
                ("prepare", 3),
                ("push", 3),
                ("status", 3),
                ("publish", 3),
            ]
            assert result["fixed_updates"] == 1458 and result["competition_entries"] == 0
        else:
            assert (
                result["status"]
                == "finite_continuation_stopped_preserve_original_reconcile_read_only"
            )
            assert not any(epoch == 3 for _, epoch in calls)
            if damage in ("first_failure", "publish_failure"):
                assert not any(name == "push" for name, _ in calls)


for damage in (None, "first_failure", "publish_failure", "downstream_failure"):
    scenario(damage)
result = {
    "status": "simulated_downstream_owner_order_and_failure_guards_passed",
    "cases": 4,
    "no_first_epoch_push": True,
    "no_resume_before_private_preservation": True,
    "actual_GPU_API_or_publication_work": False,
}
(HERE / "campaign-owner-checked.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
