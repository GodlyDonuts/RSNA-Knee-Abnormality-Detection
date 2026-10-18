"""Simulated recovery ownership: preserve original E1 before two distinct resumes."""

import contextlib
import io
import json
import os
import runpy
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

HERE = Path(__file__).parent


@pytest.mark.parametrize(
    "damage", [None, "uncertain_publication", "training_failure", "unknown_once"]
)
def test_one_early_private_preservation_and_no_E1_GPU(damage):
    calls = []
    seen = {}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        control = root / "control"
        control.mkdir()
        for name in ("full_once.py.txt", "publish_epoch_once.py.txt", "funding_bridge.py"):
            (control / name).write_text("# frozen synthetic control\n")
        owner = control / "finish_once.py.txt"
        owner.write_text((HERE / "finish_once.py.txt").read_text())
        old = root / "artifacts/finding-stratified-bags-20261018-v1/full"
        old.mkdir(parents=True)
        (old / "continuation-owner.result.json").write_text(
            json.dumps(
                {"status": "finite_continuation_stopped_preserve_original_reconcile_read_only"}
            )
        )
        science = root / "initiative/research/finding-stratified-bags-20261018-v1"
        science.mkdir(parents=True)

        def actual(e):
            return {
                "status": "COMPLETE",
                "epoch": e,
                "role": "influence",
                "kernel_id": 139354995 if e == 1 else 100 + e,
                "source_sha256": str(e) * 64,
                "receipt": {"completed": e * 486},
            }

        (science / "epoch-1-actual-accepted.json").write_text(json.dumps(actual(1)))

        def run(args, **kwargs):
            name = Path(args[1]).name
            calls.append(tuple([name, *args[2:]]))
            if name == "publish_epoch_once.py.txt":
                e = int(args[2])
                if damage == "uncertain_publication":
                    return subprocess.CompletedProcess(args, 1, "", "synthetic uncertainty")
                a = actual(e)
                out = {
                    "status": "actual_finding_strata_epoch_private_full_download_size_SHA_verified",
                    "epoch": e,
                    "completed": 486 * e,
                    "original_Kaggle_kernel": a["kernel_id"],
                    "original_source_sha256": a["source_sha256"],
                    "full_optimizer_privately_verified": True,
                    "revision": "synthetic",
                }
                (science / f"epoch-{e}-private-verified.json").write_text(json.dumps(out))
            else:
                phase = args[2]
                e = int(args[3])
                assert e in (2, 3)
                assert (science / f"epoch-{e - 1}-private-verified.json").exists()
                if phase == "prepare":
                    out = {"status": "finding_strata_epoch_prepared_no_GPU"}
                elif phase == "push":
                    out = {"status": "one_finding_strata_full_epoch_pushed"}
                else:
                    seen[e] = seen.get(e, 0) + 1
                    if damage == "training_failure":
                        out = {"status": "ERROR"}
                    elif damage == "unknown_once" and seen[e] == 1:
                        out = {"status": None}
                    else:
                        out = actual(e)
                        (science / f"epoch-{e}-actual-accepted.json").write_text(json.dumps(out))
            return subprocess.CompletedProcess(args, 0, json.dumps(out), "")

        before = Path.cwd()
        try:
            os.chdir(root)
            with (
                patch("subprocess.run", side_effect=run),
                patch("time.sleep"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if damage in ("uncertain_publication", "training_failure"):
                    with pytest.raises((AssertionError, RuntimeError)):
                        runpy.run_path(str(owner))
                else:
                    runpy.run_path(str(owner))
        finally:
            os.chdir(before)
        assert calls[0] == ("publish_epoch_once.py.txt", "1")
        assert not any(
            name == "full_once.py.txt" and e == "1" for name, *rest in calls for e in rest[-1:]
        )
        if damage == "uncertain_publication":
            assert len(calls) == 1
        if damage in (None, "unknown_once"):
            result = json.loads(
                (
                    root
                    / "artifacts/finding-full-funding-recovery-20261018-v2/continuation-owner.result.json"
                ).read_text()
            )
            assert (
                result["status"]
                == "one_original_finding_strata_fixed_final3_fit_and_all_three_private_epochs_verified"
            )
            assert [c for c in calls if c[0] == "publish_epoch_once.py.txt"] == [
                ("publish_epoch_once.py.txt", str(e)) for e in (1, 2, 3)
            ]
            assert len([c for c in calls if c[0] == "full_once.py.txt" and c[1] == "push"]) == 2
