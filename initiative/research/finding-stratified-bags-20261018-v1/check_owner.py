"""Check side-effect order and publication failure boundaries without external work."""

import ast
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
    "damage", [None, "GPU_failure", "publication_failure", "unverified_optimizer"]
)
def test_single_fixed_campaign_preserves_before_each_resume(damage):
    calls = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        local = root / "artifacts/finding-stratified-bags-20261018-v1/full"
        local.mkdir(parents=True)
        owner = root / "finish_full_once.py.txt"
        owner.write_text((HERE / "finish_full_once.py.txt").read_text())
        for name in (
            "full_once.py.txt",
            "full_program.py.txt",
            "publish_epoch_once.py.txt",
            "publish_epoch_program.py.txt",
            "epoch_verify.py",
            "production.py",
            "sampling.py",
            "fit.py",
        ):
            (root / name).write_text("# simulated immutable helper\n")

        def actual(epoch):
            return {
                "status": "COMPLETE",
                "epoch": epoch,
                "role": "influence",
                "kernel_id": 100 + epoch,
                "source_sha256": str(epoch) * 64,
            }

        def run(args, **kwargs):
            name, epoch = Path(args[1]).name, int(args[-1])
            if name == "publish_epoch_once.py.txt":
                calls.append(("publish", epoch))
                if damage == "publication_failure":
                    return subprocess.CompletedProcess(args, 1, "", "synthetic failure")
                a = actual(epoch)
                out = {
                    "status": "actual_finding_strata_epoch_private_full_download_size_SHA_verified",
                    "epoch": epoch,
                    "completed": 486 * epoch,
                    "original_Kaggle_kernel": a["kernel_id"],
                    "original_source_sha256": a["source_sha256"],
                    "revision": "synthetic",
                    "full_optimizer_privately_verified": damage != "unverified_optimizer",
                }
                (root / f"epoch-{epoch}-private-verified.json").write_text(json.dumps(out))
            else:
                assert name == "full_once.py.txt"
                phase = args[2]
                calls.append((phase, epoch))
                if epoch > 1:
                    proof = json.loads(
                        (root / f"epoch-{epoch - 1}-private-verified.json").read_text()
                    )
                    assert proof["full_optimizer_privately_verified"] is True
                if phase == "prepare":
                    out = {"status": "finding_strata_epoch_prepared_no_GPU"}
                    (root / f"epoch-{epoch}-prepared.json").write_text(json.dumps(out))
                elif phase == "push":
                    out = {"status": "one_finding_strata_full_epoch_pushed"}
                    (root / f"epoch-{epoch}-pushed.json").write_text(json.dumps(out))
                else:
                    out = (
                        {"status": "ERROR"}
                        if damage == "GPU_failure"
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
                if damage is None:
                    runpy.run_path(str(owner))
                else:
                    with pytest.raises((AssertionError, RuntimeError)):
                        runpy.run_path(str(owner))
        finally:
            os.chdir(old)
        result = json.loads((local / "continuation-owner.result.json").read_text())
        if damage is None:
            assert calls == [
                (p, e) for e in (1, 2, 3) for p in ("prepare", "push", "status", "publish")
            ]
            assert result["fixed_updates"] == 1458 and result["competition_entries"] == 0
        else:
            assert not any(epoch > 1 for _, epoch in calls)


def test_original_transport_function_is_byte_equivalent_and_optimizer_is_not_omitted():
    old = (
        HERE.parents[1]
        / "experiments/competitive-v1/multisource-influence-v1/publish_epoch_worker.py.txt"
    ).read_text()
    new = (HERE / "publish_epoch_program.py.txt").read_text()

    def functions(s):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(s).body
            if isinstance(n, ast.FunctionDef)
        }

    assert functions(old)["download"] == functions(new)["download"]
    assert 'files["optimizer-checkpoint.pt"]=RAM/"optimizer-checkpoint.pt"' in new
    assert "dir=str(RAM)" in new and "shutil.rmtree(RAM)" in new
    assert 'dir="/tmp"' not in new
    assert "shared_publication_lock()" in new and "parent_commit=info.sha" in new
    assert "force_download=True" in new and "original_Kaggle" in new
    compile(new, "actual_finding_epoch_RAM_publisher", "exec")
