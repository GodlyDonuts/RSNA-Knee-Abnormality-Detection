"""Actual new recovery worker/decoder, fresh processes and adversarial state changes."""

import importlib.util
import json
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest
import torch

D = Path("initiative/research/teacher-pair-real-recovery-20261019-v1").resolve()


def module(name):
    loader = SourceFileLoader("teacher_pair_recovery_" + name, str(D / (name + ".py.txt")))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def actual_fresh_processes(tmp_path_factory):
    root = tmp_path_factory.mktemp("new-actual-recovery-engine")
    entry = root / "fixture.py"
    entry.write_text("""import sys,torch
from pathlib import Path
from test_teacher_pair import Model,data,engine
from test_teacher_pair_real_recovery import module
from rsna.schema import TARGETS
class OptionalModel(Model):
 def forward(self,x,logq=None):
  if logq is None:logq=torch.zeros(x.shape[:2])
  return super().forward(x,logq)
torch.set_num_threads(1);torch.manual_seed(77)
root=Path(sys.argv[1]);d=data(32)
ctx={'model':OptionalModel(),'fit':engine(root).fit,'original':d.original,'data':d,'device':torch.device('cpu'),'engineering_indices':list(range(32)),'targets':TARGETS,'source_binding':{'synthetic':True,'GPU_acceptance':False}}
module('worker').run(ctx,sys.argv[2],sys.argv[3],root)
""")
    phases = []
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(Path.cwd()) + os.pathsep + str(Path.cwd() / "tests"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for objective in ("pointwise", "teacher_rank"):
        for phase in ("full", "stop1", "resume"):
            x = subprocess.run(
                [sys.executable, str(entry), str(root), phase, objective],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert x.returncode == 0, x.stderr
            phases.append((objective, phase))
    return root, phases


def test_actual_worker_six_fresh_processes_all_decoded_states_snapshots_CSV_equal(
    actual_fresh_processes,
):
    root, phases = actual_fresh_processes
    assert len(phases) == 6
    driver = module("driver")
    a = driver.compare(root, "pointwise")
    b = driver.compare(root, "teacher_rank")
    assert a["initial_state_sha256"] == b["initial_state_sha256"]
    assert a["final_state_digest"] != b["final_state_digest"]
    assert (
        a["exact_16x12_two_view_full94_MRI_CSV_bytes_equal"]
        and b["all_epoch_snapshot_states_equal"]
    )
    for role in ("pointwise", "teacher_rank"):
        file = root / role / "full/engineering-predictions.private.csv"
        assert len(file.read_text().splitlines()) == 17
        receipt = json.loads((root / role / "full/full-process.json").read_text())
        assert receipt["device_type"] == "cpu" and receipt["trainable_parameters"] < 73136096


@pytest.mark.parametrize("tamper", ["AdamW", "RNG", "epoch_snapshot", "prediction"])
def test_actual_decoder_rejects_real_state_or_prediction_changes(actual_fresh_processes, tamper):
    root, _ = actual_fresh_processes
    d = root / "teacher_rank/resumed"
    path = d / "checkpoint.pt"
    if tamper == "epoch_snapshot":
        path = d / "epoch-checkpoints/epoch-002.pt"
    elif tamper == "prediction":
        path = d / "engineering-predictions.private.csv"
    original = path.read_bytes()
    try:
        if tamper == "prediction":
            path.write_bytes(original + b"changed\n")
        else:
            x = torch.load(path, map_location="cpu", weights_only=True)
            if tamper == "AdamW":
                next(iter(x["optimizer"]["state"].values()))["exp_avg"].add_(1)
            elif tamper == "RNG":
                x["rng_by_rank"][0]["torch"][0] ^= 1
            else:
                x["completed"] += 1
            torch.save(x, path)
        with pytest.raises(AssertionError):
            module("driver").compare(root, "teacher_rank")
    finally:
        path.write_bytes(original)
