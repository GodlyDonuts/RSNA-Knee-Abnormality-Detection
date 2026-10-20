"""Exercise the new actual paired-view loop, including a padded two-rank batch."""

import base64
import hashlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import asdict
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest
import torch
from test_teacher_pair import same

from rsna.competitive import licensed_importance, paired_source_zero

D = Path("initiative/research/teacher-only-T4-recovery-20261020-v1")


def compiler():
    loader = SourceFileLoader("paired_zero_compiler", str(D / "engine.py.txt"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    value = importlib.util.module_from_spec(spec)
    loader.exec_module(value)
    return value


def test_new_recipe_changes_only_explicit_source_weight_and_role():
    reference = licensed_importance.Recipe(epochs=2, role="influence")
    candidate = paired_source_zero.Recipe(epochs=2)
    assert asdict(candidate) == {**asdict(reference), "source_weight": 0.0}
    with pytest.raises(ValueError):
        licensed_importance.Recipe(epochs=2, source_weight=0.0)
    for value in (False, 0.25, float("nan")):
        with pytest.raises(ValueError):
            paired_source_zero.Recipe(source_weight=value)
    with pytest.raises(ValueError):
        paired_source_zero.Recipe(role="uniform")


def test_compiler_refuses_changed_original_engine():
    original = Path("rsna/competitive/licensed_importance_train.py").read_text()
    compile(compiler().build(original), "actual-paired-zero-source", "exec")
    with pytest.raises(ValueError):
        compiler().build(original + "\n")


def test_private_notebook_compiler_preserves_model_prefix_and_binds_new_loop(monkeypatch):
    loader = SourceFileLoader("paired_zero_notebook", str(D / "compile.py.txt"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    c = importlib.util.module_from_spec(spec)
    loader.exec_module(c)
    prefix = "def _whole_model(x):return x\n_PROOFS=[]\n_BINDING={'old':True}\n_ROLE='old'\n"
    private = prefix + "def main():pass\nif __name__=='__main__':main()\n"
    original = "\n".join(
        f"{name}={value!r}"
        for name, value in {
            "FILES": {"rsna/competitive/old.py": "# original own module"},
            "WORKER": base64.b64encode(private.encode()).decode(),
            "PERMISSION": base64.b64encode(b"synthetic permission").decode(),
            "BINDING": {"old": True},
        }.items()
    )
    monkeypatch.setattr(c, "BASE_SHA", hashlib.sha256(original.encode()).hexdigest())
    additions = {name: "# synthetic own source" for name in c.ADDITIONS}
    main = (D / "worker.py.txt").read_text()
    bootstrap = (D / "bootstrap.py.txt").read_text()
    source, metadata = c.build(original, additions, main, bootstrap, "a" * 40)
    assert metadata["competition_entries"] == 0 and metadata["maximum_VM_seconds"] == 3600
    import ast

    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("FILES", "WORKER", "BINDING")
    }
    worker = base64.b64decode(values["WORKER"]).decode()
    assert "def _whole_model(x):return x" in worker and "_PROOFS=[]" in worker
    assert "_BINDING={'old':True}" not in worker and "_ROLE='old'" not in worker
    assert values["BINDING"]["source_loss_weight"] == 0.0
    assert "--nproc_per_node=2" in source and '"stop1"' in source
    assert "range(17, 32)" in worker and "range(17)" in worker
    with pytest.raises(ValueError):
        c.build(original + "\n", additions, main, bootstrap, "a" * 40)
    with pytest.raises(ValueError):
        c.build(original, {}, main, bootstrap, "a" * 40)


WORKER = """import csv,importlib.util,json,random,sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rsna.competitive.paired_source_zero import Recipe
from rsna.competitive.licensed_parent_train import PARENT_SHA
from test_teacher_pair import Model,data

def run(rank,world,root,phase):
 torch.set_num_threads(1)
 if world==2:
  dist.init_process_group('gloo',init_method='file://'+str(root/(phase+'.rendezvous')),rank=rank,world_size=world)
 spec=importlib.util.spec_from_file_location('actual_paired_zero_engine',root/'engine.py')
 engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
 random.seed(71);np.random.seed(71);torch.manual_seed(71)
 model=Model()
 if root.name.startswith('normalization'):
  model.backbone[1]=torch.nn.Identity()
 random.seed(100+rank);np.random.seed(100+rank);torch.manual_seed(100+rank)
 population=data(17)
 output=root/('full' if phase=='full' else 'resumed')
 result=engine.fit(model,population,output,{'parent_checkpoint_sha256':PARENT_SHA,'synthetic_CPU_only':True},Recipe(epochs=2),device=torch.device('cpu'),resume=phase=='resume',stop_after=1 if phase=='stop1' else None)
 calls=population.original.calls
 assert len(calls)%2==0
 for a,b in zip(calls[::2],calls[1::2]):
  assert a[:2]==b[:2] and a[2] is not b[2]
 if rank==0 and phase!='stop1':
  model.eval();rows=[]
  engineering=data(32).original
  with torch.no_grad():
   for i in range(17,32):
    views=[]
    for mirror in (False,True):
     case=engineering.case(i,list(range(94)),mirror)
     views.append(model(case['images'][None],torch.zeros(1,94)).sigmoid())
    rows.append([engineering.uids[i],*((views[0]+views[1])/2).flatten().tolist()])
  with (output/'predictions.csv').open('w',newline='') as f:
   csv.writer(f).writerows(rows)
  (output/'actual.json').write_text(json.dumps(result))
 if world==2:dist.destroy_process_group()

if __name__=='__main__':
 root=Path(sys.argv[1]);world=int(sys.argv[2]);phase=sys.argv[3]
 if world==2:mp.spawn(run,args=(world,root,phase),nprocs=2,join=True)
 else:run(0,world,root,phase)
"""


def execute(root, world, phases):
    root.mkdir()
    (root / "engine.py").write_text(
        compiler().build(Path("rsna/competitive/licensed_importance_train.py").read_text())
    )
    entry = root / "worker.py"
    entry.write_text(WORKER)
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()) + os.pathsep + str(Path.cwd() / "tests"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for phase in phases:
        result = subprocess.run(
            [sys.executable, str(entry), str(root), str(world), phase],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return root


@pytest.mark.parametrize("world", [1, 2])
def test_actual_loop_fresh_resume_all_state_snapshots_and_predictions_equal(tmp_path, world):
    if world == 2 and not dist_available():
        pytest.skip("Actual Gloo backend unavailable")
    root = execute(tmp_path / f"recovery{world}", world, ("full", "stop1", "resume"))

    def load(p):
        return torch.load(p, map_location="cpu", weights_only=True)

    full = load(root / "full/checkpoint.pt")
    resumed = load(root / "resumed/checkpoint.pt")
    same(full, resumed)
    assert full["completed"] == 6 and len(full["rng_by_rank"]) == world
    assert [x["studies"] for x in full["history"]] == [8, 8, 1, 8, 8, 1]
    assert [x["anatomical_views"] for x in full["history"]] == [16, 16, 2, 16, 16, 2]
    assert full["binding"]["source_loss_weight"] == 0.0
    assert full["binding"]["anatomical_views_per_study"] == 2
    for epoch in (1, 2):
        name = f"epoch-checkpoints/epoch-{epoch:03d}.pt"
        same(load(root / "full" / name), load(root / "resumed" / name))
    assert (root / "full/predictions.csv").read_bytes() == (
        root / "resumed/predictions.csv"
    ).read_bytes()


def dist_available():
    return torch.distributed.is_available() and torch.distributed.is_gloo_available()


def test_two_rank_mean_matches_single_rank_including_final_dummy(tmp_path):
    if not dist_available():
        pytest.skip("Actual Gloo backend unavailable")
    one = execute(tmp_path / "normalization1", 1, ("full",))
    two = execute(tmp_path / "normalization2", 2, ("full",))
    a, b = (
        torch.load(p / "full/checkpoint.pt", map_location="cpu", weights_only=True)
        for p in (one, two)
    )
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], atol=1e-7, rtol=1e-6)
    for x, y in zip(a["history"], b["history"], strict=True):
        assert x["loss"] == pytest.approx(y["loss"], abs=1e-7)
        assert x["gradient_norm"] == pytest.approx(y["gradient_norm"], abs=1e-7)
