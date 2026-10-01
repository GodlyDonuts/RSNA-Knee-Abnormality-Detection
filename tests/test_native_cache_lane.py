"""Finite CPU lanes cover the cohort once and stop safely at a failed shard."""

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.newton import run_native_cache_lane as lane
from scripts.newton import run_native_cache_shard as shard


def cohort(tmp_path, monkeypatch):
    base = tmp_path / "private"
    root = base / "batch"
    common = root / "common"
    common.mkdir(parents=True)
    runner = common / "runner.py"
    runner.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[2]);x=json.loads((p/'launch.json').read_text())\n"
        "token=json.load(sys.stdin);assert token['token']=='hf_test'\n"
        "n=x['shard'];(p/'called').write_text(str(n))\n"
        "r={'status':'original_source_paired_shard_and_private_preservation_complete',"
        "'shard':n,'job_id':'123','scratch_reclaimed_after_full_private_verification':True}\n"
        "(p/'result.json').write_text(json.dumps(r))\n"
    )
    units = {}
    for n in range(1, 75):
        control = root / f"shard-{n:03d}"
        control.mkdir()
        value = {
            "shard": n,
            "files": {"runner.py": {"path": str(runner), "sha256": lane.digest(runner)}},
        }
        (control / "launch.json").write_text(json.dumps(value))
        units[str(n)] = {
            "control": str(control),
            "launch_sha256": lane.digest(control / "launch.json"),
        }
    plan = {
        "source_git": "pinned",
        "lane_driver_sha256": lane.digest(Path(lane.__file__)),
        "units": units,
        "shards_by_lane": {"0": list(range(1, 75, 2)), "1": list(range(2, 75, 2))},
    }
    (root / "batch-plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(lane, "BASE", base)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    real_run = subprocess.run

    def run(command, **kwargs):
        return real_run([sys.executable, *command[1:]], **kwargs)

    monkeypatch.setattr(lane.subprocess, "run", run)
    return root, plan


def invoke(root, n, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lane", "--control", str(root), "--lane", str(n)])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"token":"hf_test"}'))
    lane.main()


def test_two_real_lanes_cover_all_74_shards_once(tmp_path, monkeypatch):
    root, plan = cohort(tmp_path, monkeypatch)
    for n in (0, 1):
        invoke(root, n, monkeypatch)
        result = json.loads((root / f"lane-{n}/complete.json").read_text())
        assert result["shards"] == plan["shards_by_lane"][str(n)]
        assert result["full_cohort_complete"] is False
    assert len(list(root.glob("shard-*/called"))) == 74
    with pytest.raises(FileExistsError, match="existing_lane"):
        invoke(root, 0, monkeypatch)


def test_changed_plan_stops_before_any_shard(tmp_path, monkeypatch):
    root, _ = cohort(tmp_path, monkeypatch)
    (root / "shard-003/launch.json").write_text("{}")
    with pytest.raises(ValueError, match="prebound_unit"):
        invoke(root, 0, monkeypatch)
    assert not list(root.glob("shard-*/called"))


def test_runner_failure_stops_lane_without_repeating_completed_shards(tmp_path, monkeypatch):
    root, _ = cohort(tmp_path, monkeypatch)
    real_run = lane.subprocess.run

    def fail_three(command, **kwargs):
        if command[-1].endswith("shard-003"):
            raise subprocess.CalledProcessError(2, command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(lane.subprocess, "run", fail_three)
    with pytest.raises(subprocess.CalledProcessError):
        invoke(root, 0, monkeypatch)
    assert (root / "shard-001/called").exists()
    assert not (root / "shard-005/called").exists()
    assert json.loads((root / "lane-0/failure.json").read_text())["shard"] == 3
    assert (root / "lane-0/shard-001-complete.json").exists()


def test_partition_rejects_missing_duplicate_and_overlapping_assignments():
    plan = {
        "units": {str(i): {"control": str(i)} for i in range(1, 75)},
        "shards_by_lane": {"0": list(range(1, 75, 2)), "1": list(range(2, 75, 2))},
    }
    assert set(lane.validate_lane(plan, 0)).isdisjoint(lane.validate_lane(plan, 1))
    plan["units"]["74"]["control"] = "1"
    with pytest.raises(ValueError, match="distinct_shard"):
        lane.validate_lane(plan, 0)
    plan["units"]["74"]["control"] = "74"
    plan["shards_by_lane"]["1"].append(1)
    with pytest.raises(ValueError, match="exact_two_lane"):
        lane.validate_lane(plan, 1)


def test_shared_input_byte_binding_and_private_path(tmp_path):
    base = tmp_path / "private"
    control = base / "unit"
    common = base / "common"
    control.mkdir(parents=True)
    common.mkdir()
    target = common / "worker.py"
    target.write_bytes(b"pinned source")
    files = {
        "worker.py": {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    }
    assert shard.staged_paths(control, files, base)["worker.py"] == target
    target.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="staged_launch"):
        shard.staged_paths(control, files, base)
    target.unlink()
    target.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="staged_launch"):
        shard.staged_paths(control, files, base)


def test_publication_lock_serializes_actual_processes(tmp_path):
    base = tmp_path / "private"
    control = base / "batch"
    control.mkdir(parents=True)
    lock = control / "publication.lock"
    probe = (
        "from pathlib import Path;from scripts.newton.run_native_cache_shard import publication_lock;"
        "import sys;"
        "ctx=publication_lock(Path(sys.argv[1]),Path(sys.argv[2]),timeout=.05,poll=.01);"
        "ctx.__enter__();ctx.__exit__(None,None,None)"
    )
    with shard.publication_lock(lock, base):
        assert (
            subprocess.run(
                [sys.executable, "-c", probe, str(lock), str(base)], capture_output=True
            ).returncode
            != 0
        )
    assert (
        subprocess.run(
            [sys.executable, "-c", probe, str(lock), str(base)], capture_output=True
        ).returncode
        == 0
    )
    lock.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="private_owned_publication"):
        with shard.publication_lock(lock, base):
            pytest.fail("symlink lock accepted")
