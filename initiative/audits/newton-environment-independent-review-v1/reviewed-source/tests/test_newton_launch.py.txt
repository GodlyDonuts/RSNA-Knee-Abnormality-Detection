"""Offline launch safety and actual signal propagation; never talks to a scheduler."""

import hashlib
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / "newton" / (name + ".py")
    spec = importlib.util.spec_from_file_location("newton_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch, preflight = load_script("launch"), load_script("preflight")
stage_weights, supervise = load_script("stage_weights"), load_script("supervise")


def arguments(tmp_path, **changes):
    args = SimpleNamespace(
        account="research",
        partition="gpu",
        time="02:00:00",
        job_name="folds",
        repo="/project/repo with spaces",
        python="/project/env/bin/python",
        studies="/project/data/studies.parquet",
        series="/project/data/series.csv",
        cache="/scratch/cache",
        cache_manifest="/project/data/cache-index.parquet",
        folds="/project/folds.json",
        artifact_root="/project/repo with spaces",
        logs="/project/logs with spaces",
        expected_git="a" * 40,
        config=["/project/fold0.json"],
        output=["/project/run 0"],
        resume=None,
        cpus=8,
        memory_gb=32,
        min_free_gb=20,
        gpus=1,
        gpu_type=None,
        mode="independent",
        signal_seconds=300,
        checkpoint_grace_seconds=120,
        write_script=tmp_path / "job with spaces.sbatch",
        dry_run=True,
    )
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def test_generator_is_dry_run_and_safe_for_spaces(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no process allowed"))
    args = arguments(tmp_path, config=["/project/config ; touch BAD.json"])
    plan = launch.generate(args)
    assert plan["submitted"] is False
    assert not args.write_script.exists()
    assert shlex.join(["cd", "--", args.repo]) in plan["script"]
    assert "'/project/config ; touch BAD.json'" in plan["script"]
    assert shlex.split(plan["manual_submission_command"]) == ["sbatch", str(args.write_script)]
    assert shlex.split(plan["before_manual_submission"])[-1] == args.logs
    assert "sbatch " not in plan["script"]
    assert "--device cuda" in plan["script"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("account", "CHANGEME"),
        ("partition", "gpu;bad"),
        ("repo", "/scratch/$USER/repo"),
        ("logs", "/scratch/logs\n#SBATCH --x"),
        ("expected_git", "main"),
        ("expected_git", "0" * 40),
        ("time", "00:01:00"),
        ("signal_seconds", 60),
        ("cpus", 0),
        ("memory_gb", 0),
    ],
)
def test_unresolved_or_unsafe_parameters_fail_closed(tmp_path, field, value):
    with pytest.raises(ValueError):
        launch.generate(arguments(tmp_path, **{field: value}))


def test_two_gpu_default_is_independent_and_ddp_explicit(tmp_path):
    args = arguments(
        tmp_path, gpus=2, config=["/p/f0.json", "/p/f1.json"], output=["/p/o0", "/p/o1"]
    )
    independent = launch.generate(args)["script"]
    assert independent.count("--nproc-per-node=1") == 2
    assert 'CUDA_VISIBLE_DEVICES="${assigned_gpus[0]}"' in independent
    assert 'CUDA_VISIBLE_DEVICES="${assigned_gpus[1]}"' in independent
    ddp = launch.generate(arguments(tmp_path, gpus=2, mode="ddp"))["script"]
    assert ddp.count("--nproc-per-node=2") == 1
    assert "--require-nccl" in ddp
    assert "--max-restarts=0" in ddp
    assert "scontrol requeue" not in ddp
    with pytest.raises(ValueError):
        launch.generate(arguments(tmp_path, gpus=2))


def test_generated_shell_syntax_and_signal_route(tmp_path):
    script = launch.generate(arguments(tmp_path))["script"]
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "#SBATCH --signal=B:USR1@300" in script
    assert "trap request_checkpoint USR1 TERM INT" in script
    assert 'kill -USR1 "$pid"' in script
    assert "scripts/newton/supervise.py" in script


def test_cuda_cannot_silently_fall_back():
    info = {
        "installed": True,
        "cuda_available": False,
        "visible_gpu_count": 0,
        "nccl_version": None,
    }
    errors = preflight.device_errors(info, "cuda", 2, True)
    assert "cuda_requested_but_unavailable" in errors
    assert "assigned_gpu_count_mismatch" in errors
    assert "nccl_required_but_unavailable" in errors
    assert not preflight.device_errors(info, "cpu", 0, False)


def test_provenance_removes_paths_tokens_urls():
    value = preflight.safe_output(
        "GPU /Users/person/private/file hf_" + "x" * 30 + " https://host/?token=private"
    )
    assert "/Users" not in value and "person" not in value
    assert "hf_" not in value and "token=private" not in value


def test_encoder_preflight_hashes_local_files_without_paths(tmp_path):
    config, weights = tmp_path / "config.json", tmp_path / "weights.pt"
    config.write_text("{}")
    weights.write_bytes(b"local-weights")
    encoder = {
        "architecture": "dinov2_b14_reg",
        "source_id": "org/model",
        "source_revision": "a" * 40,
        "license": "apache-2.0",
        "transformers_version": "5.6.1",
        "state_dict_format": "pytorch",
        "config_path": str(config),
        "weights_path": str(weights),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    versions = {"transformers": "5.6.1", "safetensors": "0.7.0"}
    result = preflight.encoder_summary({"encoder": encoder}, versions)
    assert str(tmp_path) not in json.dumps(result)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="model_file_hash_mismatch"):
        preflight.encoder_summary({"encoder": encoder}, versions)


@pytest.mark.skipif(os.name != "posix", reason="Slurm signal contract is POSIX")
def test_actual_usr1_reaches_worker_and_preserves_fresh_checkpoint(tmp_path):
    ready, checkpoint, worker, fake = signal_workers(tmp_path)
    supervisor = Path(__file__).parents[1] / "scripts/newton/supervise.py"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(supervisor),
            "--grace-seconds",
            "5",
            "--checkpoint",
            str(checkpoint),
            "--",
            sys.executable,
            str(fake),
            str(worker),
            str(checkpoint),
            str(ready),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        proc.send_signal(signal.SIGUSR1)
        stdout, stderr = proc.communicate(timeout=15)
        assert proc.returncode == 75, (stdout, stderr)
        assert checkpoint.read_bytes() == b"atomic-test-checkpoint"
        assert json.loads(stdout)["workers_signaled"] == 1
        assert json.loads(stdout)["checkpoint_stop_verified"] is True
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=20)


def signal_workers(tmp_path):
    ready, checkpoint = tmp_path / "ready", tmp_path / "checkpoint.pt"
    worker = tmp_path / "worker.py"
    worker.write_text("""import hashlib,json,pathlib,signal,sys,time
checkpoint,ready=map(pathlib.Path,sys.argv[1:3])
def stop(*args):
    temporary=checkpoint.with_suffix('.partial')
    temporary.write_bytes(b'atomic-test-checkpoint')
    temporary.replace(checkpoint)
    binding={'test':True}
    (checkpoint.parent/'run.json').write_text(json.dumps({'binding':binding}))
    receipt={'status':'checkpointed_for_resume','exit_code':75,
        'binding_sha256':hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (checkpoint.parent/'invocation.json').write_text(json.dumps(receipt))
    sys.exit(75)
signal.signal(signal.SIGUSR1,stop)
ready.write_text('ready')
while True: time.sleep(.02)
""")
    fake = tmp_path / "fake_torchrun.py"
    fake.write_text("""import subprocess,sys
child=subprocess.Popen([sys.executable,sys.argv[1],*sys.argv[2:],'rsna.competitive.train'])
sys.exit(child.wait())
""")
    return ready, checkpoint, worker, fake


@pytest.mark.skipif(os.name != "posix", reason="Slurm signal contract is POSIX")
@pytest.mark.parametrize("jobs", [1, 2])
def test_generated_batch_shell_forwards_usr1_and_returns75(tmp_path, jobs):
    scripts = tmp_path / "scripts/newton"
    scripts.mkdir(parents=True)
    # Hardware and trainer are test doubles; the generated Bash and supervisor are real.
    (scripts / "preflight.py").write_text("raise SystemExit(0)\n")
    real_supervisor = Path(__file__).parents[1] / "scripts/newton/supervise.py"
    shim = """import pathlib,runpy,sys
checkpoint=pathlib.Path(sys.argv[sys.argv.index('--checkpoint')+1])
root=checkpoint.parent
sys.argv=[sys.argv[0],'--grace-seconds','5','--checkpoint',str(checkpoint),'--',
sys.executable,str(root/'fake_torchrun.py'),str(root/'worker.py'),str(checkpoint),str(root/'ready')]
runpy.run_path(SUPERVISOR,run_name='__main__')
""".replace("SUPERVISOR", repr(str(real_supervisor)))
    (scripts / "supervise.py").write_text(shim)
    outputs, ready_files = [], []
    for index in range(jobs):
        output = tmp_path / ("output with spaces " + str(index))
        output.mkdir()
        ready, _, _, _ = signal_workers(output)
        outputs.append(str(output))
        ready_files.append(ready)
    args = arguments(
        tmp_path,
        repo=str(tmp_path),
        python=sys.executable,
        gpus=jobs,
        config=[str(tmp_path / ("config" + str(i) + ".json")) for i in range(jobs)],
        output=outputs,
    )
    job = tmp_path / "test-job.sh"
    job.write_text(launch.generate(args)["script"])
    env = {**os.environ, "SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0,1" if jobs == 2 else "0"}
    proc = subprocess.Popen(
        ["bash", str(job)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not all(p.exists() for p in ready_files)
            and proc.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert all(p.exists() for p in ready_files)
        proc.send_signal(signal.SIGUSR1)
        stdout, stderr = proc.communicate(timeout=20)
        assert proc.returncode == 75, (stdout, stderr)
        assert (
            sum(json.loads(line)["checkpoint_stop_verified"] for line in stdout.splitlines())
            == jobs
        )
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=20)


def test_periodic_checkpoint_is_not_proof_of_clean_preemption(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"periodic-only")
    (tmp_path / "run.json").write_text(json.dumps({"binding": {}}))
    assert not supervise.checkpoint_stop_verified(checkpoint, 0)
    receipt = {
        "status": "checkpointed_for_resume",
        "exit_code": 75,
        "binding_sha256": hashlib.sha256(b"{}").hexdigest(),
        "checkpoint_sha256": hashlib.sha256(b"periodic-only").hexdigest(),
    }
    path = tmp_path / "invocation.json"
    path.write_text(json.dumps(receipt))
    assert supervise.checkpoint_stop_verified(checkpoint, 0)
    assert not supervise.checkpoint_stop_verified(checkpoint, time.time() + 10)
    checkpoint.write_bytes(b"unrelated-checkpoint")
    assert not supervise.checkpoint_stop_verified(checkpoint, 0)


def staged_manifest(tmp_path):
    config = tmp_path / "configs/competitive/weights/model.config.json"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"{}")
    payload = b"synthetic-local-weights"
    manifest = tmp_path / "manifest.json"
    value = {
        "schema_version": 1,
        "encoder": {
            "architecture": "dinov2_b14_reg",
            "source_id": "facebook/dinov2-with-registers-base",
            "source_revision": "a" * 40,
            "license": "apache-2.0",
            "transformers_version": "5.6.1",
            "state_dict_format": "safetensors",
            "config_path": str(config.relative_to(tmp_path)),
            "weights_path": "artifacts/encoder-weights/model/model.safetensors",
            "config_sha256": hashlib.sha256(b"{}").hexdigest(),
            "weights_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "download": {
            "config_filename": "config.json",
            "config_bytes": 2,
            "weights_filename": "model.safetensors",
            "weights_bytes": len(payload),
        },
    }
    manifest.write_text(json.dumps(value))
    return manifest, tmp_path / value["encoder"]["weights_path"], payload


def test_weight_staging_is_offline_by_default_and_no_overwrite(tmp_path, monkeypatch):
    manifest, weights, payload = staged_manifest(tmp_path)
    monkeypatch.setattr(
        stage_weights.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network forbidden")
    )
    with pytest.raises(ValueError, match="offline"):
        stage_weights.stage(manifest, tmp_path)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(payload)
    report = stage_weights.stage(manifest, tmp_path)
    assert report["read_only"] and not report["downloaded"]
    assert str(tmp_path) not in json.dumps(report)
    weights.write_bytes(b"existing-wrong-file")
    with pytest.raises(ValueError):
        stage_weights.stage(manifest, tmp_path, download=True)
    assert weights.read_bytes() == b"existing-wrong-file"


def test_weight_download_requires_allocation_and_hashes_stream(tmp_path, monkeypatch):
    import io

    manifest, weights, payload = staged_manifest(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="allocated"):
        stage_weights.stage(manifest, tmp_path, download=True)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        stage_weights.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload)
    )
    assert stage_weights.stage(manifest, tmp_path, download=True)["downloaded"]
    assert weights.read_bytes() == payload
    assert not list(weights.parent.glob(".stage-*"))
    assert not weights.with_name(weights.name + ".stage-lock").exists()


def test_bad_weight_stream_and_existing_lock_fail_closed(tmp_path, monkeypatch):
    import io

    manifest, weights, payload = staged_manifest(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        stage_weights.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x" * len(payload))
    )
    with pytest.raises(ValueError, match="sha256"):
        stage_weights.stage(manifest, tmp_path, download=True)
    assert not weights.exists()
    lock = weights.with_name(weights.name + ".stage-lock")
    lock.write_text("interrupted attempt")
    with pytest.raises(FileExistsError):
        stage_weights.stage(manifest, tmp_path, download=True)
    assert lock.read_text() == "interrupted attempt"


def test_archive_launch_uses_exact_cache_path_and_cleans_on_failure(tmp_path):
    scripts = tmp_path / "scripts/newton"
    scripts.mkdir(parents=True)
    runtime_root = tmp_path / "runtime with spaces"
    record = tmp_path / "calls.jsonl"
    (scripts / "runtime_archive.py").write_text(
        "import json,pathlib,sys\n"
        f"record=pathlib.Path({str(record)!r})\n"
        'with record.open("a") as output: output.write(json.dumps(sys.argv[1:])+"\\n")\n'
        f'print(json.dumps({{"runtime_root":{str(runtime_root)!r}}}))\n'
    )
    (scripts / "preflight.py").write_text(
        "import json,pathlib,sys\n"
        f'with pathlib.Path({str(record)!r}).open("a") as output: output.write(json.dumps(sys.argv[1:])+"\\n")\n'
        "raise SystemExit(7)\n"
    )
    args = arguments(
        tmp_path,
        repo=str(tmp_path),
        python=sys.executable,
        cache=None,
        cache_archive="/project/cache archive.tar",
        cache_archive_sha256="b" * 64,
        cache_archive_files=2248,
        cache_unpacked_bytes=4077105588,
        scratch_base="/tmp",
    )
    script = launch.generate(args)["script"]
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"},
    )
    assert result.returncode == 7, result.stderr
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert calls[0][0] == "extract"
    assert calls[0][calls[0].index("--archive") + 1] == args.cache_archive
    assert calls[1][calls[1].index("--cache") + 1] == str(runtime_root / "cache")
    assert calls[2] == ["cleanup", "--root", str(runtime_root)]
    assert "--rdzv-endpoint=127.0.0.1:0" in script
    assert "--standalone" not in script


@pytest.mark.parametrize(
    "field,value",
    [
        ("cache_archive_sha256", None),
        ("cache_archive_files", 0),
        ("cache_unpacked_bytes", -1),
        ("scratch_base", "/tmp/$USER"),
        ("cache", "/also/direct/cache"),
    ],
)
def test_archive_launch_requires_unambiguous_pinned_bounds(tmp_path, field, value):
    args = arguments(
        tmp_path,
        cache=None,
        cache_archive="/p/cache.tar",
        cache_archive_sha256="b" * 64,
        cache_archive_files=2248,
        cache_unpacked_bytes=4077105588,
        scratch_base="/tmp",
    )
    setattr(args, field, value)
    with pytest.raises(ValueError):
        launch.generate(args)


def load_environment_script(monkeypatch):
    monkeypatch.setitem(sys.modules, "runtime_archive", load_script("runtime_archive"))
    return load_script("runtime_environment")


def environment_tar(tmp_path, *, link_target=None, extra=None):
    import io
    import tarfile

    path = tmp_path / "environment.tar"
    regular = [
        ("environment/pyvenv.cfg", b"include-system-site-packages = false\n"),
        ("environment/lib/package.py", b"value=1\n"),
    ]
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in regular:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        for name, target in [
            ("environment/bin/python", link_target or sys.executable),
            ("environment/bin/python3", "python"),
            ("environment/lib64", "lib"),
        ]:
            member = tarfile.TarInfo(name)
            member.type = tarfile.SYMTYPE
            member.linkname = target
            archive.addfile(member)
        if extra:
            member = tarfile.TarInfo(extra[0])
            member.type = tarfile.SYMTYPE
            member.linkname = extra[1]
            archive.addfile(member)
    return (
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        len(regular),
        sum(len(p) for _, p in regular),
    )


def environment_args(tmp_path, monkeypatch, **changes):
    module = load_environment_script(monkeypatch)
    monkeypatch.setattr(module, "local_disk", lambda p: True)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: SimpleNamespace(free=200 * 2**30))
    archive, sha, count, size = environment_tar(tmp_path)
    args = dict(
        archive=archive,
        sha256=sha,
        expected_files=count,
        expected_bytes=size,
        scratch_base=tmp_path,
        min_free_gib=20,
        shared_python=Path(sys.executable),
        job_id="123",
    )
    args.update(changes)
    return module, args


def test_environment_archive_uses_only_exact_python_and_cleans_owned_tree(tmp_path, monkeypatch):
    module, args = environment_args(tmp_path, monkeypatch)
    result = module.extract(**args)
    root = Path(result["runtime_root"])
    assert Path(result["python"]).resolve() == Path(sys.executable).resolve()
    assert (root / "environment/lib64/package.py").read_bytes() == b"value=1\n"
    assert root.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ValueError):
        module.cleanup(root, "124")
    module.cleanup(root, "123")
    assert not root.exists()


@pytest.mark.parametrize(
    "bad", ["wrong_hash", "wrong_count", "wrong_bytes", "external_link", "escaping_link"]
)
def test_environment_archive_rejects_unsafe_or_unbound_bytes(tmp_path, monkeypatch, bad):
    module, args = environment_args(tmp_path, monkeypatch)
    if bad == "wrong_hash":
        args["sha256"] = "0" * 64
    elif bad == "wrong_count":
        args["expected_files"] += 1
    elif bad == "wrong_bytes":
        args["expected_bytes"] += 1
    else:
        extra = (
            ("environment/bin/other", sys.executable)
            if bad == "external_link"
            else ("environment/escape", "../../outside")
        )
        archive, sha, count, size = environment_tar(tmp_path, extra=extra)
        args.update(archive=archive, sha256=sha, expected_files=count, expected_bytes=size)
    with pytest.raises(ValueError):
        module.extract(**args)
    assert not list(tmp_path.glob("rsna-environment-*"))


def test_environment_bootstrap_launch_uses_runtime_python_and_cleans_both_roots(tmp_path):
    scripts = tmp_path / "scripts/newton"
    scripts.mkdir(parents=True)
    env_root = tmp_path / "environment root with spaces"
    (env_root / "environment/bin").mkdir(parents=True)
    (env_root / "environment/bin/python").symlink_to(sys.executable)
    cache_root = tmp_path / "cache root with spaces"
    record = tmp_path / "calls.jsonl"
    for helper, root in [("runtime_environment", env_root), ("runtime_archive", cache_root)]:
        (scripts / (helper + ".py")).write_text(
            "import json,pathlib,sys\n"
            f'with pathlib.Path({str(record)!r}).open("a") as out: out.write(json.dumps([{helper!r}]+sys.argv[1:])+"\\n")\n'
            f'print(json.dumps({{"runtime_root":{str(root)!r}}}))\n'
        )
    for helper in ["preflight", "supervise"]:
        (scripts / (helper + ".py")).write_text(
            "import json,pathlib,sys\n"
            f'with pathlib.Path({str(record)!r}).open("a") as out: out.write(json.dumps([{helper!r},sys.executable]+sys.argv[1:])+"\\n")\n'
        )
    args = arguments(
        tmp_path,
        repo=str(tmp_path),
        python=sys.executable,
        cache=None,
        cache_archive="/p/cache.tar",
        cache_archive_sha256="b" * 64,
        cache_archive_files=2248,
        cache_unpacked_bytes=4077105588,
        scratch_base="/tmp",
        environment_archive="/p/env.tar",
        environment_archive_sha256="c" * 64,
        environment_archive_files=123,
        environment_unpacked_bytes=1234567,
    )
    script = launch.generate(args)["script"]
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"},
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert [row[0] for row in calls] == [
        "runtime_environment",
        "runtime_archive",
        "preflight",
        "supervise",
        "runtime_archive",
        "runtime_environment",
    ]
    assert calls[2][1] == str(env_root / "environment/bin/python")
    assert calls[3][1] == str(env_root / "environment/bin/python")
    assert calls[-2] == ["runtime_archive", "cleanup", "--root", str(cache_root)]
    assert calls[-1] == ["runtime_environment", "cleanup", "--root", str(env_root)]


def test_acceptance_launch_runs_bounded_helper_and_forwards_warning(tmp_path):
    scripts = tmp_path / "scripts"
    (scripts / "newton").mkdir(parents=True)
    (scripts / "newton/preflight.py").write_text("raise SystemExit(0)\n")
    ready = tmp_path / "ready"
    record = tmp_path / "accepted-args.json"
    (scripts / "local_competitive_acceptance.py").write_text(
        "import pathlib,signal,sys,json,time\n"
        "signal.signal(signal.SIGUSR1,lambda *args: sys.exit(2))\n"
        f"pathlib.Path({str(record)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        f'pathlib.Path({str(ready)!r}).write_text("ready")\n'
        "while True:time.sleep(.01)\n"
    )
    args = arguments(
        tmp_path, repo=str(tmp_path), python=sys.executable, gpus=2, mode="ddp", acceptance=True
    )
    script = launch.generate(args)["script"]
    assert "scripts/newton/supervise.py" not in script
    assert "scripts/local_competitive_acceptance.py" in script
    job = tmp_path / "job.sh"
    job.write_text(script)
    process = subprocess.Popen(
        ["bash", str(job)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0,1"},
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.01)
        assert ready.exists()
        process.send_signal(signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 2, (stdout, stderr)
        received = json.loads(record.read_text())
        assert received[received.index("--world-size") + 1] == "2"
        assert received[received.index("--device") + 1] == "cuda"
        assert "--skip-public-examples" in received
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    args.mode = "independent"
    with pytest.raises(ValueError):
        launch.generate(args)
