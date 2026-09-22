"""Independent small archive/subprocess checks; no SSH, scheduler or real weights."""

import hashlib
import importlib.util
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script(name):
    path = Path(__file__).parents[1] / "scripts/newton" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"review_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stream = load_script("stream_cache_archive")
weights = load_script("stage_weights")
runtime = load_script("runtime_archive")


def archive_bytes(entries=None):
    output = io.BytesIO()
    entries = entries or [("cache/study/series.npz", b"synthetic-volume")]
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            if payload is None:
                info.type = tarfile.SYMTYPE
                info.linkname = "/outside"
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def receive(target, payload, count=1):
    return subprocess.run(
        [sys.executable, "-c", stream.RECEIVE, str(target), str(count)],
        input=payload,
        capture_output=True,
        timeout=10,
    )


def commit(target, payload):
    return subprocess.run(
        [
            sys.executable,
            "-c",
            stream.COMMIT,
            str(target),
            hashlib.sha256(payload).hexdigest(),
            str(len(payload)),
        ],
        capture_output=True,
        timeout=10,
    )


def test_actual_receiver_and_commit_preserve_bytes_permissions_and_no_overwrite(tmp_path):
    target = tmp_path / "cache.tar"
    payload = archive_bytes()
    result = receive(target, payload)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["file_members"] == 1
    assert receipt["unpacked_file_bytes"] == len(b"synthetic-volume")
    assert not target.exists()
    partial = Path(str(target) + ".partial")
    assert partial.read_bytes() == payload
    assert stat.S_IMODE(partial.stat().st_mode) == 0o600
    published = commit(target, payload)
    assert published.returncode == 0, published.stderr
    assert json.loads(published.stdout)["status"] == "published_verified_archive"
    assert target.read_bytes() == payload and not partial.exists()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert receive(target, archive_bytes()).returncode != 0
    assert target.read_bytes() == payload


@pytest.mark.parametrize(
    "entries,count",
    [
        ([("/absolute", b"x")], 1),
        ([("cache/../escaped", b"x")], 1),
        ([("other/volume.npz", b"x")], 1),
        ([("cache/link", None)], 0),
        ([("cache/same.npz", b"a"), ("cache/same.npz", b"b")], 2),
        ([("cache/good.npz", b"x")], 2),
    ],
)
def test_receiver_rejects_unsafe_duplicate_or_incomplete_members(tmp_path, entries, count):
    target = tmp_path / "cache.tar"
    payload = archive_bytes(entries)
    assert receive(target, payload, count).returncode != 0
    assert not target.exists()
    assert Path(str(target) + ".partial").read_bytes() == payload
    assert not Path(str(target) + ".partial.verified.json").exists()


def test_truncated_payload_is_preserved_but_never_verified_or_published(tmp_path):
    target = tmp_path / "cache.tar"
    payload = archive_bytes([("cache/series.npz", b"x" * 3000)])[:1024]
    assert receive(target, payload).returncode != 0
    assert not target.exists()
    assert Path(str(target) + ".partial").read_bytes() == payload
    assert not Path(str(target) + ".partial.verified.json").exists()


def test_existing_partial_is_never_overwritten(tmp_path):
    target = tmp_path / "cache.tar"
    partial = Path(str(target) + ".partial")
    partial.write_bytes(b"previous-interrupted-attempt")
    assert receive(target, archive_bytes()).returncode != 0
    assert partial.read_bytes() == b"previous-interrupted-attempt"
    assert not target.exists()


@pytest.mark.parametrize("change", ["bytes", "replacement", "requested_digest", "existing_final"])
def test_commit_rejects_changed_partial_or_wrong_sender_binding(tmp_path, change):
    target = tmp_path / "cache.tar"
    payload = archive_bytes()
    assert receive(target, payload).returncode == 0
    partial = Path(str(target) + ".partial")
    if change == "bytes":
        original = partial.stat()
        with partial.open("r+b") as handle:
            handle.write(b"changed")
        os.utime(partial, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
    elif change == "replacement":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(payload)
        os.replace(replacement, partial)
    elif change == "requested_digest":
        payload = b"x" * len(payload)
    else:
        target.write_bytes(b"existing-final")
    assert commit(target, payload).returncode != 0
    assert partial.exists()
    if change == "existing_final":
        assert target.read_bytes() == b"existing-final"
    else:
        assert not target.exists()


@pytest.mark.parametrize(
    "destination,authorization",
    [
        ("https://huggingface.co/other", True),
        ("https://cas-bridge.xethub.hf.co/file", False),
        ("https://cdn-lfs.huggingface.co/file", False),
    ],
)
def test_authenticated_redirect_credentials_stay_on_exact_hub(destination, authorization):
    request = urllib.request.Request(
        "https://huggingface.co/source", headers={"Authorization": "Bearer synthetic"}
    )
    redirected = weights.TokenSafeRedirect().redirect_request(
        request, None, 302, "Found", {}, destination
    )
    assert (redirected.get_header("Authorization") is not None) is authorization


@pytest.mark.parametrize(
    "destination",
    [
        "http://huggingface.co/file",
        "https://huggingface.co.evil.test/file",
        "https://evil.test/file",
    ],
)
def test_authenticated_redirect_rejects_unapproved_origins(destination):
    request = urllib.request.Request("https://huggingface.co/source")
    with pytest.raises(ValueError, match="redirect"):
        weights.TokenSafeRedirect().redirect_request(request, None, 302, "Found", {}, destination)


def test_interrupted_weight_response_does_not_publish_or_leave_unverified_weights(
    tmp_path, monkeypatch
):
    class Interrupted(io.BytesIO):
        def read(self, _n=-1):
            raise OSError("synthetic interrupted response")

    monkeypatch.setattr(weights.urllib.request, "urlopen", lambda *a, **k: Interrupted())
    target = tmp_path / "model.safetensors"
    with pytest.raises(OSError, match="interrupted"):
        weights.download_missing("https://huggingface.co/test", target, "a" * 64, 1024)
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def local_archive(tmp_path, monkeypatch):
    payload = archive_bytes()
    archive = tmp_path / "source.tar"
    archive.write_bytes(payload)
    monkeypatch.setattr(runtime, "local_disk", lambda path: True)
    monkeypatch.setattr(
        runtime.shutil, "disk_usage", lambda path: SimpleNamespace(free=200 * 2**30)
    )
    args = dict(
        archive=archive,
        sha256=hashlib.sha256(payload).hexdigest(),
        expected_files=1,
        expected_bytes=len(b"synthetic-volume"),
        scratch_base=tmp_path,
        min_free_gib=20,
        job_id="123",
    )
    return archive, payload, args


def test_job_local_extraction_exact_bytes_private_modes_and_owned_cleanup(tmp_path, monkeypatch):
    archive, payload, args = local_archive(tmp_path, monkeypatch)
    result = runtime.extract(**args)
    root = Path(result["runtime_root"])
    assert result["archive_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["file_members"] == 1
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    extracted = Path(result["cache_path"]) / "study/series.npz"
    assert extracted.read_bytes() == b"synthetic-volume"
    assert stat.S_IMODE(extracted.stat().st_mode) == 0o600
    with pytest.raises(ValueError):
        runtime.cleanup(root, "124")
    assert root.exists()
    runtime.cleanup(root, "123")
    assert not root.exists()
    assert archive.read_bytes() == payload


@pytest.mark.parametrize("change", ["hash", "file_count", "byte_count", "capacity"])
def test_extraction_failures_do_not_leave_a_usable_partial_cache(tmp_path, monkeypatch, change):
    archive, payload, args = local_archive(tmp_path, monkeypatch)
    if change == "hash":
        args["sha256"] = "a" * 64
    elif change == "file_count":
        args["expected_files"] = 2
    elif change == "byte_count":
        args["expected_bytes"] -= 1
    else:
        monkeypatch.setattr(runtime.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(ValueError):
        runtime.extract(**args)
    assert not list(tmp_path.glob("rsna-knee-*"))
    assert archive.read_bytes() == payload


def test_archive_changed_between_precheck_and_extraction_cannot_claim_old_digest(
    tmp_path, monkeypatch
):
    archive, _payload, args = local_archive(tmp_path, monkeypatch)
    changed = archive_bytes([("cache/study/series.npz", b"different-volume")])
    assert len(b"different-volume") == args["expected_bytes"]
    original_open = runtime.tarfile.open

    def replace_before_extract(*a, **kw):
        archive.write_bytes(changed)
        return original_open(*a, **kw)

    monkeypatch.setattr(runtime.tarfile, "open", replace_before_extract)
    with pytest.raises(ValueError, match="(sha256|changed|digest)"):
        runtime.extract(**args)
    assert not list(tmp_path.glob("rsna-knee-*"))


def campaign_fixture(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "stage_weights", weights)
    materializer = load_script("materialize_config")
    config = tmp_path / "configs/competitive/weights/config.json"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"{}")
    license_path = config.with_name("LICENSE.md")
    license_path.write_bytes(b"synthetic-license-fixture")
    model_path = tmp_path / "artifacts/encoder-weights/model/model.safetensors"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"synthetic-weights-fixture")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    encoder = {
        "architecture": "dinov3_l16",
        "source_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "source_revision": "a" * 40,
        "license": "other",
        "transformers_version": "5.6.1",
        "state_dict_format": "safetensors",
        "config_path": config.relative_to(tmp_path).as_posix(),
        "weights_path": model_path.relative_to(tmp_path).as_posix(),
        "config_sha256": digest(config),
        "weights_sha256": digest(model_path),
    }
    manifest = {
        "schema_version": 1,
        "encoder": encoder,
        "download": {
            "config_filename": "config.json",
            "config_bytes": config.stat().st_size,
            "weights_filename": "model.safetensors",
            "weights_bytes": model_path.stat().st_size,
        },
        "license_policy": {
            "name": weights.DINOV3_LICENSE_NAME,
            "url": weights.DINOV3_LICENSE_URL,
            "path": license_path.relative_to(tmp_path).as_posix(),
            "sha256": digest(license_path),
            "bytes": license_path.stat().st_size,
        },
    }
    recipe = {
        **{
            key: encoder[key]
            for key in ("architecture", "source_id", "source_revision", "weights_sha256")
        },
        "weights_bytes": model_path.stat().st_size,
        "configuration_sha256": encoder["config_sha256"],
        "train_config_without_encoder": {
            "epochs": 10,
            "submission_every_epochs": 2,
            "precision": "bf16",
        },
        "pilot": {
            "submission_eligible": False,
            "epochs": 2,
            "diagnostic_study_limit": 100,
            "initial_precision": "fp32",
        },
    }
    recipe_path, manifest_path = tmp_path / "recipe.json", tmp_path / "manifest.json"
    recipe_path.write_text(json.dumps(recipe))
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        weights.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no network")
    )
    return materializer, recipe_path, manifest_path, model_path, license_path


@pytest.mark.parametrize("mode", ["pilot", "production"])
def test_materializer_verifies_offline_bytes_and_preserves_existing_config(
    tmp_path, monkeypatch, mode
):
    materializer, recipe, manifest, _model, _license = campaign_fixture(tmp_path, monkeypatch)
    output = tmp_path / "config.json"
    result = materializer.materialize(recipe, manifest, tmp_path, output, mode)
    data = json.loads(output.read_text())
    assert result["download_performed"] is False
    assert Path(data["encoder"]["weights_path"]).is_absolute()
    if mode == "pilot":
        assert data["epochs"] == 2 and data["diagnostic_study_limit"] == 100
        assert data["precision"] == "fp32" and data["submission_every_epochs"] == 0
    else:
        assert data["epochs"] == 10 and data["diagnostic_study_limit"] is None
        assert data["submission_every_epochs"] == 2
    assert (
        materializer.materialize(recipe, manifest, tmp_path, output, mode)["status"]
        == "verified_existing_identical"
    )
    output.write_bytes(b"unrelated-existing-config")
    with pytest.raises(ValueError, match="existing"):
        materializer.materialize(recipe, manifest, tmp_path, output, mode)
    assert output.read_bytes() == b"unrelated-existing-config"


@pytest.mark.parametrize(
    "change", ["missing_weights", "bad_weights", "bad_license", "wrong_campaign", "eligible_pilot"]
)
def test_materializer_refuses_incomplete_or_inconsistent_pins(tmp_path, monkeypatch, change):
    materializer, recipe, manifest, model, license_path = campaign_fixture(tmp_path, monkeypatch)
    if change == "missing_weights":
        model.unlink()
    elif change == "bad_weights":
        model.write_bytes(b"wrong")
    elif change == "bad_license":
        license_path.write_bytes(b"wrong")
    else:
        data = json.loads(recipe.read_text())
        if change == "wrong_campaign":
            data["source_revision"] = "b" * 40
        else:
            data["pilot"]["submission_eligible"] = True
        recipe.write_text(json.dumps(data))
    output = tmp_path / "config.json"
    with pytest.raises(ValueError):
        materializer.materialize(recipe, manifest, tmp_path, output, "pilot")
    assert not output.exists()


def test_sigterm_during_cli_extraction_cleans_owned_partial_tree(tmp_path):
    payload = archive_bytes()
    archive = tmp_path / "source.tar"
    archive.write_bytes(payload)
    ready = tmp_path / "copy-started"
    script_dir = Path(__file__).parents[1] / "scripts/newton"
    code = """
import sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
import runtime_archive
runtime_archive.local_disk=lambda path:True
runtime_archive.shutil.disk_usage=lambda path:SimpleNamespace(free=200*2**30)
ready=Path(sys.argv[2])
def stalled_copy(source,output,length):
    output.write(b'x');output.flush()
    ready.write_text('ready')
    while True:time.sleep(0.05)
runtime_archive.shutil.copyfileobj=stalled_copy
sys.argv=['runtime_archive',*sys.argv[3:]]
runtime_archive.main()
"""
    command = [
        sys.executable,
        "-c",
        code,
        str(script_dir),
        str(ready),
        "extract",
        "--archive",
        str(archive),
        "--sha256",
        hashlib.sha256(payload).hexdigest(),
        "--expected-files",
        "1",
        "--expected-bytes",
        str(len(b"synthetic-volume")),
        "--scratch-base",
        str(tmp_path),
    ]
    child = subprocess.Popen(
        command,
        env={**os.environ, "SLURM_JOB_ID": "123"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        assert list(tmp_path.glob("rsna-knee-123-*"))
        child.send_signal(signal.SIGTERM)
        child.communicate(timeout=5)
        assert child.returncode != 0
        assert not list(tmp_path.glob("rsna-knee-123-*"))
        assert archive.read_bytes() == payload
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


@pytest.mark.parametrize(
    "training_status,cleanup_status,expected",
    [(0, 0, 0), (75, 0, 75), (8, 0, 8), (0, 2, 2)],
)
def test_archive_job_cleanup_and_training_exit_status_are_preserved(
    tmp_path, training_status, cleanup_status, expected
):
    launch = load_script("launch")
    scripts = tmp_path / "scripts/newton"
    scripts.mkdir(parents=True)
    runtime_root = tmp_path / "selected runtime with spaces"
    calls_path = tmp_path / "calls.jsonl"
    record = (
        "import json,pathlib,sys\n"
        f"with pathlib.Path({str(calls_path)!r}).open('a') as handle: "
        "handle.write(json.dumps([pathlib.Path(__file__).name,*sys.argv[1:]])+'\\n')\n"
    )
    (scripts / "runtime_archive.py").write_text(
        record
        + "if sys.argv[1]=='extract':\n"
        + f" print(json.dumps({{'runtime_root':{str(runtime_root)!r}}}))\n"
        + f"else: raise SystemExit({cleanup_status})\n"
    )
    (scripts / "preflight.py").write_text(record)
    (scripts / "supervise.py").write_text(record + f"raise SystemExit({training_status})\n")
    args = SimpleNamespace(
        account="test",
        partition="test",
        time="02:00:00",
        job_name="test",
        repo=str(tmp_path),
        python=sys.executable,
        studies="/synthetic/studies.parquet",
        series="/synthetic/series.csv",
        cache=None,
        cache_archive="/synthetic/cache.tar",
        cache_archive_sha256="b" * 64,
        cache_archive_files=1,
        cache_unpacked_bytes=16,
        scratch_base="/tmp",
        cache_manifest="/synthetic/index.parquet",
        folds="/synthetic/folds.json",
        artifact_root=str(tmp_path),
        logs=str(tmp_path),
        expected_git="a" * 40,
        config=["/synthetic/config.json"],
        output=["/synthetic/output"],
        resume=None,
        cpus=2,
        memory_gb=4,
        min_free_gb=20,
        gpus=1,
        gpu_type=None,
        mode="independent",
        signal_seconds=300,
        checkpoint_grace_seconds=120,
        write_script=None,
    )
    script = launch.generate(args)["script"]
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"},
        timeout=10,
    )
    assert result.returncode == expected, result.stderr
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert [row[0] for row in calls] == [
        "runtime_archive.py",
        "preflight.py",
        "supervise.py",
        "runtime_archive.py",
    ]
    for row in calls[1:3]:
        assert row[row.index("--cache") + 1] == str(runtime_root / "cache")
    assert calls[-1][1:] == ["cleanup", "--root", str(runtime_root)]


@pytest.mark.parametrize("gpus", [1, 2])
def test_environment_archive_job_forwards_usr1_and_cleans_both_roots(tmp_path, gpus):
    launch = load_script("launch")
    scripts = tmp_path / "scripts/newton"
    scripts.mkdir(parents=True)
    calls = tmp_path / "calls.jsonl"
    cache_root, env_root = tmp_path / "cache root with spaces", tmp_path / "env root with spaces"
    record = (
        "import json,pathlib,sys,shutil\n"
        f"record=pathlib.Path({str(calls)!r})\n"
        "def note(value):\n"
        " with record.open('a') as handle: handle.write(json.dumps(value)+'\\n')\n"
        "note([pathlib.Path(__file__).name,*sys.argv[1:]])\n"
    )
    for name, root, environment in [
        ("runtime_archive.py", cache_root, False),
        ("runtime_environment.py", env_root, True),
    ]:
        body = record + f"root=pathlib.Path({str(root)!r})\n"
        body += "if sys.argv[1]=='extract':\n"
        if environment:
            body += " (root/'environment/bin').mkdir(parents=True)\n"
            body += " (root/'environment/bin/python').symlink_to(sys.executable)\n"
        else:
            body += " (root/'cache').mkdir(parents=True)\n"
        body += " print(json.dumps({'runtime_root':str(root)}))\n"
        body += "else: shutil.rmtree(root)\n"
        (scripts / name).write_text(body)
    (scripts / "preflight.py").write_text(record)
    (scripts / "supervise.py").write_text(
        record
        + "import signal,time\n"
        + "def requested(signum,frame):\n"
        + " note(['received',signum,sys.executable])\n"
        + " raise SystemExit(75)\n"
        + "signal.signal(signal.SIGUSR1,requested)\n"
        + "checkpoint=pathlib.Path(sys.argv[sys.argv.index('--checkpoint')+1])\n"
        + f"(pathlib.Path({str(tmp_path)!r})/('ready-'+checkpoint.parent.name)).write_text('ready')\n"
        + "while True: time.sleep(0.05)\n"
    )
    args = SimpleNamespace(
        account="test",
        partition="test",
        time="02:00:00",
        job_name="test",
        repo=str(tmp_path),
        python=sys.executable,
        studies="/synthetic/studies.parquet",
        series="/synthetic/series.csv",
        cache=None,
        cache_archive="/synthetic/cache.tar",
        cache_archive_sha256="b" * 64,
        cache_archive_files=1,
        cache_unpacked_bytes=16,
        scratch_base="/tmp",
        environment_archive="/synthetic/environment.tar.gz",
        environment_archive_sha256="c" * 64,
        environment_archive_files=1,
        environment_unpacked_bytes=16,
        cache_manifest="/synthetic/index.parquet",
        folds="/synthetic/folds.json",
        artifact_root=str(tmp_path),
        logs=str(tmp_path),
        expected_git="a" * 40,
        config=[f"/synthetic/config{i}.json" for i in range(gpus)],
        output=[f"/synthetic/out{i}" for i in range(gpus)],
        resume=None,
        cpus=4,
        memory_gb=4,
        min_free_gb=20,
        gpus=gpus,
        gpu_type=None,
        mode="independent",
        signal_seconds=300,
        checkpoint_grace_seconds=120,
        write_script=None,
    )
    script = tmp_path / "job.sh"
    script.write_text(launch.generate(args)["script"])
    child = subprocess.Popen(
        ["bash", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={
            **os.environ,
            "SLURM_JOB_ID": "123",
            "CUDA_VISIBLE_DEVICES": "0,1" if gpus == 2 else "0",
        },
    )
    try:
        deadline = time.monotonic() + 5
        while (
            len(list(tmp_path.glob("ready-*"))) < gpus
            and child.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert len(list(tmp_path.glob("ready-*"))) == gpus
        child.send_signal(signal.SIGUSR1)
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 75, (stdout, stderr)
        values = [json.loads(line) for line in calls.read_text().splitlines()]
        signals = [row for row in values if row[0] == "received"]
        assert len(signals) == gpus
        assert all(row[1] == signal.SIGUSR1 for row in signals)
        assert all(Path(row[2]) == env_root / "environment/bin/python" for row in signals)
        assert values[-2:] == [
            ["runtime_archive.py", "cleanup", "--root", str(cache_root)],
            ["runtime_environment.py", "cleanup", "--root", str(env_root)],
        ]
        assert not cache_root.exists() and not env_root.exists()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate(timeout=5)


@pytest.mark.parametrize(
    "trigger", ["usr1", "term", "timeout", "launcher_failure", "launcher_success"]
)
def test_acceptance_child_group_termination_leaves_no_term_ignoring_grandchild(tmp_path, trigger):
    worker_ready = tmp_path / "worker.json"
    primary_ready = tmp_path / "primary.json"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json,os,pathlib,signal,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        f"pathlib.Path({str(worker_ready)!r}).write_text(json.dumps({{'pid':os.getpid(),'pgid':os.getpgrp()}}))\n"
        "while True:time.sleep(.05)\n"
    )
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import json,os,pathlib,subprocess,sys,time\n"
        f"child=subprocess.Popen([sys.executable,{str(worker)!r}])\n"
        f"pathlib.Path({str(primary_ready)!r}).write_text(json.dumps({{'pid':os.getpid()}}))\n"
        f"while not pathlib.Path({str(worker_ready)!r}).exists():time.sleep(.01)\n"
        + (
            f"raise SystemExit({7 if trigger == 'launcher_failure' else 0})\n"
            if trigger.startswith("launcher_")
            else "while True:time.sleep(.05)\n"
        )
    )
    outcome = tmp_path / "outcome.json"
    script_dir = Path(__file__).parents[1] / "scripts"
    code = """
import json,pathlib,signal,sys
sys.path.insert(0,sys.argv[1])
import local_competitive_acceptance as helper
signals=(signal.SIGUSR1,signal.SIGTERM,signal.SIGINT)
before=[signal.getsignal(s)for s in signals]
with pathlib.Path(sys.argv[3]).open('w')as handle:
    try:
        result={'returned':helper.run_child([sys.executable,sys.argv[2]],handle,timeout=int(sys.argv[5]))}
    except BaseException as error:
        result={'raised':type(error).__name__}
result['handlers_restored']=[signal.getsignal(s)for s in signals]==before
pathlib.Path(sys.argv[4]).write_text(json.dumps(result))
"""
    controller = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(script_dir),
            str(launcher),
            str(tmp_path / "child.log"),
            str(outcome),
            "1" if trigger == "timeout" else "20",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    owned_group = None

    def is_running(pid):
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, timeout=5
        )
        value = result.stdout.strip()
        return bool(value) and not value.startswith("Z")

    try:
        deadline = time.monotonic() + 15
        while (
            not worker_ready.exists() and controller.poll() is None and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert worker_ready.exists()
        identity = json.loads(worker_ready.read_text())
        owned_group = identity["pgid"]
        assert owned_group == json.loads(primary_ready.read_text())["pid"]
        if trigger in {"usr1", "term"}:
            controller.send_signal(signal.SIGUSR1 if trigger == "usr1" else signal.SIGTERM)
        stdout, stderr = controller.communicate(timeout=15)
        assert controller.returncode == 0, (stdout, stderr)
        record = json.loads(outcome.read_text())
        assert record["handlers_restored"]
        if trigger in {"usr1", "term"}:
            assert record["raised"] == "InterruptedError"
        elif trigger == "timeout":
            assert record["raised"] == "TimeoutExpired"
        else:
            assert record["returned"] == (7 if trigger == "launcher_failure" else 0)
        deadline = time.monotonic() + 3
        while is_running(identity["pid"]) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not is_running(identity["pid"]), "failed diagnostic left an active descendant"
    finally:
        if owned_group is not None:
            try:
                os.killpg(owned_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if controller.poll() is None:
            controller.kill()
            controller.communicate(timeout=5)
