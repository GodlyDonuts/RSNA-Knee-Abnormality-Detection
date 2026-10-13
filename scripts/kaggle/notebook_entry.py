"""Single-file offline Kaggle entry; workflow prepends EXPECTED_MANIFEST_SHA256."""

import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def bounded_process(command, *, env, timeout):
    """Stop the entire resident process group when its pipeline budget expires."""
    if timeout <= 0:
        raise ValueError("Notebook bootstrap exhausted the execution budget")
    child = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = child.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        raise


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attached_dataset_files(inputs, name):
    """Find attached payloads without traversing the competition MRI tree."""
    if Path(name).name != name:
        raise ValueError("Attached payload name must be a basename")
    scoped = globals().get("ATTACHED_PAYLOAD_ROOTS")
    if scoped is not None:
        if not isinstance(scoped, dict) or name not in scoped:
            raise ValueError("Known versioned payload scope required")
        item = scoped[name]
        if (
            set(item) != {"kind", "slug", "version", "relative"}
            or item["kind"] not in {"datasets", "notebooks"}
            or item["version"] != 1
        ):
            raise ValueError("Exact original attached payload identity required")
        parts = item["slug"].split("/")
        relative = Path(item["relative"])
        if (
            len(parts) != 2
            or any(not p or p in {".", ".."} for p in parts)
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.name != name
        ):
            raise ValueError("Attached payload identity/path escapes its scope")
        owner, slug = parts
        roots = [inputs / slug, inputs / owner / slug, inputs / item["kind"] / owner / slug]
        seen_paths = set()
        for root in roots:
            for version_root in (root, root / "1", root / "v1"):
                path = version_root / relative
                if path.is_file() and not path.is_symlink():
                    canonical = path.resolve()
                    if not canonical.is_relative_to(inputs.resolve()):
                        raise ValueError("Attached payload escapes the input filesystem")
                    if canonical not in seen_paths:
                        seen_paths.add(canonical)
                        yield path
        return
    seen = 0
    for directory, children, files in os.walk(inputs, followlinks=False):
        directory = Path(directory)
        seen += 1
        depth = len(directory.relative_to(inputs).parts)
        if seen > 2048:
            raise ValueError("Attached Dataset discovery exceeded its metadata budget")
        if directory.name in {"competitions", "rsna-knee-abnormality-detection"} or (
            "train_series" in children or "test_series" in children
        ):
            children[:] = []
            continue
        children[:] = [
            child
            for child in children
            if depth < 8
            and child
            not in {
                "competitions",
                "rsna-knee-abnormality-detection",
                "code",
                "artifacts",
                "cache",
                "witness-training-source",
                "witness-inputs-admission",
                "resident-performance",
            }
        ]
        if name in files:
            path = directory / name
            if not path.is_symlink() and path.is_file():
                yield path


def resolve_wheels(
    bundle, manifest, inputs=Path("/kaggle/input"), destination=Path("/tmp/rsna-shared-wheels")
):
    expected = json.loads((bundle / "wheelhouse.json").read_text())
    shared = manifest.get("shared_wheelhouse")
    if shared is None:
        return bundle / "wheels", expected
    name = shared["archive_file"]
    if Path(name).name != name:
        raise ValueError("Invalid shared archive path")
    candidates = [
        p
        for p in attached_dataset_files(inputs, name)
        if p.is_file()
        and p.stat().st_size == shared["archive_bytes"]
        and sha256(p) == shared["archive_sha256"]
    ]
    if len(candidates) != 1:
        raise ValueError("Exactly one full-byte-bound shared wheel archive required")
    destination.mkdir(exist_ok=False)
    with tarfile.open(candidates[0]) as handle:
        members = handle.getmembers()
        if len(members) > 1000:
            raise ValueError("Unexpected shared archive size")
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != "wheelhouse"
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unexpected shared archive member")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
            else:
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with handle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    wheels = destination / "wheelhouse"
    if sha256(wheels / "wheelhouse.json") != shared["manifest_sha256"]:
        raise ValueError("Shared wheel manifest bytes differ")
    found = json.loads((wheels / "wheelhouse.json").read_text())
    if found != expected:
        raise ValueError("Shared runtime differs from checkpoint package dependency contract")
    actual = {str(p.relative_to(wheels)) for p in wheels.rglob("*") if p.is_file()}
    if actual != set(found["files"]) | {"wheelhouse.json"}:
        raise ValueError("Shared archive contains missing or extraneous files")
    for name, row in found["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid wheel payload path")
        path = wheels / name
        if sha256(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Shared wheel payload bytes differ")
    return wheels, found


COMMON_BINDING_KEYS = {
    "schema_version",
    "dataset_slug",
    "dataset_id",
    "dataset_version",
    "manifest_sha256",
    "manifest_bytes",
    "files_sha256",
    "member_count",
    "total_bytes",
    "verification_sha256",
    "checkpoint_format",
    "license_source",
    "license_sha256",
    "encoder_spec_sha256",
    "code_sha256",
    "wheelhouse_manifest_sha256",
}
CHECKPOINT_ONLY_MEMBERS = {
    "checkpoint.pt",
    "snapshot.json",
    "fitting-admission.json",
    "baseline-proof.json",
}


def object_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def copy_exact(root, name, record, destination):
    relative = Path(name)
    source = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or source.is_symlink()
        or not source.is_file()
        or not source.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("Component member path escapes its mounted Dataset")
    target = destination / relative
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    with source.open("rb") as handle, target.open("xb") as output:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
            count += len(block)
            output.write(block)
    if digest.hexdigest() != record["sha256"] or count != record["bytes"]:
        raise ValueError("Component member bytes differ from the bound manifest")


def compose_common_bundle(
    bundle, manifest, inputs=Path("/kaggle/input"), destination=Path("/tmp/rsna-composed-bundle")
):
    """Standalone manifests pass through; component payloads need one exact common bundle."""
    record = manifest.get("common_bundle")
    if record is None:
        return bundle, manifest
    if (
        not isinstance(record, dict)
        or set(record) != COMMON_BINDING_KEYS
        or record["schema_version"] != 1
        or record["dataset_version"] != 1
        or record["checkpoint_format"] != manifest.get("checkpoint_format", "full")
        or record["license_sha256"] != manifest.get("license_sha256")
        or record["license_source"] != manifest.get("license_source")
    ):
        raise ValueError("Exact versioned common bundle binding required")
    found = [
        p
        for p in attached_dataset_files(inputs, "common-manifest.json")
        if p.stat().st_size == record["manifest_bytes"] and sha256(p) == record["manifest_sha256"]
    ]
    if len(found) != 1:
        raise ValueError("Exactly one byte-bound common bundle must be attached")
    root = found[0].parent
    common = json.loads(found[0].read_text())
    files = common.get("files")
    candidate = manifest["files"]
    if (
        common.get("schema_version") != 1
        or common.get("kind") != "rsna_common_encoder_code_bundle"
        or common.get("diagnostic_only") is not False
        or common.get("competition") != manifest.get("competition")
        or common.get("checkpoint_format") != record["checkpoint_format"]
        or common.get("shared_wheelhouse") != manifest.get("shared_wheelhouse")
        or not isinstance(files, dict)
        or object_sha256(files) != record["files_sha256"]
        or len(files) != record["member_count"]
        or files.get("DINOv3-LICENSE.md", {}).get("sha256") != record["license_sha256"]
        or files.get("encoder.json", {}).get("sha256") != record["encoder_spec_sha256"]
        or files.get("wheelhouse.json", {}).get("sha256") != record["wheelhouse_manifest_sha256"]
    ):
        raise ValueError("Mounted common bundle differs from its bound version")
    if not set(candidate) <= CHECKPOINT_ONLY_MEMBERS or "checkpoint.pt" not in candidate:
        raise ValueError("Checkpoint-only payload carries an unexpected member")
    common_names = {name.casefold() for name in files}
    if any(
        name.casefold() in common_names or name.casefold() == "common-manifest.json"
        for name in candidate
    ):
        raise ValueError("Conflicting checkpoint/common member path")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(files) | {"common-manifest.json"}:
        raise ValueError("Common bundle contains missing or extraneous members")
    destination.mkdir(mode=0o700, exist_ok=False)
    for name, row in sorted(files.items()):
        copy_exact(root, name, row, destination)
    for name, row in sorted(candidate.items()):
        copy_exact(bundle, name, row, destination)
    manifest_file = bundle / "bundle-manifest.json"
    copy_exact(
        bundle,
        "bundle-manifest.json",
        {"sha256": sha256(manifest_file), "bytes": manifest_file.stat().st_size},
        destination,
    )
    return destination, {**manifest, "files": dict(sorted({**files, **candidate}.items()))}


def resolve_competition(inputs, slug):
    candidates = [inputs / slug, inputs / "competitions" / slug]
    present = [path for path in candidates if (path / "test.csv").is_file()]
    if len(present) != 1:
        raise ValueError("Cannot identify exactly one official competition input mount")
    competition = present[0]
    for name in ("test_series.csv", "test_series"):
        if not (competition / name).exists():
            raise ValueError("Official runtime competition input is missing")
    return competition


def validate_private_training_sources(sources, hashes, accepted_sha):
    """Bound standalone wrappers may carry two explicitly hashed source overlays."""
    if (
        not isinstance(sources, dict)
        or set(sources) != {"witness_launch.py", "witness_recovery.py"}
        or not all(isinstance(source, str) for source in sources.values())
        or not isinstance(hashes, dict)
        or set(hashes) != set(sources)
        or sum(len(source.encode()) for source in sources.values()) > 128 * 1024
        or not isinstance(accepted_sha, str)
        or len(accepted_sha) != 64
        or any(c not in "0123456789abcdef" for c in accepted_sha)
    ):
        raise ValueError("Exact private trainer wrapper and actual CPU receipt required")
    for name, source in sources.items():
        if hashlib.sha256(source.encode()).hexdigest() != hashes[name]:
            raise ValueError("Private recovery wrapper source differs")
        compile(source, name, "exec")


def encoder_arguments(bundle, working, manifest):
    """Bind mounted original weights for delta inference without an online fetch."""
    spec = json.loads((bundle / "encoder.json").read_text())
    storage = manifest.get("checkpoint_format", "full")
    if storage not in {"full", "frozen_base_delta"}:
        raise ValueError("Unknown inference checkpoint storage")
    for kind in ("config", "weights") if storage == "frozen_base_delta" else ("config",):
        relative = Path(spec[f"{kind}_path"])
        path = bundle / relative
        record = manifest["files"].get(relative.as_posix(), {})
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(bundle.resolve())
            or not path.is_file()
            or record.get("sha256") != spec.get(f"{kind}_sha256")
            or sha256(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")
        ):
            raise ValueError("Bundled encoder bytes/path differ from the pinned source")
        spec[f"{kind}_path"] = str(path.resolve())
    if storage == "full":
        return ["--encoder-config", spec["config_path"]]
    runtime_spec = working / "encoder-runtime.json"
    with runtime_spec.open("x") as handle:
        json.dump(spec, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return ["--encoder-spec", str(runtime_spec)]


def stage_prefetch_sources(bundle, working, sources, hashes):
    """Add orchestration helpers without replacing any immutable trained rsna source."""
    names = {
        "prefetch_entry.py",
        "prefetch_preparation.py",
        "parallel_preparation.py",
        "process_memory.py",
        "resident_infer.py",
        "resident_pool.py",
        "resident_worker.py",
        "chunked_infer.py",
    }
    if (
        not isinstance(sources, dict)
        or not isinstance(hashes, dict)
        or set(sources) != names
        or set(hashes) != names
        or not all(isinstance(value, str) for value in sources.values())
        or sum(len(value.encode()) for value in sources.values()) > 256 * 1024
        or not (bundle / "code/rsna").is_dir()
    ):
        raise ValueError("Exact bounded prefetch helpers and original trained source required")
    for name, source in sources.items():
        if hashlib.sha256(source.encode()).hexdigest() != hashes[name]:
            raise ValueError("Prefetch helper bytes changed")
        compile(source, name, "exec")
    root = working / "prefetch-runtime-source"
    root.mkdir(mode=0o700)
    for name, source in sources.items():
        with (root / name).open("x") as handle:
            handle.write(source)
    return root


def diagnostic_probe_command(
    bundle, working, manifest, competition, python, *, probe_source=None, probe_source_sha256=None
):
    """Separate engineering execution from the competition output path."""
    if (
        manifest.get("diagnostic_only") is not True
        or manifest.get("competition_submission_eligible") is not False
        or manifest.get("checkpoint_format") != "frozen_base_delta"
    ):
        raise ValueError("Only an explicitly ineligible delta diagnostic can use this entry")
    probe_path = bundle / "code/delta_probe.py"
    if probe_source is not None or probe_source_sha256 is not None:
        # A corrected private engineering wrapper can reuse immutable, fully
        # verified base/delta inputs. Its exact source is part of the saved
        # notebook, separately hashed, and never replaces mounted Dataset bytes.
        if (
            not isinstance(probe_source, str)
            or len(probe_source.encode()) > 128 * 1024
            or hashlib.sha256(probe_source.encode()).hexdigest() != probe_source_sha256
        ):
            raise ValueError("Diagnostic replacement source is not byte-bound")
        chunked = bundle / "code/chunked_infer.py"
        record = manifest["files"].get("code/chunked_infer.py", {})
        if (
            chunked.is_symlink()
            or not chunked.is_file()
            or not chunked.resolve().is_relative_to(bundle.resolve())
            or sha256(chunked) != record.get("sha256")
            or chunked.stat().st_size != record.get("bytes")
        ):
            raise ValueError("Diagnostic shared inference source differs")
        runtime_code = working / "delta-probe-runtime-code"
        runtime_code.mkdir(mode=0o700)
        probe_path = runtime_code / "delta_probe.py"
        with probe_path.open("x") as handle:
            handle.write(probe_source)
        shutil.copyfile(chunked, runtime_code / "chunked_infer.py")
    encoder = encoder_arguments(bundle, working, manifest)
    if encoder[0] != "--encoder-spec":
        raise ValueError("Diagnostic must load the pinned original base")
    return [
        str(python),
        str(probe_path),
        "--bundle",
        str(bundle),
        *encoder,
        "--competition",
        str(competition),
        "--output",
        str(working / "delta-diagnostic"),
    ]


def run():
    notebook_started = time.monotonic()
    expected = globals().get("EXPECTED_MANIFEST_SHA256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Prepared notebook manifest binding is required")
    matches = [
        p
        for p in attached_dataset_files(Path("/kaggle/input"), "bundle-manifest.json")
        if sha256(p) == expected
    ]
    if len(matches) != 1:
        raise ValueError("Exactly one byte-bound inference bundle must be attached")
    manifest = json.loads(matches[0].read_text())
    bundle = matches[0].parent
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Pinned Linux x86_64 runtime required")
    if ".".join(platform.python_version_tuple()[:2]) != manifest["python"]:
        raise ValueError("Kaggle Python ABI differs from the accepted offline wheelhouse")
    for name, record in manifest["files"].items():
        relative = Path(name)
        path = bundle / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.resolve().is_relative_to(bundle.resolve())
        ):
            raise ValueError("Invalid bundle path")
        if (
            not path.is_file()
            or sha256(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError("Bundle bytes differ from the prepared checkpoint/environment")
    bundle, manifest = compose_common_bundle(bundle, manifest)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_XET": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
    wheels, dependencies = resolve_wheels(bundle, manifest)
    runtime = Path("/tmp/rsna-knee-offline-environment")
    if runtime.exists():
        raise ValueError("Offline runtime must be newly created")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(runtime)], check=True)
    python = runtime / "bin/python"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--no-index",
            "--require-hashes",
            "--find-links",
            str(wheels),
            "-r",
            str(wheels / "requirements.lock"),
        ],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    installed = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import importlib.metadata,json,sys; "
                "print(json.dumps({n:importlib.metadata.version(n) for n in json.loads(sys.argv[1])}))",
                json.dumps(list(dependencies["resolved_versions"])),
            ],
            text=True,
        )
    )
    if installed != dependencies["resolved_versions"]:
        raise ValueError("Installed package differs from the accepted dependency lock")
    working = Path("/kaggle/working")
    sys.path.insert(0, str(bundle / "code"))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(bundle / "code")
    if globals().get("WITNESS_T4_RECOVERY_PROBE") is True:
        sources = globals().get("WITNESS_T4_RECOVERY_SOURCES")
        hashes = globals().get("WITNESS_T4_RECOVERY_HASHES")
        accepted_sha = globals().get("WITNESS_T4_CPU_ACCEPTANCE_SHA256")
        validate_private_training_sources(sources, hashes, accepted_sha)
        accepted_files = [
            p
            for p in attached_dataset_files(
                Path("/kaggle/input"), "witness-inputs-CPU-accepted.json"
            )
            if sha256(p) == accepted_sha
        ]
        if len(accepted_files) != 1:
            raise ValueError("Attach exactly the original accepted CPU notebook output")
        source_output = accepted_files[0].parent
        control = source_output / "witness-training-source"
        cache = source_output / "witness-inputs-admission/cache"
        if not control.is_dir() or not cache.is_dir():
            raise ValueError("Original accepted source and MRI cache output missing")
        helper_dir = working / "witness-recovery-code"
        helper_dir.mkdir(mode=0o700)
        for name, source in sources.items():
            if hashlib.sha256(source.encode()).hexdigest() != hashes[name]:
                raise ValueError("Private recovery wrapper source differs")
            with (helper_dir / name).open("x") as handle:
                handle.write(source)
        env.update(
            PYTHONPATH=str(control / "code"),
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        remaining = 14400 - (time.monotonic() - notebook_started)
        command = [
            str(python),
            str(helper_dir / "witness_launch.py"),
            "--control",
            str(control),
            "--cache",
            str(cache),
            "--cpu-acceptance",
            str(accepted_files[0]),
            "--cpu-acceptance-sha256",
            accepted_sha,
            *encoder_arguments(bundle, working, manifest),
            "--output",
            str(working / "witness-T4-recovery"),
            "--worker",
            str(helper_dir / "witness_recovery.py"),
            "--remaining-seconds",
            str(remaining),
        ]
        bounded_process(command, env=env, timeout=remaining)
        return
    if globals().get("WITNESS_INPUTS_CPU_PROBE") is True:
        source = globals().get("WITNESS_INPUTS_SOURCE")
        source_sha = globals().get("WITNESS_INPUTS_SOURCE_SHA256")
        archive_sha = globals().get("WITNESS_INPUTS_ARCHIVE_SHA256")
        admission_sha = globals().get("WITNESS_INPUTS_ADMISSION_SHA256")
        if (
            not isinstance(source, str)
            or len(source.encode()) > 64 * 1024
            or hashlib.sha256(source.encode()).hexdigest() != source_sha
            or not isinstance(archive_sha, str)
            or len(archive_sha) != 64
            or not isinstance(admission_sha, str)
            or len(admission_sha) != 64
        ):
            raise ValueError("Explicit standalone CPU input source/receipts required")
        payloads = [
            p
            for p in attached_dataset_files(Path("/kaggle/input"), "CPU-admission.json")
            if sha256(p) == admission_sha
        ]
        if len(payloads) != 1:
            raise ValueError("Exactly one original private training input Dataset required")
        helper = working / "witness_inputs.py"
        with helper.open("x") as handle:
            handle.write(source)
        control = working / "witness-training-source"
        bounded_process(
            [
                str(python),
                str(helper),
                "--extract-inputs",
                "--input-directory",
                str(payloads[0].parent),
                "--destination",
                str(control),
                "--archive-sha256",
                archive_sha,
                "--admission-sha256",
                admission_sha,
            ],
            env=env,
            timeout=28800 - (time.monotonic() - notebook_started),
        )
        env.update(
            PYTHONPATH=str(control / "code"),
            PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        bounded_process(
            [
                str(python),
                str(helper),
                "--control",
                str(control),
                *encoder_arguments(bundle, working, manifest),
                "--competition",
                str(resolve_competition(Path("/kaggle/input"), manifest["competition"])),
                "--output",
                str(working / "witness-inputs-admission"),
            ],
            env=env,
            timeout=28800 - (time.monotonic() - notebook_started),
        )
        return
    if globals().get("DELTA_DIAGNOSTIC_PROBE") is True:
        competition = resolve_competition(Path("/kaggle/input"), manifest["competition"])
        subprocess.run(
            diagnostic_probe_command(
                bundle,
                working,
                manifest,
                competition,
                python,
                probe_source=globals().get("DELTA_DIAGNOSTIC_PROBE_SOURCE"),
                probe_source_sha256=globals().get("DELTA_DIAGNOSTIC_PROBE_SOURCE_SHA256"),
            ),
            env=env,
            check=True,
        )
        shutil.copyfile(
            working / "delta-diagnostic/delta-deployment-acceptance.json",
            working / "delta-deployment-acceptance.json",
        )
        return
    if globals().get("RESIDENT_PERFORMANCE_PROBE") is True:
        sources = globals().get("RESIDENT_PERFORMANCE_SOURCES")
        hashes = globals().get("RESIDENT_PERFORMANCE_HASHES")
        roster = globals().get("RESIDENT_PERFORMANCE_FITTING_UIDS")
        if not isinstance(sources, dict) or set(sources) != set(hashes or {}):
            raise ValueError("Bound standalone performance source required")
        if (
            set(sources)
            != {
                "chunked_infer.py",
                "resident_infer.py",
                "resident_pool.py",
                "resident_worker.py",
                "resident_entry.py",
                "performance_probe.py",
            }
            or sum(len(v.encode()) for v in sources.values()) > 256 * 1024
        ):
            raise ValueError("Bounded standalone performance source required")
        runtime_code = working / "resident-performance-code"
        runtime_code.mkdir(mode=0o700)
        for name, source in sources.items():
            if (
                Path(name).name != name
                or hashlib.sha256(source.encode()).hexdigest() != hashes[name]
            ):
                raise ValueError("Standalone performance source differs")
            with (runtime_code / name).open("x") as handle:
                handle.write(source)
        study_count = globals().get("RESIDENT_PERFORMANCE_STUDY_COUNT", 32)
        if study_count not in {32, 1300} or (
            not isinstance(roster, list)
            or len(roster) != study_count
            or len(set(roster)) != study_count
        ):
            raise ValueError("Exact admitted fitting performance roster required")
        roster_path = runtime_code / "fitting-roster.json"
        roster_path.write_text(json.dumps(roster))
        competition = resolve_competition(Path("/kaggle/input"), manifest["competition"])
        command = [
            str(python),
            str(runtime_code / "performance_probe.py"),
            "--bundle",
            str(bundle),
            *encoder_arguments(bundle, working, manifest),
            "--competition",
            str(competition),
            "--roster",
            str(roster_path),
            "--output",
            str(working / "resident-performance"),
            "--study-count",
            str(study_count),
        ]
        bounded_process(
            command,
            env=env,
            timeout=28800 - (time.monotonic() - notebook_started),
        )
        return
    policy = manifest.get("inference_policy", {})
    competition = resolve_competition(Path("/kaggle/input"), manifest["competition"])
    engine = policy.get("engine", "legacy")
    runtime_code = bundle / "code"
    if globals().get("INFERENCE_FP32_PREFETCH") is True:
        if (
            policy.get("cache_side") != 384
            or policy.get("max_series", 6) != 6
            or policy.get("series_sampling", "selected") != "selected"
            or manifest.get("diagnostic_only") is not False
        ):
            raise ValueError(
                "Only the admitted native384 six-series production policy is supported"
            )
        runtime_code = stage_prefetch_sources(
            bundle,
            working,
            globals().get("INFERENCE_PREFETCH_SOURCES"),
            globals().get("INFERENCE_PREFETCH_HASHES"),
        )
        env.update(
            PYTHONPATH=str(runtime_code) + os.pathsep + str(bundle / "code"),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        engine = "resident_two_gpu_fp32_prefetch"
    resident = engine in {"resident_two_gpu_fp32", "resident_two_gpu_fp32_prefetch"}
    if engine not in {"legacy", "resident_two_gpu_fp32", "resident_two_gpu_fp32_prefetch"}:
        raise ValueError("Unadmitted notebook inference engine")
    remaining = 28800 - (time.monotonic() - notebook_started)
    if resident and remaining <= 0:
        raise ValueError("Notebook bootstrap exhausted the eight-hour execution target")
    command = [
        str(python),
        str(
            runtime_code
            / (
                "prefetch_entry.py"
                if engine == "resident_two_gpu_fp32_prefetch"
                else "resident_entry.py"
                if engine == "resident_two_gpu_fp32"
                else "chunked_infer.py"
            )
        ),
        "--checkpoint",
        str(bundle / "checkpoint.pt"),
        *encoder_arguments(bundle, working, manifest),
        "--series",
        str(competition / "test_series.csv"),
        "--test-studies",
        str(competition / "test.csv"),
        "--dicom-root",
        str(competition),
        "--cache-root",
        "/tmp/rsna-knee-test-cache",
        "--output",
        str(working / "submission.csv"),
        "--workers",
        "0",
        "--device",
        "cuda",
        "--precision",
        "fp32",
        "--chunk-size",
        "8" if engine == "resident_two_gpu_fp32_prefetch" else "32",
        "--resilient-series",
        "--fallback-on-unusable-study",
        "--recover-slices",
        "--validate-model-geometry",
        "--isolate-study-errors",
        "--cache-side",
        str(policy.get("cache_side", 256)),
        "--max-series",
        str(policy.get("max_series", 6)),
        "--series-sampling",
        policy.get("series_sampling", "selected"),
    ]
    if resident:
        command.extend(
            (
                "--resident-remaining-seconds",
                str(remaining),
                "--single-decode",
            )
        )
        if engine == "resident_two_gpu_fp32":
            command.extend(("--resident-workers", "2"))
    if (competition / "sample_submission.csv").is_file():
        command.extend(("--template", str(competition / "sample_submission.csv")))
    if manifest.get("baseline_epoch_one") is True:
        command.extend(
            [
                "--baseline-epoch-one",
                "--baseline-sidecar",
                str(bundle / "snapshot.json"),
                "--baseline-proof",
                str(bundle / "baseline-proof.json"),
            ]
        )
    if manifest.get("fitting_admission_sha256") is not None:
        admission = bundle / "fitting-admission.json"
        if (
            manifest.get("baseline_epoch_one")
            or sha256(admission) != manifest["fitting_admission_sha256"]
        ):
            raise ValueError("Reviewed fitting admission differs from the exact bundle")
        command.extend(("--fitting-admission", str(admission)))
    if resident:
        bounded_process(command, env=env, timeout=remaining)
    else:
        subprocess.run(command, env=env, check=True)
    metrics = json.loads((working / "submission.metrics.json").read_text())
    if (
        metrics.get("status") != "passed"
        or metrics.get("diagnostic_only") is not False
        or metrics.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
        or metrics.get("submission_sha256") != sha256(working / "submission.csv")
        or metrics.get("chunk_count", 0) < 1
        or metrics.get("checkpoint_epoch") != manifest["epoch"]
    ):
        raise ValueError("Inference output is diagnostic or used a different checkpoint")
    # Public notebook execution normally has three examples. This validates only
    # execution: the submitted code is rerun by Kaggle on the hidden test set.
    receipt = {
        "schema_version": 1,
        "status": "offline_inference_complete",
        "bundle_manifest_sha256": expected,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "studies": metrics["studies"],
        "submission_sha256": sha256(working / "submission.csv"),
        "public_example_only": metrics["studies"] <= 3,
        "hidden_score_claimed": False,
        "checkpoint_format": manifest.get("checkpoint_format", "full"),
    }
    (working / "kaggle-inference-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2)
    )


if __name__ == "__main__":
    run()
