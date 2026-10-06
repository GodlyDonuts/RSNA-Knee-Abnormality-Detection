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
    seen = 0
    for directory, children, files in os.walk(inputs, followlinks=False):
        directory = Path(directory)
        seen += 1
        depth = len(directory.relative_to(inputs).parts)
        if seen > 2048 or depth > 8:
            raise ValueError("Attached Dataset discovery exceeded its metadata budget")
        if directory.name in {"competitions", "rsna-knee-abnormality-detection"} or (
            "train_series" in children or "test_series" in children
        ):
            children[:] = []
            continue
        children[:] = [
            child
            for child in children
            if child not in {"competitions", "rsna-knee-abnormality-detection"}
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
    if engine not in {"legacy", "resident_two_gpu_fp32"}:
        raise ValueError("Unadmitted notebook inference engine")
    remaining = 28800 - (time.monotonic() - notebook_started)
    if engine == "resident_two_gpu_fp32" and remaining <= 0:
        raise ValueError("Notebook bootstrap exhausted the eight-hour execution target")
    command = [
        str(python),
        str(
            bundle
            / (
                "code/resident_entry.py"
                if engine == "resident_two_gpu_fp32"
                else "code/chunked_infer.py"
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
        "32",
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
    if engine == "resident_two_gpu_fp32":
        command.extend(
            (
                "--resident-remaining-seconds",
                str(remaining),
                "--resident-workers",
                "2",
                "--single-decode",
            )
        )
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
    if engine == "resident_two_gpu_fp32":
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
