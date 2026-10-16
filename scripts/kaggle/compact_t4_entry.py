"""Offline bootstrap for the private compact precision probe, never a submission."""

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def run(expected_manifest_sha256):
    if platform.python_version_tuple()[:2] != ("3", "12"):
        raise ValueError("Original pinned Python3.12 image required")
    found = []
    for directory, children, files in os.walk("/kaggle/input"):
        path = Path(directory)
        if len(path.parts) > 9:
            children[:] = []
        if len(found) > 4:
            raise ValueError("Ambiguous payload discovery")
        if "bundle-manifest.json" in files:
            p = path / "bundle-manifest.json"
            if sha(p) == expected_manifest_sha256:
                found.append(p)
    if len(found) != 1:
        raise ValueError("One exact private diagnostic payload required")
    bundle = found[0].parent
    manifest = json.loads(found[0].read_text())
    if (
        manifest.get("diagnostic_only") is not True
        or manifest.get("competition_eligible") is not False
    ):
        raise ValueError("Engineering scope required")
    for name, record in manifest["files"].items():
        path = bundle / name
        if path.is_symlink() or not path.resolve().is_relative_to(bundle.resolve()):
            raise ValueError("Unsafe payload member")
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Payload differs from original node proof")
    helper = bundle / "code/scripts/kaggle/notebook_entry.py"
    spec = importlib.util.spec_from_file_location("existing_offline_bootstrap", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    wheels, dependencies = module.resolve_wheels(bundle, manifest)
    runtime = Path("/tmp/rsna-compact-offline-runtime")
    assert not runtime.exists()
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
    installed = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import importlib.metadata,json,sys;print(json.dumps({n:importlib.metadata.version(n) for n in json.loads(sys.argv[1])}))",
                json.dumps(list(dependencies["resolved_versions"])),
            ],
            text=True,
        )
    )
    if installed != dependencies["resolved_versions"]:
        raise ValueError("Original locked dependency differs")
    extra = Path("/tmp/rsna-compact-extra-wheels")
    extra.mkdir(exist_ok=False)
    added = []
    for item in manifest["extra_wheels"]:
        name = item["filename"]
        if Path(name).name != name or not name.endswith(".whl"):
            raise ValueError("Invalid extra wheel")
        source = bundle / item["payload"]
        target = extra / name
        if sha(source) != item["sha256"]:
            raise ValueError("Extra wheel changed")
        shutil.copyfile(source, target)
        added.append(str(target))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--no-deps",
            "--no-index",
            *added,
        ],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(bundle / "code")
    command = [
        str(python),
        "-B",
        str(bundle / "code/scripts/kaggle/compact_t4_probe.py"),
        "--bundle",
        str(bundle),
        "--output",
        "/kaggle/working/compact-T4-precision-accepted.json",
    ]
    subprocess.run(command, env=env, check=True, timeout=2400)
    if Path("/kaggle/working/submission.csv").exists():
        raise ValueError("Diagnostic cannot create submission.csv")


if __name__ == "__main__":
    run(globals().get("EXPECTED_MANIFEST_SHA256"))
