"""Single-file offline Kaggle entry; workflow prepends EXPECTED_MANIFEST_SHA256."""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run():
    expected = globals().get("EXPECTED_MANIFEST_SHA256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Prepared notebook manifest binding is required")
    matches = [
        p for p in Path("/kaggle/input").rglob("bundle-manifest.json") if sha256(p) == expected
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
            str(bundle / "wheels"),
            "-r",
            str(bundle / "wheels/requirements.lock"),
        ],
        check=True,
    )
    dependencies = json.loads((bundle / "wheelhouse.json").read_text())
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
    encoder = json.loads((bundle / "encoder.json").read_text())
    candidates = [
        Path("/kaggle/input") / manifest["competition"],
        Path("/kaggle/input/competitions") / manifest["competition"],
    ]
    present = [path for path in candidates if (path / "sample_submission.csv").is_file()]
    if len(present) != 1:
        raise ValueError("Cannot identify exactly one official competition input mount")
    competition = present[0]
    for name in ("test_series.csv", "sample_submission.csv", "test_series"):
        if not (competition / name).exists():
            raise ValueError("Official runtime competition input is missing")
    subprocess.run(
        [
            str(python),
            "-m",
            "rsna.competitive.infer",
            "--checkpoint",
            str(bundle / "checkpoint.pt"),
            "--encoder-config",
            str(bundle / encoder["config_path"]),
            "--series",
            str(competition / "test_series.csv"),
            "--template",
            str(competition / "sample_submission.csv"),
            "--dicom-root",
            str(competition),
            "--cache",
            "/tmp/rsna-knee-test-cache",
            "--output",
            str(working / "submission.csv"),
            "--workers",
            "0",
            "--device",
            "cuda",
            "--precision",
            "fp32",
            "--scheduled-epoch",
        ],
        env=env,
        check=True,
    )
    metrics = json.loads((working / "submission.metrics.json").read_text())
    if (
        metrics.get("diagnostic_only") is not False
        or metrics.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
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
    }
    (working / "kaggle-inference-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2)
    )


if __name__ == "__main__":
    run()
