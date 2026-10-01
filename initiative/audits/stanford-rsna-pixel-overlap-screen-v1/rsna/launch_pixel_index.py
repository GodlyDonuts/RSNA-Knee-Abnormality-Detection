"""Run a frozen CPU-only pixel indexer without labels, models or new data copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control
    source = json.loads((control / "source.json").read_text())
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.uname().nodename != source["node"]
        or sha(Path(__file__)) != source["files"]["scripts/newton/launch_pixel_index.py"]["sha256"]
        or (control / "run-start.json").exists()
    ):
        raise ValueError("Original single-dispatch one-CPU zero-GPU owning-node wrapper required")
    os.environ.update(
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
    )
    if sha(control / "source.tar") != source["tar_sha256"]:
        raise ValueError("Frozen pixel-only source closure required")
    with tempfile.TemporaryDirectory(prefix="rsna-pixel-index-code-", dir="/tmp") as temporary:
        code = Path(temporary)
        if code.stat().st_dev == control.stat().st_dev:
            raise ValueError("Private node-local code/index required")
        with tarfile.open(control / "source.tar", "r:") as archive:
            members = archive.getmembers()
            if {m.name for m in members} != set(source["files"]) or len(members) != len(
                source["files"]
            ):
                raise ValueError("Exact pixel index source closure required")
            for member in members:
                relative = Path(member.name)
                if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe code archive member")
                destination = code / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(member).read())
                if sha(destination) != source["files"][member.name]["sha256"]:
                    raise ValueError("Original source bytes changed")
        helper = code / "scripts/newton/runtime_environment.py"
        extracted = json.loads(
            subprocess.check_output(
                [
                    "/usr/bin/python3.12",
                    "-B",
                    str(helper),
                    "extract",
                    "--archive",
                    source["environment_archive"],
                    "--sha256",
                    source["environment_sha256"],
                    "--expected-files",
                    "40403",
                    "--expected-bytes",
                    "6322691335",
                    "--scratch-base",
                    "/tmp",
                    "--min-free-gib",
                    "20",
                    "--shared-python",
                    "/usr/bin/python3.12",
                ],
                text=True,
                timeout=900,
            )
        )
        try:
            output = (
                Path(tempfile.mkdtemp(prefix="rsna-private-pixel-index-retained-", dir="/tmp"))
                / "index.jsonl"
            )
            with (control / "run-start.json").open("x") as stream:
                json.dump(
                    {
                        "job_id": os.environ["SLURM_JOB_ID"],
                        "node": os.uname().nodename,
                        "index_path": str(output),
                        "runtime": extracted,
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                )
            os.environ["PYTHONPATH"] = str(code)
            subprocess.run(
                [
                    extracted["python"],
                    "-B",
                    str(code / "scripts/newton/index_source_pixels.py"),
                    "--source",
                    str(control / "source.json"),
                    "--output",
                    str(output),
                    "--receipt",
                    str(control / "index-verified.json"),
                ],
                check=True,
                timeout=6800,
            )
        finally:
            subprocess.run(
                [
                    "/usr/bin/python3.12",
                    "-B",
                    str(helper),
                    "cleanup",
                    "--root",
                    extracted["runtime_root"],
                ],
                check=True,
                timeout=60,
            )


if __name__ == "__main__":
    main()
