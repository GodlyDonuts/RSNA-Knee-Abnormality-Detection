"""Original numerical Stanford pilot on an independently verified restoration."""

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
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "2"
        or not os.environ.get("SLURM_JOB_GPUS")
    ):
        raise ValueError("Two CPUs and one allocated GPU required")
    source = json.loads((control / "source.json").read_text())
    if sha(Path(__file__)) != source["restored_wrapper_sha256"]:
        raise ValueError("Original accepted private GPU wrapper required")
    input_receipt = Path(source["input_admission_path"])
    if sha(input_receipt) != source["input_admission_sha256"]:
        raise ValueError("Exact completed actual Stanford CPU input proof required")
    acquisition = Path(source["acquisition_control"])
    if sha(acquisition / "accepted.json") != source["acquisition_receipt_sha256"]:
        raise ValueError("Exact original Stanford acquisition receipt required")
    receipt = json.loads((acquisition / "accepted.json").read_text())
    restoration_path = Path(source["restoration_path"])
    if sha(restoration_path) != source["restoration_sha256"]:
        raise ValueError("Exact actual new-node restoration required")
    restoration = json.loads(restoration_path.read_text())
    if (
        restoration["status"] != "verbatim_MRNet_private_node_restoration_verified"
        or restoration["node"] != os.uname().nodename
        or restoration["volume_files"] != 3750
        or restoration["volume_bytes"] != 7740871424
        or restoration["original_acquisition_receipt_sha256"]
        != source["acquisition_receipt_sha256"]
        or restoration["original_input_sha256"] != source["input_admission_sha256"]
        or restoration["original_eight_image_label_tensor_records_equal"] is not True
        or receipt["files"] != 3750
    ):
        raise ValueError("Original source bytes and actual tensor-equal restoration required")
    if (control / "probe-start.json").exists() or sha(control / "source.tar") != source[
        "tar_sha256"
    ]:
        raise ValueError("Existing probe or changed source requires read-only review")
    os.environ.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    with tempfile.TemporaryDirectory(prefix="rsna-mrnet-adapter-source-", dir="/tmp") as temporary:
        code = Path(temporary)
        if code.stat().st_dev == control.stat().st_dev:
            raise ValueError("Node-local source and runtime required")
        with tarfile.open(control / "source.tar", "r:") as archive:
            members = archive.getmembers()
            if {member.name for member in members} != set(source["files"]) or len(members) != len(
                source["files"]
            ):
                raise ValueError("Exact original code closure required")
            for member in members:
                path = Path(member.name)
                if not member.isfile() or path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe source archive member")
                destination = code / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(member).read())
                if sha(destination) != source["files"][member.name]["sha256"]:
                    raise ValueError("Source archive member changed")
        os.environ["PYTHONPATH"] = str(code)
        helper = code / "scripts/newton/runtime_environment.py"
        command = [
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
        ]
        extracted = json.loads(subprocess.check_output(command, text=True, timeout=900))
        environment = Path(extracted["runtime_root"])
        with (control / "runtime.json").open("x") as stream:
            json.dump(
                {
                    "source_git": source["source_git"],
                    "job_id": os.environ["SLURM_JOB_ID"],
                    "environment": extracted,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
        try:
            output = (
                Path(tempfile.mkdtemp(prefix="rsna-mrnet-adapter-retained-", dir="/tmp"))
                / "trajectories"
            )
            with (control / "probe-start.json").open("x") as stream:
                json.dump(
                    {
                        "job_id": os.environ["SLURM_JOB_ID"],
                        "node": os.uname().nodename,
                        "output": str(output),
                        "source_git": source["source_git"],
                        "private_research_only": True,
                        "node_local_checkpoint_cap_bytes": 2 * 1024**3,
                        "no_external_weight_publication": True,
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                )
            subprocess.run(
                [
                    extracted["python"],
                    "-B",
                    str(code / "scripts/newton/probe_mrnet_adapter_recovery.py"),
                    "--root",
                    restoration["root"],
                    "--acquisition-receipt",
                    str(acquisition / "accepted.json"),
                    "--input-receipt",
                    str(input_receipt),
                    "--config",
                    str(control / "config.json"),
                    "--output",
                    str(output),
                ],
                check=True,
                timeout=2700,
            )
            with (control / "recovery-verified.json").open("x") as stream:
                json.dump(
                    json.loads((output / "recovery-verified.json").read_text()),
                    stream,
                    indent=2,
                    sort_keys=True,
                )
        finally:
            subprocess.run(
                [
                    "/usr/bin/python3.12",
                    "-B",
                    str(helper),
                    "cleanup",
                    "--root",
                    str(environment),
                ],
                check=True,
                timeout=60,
            )


if __name__ == "__main__":
    main()
