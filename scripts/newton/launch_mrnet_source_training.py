"""Private full-source engine recovery or fitting on an admitted owning node."""

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
    parser.add_argument("--phase", choices=("recovery", "full"), required=True)
    args = parser.parse_args()
    control = args.control
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "2"
        or not os.environ.get("SLURM_JOB_GPUS")
    ):
        raise ValueError("Two CPUs and one allocated GPU required")
    source = json.loads((control / "source.json").read_text())
    if (
        sha(Path(__file__))
        != source["files"]["scripts/newton/launch_mrnet_source_training.py"]["sha256"]
    ):
        raise ValueError("Original accepted private GPU wrapper required")
    input_receipt = Path(source["input_admission_path"])
    if sha(input_receipt) != source["input_admission_sha256"]:
        raise ValueError("Exact completed actual Stanford CPU input proof required")
    acquisition = Path(source["acquisition_control"])
    if sha(acquisition / "accepted.json") != source["acquisition_receipt_sha256"]:
        raise ValueError("Exact original Stanford acquisition receipt required")
    receipt = json.loads((acquisition / "accepted.json").read_text())
    scratch = json.loads((acquisition / "scratch.json").read_text())
    if (
        receipt["job_id"] != scratch["job_id"]
        or scratch["node"] != os.uname().nodename
        or receipt["files"] != 3750
    ):
        raise ValueError("Run on the owning Stanford data node")
    if args.phase == "full":
        recovery = Path(source["accepted_recovery_path"])
        if sha(recovery) != source["accepted_recovery_sha256"]:
            raise ValueError("Actual exact-engine recovery receipt required")
        if source["funding"]["node_local_checkpoint_cap_bytes"] < 3 * 1024**3:
            raise ValueError("All ten original full-source epochs need separate funding")
    if (control / "run-start.json").exists() or sha(control / "source.tar") != source["tar_sha256"]:
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
    with tempfile.TemporaryDirectory(
        prefix="rsna-mrnet-full-engine-source-", dir="/tmp"
    ) as temporary:
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
                Path(tempfile.mkdtemp(prefix="rsna-mrnet-full-engine-retained-", dir="/tmp"))
                / args.phase
            )
            with (control / "run-start.json").open("x") as stream:
                json.dump(
                    {
                        "job_id": os.environ["SLURM_JOB_ID"],
                        "node": os.uname().nodename,
                        "output": str(output),
                        "source_git": source["source_git"],
                        "private_research_only": True,
                        "node_local_checkpoint_cap_bytes": source["funding"][
                            "node_local_checkpoint_cap_bytes"
                        ],
                        "phase": args.phase,
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
                    str(code / "scripts/newton/train_mrnet_private.py"),
                    "--mode",
                    args.phase,
                    "--root",
                    scratch["path"],
                    "--acquisition-receipt",
                    str(acquisition / "accepted.json"),
                    "--input-receipt",
                    str(input_receipt),
                    "--config",
                    str(control / "config.json"),
                    "--output",
                    str(output),
                    *(
                        ["--recovery-receipt", source["accepted_recovery_path"]]
                        if args.phase == "full"
                        else []
                    ),
                ],
                check=True,
                timeout=2700 if args.phase == "recovery" else 34500,
            )
            result_name = "recovery-verified.json" if args.phase == "recovery" else "terminal.json"
            with (control / result_name).open("x") as stream:
                json.dump(
                    json.loads((output / result_name).read_text()), stream, indent=2, sort_keys=True
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
