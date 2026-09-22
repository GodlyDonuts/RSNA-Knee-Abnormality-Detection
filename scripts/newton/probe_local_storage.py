"""Tiny allocated-node filesystem probe; no GPU operation or persistent scratch files."""

import json
import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "stdout": result.stdout[:4000]}
    except (OSError, subprocess.SubprocessError) as error:
        return {"error_type": type(error).__name__}


def main():
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        raise SystemExit("An allocated Slurm job is required")
    paths = [os.environ.get(name) for name in ("SLURM_TMPDIR", "TMPDIR")]
    paths.extend(["/tmp", "/var/tmp", "/scratch", "/local", "/localscratch", "/dev/shm"])
    home_device = Path.home().stat().st_dev
    filesystems = []
    for value in dict.fromkeys(path for path in paths if path):
        path = Path(value)
        if not path.is_dir():
            continue
        usage = shutil.disk_usage(path)
        mount = command(
            ["findmnt", "--noheadings", "--output", "FSTYPE,SOURCE,TARGET", "--target", str(path)]
        )
        fs_type = mount.get("stdout", "").split()[0] if mount.get("stdout", "").split() else None
        local_disk = path.stat().st_dev != home_device and fs_type not in {
            None,
            "lustre",
            "nfs",
            "nfs4",
            "tmpfs",
            "ramfs",
        }
        row = {
            "path": str(path),
            "resolved_path": str(path.resolve()),
            "filesystem_type": fs_type,
            "filesystem_free_bytes": usage.free,
            "filesystem_total_bytes": usage.total,
            "same_device_as_home": path.stat().st_dev == home_device,
            "local_disk_candidate": local_disk,
            "writable": os.access(path, os.W_OK),
            "at_least_100_gib_free": usage.free >= 100 * 2**30,
            "write_test": "not_attempted",
        }
        if local_disk and row["writable"]:
            try:
                with tempfile.TemporaryDirectory(prefix="rsna-storage-probe-", dir=path) as scratch:
                    probe = Path(scratch) / "probe.bin"
                    probe.write_bytes(b"\0" * 65536)
                    row["write_test"] = "65536_bytes_written_and_removed"
            except OSError as error:
                row["write_test"] = type(error).__name__
        filesystems.append(row)
    tools = {}
    for name in (
        "apptainer",
        "singularity",
        "squashfuse",
        "squashfuse_ll",
        "fusermount",
        "fusermount3",
        "mksquashfs",
        "unsquashfs",
        "tar",
    ):
        tools[name] = bool(shutil.which(name))
    module_result = command(["bash", "-lc", "module -t avail 2>&1"])
    modules = [
        line
        for line in module_result.get("stdout", "").splitlines()
        if any(name in line.lower() for name in ("apptainer", "singularity", "squash", "fuse"))
    ]
    fuse = Path("/dev/fuse")
    print(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": job,
                "hostname": socket.gethostname(),
                "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "requested_gpu_count": 0,
                "job_gpu_allocation": os.environ.get("SLURM_JOB_GPUS", ""),
                "filesystems": filesystems,
                "tools_available": tools,
                "container_modules": modules,
                "dev_fuse_exists": fuse.exists(),
                "dev_fuse_read_write": os.access(fuse, os.R_OK | os.W_OK),
                "persistent_scratch_files_created": 0,
                "policy_limit": "Capacity and writability are measurements, not an ARCC retention or permitted-use policy.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
