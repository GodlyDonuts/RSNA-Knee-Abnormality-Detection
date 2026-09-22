"""Run inside a bounded zero-GPU Slurm job; persist only archives and small proofs."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import tomllib

FLOOR = 20 * 1024**3


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def run(command, log, env=None, timeout=1800):
    with log.open("ab") as handle:
        subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=True
        )


def wheel_metadata(wheel):
    with zipfile.ZipFile(wheel) as handle:
        names = [
            n for n in handle.namelist() if n.count("/") == 1 and n.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise ValueError("Ambiguous top-level wheel metadata")
        return email.message_from_bytes(handle.read(names[0]))


def archive(source, destination, arcname):
    if destination.exists() or destination.with_suffix(".partial").exists():
        raise ValueError("Refusing to replace an environment archive")
    size = sum(p.stat().st_size for p in source.rglob("*") if p.is_file() and not p.is_symlink())
    if shutil.disk_usage(destination.parent).free < FLOOR + size:
        raise ValueError("Archive would violate free-space reserve")
    temporary = destination.with_suffix(".partial")
    with tarfile.open(temporary, "w") as handle:
        handle.add(source, arcname=arcname)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.rename(temporary, destination)
    with tarfile.open(destination) as handle:
        members = handle.getmembers()
    return {
        "file": destination.name,
        "sha256": sha(destination),
        "bytes": destination.stat().st_size,
        "regular_files": sum(m.isfile() for m in members),
        "unpacked_bytes": sum(m.size for m in members if m.isfile()),
        "links": [
            {"name": m.name, "target": m.linkname, "kind": "symlink" if m.issym() else "hardlink"}
            for m in members
            if m.issym() or m.islnk()
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID", "").isdigit() or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Run in an allocated CPU-only Slurm job")
    if (
        sys.version_info[:2] != (3, 12)
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
    ):
        raise ValueError("Linux x86_64 Python3.12 is required")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) not in (1, 2, 3, 4):
        raise ValueError("Use at most four allocated CPU cores")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    log = args.output / "build-private.log"
    project = tomllib.loads(args.pyproject.read_text())
    requirements = (
        project["project"]["dependencies"]
        + project["project"]["optional-dependencies"]["competitive"]
    )
    requirements += ["pytest==8.3.4", "pip==25.0.1"]
    if any("==" not in item for item in requirements):
        raise ValueError("Every requested top-level package must be exactly pinned")
    started = time.time()
    os.environ.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_CACHE_DIR": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if shutil.disk_usage("/tmp").free < FLOOR + 40 * 1024**3:
        raise ValueError("Node-local preparation reserve is insufficient")
    with tempfile.TemporaryDirectory(prefix="rsna-wheelhouse-", dir="/tmp") as tmp:
        root = Path(tmp)
        wheels = root / "wheelhouse"
        wheels.mkdir()
        resolver = root / "resolver"
        run([sys.executable, "-m", "venv", str(resolver)], log)
        python = resolver / "bin/python"
        request_file = root / "requested.txt"
        request_file.write_text("\n".join(requirements) + "\n")
        run(
            [
                str(python),
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--index-url",
                "https://pypi.org/simple",
                "--dest",
                str(wheels),
                "-r",
                str(request_file),
            ],
            log,
        )
        resolved = {}
        for wheel in sorted(wheels.glob("*.whl")):
            meta = wheel_metadata(wheel)
            name = meta["Name"].lower().replace("_", "-")
            if name in resolved:
                raise ValueError("Duplicate distribution in resolved wheelhouse")
            resolved[name] = {"version": meta["Version"], "sha256": sha(wheel), "wheel": wheel.name}
        if not resolved:
            raise ValueError("No binary dependencies downloaded")
        lock = wheels / "requirements.lock"
        lock.write_text(
            "\n".join(
                f"{name}=={row['version']} --hash=sha256:{row['sha256']}"
                for name, row in sorted(resolved.items())
            )
            + "\n"
        )
        environment = root / "environment"
        run([sys.executable, "-m", "venv", str(environment)], log)
        python = environment / "bin/python"
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--require-hashes",
                "--find-links",
                str(wheels),
                "-r",
                str(lock),
            ],
            log,
        )
        run([str(python), "-m", "pip", "check"], log)
        check = r"""
import importlib.metadata, json, platform, sys
import torch, numpy, pandas, pydicom, transformers, safetensors, pyarrow, sklearn
from pydicom.pixels import get_decoder
from pydicom.uid import JPEGBaseline8Bit, JPEGLosslessSV1, JPEG2000Lossless
from transformers import DINOv3ViTConfig, DINOv3ViTModel
torch.set_num_threads(2)
x=torch.ones(2,3,requires_grad=True); (x.square().sum()).backward()
assert x.grad.eq(2).all()
config=DINOv3ViTConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,image_size=32,patch_size=16,num_register_tokens=4)
model=DINOv3ViTModel(config)
with torch.no_grad(): value=model(pixel_values=torch.zeros(1,3,32,32)).pooler_output
assert value.shape==(1,32) and torch.isfinite(value).all()
decoders={str(uid):get_decoder(uid).is_available for uid in (JPEGBaseline8Bit,JPEGLosslessSV1,JPEG2000Lossless)}
assert all(decoders.values())
print(json.dumps({'python':platform.python_version(),'base_python':sys.base_prefix,'torch':str(torch.__version__),'torch_cuda':torch.version.cuda,'compiled_cuda_arches':torch._C._cuda_getArchFlags(),'versions':{d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()},'decoder_plugins_available':decoders,'cpu_tensor_backward':True,'synthetic_dinov3_cpu_forward':True,'cuda_forward_performed':False,'pretrained_weights_accessed':False},sort_keys=True))
"""
        result = subprocess.run(
            [str(python), "-c", check], capture_output=True, text=True, timeout=180, check=True
        )
        acceptance = json.loads(result.stdout)
        if acceptance["torch"].split("+")[0] != "2.14.0" or acceptance["torch_cuda"] != "13.0":
            raise ValueError("Installed torch/CUDA differs from the checked PyPI expectation")
        if not all(x in acceptance["compiled_cuda_arches"].split() for x in ("sm_75", "sm_90")):
            raise ValueError("Compiled wheel must contain T4 and H100 architectures")
        acceptance.update(
            schema_version=1,
            status="cpu_offline_install_passed",
            job_id=os.environ["SLURM_JOB_ID"],
            pyproject_sha256=sha(args.pyproject),
            builder_sha256=sha(Path(__file__)),
            requirements=requirements,
            requirement_lock_sha256=sha(lock),
            gpu_runtime_accepted=False,
        )
        write(wheels / "acceptance.json", acceptance)
        manifest = {
            "schema_version": 1,
            "platform": "linux_x86_64",
            "python": "3.12",
            "cuda": "13.0",
            "offline_install_verified": True,
            "gpu_forward_verified": False,
            "source_index": "https://pypi.org/simple",
            "pyproject_sha256": sha(args.pyproject),
            "files": {
                p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                for p in sorted(wheels.iterdir())
            },
            "resolved_versions": {name: row["version"] for name, row in sorted(resolved.items())},
        }
        write(wheels / "wheelhouse.json", manifest)
        wheel_archive = archive(wheels, args.output / "wheelhouse-cp312-cu130.tar", "wheelhouse")
        env_archive = archive(
            environment, args.output / "environment-cp312-cu130.tar", "environment"
        )
        # Verify actual relocation by extracting the prepared environment elsewhere
        # and invoking its interpreter directly, not its original absolute scripts.
        relocated = root / "relocated"
        relocated.mkdir()
        with tarfile.open(args.output / env_archive["file"]) as handle:
            handle.extractall(relocated)
        relocated_result = subprocess.run(
            [str(relocated / "environment/bin/python"), "-c", check],
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        relocated_acceptance = json.loads(relocated_result.stdout)
        if relocated_acceptance != json.loads(result.stdout):
            raise ValueError("Relocated environment acceptance differs")
        final = {
            "schema_version": 1,
            "status": "passed",
            "job_id": os.environ["SLURM_JOB_ID"],
            "requested_gpus": 0,
            "python": "3.12",
            "cuda": "13.0",
            "wheelhouse": wheel_archive,
            "environment": env_archive,
            "acceptance": acceptance,
            "relocated_environment_verified": True,
            "relocation_method": "extract archive then call environment/bin/python directly; do not invoke archived pip/pytest shebangs",
            "shared_base_python_required": sys.base_prefix,
            "elapsed_seconds": time.time() - started,
            "gpu_forward_verified": False,
        }
        write(args.output / "receipt.json", final)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "job_id": final["job_id"],
                    "wheelhouse": wheel_archive,
                    "environment": env_archive,
                    "gpu_forward_verified": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
