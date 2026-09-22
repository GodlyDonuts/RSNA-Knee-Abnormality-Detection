"""Read-only allocation and local-artifact preflight; stdout contains no path/env dump."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path


def fingerprint(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError("regular_artifact_required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def safe_output(value: str) -> str:
    """Defense in depth for hardware strings, never intended for config/argv dumps."""
    value = re.sub(r"(?:hf_|KGAT_)[A-Za-z0-9]{12,}", "[redacted]", value)
    value = re.sub(r"https?://\S+", "[redacted-url]", value)
    value = re.sub(r"(?<![A-Za-z0-9])/(?:[^\s,;]+)", "[redacted-path]", value)
    return value[:16384]


def command(argv: list[str], cwd: Path | None = None) -> dict:
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=15)
        return {
            "available": True,
            "returncode": result.returncode,
            "output": safe_output(result.stdout) if result.returncode == 0 else "",
        }
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "returncode": None, "output": ""}


def probe_torch(torch_module: object | None = None) -> dict:
    try:
        if torch_module is None:
            import torch as torch_module
        torch = torch_module
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        devices = []
        for index in range(count):
            prop = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": safe_output(prop.name),
                    "total_memory_bytes": int(prop.total_memory),
                    "compute_capability": [int(prop.major), int(prop.minor)],
                }
            )
        nccl = None
        if available and torch.distributed.is_nccl_available():
            nccl = list(torch.cuda.nccl.version())
        return {
            "installed": True,
            "version": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "cuda_available": available,
            "visible_gpu_count": count,
            "devices": devices,
            "nccl_version": nccl,
            "bf16_supported": bool(available and torch.cuda.is_bf16_supported()),
        }
    except Exception:
        return {
            "installed": False,
            "cuda_available": False,
            "visible_gpu_count": 0,
            "nccl_version": None,
            "bf16_supported": False,
            "failure": "torch_probe_failed",
        }


def device_errors(info: dict, device: str, expected_gpus: int, require_nccl: bool) -> list[str]:
    errors = []
    if not info["installed"]:
        errors.append("torch_unavailable")
    if device == "cuda":
        if not info["cuda_available"]:
            errors.append("cuda_requested_but_unavailable")
        if info["visible_gpu_count"] != expected_gpus:
            errors.append("assigned_gpu_count_mismatch")
    if require_nccl and info.get("nccl_version") is None:
        errors.append("nccl_required_but_unavailable")
    return errors


def package_versions() -> dict:
    versions = {}
    for name in ("torch", "transformers", "safetensors", "numpy", "pandas", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def encoder_summary(config: dict, versions: dict) -> dict:
    encoder = config.get("encoder", {})
    architecture = encoder.get("architecture")
    if architecture == "diagnostic_tiny":
        limit = config.get("diagnostic_study_limit")
        if type(limit) is not int or limit <= 0:
            raise ValueError("tiny_encoder_requires_bounded_diagnostic")
        return {"architecture": architecture, "diagnostic_only": True}
    if architecture not in {"dinov2_b14_reg", "dinov3_b16", "dinov3_l16", "convnext_tiny"}:
        raise ValueError("unsupported_encoder_architecture")
    for name, version in (("transformers", "5.6.1"), ("safetensors", "0.7.0")):
        if versions[name] != version:
            raise ValueError("encoder_dependency_version_mismatch")
    if encoder.get("transformers_version") != "5.6.1":
        raise ValueError("encoder_transformers_pin_mismatch")
    if encoder.get("state_dict_format") not in {"safetensors", "pytorch"}:
        raise ValueError("encoder_state_dict_format_required")
    if not re.fullmatch(r"[0-9a-f]{40}", str(encoder.get("source_revision", ""))):
        raise ValueError("immutable_model_revision_required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(encoder.get("source_id", ""))):
        raise ValueError("model_source_identifier_required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(encoder.get("license", ""))):
        raise ValueError("model_license_identifier_required")
    result = {k: encoder[k] for k in ("architecture", "source_id", "source_revision", "license")}
    for name in ("config", "weights"):
        value = fingerprint(Path(encoder[name + "_path"]))
        if value["sha256"] != encoder.get(name + "_sha256"):
            raise ValueError("model_file_hash_mismatch")
        result[name] = value
    return result


def ram_bytes() -> dict:
    path = Path("/proc/meminfo")
    if path.exists():
        rows = dict(line.split(":", 1) for line in path.read_text().splitlines())
        return {
            name: int(rows[key].split()[0]) * 1024
            for name, key in (("total", "MemTotal"), ("available", "MemAvailable"))
            if key in rows
        }
    try:
        return {"total": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")}
    except (OSError, ValueError):
        return {}


def inspect(args: argparse.Namespace) -> dict:
    errors: list[str] = []
    versions = package_versions()
    hardware = probe_torch()
    errors.extend(device_errors(hardware, args.device, args.expected_gpus, args.require_nccl))
    actual_git = command(["git", "rev-parse", "HEAD"], args.repo)
    commit = actual_git["output"].strip()
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_git) or commit != args.expected_git:
        errors.append("code_git_pin_mismatch")
    dirty = command(["git", "status", "--porcelain", "--untracked-files=normal"], args.repo)
    if dirty["returncode"] != 0 or dirty["output"].strip():
        errors.append("code_checkout_not_clean")
    artifacts = {}
    try:
        artifacts["cache_manifest"] = fingerprint(args.cache_manifest)
        if not args.cache.is_dir():
            errors.append("cache_directory_missing")
        storage = shutil.disk_usage(args.storage)
        if storage.free < args.min_free_gb * 2**30:
            errors.append("insufficient_free_storage")
        free = storage.free
    except (OSError, ValueError):
        free = None
        errors.append("cache_or_storage_check_failed")
    configs = []
    for path in args.config:
        try:
            config = json.loads(path.read_text())
            if any(type(config.get(k)) is not int or config[k] < 0 for k in ("fold", "seed")):
                raise ValueError("integer_fold_and_seed_required")
            if config.get("precision") not in {"fp32", "bf16"}:
                raise ValueError("explicit_precision_required")
            summary = {"artifact": fingerprint(path), "encoder": encoder_summary(config, versions)}
            summary.update({k: config[k] for k in ("fold", "seed", "precision")})
            if config["precision"] == "bf16" and not hardware["bf16_supported"]:
                errors.append("bf16_requested_but_unsupported")
            configs.append(summary)
        except (OSError, ValueError, KeyError, TypeError):
            errors.append("config_or_encoder_preflight_failed")
    return {
        "schema_version": 1,
        "read_only": True,
        "status": "blocked" if errors else "passed",
        "errors": sorted(set(errors)),
        "code": {
            "git_commit": commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None,
            "clean": "code_checkout_not_clean" not in errors,
        },
        "node": safe_output(socket.gethostname()),
        "platform": platform.system(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes(),
        "torch": hardware,
        "package_versions": versions,
        "free_storage_bytes": free,
        "cache_artifacts": artifacts,
        "runs": configs,
        "nvidia_driver_gpu_inventory": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,pci.bus_id",
                "--format=csv,noheader",
            ]
        ),
        "gpu_topology": command(["nvidia-smi", "topo", "-m"]),
        "cpu_topology": command(["lscpu", "--json"]),
        "scope": "No training, downloads, scheduler mutations, environment dump or payload rewrite.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--expected-gpus", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--require-nccl", action="store_true")
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--expected-git", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--cache-manifest", type=Path, required=True)
    p.add_argument("--storage", type=Path, required=True)
    p.add_argument("--min-free-gb", type=int, required=True)
    p.add_argument("--config", type=Path, action="append", required=True)
    args = p.parse_args()
    if args.min_free_gb <= 0 or (args.device == "cuda" and args.expected_gpus == 0):
        p.error("positive storage floor and nonzero CUDA GPU request required")
    try:
        report = inspect(args)
    except Exception:
        report = {"status": "blocked", "errors": ["preflight_failed"], "read_only": True}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
