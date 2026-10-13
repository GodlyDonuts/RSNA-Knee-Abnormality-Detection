"""Prepared owning-node CPU wrapper for the streaming public ordered-triplet cohort cache.

NOT RUN by the authoring session and never a scheduler dispatch. It refuses to start
unless one SHA256-pinned binding matches the node, CPU-only environment, exact Git
source and module files, runtime versions, every input file and the storage envelope.
Original DICOM objects are copied one study at a time from the retained owning-node
inventory root (no network, no re-acquisition). The cache, intent, journal, scratch
and study IDs stay private on the node; stdout/--out receive only sanitized counts,
hashes and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPECTED_COUNTS = (2247, 1156)
INPUTS = ("source", "member_manifest", "studies", "folds", "series", "inventory")
RUNTIME = ("python", "numpy", "torch", "pydicom", "pandas")
MODULES = (
    "rsna/competitive/public_input.py",
    "rsna/competitive/public_input_cache.py",
    "scripts/newton/run_public_triplet_cache.py",
    "tools/research/public_dinov2_input_reference/dino_adapter.py",
    "tools/research/public_dinov2_input_reference/reference.py",
    "tools/research/public_dinov2_input_reference/provenance.py",
    "tools/research/public_dinov2_input_reference/contract.py",
)
GPU_ENV = ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
SCHEDULER_SUBMIT = ("SLURM_SUBMIT_COMMAND",)
BINDING_KEYS = {
    "node",
    "git_head",
    "files",
    "runtime",
    "inputs",
    "caps",
    "expected_counts",
    "fitting_uids_sha256",
    "evaluation_uids_sha256",
    "source_revision",
    "raw_source_revision",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_binding(path: Path, expected_sha256: str) -> dict:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Wrapper binding bytes differ from the pinned SHA256")
    binding = json.loads(payload)
    if (
        not isinstance(binding, dict)
        or set(binding) != BINDING_KEYS
        or set(binding["files"]) != set(MODULES)
        or set(binding["runtime"]) != set(RUNTIME)
        or set(binding["inputs"]) != set(INPUTS)
        or tuple(binding["expected_counts"]) != EXPECTED_COUNTS
    ):
        raise ValueError("Wrapper binding fields differ")
    return binding


def check_environment(binding: dict, environ: Mapping[str, str], nodename: str, cuda: bool):
    if nodename != binding["node"]:
        raise ValueError("Original owning node required")
    if cuda or any(environ.get(name, "") not in ("", "-1", "none") for name in GPU_ENV):
        raise ValueError("CPU-only execution required; GPU visible")
    if any(name in environ for name in SCHEDULER_SUBMIT):
        raise ValueError("Wrapper never submits or nests scheduler jobs")
    if environ.get("OMP_NUM_THREADS") != "1" or environ.get("SLURM_CPUS_PER_TASK", "1") != "1":
        raise ValueError("Single-thread CPU envelope required")


def check_runtime(binding: dict, versions: Mapping[str, str]) -> None:
    if dict(versions) != binding["runtime"]:
        raise ValueError("Runtime versions differ from the binding")


def runtime_versions() -> dict:
    import numpy
    import pandas
    import pydicom
    import torch

    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "pydicom": pydicom.__version__,
        "pandas": pandas.__version__,
    }


def check_source(binding: dict, root: Path, head: Callable[[Path], str]) -> None:
    if head(root) != binding["git_head"]:
        raise ValueError("Checkout differs from the bound Git source")
    for relative, digest in binding["files"].items():
        path = root / relative
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError("Bound source module bytes differ")


def check_loaded_modules(root: Path) -> None:
    from rsna.competitive import public_input, public_input_cache
    from tools.research.public_dinov2_input_reference import provenance, reference

    for module in (public_input, public_input_cache, provenance, reference):
        if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise ValueError("Imported module resolved outside the bound checkout")


def check_inputs(binding: dict, paths: Mapping[str, Path]) -> None:
    for name in INPUTS:
        path = Path(paths[name])
        if path.is_symlink() or not path.is_file() or sha256(path) != binding["inputs"][name]:
            raise ValueError("Bound input bytes differ")


def check_storage(
    binding: dict,
    output: Path,
    scratch: Path,
    resume: bool,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> None:
    caps = binding["caps"]
    for path in (output, scratch):
        if not path.is_absolute() or path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Private node-local output outside the checkout required")
    if scratch.exists() or (output.exists() and not resume) or (resume and not output.is_dir()):
        raise ValueError("Fresh scratch and an explicit resume decision required")
    probe = output if output.exists() else output.parent
    need = caps["max_member_bytes"] if resume else caps["max_cache_bytes"]
    if disk_usage(probe).free < need + caps["reserve_bytes"]:
        raise ValueError("Output storage envelope unavailable")
    if disk_usage(scratch.parent).free < caps["max_study_source_bytes"] + caps["reserve_bytes"]:
        raise ValueError("Scratch storage envelope unavailable")


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() if not status.stdout.strip() else "dirty"


def inventory_objects(inventory: Path, cohort: set[str]) -> dict[str, list]:
    import pandas as pd

    frame = pd.read_parquet(inventory)
    frame = frame[frame.study_uid.astype(str).isin(cohort)]
    objects: dict[str, list] = {uid: [] for uid in sorted(cohort)}
    for record in frame.to_dict("records"):
        value = {k: str(record[k]) for k in record if k != "bytes"}
        value["bytes"] = int(record["bytes"])
        objects[value["study_uid"]].append(value)
    return objects


def run(args: argparse.Namespace) -> dict:
    binding = load_binding(args.binding, args.binding_sha256)
    import torch

    check_environment(binding, os.environ, os.uname().nodename, torch.cuda.is_available())
    torch.set_num_threads(1)
    check_runtime(binding, runtime_versions())
    check_source(binding, ROOT, git_head)
    check_loaded_modules(ROOT)
    paths = {name: getattr(args, name) for name in INPUTS}
    check_inputs(binding, paths)
    check_storage(binding, args.output_root, args.scratch, args.resume)

    import pandas as pd

    from rsna.competitive import public_input as pi
    from rsna.competitive import public_input_cache as pc
    from rsna.competitive.folds import load_folds
    from rsna.competitive.train import select_development
    from rsna.release import load_verified_release
    from scripts.newton.compare_public_dinov2_inputs import member_pixel_group
    from tools.research.public_dinov2_input_reference import reference

    frame, release = load_verified_release(Path(args.studies), Path(args.artifact_root))
    folds = load_folds(Path(args.folds), frame, release)
    fitting, outer = select_development(frame, folds, 0)
    fitting_uids = list(map(str, fitting.study_uid))
    evaluation_uids = list(map(str, outer.study_uid))
    pc.validate_rosters(fitting_uids, evaluation_uids, EXPECTED_COUNTS)
    if (
        pi.roster_sha256(fitting_uids) != binding["fitting_uids_sha256"]
        or pi.roster_sha256(evaluation_uids) != binding["evaluation_uids_sha256"]
        or release["source_revision"] != binding["source_revision"]
    ):
        raise ValueError("Accepted fitting/evaluation rosters differ from the binding")
    pixel_group = member_pixel_group(Path(args.member_manifest).read_bytes())["pixel_group"]
    namespace, receipt = reference.load(Path(args.source).read_bytes())
    reference.configure(namespace, pixel_group)
    series = pd.read_csv(args.series)
    plane_map = dict(zip(series.SeriesInstanceUID.astype(str), series.Anatomical_Plane))
    renderer = pc.BoundRenderer(namespace, receipt, pixel_group, plane_map)
    objects = inventory_objects(args.inventory, set(fitting_uids) | set(evaluation_uids))
    summary = pc.produce(
        args.output_root,
        render=renderer,
        renderer_binding=renderer.binding,
        transport=pc.LocalTransport(args.dicom_root),
        objects=objects,
        fitting_uids=fitting_uids,
        evaluation_uids=evaluation_uids,
        expected_counts=EXPECTED_COUNTS,
        pixel_group=pixel_group,
        source_revision=binding["source_revision"],
        raw_source_revision=binding["raw_source_revision"],
        scratch=args.scratch,
        caps=pc.Caps(**binding["caps"]),
        provenance={"wrapper_binding_sha256": args.binding_sha256},
        resume=args.resume,
    )
    return summary | {"wrapper_binding_sha256": args.binding_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    for name in INPUTS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in ("artifact-root", "dicom-root", "output-root", "scratch"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    from rsna.competitive.public_input_cache import ProducerFailure

    try:
        result = run(args)
        code = 0
    except ProducerFailure as failure:
        result = {
            "status": "public_triplet_cohort_cache_incomplete_private",
            "stage": failure.stage,
            "error_type": failure.error_type,
            "completed_members": failure.completed,
        }
        code = 2
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with args.out.open("x") as stream:
            stream.write(text + "\n")
    print(text)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
