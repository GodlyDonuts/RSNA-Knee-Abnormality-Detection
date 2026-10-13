"""Single-T4 recovery of the exact final fitting scope; engineering, never accuracy."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from rsna.competitive.data import CompetitiveStudyDataset  # noqa: E402
from rsna.competitive.folds import load_folds  # noqa: E402
from rsna.competitive.supervision import apply_fitting_supervision  # noqa: E402
from rsna.competitive.train import select_development  # noqa: E402
from rsna.release import load_verified_release  # noqa: E402
from scripts.newton.probe_final_public_source_recovery import (  # noqa: E402
    engineering_config,
    validate_final_checkpoint,
)
from scripts.newton.probe_patch_witness_trainer_recovery import (  # noqa: E402
    digest,
    numerical_updates,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(source, manifest, cpu):
    expected = {
        name.removeprefix("code/"): value
        for name, value in cpu["source_modules"].items()
        if name.startswith("code/rsna/")
    }
    if not expected:
        raise ValueError("Original complete numerical source required")
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError("Original CPU-admitted scientific source differs")
    for name, value in manifest.items():
        path = source / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(source.resolve())
            or sha(path) != value
        ):
            raise ValueError("Recovery source path or bytes differ")
    return expected


def mounted_config(base, spec):
    for key in set(base["encoder"]) | set(spec):
        if key not in {"weights_path", "config_path"} and base["encoder"].get(key) != spec.get(key):
            raise ValueError("Mounted official encoder policy differs")
    return engineering_config(dict(base, encoder=spec, min_free_gib=3))


def admit_inputs(artifact, cache, overlay, values, cpu):
    if cpu.get("status") != "actual_final_public_source_CPU_roster_and_16MRI_inputs_verified" or (
        cpu.get("training_studies"),
        cpu.get("outer_studies"),
        cpu.get("masked_out_source_studies"),
    ) != (3403, 0, 0):
        raise ValueError("Actual original final-scope CPU admission required")
    studies = artifact / "initiative/releases/report-supervision-v1/studies.parquet"
    folds_path = artifact / "initiative/experiments/competitive-v1/folds"
    series_path = (
        artifact / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
    )
    frame, binding = load_verified_release(studies, artifact)
    if (
        binding["manifest_sha256"] != cpu["release_sha256"]
        or sha(folds_path / "fold.csv") != cpu["folds_sha256"]
    ):
        raise ValueError("Original release/fold bindings differ")
    folds = load_folds(folds_path, frame, binding)
    train, outer = select_development(frame, folds, 0, "all_development")
    if (
        len(train) != 3403
        or not outer.empty
        or digest(sorted(train.study_uid)) != cpu["fitting_uid_sha256"]
    ):
        raise ValueError("Original complete final-fitting roster differs")
    train, _ = apply_fitting_supervision(
        train,
        outer,
        overlay,
        values["fitting_supervision"],
        release_sha256=binding["manifest_sha256"],
        folds_sha256=sha(folds_path / "fold.csv"),
        fold=0,
    )
    data = CompetitiveStudyDataset(
        train.head(16),
        pd.read_csv(series_path),
        cache,
        slices=16,
        side=384,
        max_series=6,
        soft_targets=True,
    )
    images, labels = hashlib.sha256(), hashlib.sha256()
    for index in range(16):
        item = data[index]
        if not torch.isfinite(item["images"]).all():
            raise ValueError("Nonfinite original MRI input")
        images.update(item["images"].numpy().tobytes())
        labels.update(item["labels"].numpy().tobytes())
        labels.update(item["label_mask"].numpy().tobytes())
    if (
        images.hexdigest() != cpu["image_tensor_sha256"]
        or labels.hexdigest() != cpu["label_tensor_sha256"]
    ):
        raise ValueError("Mounted final16 image/label tensors differ from original CPU admission")
    return studies, folds_path, series_path, digest(sorted(train.head(16).study_uid))


def run(args):
    started = time.monotonic()
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("Exactly one visible T4 required")
    if shutil.disk_usage(args.output.parent).free < 9 * 2**30:
        raise ValueError("Three GiB reserve and six GiB recovery bound required")
    cpu = json.loads(args.cpu_receipt.read_text())
    if sha(args.base_config) != cpu["config_sha256"] or sha(args.overlay) != cpu["overlay_sha256"]:
        raise ValueError("Original full final config/overlay bytes differ")
    manifest = json.loads(args.source_manifest.read_text())
    source_identity(args.source, manifest, cpu)
    values = mounted_config(
        json.loads(args.base_config.read_text()), json.loads(args.encoder_spec.read_text())
    )
    studies, folds, series, roster = admit_inputs(
        args.control, args.cache, args.overlay, values, cpu
    )
    args.output.mkdir(mode=0o700)
    config = args.output / "config.json"
    config.write_text(json.dumps(values, sort_keys=True, indent=2) + "\n")
    env = dict(
        os.environ,
        PYTHONPATH=str(args.source),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        WORLD_SIZE="1",
        RANK="0",
        LOCAL_RANK="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    command = [
        sys.executable,
        "-B",
        str(args.source / "scripts/newton/enforce_deterministic_runtime.py"),
        "--module",
        "rsna.competitive.train",
        "--config",
        str(config),
        "--device",
        "cuda",
        "--fitting-supervision",
        str(args.overlay),
        "--studies",
        str(studies),
        "--artifact-root",
        str(args.control),
        "--folds",
        str(folds),
        "--series",
        str(series),
        "--cache",
        str(args.cache),
    ]

    def execute(extra, name, expected):
        if sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file()) > 6 * 2**30:
            raise ValueError("Recovery scratch cap exceeded")
        remaining = args.remaining_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Whole recovery deadline exhausted")
        print(
            json.dumps({"phase": name, "elapsed_seconds": time.monotonic() - started}), flush=True
        )
        with (
            (args.output / (name + ".stdout")).open("x") as out,
            (args.output / (name + ".stderr")).open("x") as err,
        ):
            result = subprocess.run(
                command + extra, env=env, cwd=args.source, stdout=out, stderr=err, timeout=remaining
            )
        if result.returncode != expected:
            raise ValueError("Actual trainer failed; retain original logs")

    full, replay = args.output / "full", args.output / "replay"
    execute(["--output", str(full)], "full", 0)
    execute(["--output", str(replay), "--stop-after-updates", "1"], "stop", 75)
    partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
    first = json.loads((replay / "invocation.json").read_text())
    execute(["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], "resume", 0)
    whole = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
    history = json.loads((full / "invocation.json").read_text())
    last = json.loads((replay / "invocation.json").read_text())
    state = validate_final_checkpoint(whole, resumed, partial, history, first, last)
    if (
        whole["binding"]["training_uids_sha256"] != roster
        or whole["binding"]["code_sha256"]
        != {n.removeprefix("rsna/"): h for n, h in manifest.items() if n.startswith("rsna/")}
        or any(p.exists() for d in (full, replay) for p in (d / "outer_predictions.csv",))
    ):
        raise ValueError("Actual source/roster or no-evaluation final scope differs")
    receipt = dict(
        status="actual_single_T4_final_public_source_recovery_accepted",
        CPU_receipt_sha256=sha(args.cpu_receipt),
        science_git=cpu["source_git"],
        complete_state_sha256=state,
        finite_updates=numerical_updates(history),
        amp_scaler=whole["amp_scaler"],
        exact_complete_state_scaler_RNG_recovery=True,
        elapsed_seconds=time.monotonic() - started,
        diagnostic_only=True,
        clinical_accuracy_evidence=False,
        competition_submission=False,
        GPUs_used_concurrently=1,
    )
    Path("/kaggle/working/final-public-source-recovery-accepted.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "control",
        "cache",
        "cpu-receipt",
        "base-config",
        "encoder-spec",
        "source",
        "source-manifest",
        "output",
        "overlay",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--remaining-seconds", type=float, required=True)
    run(parser.parse_args())
