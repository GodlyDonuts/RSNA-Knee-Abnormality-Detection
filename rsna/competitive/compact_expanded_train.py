"""Additional345 population contrast using the unchanged compact numerical fit.

The original loader and production jobs stay frozen. This entry point admits an
explicit2592 cohort, metadata union, and new supervision hash. Expanded16 passes
cost5184 updates; it must never be described as compute-matched to4496.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.competitive.cache_union import MetadataUnion
from rsna.competitive.checkpoint import atomic_json, file_sha256
from rsna.competitive.compact_bag import (
    MODEL_NAME,
    TIMM_VERSION,
    WEIGHT_SHA256,
    CompactBagClassifier,
    load_encoder,
)
from rsna.competitive.compact_train import Settings, fit, predict
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.expanded_fitting import select_population
from rsna.competitive.folds import load_folds
from rsna.competitive.label_source import load_fitting_label_source
from rsna.competitive.train import select_development
from rsna.release import load_verified_release
from rsna.schema import TARGETS

ADMISSION_SHA = "235538195accbac10369779324f10c3fbd7eea42981c95a7164144a17591c3de"
OVERLAY_SHA = "4ca82ceb772e0cc59e657b6281a3c6ee49e2f4803d77148828054607ac51b8c7"
INDEX_SHA = "1887a3e3e0c1f3422a61e66d2767b06ebe1af6756a8ee047f2890412dc164288"
POPULATION = "original_N3_plus345_fitting_only_v1"


def validate_admission(value):
    if (
        value.get("status") != "actual_expanded2592_source_and_native_inputs_CPU_verified"
        or value.get("node") != "evc33"
        or value.get("population", {}).get("expanded_fitting_count") != 2592
        or value.get("population", {}).get("baseline_fitting_count") != 2247
        or value.get("population", {}).get("additional_fitting_count") != 345
        or value.get("population", {}).get("outer_endpoint_count") != 1156
        or value.get("population", {}).get("excluded_all_fold_studies") != 1301
        or value.get("overlay", {}).get("overlay_sha256") != OVERLAY_SHA
        or value.get("metadata_union_index_sha256") != INDEX_SHA
        or value.get("old_image_tensor_parity") is not True
        or value.get("all_old_labels_masks_and_outer_unchanged") is not True
        or value.get("encoder_optimizer_GPU_or_accuracy_executed") is not False
        or value.get("source_MRI_payload_bytes_copied") != 0
    ):
        raise ValueError(
            "Exact actual2592 native-input admission with unchanged original endpoint required"
        )


def engineering_rows(fitting, baseline_uids):
    """Eight original plus eight added cases, with no label-dependent selection."""
    old = fitting.loc[fitting.study_uid.isin(baseline_uids)].sort_values("study_uid").head(8)
    added = fitting.loc[~fitting.study_uid.isin(baseline_uids)].copy()
    added["__engineering_order"] = added.study_uid.map(
        lambda uid: hashlib.sha256(uid.encode()).hexdigest()
    )
    added = added.sort_values("__engineering_order").head(8).drop(columns="__engineering_order")
    if len(old) != 8 or len(added) != 8:
        raise ValueError("Real original-eight and additional-eight engineering studies required")
    return (
        pd.concat([old, added], ignore_index=True).sort_values("study_uid").reset_index(drop=True)
    )


def admitted_datasets(args):
    if file_sha256(args.input_admission) != ADMISSION_SHA:
        raise ValueError("Original actual input receipt whole-byte identity differs")
    admission = json.loads(args.input_admission.read_text())
    validate_admission(admission)
    root = args.artifact_root
    studies, release = load_verified_release(
        root / "initiative/releases/report-supervision-v1/studies.parquet", root
    )
    fold_root = root / "initiative/experiments/competitive-v1/folds"
    folds = load_folds(fold_root, studies, release)
    baseline, original_outer = select_development(studies, folds, 0)
    fitting, outer, population = select_population(studies, folds)
    if population != admission["population"]:
        raise ValueError("Expanded source/release/fold membership differs from actual CPU proof")
    pd.testing.assert_frame_equal(outer, original_outer)
    if os.uname().nodename != admission["node"]:
        raise ValueError("Actual original owning-node private metadata union required")
    overlay = Path(admission["private_overlay"])
    fitting, supervision = load_fitting_label_source(
        fitting,
        outer,
        overlay,
        OVERLAY_SHA,
        release_sha256=release["manifest_sha256"],
        folds_sha256=file_sha256(fold_root / "fold.csv"),
        fold=0,
    )
    index = Path(admission["private_metadata_union"])
    union = MetadataUnion.load(index, INDEX_SHA)
    if union.value["studies"] != 3748 or union.value["series"] != 20675:
        raise ValueError("Exact actual native source membership required")
    path = root / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
    if file_sha256(path) != release["series_sha256"]:
        raise ValueError("Accepted original series table differs")
    series = pd.read_csv(path)
    if args.engineering:
        fitting = engineering_rows(fitting, set(baseline.study_uid))
        outer = outer.head(16)
    kwargs = dict(
        cache=union.cache,
        volume_provider=union,
        slices=16,
        side=384,
        max_series=6,
        augment=False,
        slice_sampling="uniform",
        series_sampling="selected",
    )
    train = CompetitiveStudyDataset(fitting, series, soft_targets=True, **kwargs)
    evaluation = CompetitiveStudyDataset(outer, series, soft_targets=False, **kwargs)
    binding = {
        "release": release,
        "supervision": supervision,
        "fitting_uids_sha256": hashlib.sha256(
            json.dumps(sorted(fitting.study_uid)).encode()
        ).hexdigest(),
        "outer_uids_sha256": hashlib.sha256(
            json.dumps(sorted(outer.study_uid)).encode()
        ).hexdigest(),
        "folds_file_sha256": file_sha256(fold_root / "fold.csv"),
        "cache": str(union.cache.resolve()),
        "diagnostic_only": args.engineering,
        "initialization_sha256": WEIGHT_SHA256,
        "initialization_revision": "2c31d82376cc244c907c064332941e6c65be81f1",
        "architecture": MODEL_NAME,
        "timm_version": TIMM_VERSION,
        "targets": list(TARGETS),
        "native_input": {
            "side": 384,
            "slices": 16,
            "max_series": 6,
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
        },
        "fitting_population": {
            "version": POPULATION,
            "population": population,
            "CPU_input_sha256": ADMISSION_SHA,
            "metadata_union_sha256": INDEX_SHA,
            "sixteen_passes_updates": 5184,
            "compute_matched_to_original": False,
        },
    }
    return train, evaluation, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "input-admission", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-updates", type=int)
    args = parser.parse_args()
    if (args.output / "result.json").exists() or (
        args.stop_after_updates is not None and not args.engineering
    ):
        parser.error("Do not repeat finalized fitting; explicit update stops are engineering-only")
    torch.set_num_threads(1)
    train, outer, binding = admitted_datasets(args)
    source = Path(__file__).resolve().parents[2]
    binding["source_sha256"] = {
        str(p.relative_to(source)): file_sha256(p) for p in sorted((source / "rsna").rglob("*.py"))
    }
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or (
            torch.cuda.device_count() != 1 or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        )
    ):
        raise ValueError(
            "One actually allocated BF16 CUDA device and deterministic workspace required"
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(20260922)
    np.random.seed(20260922)
    torch.manual_seed(20260922)
    torch.cuda.manual_seed_all(20260922)
    model = CompactBagClassifier(load_encoder(args.weights))
    head = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith("encoder."):
            head.update(name.encode())
            head.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    binding["initial_head_sha256"] = head.hexdigest()
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda _s, _f: stop.__setitem__(0, True))
    settings = Settings(epochs=2 if args.engineering else 16)
    started = time.monotonic()
    result = fit(
        model,
        train,
        args.output,
        binding,
        settings,
        device=torch.device("cuda"),
        resume=args.resume,
        stop_after_updates=args.stop_after_updates,
        should_stop=lambda: stop[0],
    )
    if result["status"] != "fixed_final_fit_complete":
        atomic_json(args.output / "stopped.json", result)
        raise SystemExit(75)
    predictions = predict(model, outer, torch.device("cuda"))
    predictions.to_csv(args.output / "outer_predictions.csv", index=False)
    from rsna.model import auc_metrics

    aligned = outer.studies.set_index("study_uid").loc[predictions.study_uid]
    metrics = auc_metrics(
        aligned[list(TARGETS)].to_numpy(dtype=float),
        predictions[list(TARGETS)].to_numpy(dtype=float),
        aligned[[t + "__mask" for t in TARGETS]].to_numpy(dtype=float),
    )
    atomic_json(
        args.output / "result.json",
        {
            **result,
            "metrics": metrics,
            "elapsed_seconds": time.monotonic() - started,
            "diagnostic_only": args.engineering,
            "predictions_sha256": file_sha256(args.output / "outer_predictions.csv"),
            "checkpoint_sha256": file_sha256(args.output / "checkpoint.pt"),
        },
    )


if __name__ == "__main__":
    main()
