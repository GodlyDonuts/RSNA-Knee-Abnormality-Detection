"""Bounded engineering profile of one throughput setting, in a fresh process.

Run once per setting (never two settings in one interpreter) on actual original
MRI. Writes one new JSON receipt; no checkpoint, prediction or accuracy is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rsna.competitive import throughput
from rsna.competitive.train import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "settings", "studies", "series", "cache", "folds", "artifact-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--measured-updates", type=int, required=True)
    parser.add_argument("--warmup-updates", type=int, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-series-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-torch-version", required=True)
    parser.add_argument("--expected-device-name", required=True)
    parser.add_argument("--evict-page-cache", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("--receipt must be a new path")
    raise SystemExit(run(args))


def run(args: argparse.Namespace, factories: dict | None = None) -> int:
    if factories is None:
        from rsna.competitive.backbones import create_encoder
        from rsna.competitive.data import CompetitiveStudyDataset, collate_studies
        from rsna.competitive.folds import load_folds
        from rsna.competitive.model import CompetitiveStudyNet

        factories = dict(
            create_encoder=create_encoder,
            dataset_class=CompetitiveStudyDataset,
            model_class=CompetitiveStudyNet,
            load_folds=load_folds,
            collate=collate_studies,
        )
    fresh = args.device != "cuda" or not torch.cuda.is_initialized()
    if not fresh:
        raise ValueError("Each setting requires a fresh process")
    config_bytes = args.config.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != args.expected_config_sha256:
        raise ValueError("Recipe config differs from its explicit binding")
    base = TrainConfig(**json.loads(config_bytes))
    base.validate()
    settings = json.loads(args.settings.read_text())["settings"]
    if args.setting not in settings:
        raise ValueError("Unknown throughput setting")
    candidate = throughput.apply_setting(base, settings[args.setting]["overrides"])
    caps = throughput.validate_caps(
        args.measured_updates, args.warmup_updates, args.max_wall_seconds
    )
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but no assigned CUDA device is available")
        torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    if str(torch.__version__) != args.expected_torch_version:
        raise ValueError("Runtime torch version differs from its explicit binding")
    if args.expected_device_name not in device_name:
        raise ValueError("Assigned device differs from its explicit binding")
    state = throughput.build_profile_state(
        args,
        candidate,
        device,
        factories["create_encoder"],
        factories["dataset_class"],
        factories["model_class"],
        factories["load_folds"],
    )
    release = state["release_binding"]
    for key, expected in (
        ("manifest_sha256", args.expected_manifest_sha256),
        ("series_sha256", args.expected_series_sha256),
        ("source_revision", args.expected_source_revision),
    ):
        if release.get(key) != expected:
            raise ValueError(f"Release {key} differs from its explicit binding")
    profile = throughput.profile_updates(state, candidate, device, caps, factories["collate"])
    profiled = {uid for record in profile["records"] for uid in record["study_uids"]}
    data = state["training_data"]
    indices = [i for i, uid in enumerate(data.studies.study_uid) if uid in profiled]
    decomposition = (
        throughput.decompose_inputs(data, indices, args.evict_page_cache)
        if hasattr(data, "_metadata")
        else None
    )
    blockers = []
    if device.type != "cuda":
        blockers.append("not_cuda")
    if candidate.diagnostic_study_limit is not None:
        blockers.append("diagnostic_study_limit")
    if candidate.encoder.get("architecture") != "dinov3_l16":
        blockers.append("not_canonical_encoder")
    if profile["stop_reason"] != "measured_updates_completed":
        blockers.append(profile["stop_reason"])
    if decomposition is None:
        blockers.append("input_decomposition_unavailable")
    receipt = throughput.build_receipt(
        setting=args.setting,
        base=base,
        candidate=candidate,
        config_sha256=config_sha256,
        caps=caps,
        profile=profile,
        bindings={
            "config_sha256": config_sha256,
            "settings_sha256": hashlib.sha256(args.settings.read_bytes()).hexdigest(),
            "manifest_sha256": args.expected_manifest_sha256,
            "series_sha256": args.expected_series_sha256,
            "source_revision": args.expected_source_revision,
            "torch_version": args.expected_torch_version,
            "device_name": args.expected_device_name,
            "evict_page_cache": bool(args.evict_page_cache),
            "fresh_process": fresh,
        },
        release_binding=release,
        train=state["train"],
        device=device,
        decomposition=decomposition,
        evidence_blockers=blockers,
    )
    throughput.write_new_receipt(args.receipt, receipt)
    return 0


if __name__ == "__main__":
    main()
