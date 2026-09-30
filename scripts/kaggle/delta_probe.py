"""Private offline T4 engineering probe. No labels, leaderboard or submission call.

The bound notebook bootstrap installs the exact offline runtime before launching
this module. The diagnostic is never passed to the competition coordinator.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.infer import validate_inference_checkpoint
from rsna.competitive.model import CompetitiveStudyNet
from rsna.schema import TARGETS


def compare_outputs(delta: Path, full: Path) -> dict:
    """Require every MRI probability and UID to survive the alternate storage route."""
    left, right = pd.read_csv(delta), pd.read_csv(full)
    columns = ["StudyInstanceUID", *TARGETS]
    if (
        list(left.columns) != columns
        or list(right.columns) != columns
        or left.empty
        or left.StudyInstanceUID.duplicated().any()
        or left.StudyInstanceUID.tolist() != right.StudyInstanceUID.tolist()
    ):
        raise ValueError("Reference/delta MRI output roster differs")
    values = left[list(TARGETS)].to_numpy()
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.array_equal(values, right[list(TARGETS)].to_numpy())
        or delta.read_bytes() != full.read_bytes()
    ):
        raise ValueError("Reference/delta MRI probabilities or output bytes differ")
    return {"studies": len(left), "prediction_values_equal": True, "csv_bytes_equal": True}


def expanded_reference(checkpoint: Path, spec: Path, output: Path) -> dict:
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    binding = validate_inference_checkpoint(saved, allow_diagnostic=True)
    config = binding["config"]
    if (
        saved["schema_version"] != 2
        or config.get("diagnostic_study_limit") != 16
        or config.get("slice_sampling") != "all"
        or config.get("series_sampling") != "all"
        or config.get("slice_context") != "neighbor3"
        or saved.get("step") != 4
    ):
        raise ValueError("Only the exact bounded whole-series delta diagnostic is allowed")
    encoder, dimension, provenance = create_encoder(json.loads(spec.read_text()))
    if provenance != config["encoder"]:
        raise ValueError("Diagnostic original encoder provenance differs")
    configure_encoder_training(encoder, config["freeze_encoder"])
    model = CompetitiveStudyNet(
        encoder,
        dimension,
        pooling=config["pooling"],
        hidden_dim=config["hidden_dim"],
        encode_chunk_size=config["encode_chunk_size"],
        slice_context=config["slice_context"],
    )
    guard = FrozenBaseDelta(model, provenance)
    guard.restore(model, saved)
    # The official backbone has 415 tensors. The grayscale wrapper also stores
    # two normalization buffers; delta restoration binds all 417 frozen values.
    backbone = [name for name in guard.frozen if name.startswith("encoder.backbone.")]
    normalization = set(guard.frozen) - set(backbone)
    if (
        len(backbone) != 415
        or normalization != {"encoder.mean", "encoder.std"}
        or len(provenance["adapter"]["target_modules"]) != 120
    ):
        raise ValueError("Original frozen base or adapter partition differs")
    full = copy.deepcopy(
        {key: value for key, value in saved.items() if key not in {"model", "model_storage"}}
    )
    full["schema_version"] = 1
    full["binding"]["config"]["checkpoint_format"] = "full"
    full["model"] = model.state_dict()
    # This is disposable Kaggle scratch, not Newton's shared quota or the Mac's
    # disk reserve. Budget the full state plus the delta envelope (an upper bound
    # on optimizer/metadata), one bounded MRI chunk and separate safety headroom.
    reference_bound = (
        sum(tensor.numel() * tensor.element_size() for tensor in full["model"].values())
        + checkpoint.stat().st_size
        + 64 * 1024**2
    )
    scratch = {
        "reference_upper_bound_bytes": reference_bound,
        "mri_chunk_budget_bytes": 1024**3,
        "safety_headroom_bytes": 2 * 1024**3,
        "free_before_reference_bytes": shutil.disk_usage(output.parent).free,
    }
    scratch["required_free_bytes"] = sum(
        scratch[key]
        for key in (
            "reference_upper_bound_bytes",
            "mri_chunk_budget_bytes",
            "safety_headroom_bytes",
        )
    )
    print(json.dumps({"reference_scratch_preflight": scratch}, sort_keys=True), flush=True)
    if output.exists() or output.is_symlink():
        raise ValueError("Reference scratch must be new")
    if scratch["free_before_reference_bytes"] < scratch["required_free_bytes"]:
        raise ValueError("Reference scratch lacks the measured workload budget")
    with output.open("xb") as handle:
        torch.save(full, handle)
    if output.stat().st_size > reference_bound:
        raise ValueError("Expanded reference exceeds its measured scratch bound")
    return {
        "frozen_base_tensors_verified": len(backbone),
        "frozen_encoder_tensors_verified": len(guard.frozen),
        "normalization_buffers_verified": len(normalization),
        "adapter_modules_verified": 120,
        "reference_scratch": scratch,
    }


def run(args: Namespace) -> dict:
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Diagnostic output must be new")
    args.output.mkdir(mode=0o700)
    bundle = args.bundle
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    if (
        manifest.get("diagnostic_only") is not True
        or manifest.get("competition_submission_eligible") is not False
    ):
        raise ValueError("Probe bundle must remain explicitly diagnostic and ineligible")
    spec = args.encoder_spec
    checkpoint = bundle / "checkpoint.pt"
    full_file = args.output / "full-reference.pt"
    partition = expanded_reference(checkpoint, spec, full_file)
    loaded = importlib.util.spec_from_file_location(
        "bounded_mri_inference", Path(__file__).with_name("chunked_infer.py")
    )
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    workloads = [
        {
            "name": "public",
            "series": args.competition / "test_series.csv",
            "roster": args.competition / "test.csv",
            "source": args.competition,
        }
    ]
    plan = json.loads((bundle / "probe-plan.json").read_text())
    ids = plan["largest_training_study_ids"]
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError("Exactly three previously profiled training-image identities required")
    # Training series metadata supplies identities/scan features only; no report,
    # label CSV, gold or confirmation target is opened anywhere in this probe.
    training = pd.read_csv(args.competition / "train_series.csv")
    training = training.loc[training.StudyInstanceUID.isin(ids)].copy()
    if set(training.StudyInstanceUID) != set(ids):
        raise ValueError("Largest profiled training MRI inputs are absent")
    alias = args.output / "training-mri-source" / "test_series"
    for uid, sid in training[["StudyInstanceUID", "SeriesInstanceUID"]].itertuples(
        index=False, name=None
    ):
        source = args.competition / "train_series" / uid / sid
        if not source.is_dir() or Path(uid).name != uid or Path(sid).name != sid:
            raise ValueError("Training MRI path is invalid or absent")
        destination = alias / uid / sid
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve(), target_is_directory=True)
    series_file = args.output / "largest-series.csv"
    training.to_csv(series_file, index=False)
    roster_file = args.output / "largest-roster.csv"
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(roster_file, index=False)
    workloads.append(
        {
            "name": "largest-training-images",
            "series": series_file,
            "roster": roster_file,
            "source": alias.parent,
        }
    )
    results = []
    for work in workloads:
        for storage in ("delta", "full"):
            parameters = Namespace(
                checkpoint=checkpoint if storage == "delta" else full_file,
                encoder_config=None
                if storage == "delta"
                else Path(json.loads(spec.read_text())["config_path"]),
                encoder_spec=spec if storage == "delta" else None,
                series=work["series"],
                test_studies=work["roster"],
                template=None,
                cache_root=args.output / (work["name"] + "-" + storage + "-cache"),
                output=args.output / (work["name"] + "-" + storage + ".csv"),
                dicom_root=work["source"],
                source_revision="kaggle-delta-diagnostic-" + work["name"],
                chunk_size=3,
                workers=0,
                device="cuda",
                precision="fp32",
                allow_diagnostic=True,
                resilient_series=True,
                fallback_on_unusable_study=True,
                recover_slices=True,
                validate_model_geometry=True,
                isolate_study_errors=True,
                cache_side=256,
                max_series=6,
                series_sampling="all",
            )
            result = module.run(parameters, invoke=subprocess.run)
            if result["studies"] != 3 or sum(
                c.get("fallback_studies", 0) for c in result["chunks"]
            ):
                raise ValueError("Probe requires three MRI-backed studies without neutral fallback")
        compared = compare_outputs(
            args.output / (work["name"] + "-delta.csv"), args.output / (work["name"] + "-full.csv")
        )
        results.append({"name": work["name"], **compared})
    receipt = {
        "schema_version": 1,
        "status": "verified_offline_base_delta_fp32_equivalence",
        "diagnostic_only": True,
        "competition_submission_eligible": False,
        "competition_submission_created": False,
        "labels_read": False,
        "kaggle_score": None,
        "bundle_manifest_sha256": module.sha256(bundle / "bundle-manifest.json"),
        "checkpoint_sha256": module.sha256(checkpoint),
        "precision": "fp32",
        "workloads": results,
        "original_weights_sha256": json.loads(spec.read_text())["weights_sha256"],
        "probe_source_sha256": module.sha256(Path(__file__)),
        "chunked_inference_sha256": module.sha256(Path(__file__).with_name("chunked_infer.py")),
        **partition,
    }
    (args.output / "delta-deployment-acceptance.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    # Remove only the newly generated diagnostic reference; the immutable inputs
    # remain on the private attached Dataset, and small evidence remains exportable.
    full_file.unlink()
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "encoder-spec", "competition", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
