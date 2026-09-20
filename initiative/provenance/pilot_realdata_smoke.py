"""Reproduce the bounded real-data pipeline check; never release training labels.

Run from the repository with --cache pointing to the independently audited
153-study hardlink snapshot. This diagnostic uses reviewed batches 000..002.
Its tiny model and checkpoints are pipeline evidence, not competition weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.labels import (
    build_studies,
    file_sha256,
    read_reviews,
    report_group,
    validate_dicom_audit,
)
from rsna.model import StudyNet
from rsna.review_progress import collect
from rsna.schema import TARGETS


def assert_identical(expected, actual) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_identical(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_identical(left, right)
    else:
        assert expected == actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous smoke outputs; use a new output directory")
    source = Path("initiative/source")
    reviews_root = Path("initiative/reviews-v2")
    image_audit = Path("initiative/audits/pilot-realdata-v3")
    source_revision = "52f995f4568e36bfa5fbbaf69a89af41226b4f5d"
    progress = collect(
        Path("initiative/reviews/inputs"), reviews_root, Path("protocols/report_weak_v2.md")
    )
    selected = {f"batch-{i:03d}" for i in range(3)}
    for batch in progress["batches"]:
        if batch["batch"] in selected:
            assert all(batch[stage]["status"] == "valid" for stage in ("draft", "reviewed"))
    frames = []
    for batch in sorted(selected):
        records = json.loads(Path(f"initiative/reviews/inputs/{batch}.json").read_text())
        reports = {row["study_uid"]: row["report"] for row in records}
        frames.append(read_reviews(reviews_root / "reviewed", reports, f"{batch}-reviewed"))
    reviewed = pd.concat(frames, ignore_index=True)
    original = pd.read_csv(source / "train.csv")
    pilot_source = original[original.StudyInstanceUID.isin(reviewed.study_uid)].copy()
    inventory = pd.read_parquet(image_audit / "pilot_inventory.parquet")
    pairs = validate_dicom_audit(
        image_audit / "dicom_audit.json",
        inventory,
        set(pilot_source.StudyInstanceUID),
        source_revision,
    )
    if pairs:
        raise ValueError("New pilot image duplicates need an explicit component policy")
    studies = build_studies(pilot_source, reviewed)

    # Bind pilot splits to normalized reports across ALL source studies, including
    # official holdouts outside the pilot. No official label values are analyzed.
    original["report_group"] = original.Report.map(report_group)
    group_root = original.groupby("report_group").StudyInstanceUID.min()
    studies["leakage_group"] = studies.report_group.map(group_root).map(
        lambda uid: hashlib.sha256(uid.encode()).hexdigest()
    )
    studies["split"] = studies.leakage_group.map(
        lambda group: "weak_validation" if int(group, 16) % 10 == 0 else "train"
    )
    official = original[list(TARGETS)].notna().any(axis=1)
    gold_report_groups = set(original.loc[official, "report_group"])
    studies.loc[studies.report_group.isin(gold_report_groups), "split"] = "quarantine"
    studies.loc[studies.study_uid.isin(original.loc[official, "StudyInstanceUID"]), "split"] = (
        "gold_evaluation"
    )
    fit = studies[studies.split.isin(["train", "weak_validation"])]
    assert not set(fit.report_group) & gold_report_groups
    assert studies.groupby("leakage_group").split.nunique().max() == 1
    args.output.mkdir(parents=True)
    manifest = args.output / "pilot_studies.parquet"
    studies.to_parquet(manifest, index=False)
    scope = {
        "scope": "Pipeline diagnostic on independently reviewed batches 000..002 only",
        "training_ready": False,
        "competition_weights": False,
        "source_revision": source_revision,
        "pilot_studies": len(studies),
        "splits": studies.split.value_counts().to_dict(),
        "global_source_studies_checked_for_report_holdout": len(original),
        "limitations": [
            "Full decoded-pixel duplicate graph is unfinished; cross-pilot matches unknown.",
            "Pseudonymous PatientIDs do not establish true patient disjointness.",
            "Small CPU model tests the pipeline, not diagnostic performance.",
            "Gold agreement is not inspected; this run cannot tune the frozen rubric.",
            "Final released manifests require another smoke run after all release gates.",
        ],
        "input_hashes": {
            str(path): file_sha256(path)
            for path in [
                source / "train.csv",
                source / "train_series.csv",
                source / "test_series.csv",
                source / "sample_submission.csv",
                image_audit / "dicom_audit.json",
                image_audit / "pilot_inventory.parquet",
                manifest,
                Path(__file__),
                *[
                    reviews_root / "reviewed" / f"{batch}-reviewed.jsonl"
                    for batch in sorted(selected)
                ],
            ]
        },
    }
    (args.output / "scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    commands = []

    def run(command: list[str]) -> None:
        commands.append(command)
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        subprocess.run(command, check=True)

    common = [
        sys.executable,
        "-m",
        "rsna.train",
        "--studies",
        str(manifest),
        "--series",
        str(source / "train_series.csv"),
        "--cache",
        str(args.cache),
        "--width",
        "8",
        "--slices",
        "4",
        "--side",
        "64",
        "--max-series",
        "3",
        "--batch-size",
        "2",
        "--workers",
        "0",
        "--device",
        "cpu",
    ]
    continuous, resumed = args.output / "continuous", args.output / "resumed"
    run(common + ["--epochs", "2", "--output", str(continuous)])
    run(common + ["--epochs", "1", "--output", str(resumed)])
    run(
        common
        + [
            "--epochs",
            "2",
            "--output",
            str(resumed),
            "--resume",
            str(resumed / "checkpoint.pt"),
        ]
    )
    expected = torch.load(continuous / "checkpoint.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoint.pt", weights_only=True)
    assert_identical(expected, actual)
    assert actual["epoch_complete"] and actual["epoch"] == 1 and actual["step"] > 0
    for tensor in actual["model"].values():
        assert torch.isfinite(tensor).all()
    torch.manual_seed(42)
    initial = StudyNet(8).state_dict()
    changed = sum(not torch.equal(initial[key], value) for key, value in actual["model"].items())
    assert changed > 0
    prediction = args.output / "sample_submission_pipeline_check.csv"
    run(
        [
            sys.executable,
            "-m",
            "rsna.infer",
            "--checkpoint",
            str(resumed / "checkpoint.pt"),
            "--series",
            str(source / "test_series.csv"),
            "--cache",
            str(args.cache),
            "--template",
            str(source / "sample_submission.csv"),
            "--output",
            str(prediction),
            "--workers",
            "0",
            "--device",
            "cpu",
        ]
    )
    predictions, template = pd.read_csv(prediction), pd.read_csv(source / "sample_submission.csv")
    assert list(predictions.columns) == list(template.columns)
    assert predictions.StudyInstanceUID.tolist() == template.StudyInstanceUID.tolist()
    values = predictions[list(TARGETS)].to_numpy()
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    result = {
        "pipeline_check_passed": True,
        "training_ready": False,
        "epochs": 2,
        "optimizer_steps": actual["step"],
        "updated_model_tensors": changed,
        "continuous_and_resumed_entire_checkpoint_exactly_equal": True,
        "sample_submission_studies": len(predictions),
        "sample_submission_target_columns": len(TARGETS),
        "submission_order_and_finite_probabilities_verified": True,
        "output_hashes": {
            str(path): file_sha256(path)
            for path in [continuous / "checkpoint.pt", resumed / "checkpoint.pt", prediction]
        },
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
