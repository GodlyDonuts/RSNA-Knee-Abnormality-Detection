from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna.correspondence import REGISTRY_PATH, apply_correspondence, inspect_correspondence
from rsna.labels import audit_labels, build_studies, file_sha256
from rsna.schema import TARGETS


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def reference(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}


@pytest.fixture
def bundle(tmp_path: Path) -> dict:
    reports = {
        "bad": "Shoulder report.",
        "gold": "Knee report.",
        "neighbor": "Knee report.",
        "safe": "Other knee.",
    }
    train = pd.DataFrame(
        [
            {
                "StudyInstanceUID": uid,
                "Report": report,
                **{t: 1 if uid == "gold" else np.nan for t in TARGETS},
            }
            for uid, report in reports.items()
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "study_uid": uid,
                **{f"{t}__state": "positive" for t in TARGETS},
                **{f"{t}__evidence": report for t in TARGETS},
            }
            for uid, report in reports.items()
        ]
    )
    source = tmp_path / "source.json"
    write(source, [{"study_uid": uid, "report": report} for uid, report in reports.items()])
    records = []
    for uid in ["bad", "gold"]:
        proof = tmp_path / f"proof-{uid}.json"
        write(proof, {"case": uid, "study_uid": uid, "status": "verified"})
        records.append(
            {
                "case": uid,
                "study_uid": uid,
                "source_path": "source.json",
                "source_batch_sha256": file_sha256(source),
                "source_report_sha256": hashlib.sha256(reports[uid].encode()).hexdigest(),
                "status": "report_image_mismatch",
                "reason": "Independently reviewed correspondence failure.",
                "scope": "Whole-study report-derived supervision only.",
                "report_derived_training_supervision_allowed": False,
                "required_report_loss_mask": [False] * len(TARGETS),
                "verified_technical_proof": reference(proof, tmp_path),
            }
        )
    policy = tmp_path / "initiative/audits/correspondence-release-dispositions-v1.json"
    write(
        policy,
        {
            "schema_version": 1,
            "stage": "root_verified_correspondence_release_requirements",
            "target_order": list(TARGETS),
            "records": records,
        },
    )
    registry = tmp_path / REGISTRY_PATH
    write(
        registry,
        {
            "schema_version": 1,
            "target_order": list(TARGETS),
            "policies": [reference(policy, tmp_path)],
        },
    )
    return {
        "root": tmp_path,
        "reports": reports,
        "train": train,
        "reviews": reviews,
        "policy": policy,
        "registry": registry,
    }


def rebind(bundle: dict) -> None:
    registry = json.loads(bundle["registry"].read_text())
    registry["policies"] = [reference(bundle["policy"], bundle["root"])]
    write(bundle["registry"], registry)


def install_identity_wrapper(bundle: dict) -> dict[str, Path]:
    """Preserve a historical UID-less review and bind it through its source handoff."""
    root = bundle["root"]
    policy = json.loads(bundle["policy"].read_text())
    record = policy["records"][0]
    handoff = root / "source-handoff.json"
    write(handoff, {"case": "bad", "study_uid": "bad"})
    review = root / "preserved-native-review.json"
    write(
        review,
        {"case": "bad", "status": "verified", "bindings": [reference(handoff, root)]},
    )
    wrapper = root / "identity-wrapper.json"
    write(
        wrapper,
        {
            "stage": "root_verified_correspondence_source_identity_binding",
            "case": "bad",
            "study_uid": "bad",
            "status": "verified",
            "source_batch": {
                "path": record["source_path"],
                "sha256": record["source_batch_sha256"],
            },
            "source_report_sha256": record["source_report_sha256"],
            "verified_review": reference(review, root),
            "source_handoff": reference(handoff, root),
        },
    )
    record["verified_technical_proof"] = reference(wrapper, root)
    write(bundle["policy"], policy)
    rebind(bundle)
    return {"wrapper": wrapper, "review": review, "handoff": handoff}


def rebind_wrapper(bundle: dict, paths: dict[str, Path]) -> None:
    """Refresh outer checksums so semantic-mismatch tests reach the inner binding."""
    wrapper = json.loads(paths["wrapper"].read_text())
    wrapper["verified_review"] = reference(paths["review"], bundle["root"])
    wrapper["source_handoff"] = reference(paths["handoff"], bundle["root"])
    write(paths["wrapper"], wrapper)
    policy = json.loads(bundle["policy"].read_text())
    policy["records"][0]["verified_technical_proof"] = reference(paths["wrapper"], bundle["root"])
    write(bundle["policy"], policy)
    rebind(bundle)


@pytest.mark.parametrize(
    "mutation", ["empty_registry", "unlisted_policy", "outside_policy", "missing_policy"]
)
def test_registry_cannot_omit_or_replace_preserved_policies(bundle: dict, mutation: str) -> None:
    registry = json.loads(bundle["registry"].read_text())
    if mutation == "empty_registry":
        registry["policies"] = []
    elif mutation == "unlisted_policy":
        extra = bundle["policy"].with_name("correspondence-release-dispositions-v2.json")
        policy = json.loads(bundle["policy"].read_text())
        policy["records"] = []
        write(extra, policy)
    elif mutation == "outside_policy":
        replacement = bundle["root"] / "outside-policy.json"
        replacement.write_bytes(bundle["policy"].read_bytes())
        registry["policies"] = [reference(replacement, bundle["root"])]
    else:
        bundle["policy"].unlink()
    write(bundle["registry"], registry)
    with pytest.raises(ValueError):
        apply_correspondence(
            build_studies(bundle["train"], bundle["reviews"]), bundle["reports"], bundle["root"]
        )


def test_explicit_empty_registry_only_without_policy_files(bundle: dict) -> None:
    bundle["policy"].unlink()
    registry = json.loads(bundle["registry"].read_text())
    registry["policies"] = []
    write(bundle["registry"], registry)
    baseline = build_studies(bundle["train"], bundle["reviews"])
    effective, trace = apply_correspondence(baseline, bundle["reports"], bundle["root"])
    pd.testing.assert_frame_equal(effective[baseline.columns], baseline)
    assert effective.report_supervision_allowed.all()
    assert trace["policy_count"] == trace["study_count"] == 0


def test_all_preserved_policy_versions_are_enforced(bundle: dict) -> None:
    first = json.loads(bundle["policy"].read_text())
    second = {**first, "records": [first["records"].pop()]}
    write(bundle["policy"], first)
    second_path = bundle["policy"].with_name("correspondence-release-dispositions-v2.json")
    write(second_path, second)
    registry = json.loads(bundle["registry"].read_text())
    registry["policies"] = [
        reference(path, bundle["root"]) for path in [bundle["policy"], second_path]
    ]
    write(bundle["registry"], registry)
    effective, trace = apply_correspondence(
        build_studies(bundle["train"], bundle["reviews"]), bundle["reports"], bundle["root"]
    )
    assert trace["policy_count"] == trace["study_count"] == 2
    assert set(effective.loc[~effective.report_supervision_allowed, "study_uid"]) == {"bad", "gold"}


@pytest.mark.parametrize("reassigned", [False, True])
def test_uidless_proof_never_inherits_policy_study(bundle: dict, reassigned: bool) -> None:
    proof = bundle["root"] / "proof-bad.json"
    write(proof, {"case": "bad", "status": "verified"})
    policy = json.loads(bundle["policy"].read_text())
    record = policy["records"][0]
    record["verified_technical_proof"] = reference(proof, bundle["root"])
    if reassigned:
        record["study_uid"] = "safe"
        record["source_report_sha256"] = hashlib.sha256(
            bundle["reports"]["safe"].encode()
        ).hexdigest()
    write(bundle["policy"], policy)
    rebind(bundle)
    with pytest.raises(ValueError, match="proof"):
        inspect_correspondence(bundle["reports"], bundle["root"])


def test_identity_wrapper_preserves_original_uidless_review(bundle: dict) -> None:
    paths = install_identity_wrapper(bundle)
    originals = {key: path.read_bytes() for key, path in paths.items()}
    effective, trace = apply_correspondence(
        build_studies(bundle["train"], bundle["reviews"]), bundle["reports"], bundle["root"]
    )
    assert "study_uid" not in json.loads(paths["review"].read_text())
    assert {key: path.read_bytes() for key, path in paths.items()} == originals
    excluded = effective.set_index("study_uid").loc["bad"]
    assert all(excluded[f"{target}__mask"] == 0 for target in TARGETS)
    assert trace["decisions"][0]["verified_technical_proof"] == reference(
        paths["wrapper"], bundle["root"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrapper_uid",
        "wrapper_case",
        "source_path",
        "source_batch_hash",
        "report_hash",
        "review_case",
        "review_uid",
        "review_unverified",
        "review_missing_handoff_binding",
        "review_wrong_handoff_binding",
        "handoff_case",
        "handoff_uid",
        "handoff_missing_uid",
        "handoff_replaced_after_review",
        "review_bytes_tampered",
        "handoff_bytes_tampered",
        "review_missing",
        "handoff_missing",
    ],
)
def test_identity_wrapper_rejects_broken_source_or_review_chain(
    bundle: dict, mutation: str
) -> None:
    paths = install_identity_wrapper(bundle)
    if mutation in {"review_bytes_tampered", "handoff_bytes_tampered"}:
        path = paths[mutation.removesuffix("_bytes_tampered")]
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation in {"review_missing", "handoff_missing"}:
        paths[mutation.removesuffix("_missing")].unlink()
    else:
        wrapper = json.loads(paths["wrapper"].read_text())
        review = json.loads(paths["review"].read_text())
        handoff = json.loads(paths["handoff"].read_text())
        if mutation == "wrapper_uid":
            wrapper["study_uid"] = "safe"
        elif mutation == "wrapper_case":
            wrapper["case"] = "other"
        elif mutation == "source_path":
            alternate = bundle["root"] / "other-source.json"
            alternate.write_bytes((bundle["root"] / "source.json").read_bytes())
            wrapper["source_batch"] = reference(alternate, bundle["root"])
        elif mutation == "source_batch_hash":
            wrapper["source_batch"]["sha256"] = "0" * 64
        elif mutation == "report_hash":
            wrapper["source_report_sha256"] = hashlib.sha256(b"Different report.").hexdigest()
        elif mutation == "review_case":
            review["case"] = "other"
        elif mutation == "review_uid":
            review["study_uid"] = "safe"
        elif mutation == "review_unverified":
            review["status"] = "pending"
        elif mutation == "review_missing_handoff_binding":
            review["bindings"] = []
        elif mutation == "review_wrong_handoff_binding":
            review["bindings"][0]["sha256"] = "0" * 64
        elif mutation == "handoff_case":
            handoff["case"] = "other"
        elif mutation == "handoff_uid":
            handoff["study_uid"] = "safe"
        elif mutation == "handoff_missing_uid":
            del handoff["study_uid"]
        elif mutation == "handoff_replaced_after_review":
            handoff["notes"] = "Different preserved handoff, same apparent study identity."
        write(paths["wrapper"], wrapper)
        write(paths["handoff"], handoff)
        if mutation in {"handoff_case", "handoff_uid", "handoff_missing_uid"}:
            review["bindings"] = [reference(paths["handoff"], bundle["root"])]
        write(paths["review"], review)
        rebind_wrapper(bundle, paths)
    with pytest.raises(ValueError):
        inspect_correspondence(bundle["reports"], bundle["root"])


def test_masks_report_loss_preserves_annotations_gold_and_duplicate_groups(bundle: dict) -> None:
    baseline = build_studies(bundle["train"], bundle["reviews"])
    unchanged = baseline.copy(deep=True)
    effective, trace = apply_correspondence(baseline, bundle["reports"], bundle["root"])
    pd.testing.assert_frame_equal(baseline, unchanged)
    before = baseline.set_index("study_uid")
    after = effective.set_index("study_uid")
    assert after.loc["neighbor", "split"] == "quarantine"
    assert after.loc["gold", "leakage_group"] == after.loc["neighbor", "leakage_group"]
    pd.testing.assert_frame_equal(
        after.loc[["gold", "neighbor", "safe"], before.columns],
        before.loc[["gold", "neighbor", "safe"]],
    )
    assert not after.loc["gold", "report_supervision_allowed"]
    assert after.loc["safe", "report_supervision_allowed"]
    for target in TARGETS:
        assert after.loc["bad", f"{target}__mask"] == 0
        assert after.loc["bad", f"{target}__state"] == "positive"
        assert after.loc["bad", f"{target}__evidence"] == "Shoulder report."
        assert after.loc["bad", target] == before.loc["bad", target]
        assert after.loc["bad", f"{target}__provenance"] == "report_correspondence_masked"
        assert after.loc["gold", f"{target}__mask"] == 1
        assert after.loc["gold", target] == 1
    assert trace["report_masked_studies"] == ["bad"]
    assert trace["official_gold_preserved_studies"] == ["gold"]
    assert trace["newly_masked_targets"] == {target: 1 for target in TARGETS}
    audit = audit_labels(effective, bundle["reviews"])
    assert all(row["gold_review_coverage"] == 0 for row in audit["targets"].values())


def test_excluded_quarantine_neighbor_stays_isolated_and_preserves_gold(bundle: dict) -> None:
    policy = json.loads(bundle["policy"].read_text())
    neighbor = copy.deepcopy(policy["records"][0])
    proof = bundle["root"] / "proof-neighbor.json"
    write(proof, {"case": "neighbor", "study_uid": "neighbor", "status": "verified"})
    neighbor.update(
        case="neighbor",
        study_uid="neighbor",
        source_report_sha256=hashlib.sha256(bundle["reports"]["neighbor"].encode()).hexdigest(),
        verified_technical_proof=reference(proof, bundle["root"]),
    )
    policy["records"].append(neighbor)
    write(bundle["policy"], policy)
    rebind(bundle)
    baseline = build_studies(bundle["train"], bundle["reviews"])
    effective, trace = apply_correspondence(baseline, bundle["reports"], bundle["root"])
    before, after = baseline.set_index("study_uid"), effective.set_index("study_uid")
    assert before.loc["neighbor", "split"] == after.loc["neighbor", "split"] == "quarantine"
    assert after.loc["neighbor", "leakage_group"] == after.loc["gold", "leakage_group"]
    pd.testing.assert_series_equal(after.loc["gold", before.columns], before.loc["gold"])
    preserved = ["split", "leakage_group", "report_group", "Report", *TARGETS]
    preserved.extend(f"{target}__{field}" for target in TARGETS for field in ["state", "evidence"])
    pd.testing.assert_series_equal(
        after.loc["neighbor", preserved], before.loc["neighbor", preserved]
    )
    assert not after.loc["neighbor", "report_supervision_allowed"]
    assert after.loc["neighbor", [f"{target}__mask" for target in TARGETS]].eq(0).all()
    assert (
        after.loc["neighbor", [f"{target}__provenance" for target in TARGETS]]
        .eq("report_correspondence_masked")
        .all()
    )
    assert trace["report_masked_studies"] == ["bad", "neighbor"]
    assert trace["official_gold_preserved_studies"] == ["gold"]
    assert trace["newly_masked_targets"] == {target: 2 for target in TARGETS}


def test_gold_comparison_counts_only_eligible_reports_without_changing_official_labels(
    bundle: dict,
) -> None:
    train, reviews = bundle["train"].copy(deep=True), bundle["reviews"].copy(deep=True)
    # Excluded positive report vs negative gold would create a spurious false positive;
    # the eligible negative report must still count as a true negative.
    train.loc[train.StudyInstanceUID.isin(["gold", "safe"]), list(TARGETS)] = 0
    reviews.loc[reviews.study_uid.eq("safe"), [f"{target}__state" for target in TARGETS]] = (
        "negative"
    )
    baseline = build_studies(train, reviews)
    effective, trace = apply_correspondence(baseline, bundle["reports"], bundle["root"])
    before, after = baseline.set_index("study_uid"), effective.set_index("study_uid")
    pd.testing.assert_frame_equal(
        after.loc[["gold", "safe"], before.columns], before.loc[["gold", "safe"]]
    )
    assert not after.loc["gold", "report_supervision_allowed"]
    assert after.loc["safe", "report_supervision_allowed"]
    assert after.loc[["gold", "safe"], "split"].eq("gold_evaluation").all()
    assert after.loc[["gold", "safe"], list(TARGETS)].eq(0).all().all()
    assert after.loc[["gold", "safe"], [f"{target}__mask" for target in TARGETS]].eq(1).all().all()
    assert (
        after.loc[["gold", "safe"], [f"{target}__provenance" for target in TARGETS]]
        .eq("official_gold")
        .all()
        .all()
    )
    assert trace["official_gold_preserved_studies"] == ["gold"]
    audit = audit_labels(effective, reviews)
    for row in audit["targets"].values():
        assert row["gold_review_coverage"] == 0.5
        assert row["gold_true_negative"] == 1
        assert (
            row["gold_true_positive"]
            == row["gold_false_positive"]
            == row["gold_false_negative"]
            == 0
        )
        assert row["gold_agreement_observed"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_registry",
        "changed_policy",
        "changed_proof",
        "source_report",
        "duplicate_study",
        "unknown_study",
        "partial_mask",
        "numeric_mask",
        "enabled_supervision",
        "wrong_proof_case",
        "wrong_order",
        "duplicate_json",
    ],
)
def test_invalid_or_stale_policy_blocks_release(bundle: dict, mutation: str) -> None:
    policy = json.loads(bundle["policy"].read_text())
    row = policy["records"][0]
    if mutation == "missing_registry":
        bundle["registry"].unlink()
    elif mutation == "changed_policy":
        bundle["policy"].write_text(bundle["policy"].read_text() + " ")
    elif mutation == "changed_proof":
        (bundle["root"] / "proof-bad.json").write_text("{}")
    elif mutation == "source_report":
        bundle["reports"]["bad"] = "Changed report."
    elif mutation == "duplicate_json":
        bundle["registry"].write_text('{"schema_version":1,"schema_version":1}')
    else:
        if mutation == "duplicate_study":
            policy["records"].append(copy.deepcopy(row))
        elif mutation == "unknown_study":
            row["study_uid"] = "unknown"
        elif mutation == "partial_mask":
            row["required_report_loss_mask"][0] = True
        elif mutation == "numeric_mask":
            row["required_report_loss_mask"] = [0] * len(TARGETS)
        elif mutation == "enabled_supervision":
            row["report_derived_training_supervision_allowed"] = True
        elif mutation == "wrong_proof_case":
            p = bundle["root"] / "proof-bad.json"
            write(p, {"case": "other", "study_uid": "bad", "status": "verified"})
            row["verified_technical_proof"] = reference(p, bundle["root"])
        elif mutation == "wrong_order":
            policy["target_order"] = list(reversed(TARGETS))
        write(bundle["policy"], policy)
        rebind(bundle)
    with pytest.raises(ValueError):
        inspect_correspondence(bundle["reports"], bundle["root"])


@pytest.mark.parametrize(
    "failure", [None, "missing_registry", "omitted_policy", "wrapper_bad_report"]
)
def test_cli_enforces_policy_before_export(
    bundle: dict, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    from rsna import adjudications, labels

    root = bundle["root"]
    bundle["train"].to_csv(root / "train.csv", index=False)
    for phase in ["draft", "reviewed"]:
        d = root / phase
        d.mkdir()
        rows = [
            {
                "study_uid": uid,
                "states": ["positive"] * len(TARGETS),
                "evidence": [report] * len(TARGETS),
                "notes": "Synthetic explicitly positive fixture.",
            }
            for uid, report in bundle["reports"].items()
        ]
        (d / f"batch-000-{phase}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    pd.DataFrame({"fixture": [1]}).to_parquet(root / "inventory.parquet")
    write(root / "audit.json", {"fixture": True})
    monkeypatch.setattr(labels, "validate_review_provenance", lambda *args: {})
    monkeypatch.setattr(labels, "validate_dicom_audit", lambda *args: [])
    monkeypatch.setattr(
        adjudications,
        "apply_adjudications",
        lambda frame, *args: (frame, {"registry_sha256": "fixture"}),
    )
    arguments = {
        "train": root / "train.csv",
        "reviews": root / "reviewed",
        "drafts": root / "draft",
        "output": root / "release",
        "dicom-audit": root / "audit.json",
        "inventory": root / "inventory.parquet",
        "source-revision": "a" * 40,
        "review-provenance": root / "unused.json",
        "rubric": root / "unused.md",
        "artifact-root": root,
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsna.labels",
            *[item for key, value in arguments.items() for item in (f"--{key}", str(value))],
        ],
    )
    if failure is not None:
        if failure == "missing_registry":
            bundle["registry"].unlink()
        elif failure == "omitted_policy":
            registry = json.loads(bundle["registry"].read_text())
            registry["policies"] = []
            write(bundle["registry"], registry)
        else:
            paths = install_identity_wrapper(bundle)
            wrapper = json.loads(paths["wrapper"].read_text())
            wrapper["source_report_sha256"] = hashlib.sha256(b"Other report.").hexdigest()
            write(paths["wrapper"], wrapper)
            rebind_wrapper(bundle, paths)
        with pytest.raises(ValueError):
            labels.main()
        assert not (root / "release").exists()
        return
    labels.main()
    out = root / "release"
    studies = pd.read_parquet(out / "studies.parquet").set_index("study_uid")
    assert studies.loc["bad", [f"{t}__mask" for t in TARGETS]].eq(0).all()
    assert studies.loc["gold", [f"{t}__mask" for t in TARGETS]].eq(1).all()
    audit = json.loads((out / "label_audit.json").read_text())
    trace = json.loads((out / "correspondence_trace.json").read_text())
    assert audit["correspondence"] == trace
    assert audit["release_inputs"]["correspondence_registry_sha256"] == file_sha256(
        bundle["registry"]
    )
    annotations = pd.read_parquet(out / "report_annotations.parquet")
    assert annotations[[f"{t}__state" for t in TARGETS]].eq("positive").all().all()
