import csv
import io
import zipfile

import pandas as pd
import pytest

from rsna.schema import TARGETS
from scripts.research.audit_public_report_labels import CAP, compare, csv_member, fitting_cohort


def cohort():
    rows = []
    for uid, group in [("a", "g1"), ("b", "g1"), ("c", "g2")]:
        row = {"study_uid": uid, "leakage_group": group, "Report": "Synthetic report"}
        for target in TARGETS:
            row.update({target: 0, f"{target}__mask": 0, f"{target}__state": "uncertain"})
            row[f"{target}__evidence"] = ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("study_uid")


def source(uids=("a", "b", "c"), change=None):
    columns = ["StudyInstanceUID"] + [
        c for t in TARGETS for c in (t, f"{t}__conf", f"{t}__verdict")
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for uid in uids:
        row = {"StudyInstanceUID": uid}
        for target in TARGETS:
            row.update({target: ".94", f"{target}__conf": ".95", f"{target}__verdict": "YES"})
        if change:
            change(uid, row)
        writer.writerow(row)
    return output.getvalue().encode()


def test_excluded_targets_never_interpreted():
    def poison(uid, row):
        if uid == "protected":
            row.update({"ACL": "invalid", "ACL__verdict": "invalid"})

    result, _ = compare(source(("protected", "a", "b", "c"), poison), cohort())
    assert result["fitting_studies"] == 3
    assert result["targets"]["ACL"]["state_verdict_counts"] == {"masked:YES": 3}


@pytest.mark.parametrize("uids", [("a", "a", "b", "c"), ("a", "b")])
def test_exact_membership_rejected(uids):
    with pytest.raises(ValueError):
        compare(source(uids), cohort())


@pytest.mark.parametrize(
    "field,value", [("ACL", "nan"), ("ACL__conf", "1.1"), ("ACL__verdict", "?")]
)
def test_invalid_fitting_targets_rejected(field, value):
    with pytest.raises(ValueError, match="invalid fitting"):
        compare(source(change=lambda uid, row: row.update({field: value})), cohort())


def test_review_deterministic_bounded_by_components_without_label_promotion():
    original = cohort()
    before = original.copy(deep=True)
    result, review = compare(source(), original)
    _, reversed_review = compare(source(("c", "b", "a")), original)
    assert review == reversed_review
    assert len(review) == 8  # four target categories, two distinct components each
    assert not result["accepted_labels_changed"]
    assert not result["clinical_accuracy_claimed"]
    pd.testing.assert_frame_equal(before, original)
    for target in ("MCL", "Effusion", "Baker's", "Fracture"):
        assert len({r["leakage_group"] for r in review if r["target"] == target}) == 2


def test_archive_bounds_and_names(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("report_labels_v2.csv", b"x" * (CAP + 1))
    with pytest.raises(ValueError, match="expanded archive"):
        csv_member(archive, "report_labels_v2.csv")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../source.csv", b"x")
    with pytest.raises(ValueError, match="unsafe"):
        csv_member(archive, "../source.csv")


def test_partition_filter_and_original_fold_component_admission(tmp_path):
    frame = cohort().reset_index()
    frame["split"] = "train"
    frame["report_supervision_allowed"] = True
    frame["ACL__mask"] = 1
    protected = frame.iloc[[0]].assign(study_uid="protected", leakage_group="gp", split="gold")
    studies = tmp_path / "studies.parquet"
    pd.concat([frame, protected]).to_parquet(studies)
    membership = frame[["study_uid", "leakage_group"]].assign(fold=[1, 1, 0])
    folds = tmp_path / "fold.csv"
    membership.to_csv(folds, index=False)
    assert list(fitting_cohort(studies, folds, 0).index) == ["a", "b"]
    membership.loc[1, "fold"] = 2
    membership.to_csv(folds, index=False)
    with pytest.raises(ValueError, match="component crosses"):
        fitting_cohort(studies, folds, 0)
