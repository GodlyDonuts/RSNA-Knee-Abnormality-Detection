"""Final fitting uses all accepted development studies, with no held-out metric."""

import copy
import json

import pandas as pd
import pytest
from test_label_source_overlay import _inputs, _sha

from rsna.competitive.supervision import apply_fitting_supervision
from rsna.competitive.train import TrainConfig, select_development
from scripts.research.build_label_source_overlay import build


def fixture(tmp_path):
    studies, folds, source = _inputs(tmp_path)
    path = tmp_path / "final.json"
    build(studies, folds, 0, source, path, "synthetic", "pinned", "all_development")
    frame = pd.read_parquet(studies)
    train, outer = select_development(frame, pd.read_csv(folds), 0, "all_development")
    return studies, folds, path, train, outer


def apply(values, kind="label_source_final_v1"):
    studies, folds, path, train, outer = values
    return apply_fitting_supervision(
        train,
        outer,
        path,
        {"kind": kind, "sha256": _sha(path), "enabled": True},
        release_sha256=_sha(studies),
        folds_sha256=_sha(folds),
        fold=0,
    )


def test_final_fit_covers_all_folds_without_modifying_release(tmp_path):
    values = fixture(tmp_path)
    original = values[3].copy(deep=True)
    result, receipt = apply(values)
    assert len(result) == 8 and values[4].empty
    assert set(result.fold) == {0, 1, 2, 3}
    assert result.loc[result.study_uid == "uid-0", "ACL"].item() == 0.5
    assert receipt["final_fit_only"] and not receipt["internal_validation_available"]
    pd.testing.assert_frame_equal(values[3], original)
    with pytest.raises(FileExistsError):
        build(
            values[0],
            values[1],
            0,
            tmp_path / "source.csv",
            values[2],
            "synthetic",
            "pinned",
            "all_development",
        )


@pytest.mark.parametrize(
    "change", ["status", "scope", "roster", "protected", "disallowed", "duplicate", "evaluation"]
)
def test_final_fit_rejects_wrong_scope_membership_or_evaluation(tmp_path, change):
    values = list(fixture(tmp_path))
    artifact = json.loads(values[2].read_text())
    if change == "status":
        artifact["status"] = "label_source_experiment_only"
    elif change == "scope":
        artifact["fit_scope"] = "outer_fold"
    elif change == "roster":
        artifact["fitting_uid_sha256"] = "a" * 64
    elif change == "protected":
        values[3].loc[0, "split"] = "gold"
    elif change == "disallowed":
        values[3].loc[0, "report_supervision_allowed"] = False
    elif change == "duplicate":
        artifact["studies"][-1] = copy.deepcopy(artifact["studies"][0])
    else:
        values[4] = values[3].iloc[[0]].copy()
    values[2].write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        apply(values)


def test_old_outer_policy_cannot_consume_final_fit_artifact(tmp_path):
    with pytest.raises(ValueError, match="scope"):
        apply(fixture(tmp_path), "label_source_v1")


def test_config_requires_explicit_final_fit_policy_and_scope():
    policy = {"kind": "label_source_final_v1", "sha256": "a" * 64, "enabled": True}
    TrainConfig(
        fit_scope="all_development", fitting_supervision=policy, diagnostic_study_limit=8
    ).validate()
    with pytest.raises(ValueError):
        TrainConfig(fitting_supervision=policy).validate()
    policy["kind"] = "label_source_v1"
    with pytest.raises(ValueError):
        TrainConfig(fit_scope="all_development", fitting_supervision=policy).validate()


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_actual_trainer_final_source_full_stop_resume_state_equality(
    tmp_path, monkeypatch, precision
):
    import hashlib
    from dataclasses import replace

    from test_competitive_training import (
        test_training_mid_epoch_resume_matches_every_checkpoint_value,
    )

    from rsna.competitive import train as trainer
    from rsna.competitive.supervision import roster_sha
    from rsna.schema import TARGETS

    real_run = trainer._run_locked
    path = tmp_path / "final-source.json"

    def execute(args, config, *positional, **keywords):
        if not path.exists():
            frame, release = trainer.load_verified_release(args.studies, args.artifact_root)
            frame["report_supervision_allowed"] = True
            release["manifest_sha256"] = "a" * 64
            monkeypatch.setattr(
                trainer, "load_verified_release", lambda *a: (frame.copy(), release)
            )
            folds = pd.DataFrame({"study_uid": frame.study_uid, "fold": [0] * 5 + [1] * 9})
            folds.to_csv(tmp_path / "fold.csv", index=False)
            pool, outer = select_development(frame, folds, 0, "all_development")
            artifact = {
                "schema_version": 1,
                "status": "label_source_final_fit_only",
                "fit_scope": "all_development",
                "fold": 0,
                "release_sha256": "a" * 64,
                "folds_sha256": _sha(tmp_path / "fold.csv"),
                "source": {"csv_sha256": "b" * 64},
                "fitting_uid_sha256": roster_sha(pool),
                "outer_uid_sha256": roster_sha(outer),
                "study_count": len(pool),
                "studies": [
                    {
                        "study_uid": row.study_uid,
                        "labels": {t: 0.2 + 0.6 * float(row[t]) for t in TARGETS},
                        "mask": {t: int(row[t + "__mask"]) for t in TARGETS},
                    }
                    for _, row in pool.iterrows()
                ],
            }
            path.write_text(json.dumps(artifact))
        policy = {
            "kind": "label_source_final_v1",
            "enabled": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        args.fitting_supervision = path
        return real_run(args, replace(config, fitting_supervision=policy), *positional, **keywords)

    monkeypatch.setattr(trainer, "_run_locked", execute)
    test_training_mid_epoch_resume_matches_every_checkpoint_value(
        tmp_path, monkeypatch, "all_development", "none", None, None, precision
    )
