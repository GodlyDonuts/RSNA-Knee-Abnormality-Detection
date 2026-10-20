import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "matched_collection", Path(__file__).with_name("collect.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rows = [
        dict(
            step=i + 1,
            studies=8 if i % 2 == 0 else 4,
            encoder_updated=i >= 3,
            LR=[3e-5] * 2,
            **dict.fromkeys(
                (
                    "exam_loss",
                    "density_loss",
                    "gradient_norm",
                    "encoder_gradient_norm",
                    "head_gradient_norm",
                ),
                float(i % 2 + 1),
            ),
        )
        for i in range(200)
    ]
    return dict(
        status=m.CASES["generative"]["status"],
        actual_optimizer_updates=600,
        fresh_initialization_equal=True,
        G100_full_stop_resume_complete_state_and_prediction_bytes_equal=True,
        clinical_or_held_out_accuracy_claim=False,
        roles=[dict(role=r, real_optimizer_history=copy.deepcopy(rows)) for r in ("C100", "G100")],
    )


def test_actual_tail_weighted_means_and_matched_windows():
    out = m.summarize(fixture(), m.CASES["generative"])
    for role in out["roles"].values():
        w = role["windows"]["last20"]
        assert w["update_means"]["exam_loss"] == 1.5
        assert w["study_weighted_means"]["exam_loss"] == pytest.approx(4 / 3)
        assert w["clipped_fraction"] == 0.5


@pytest.mark.parametrize("change", ["missing", "nonfinite", "wrong_order"])
def test_no_summary_from_incomplete_or_invalid_history(change):
    value = fixture()
    rows = value["roles"][1]["real_optimizer_history"]
    if change == "missing":
        rows.pop()
    if change == "nonfinite":
        rows[30]["exam_loss"] = float("inf")
    if change == "wrong_order":
        rows.reverse()
    with pytest.raises(AssertionError):
        m.summarize(value, m.CASES["generative"])


def test_scheduler_never_uses_another_job_or_step():
    assert (
        m.scheduler("986011|RUNNING|0:0|evc27\n986011.batch|RUNNING|0:0|evc27\n", "986011")["state"]
        == "RUNNING"
    )
    with pytest.raises(ValueError):
        m.scheduler("984784|COMPLETED|0:0|evc27\n", "986011")
