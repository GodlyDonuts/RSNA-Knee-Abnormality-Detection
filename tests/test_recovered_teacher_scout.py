"""Recovered predictions cannot stand in for failed resource or clinical acceptance."""

import copy
import hashlib
import json

import numpy as np
import pytest

from rsna.competitive import recovered_teacher_scout as scout


@pytest.fixture
def example(monkeypatch):
    uids = [f"case{i:04d}" for i in range(3887)]
    h = hashlib.sha256(json.dumps(uids, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(scout, "POPULATION_SHA", h)
    counts = {u: 1 + i % 9 for i, u in enumerate(uids)}
    costs = sorted((counts[u], u) for u in uids)
    p8 = np.linspace(0, 3886, 8).round().astype(int).tolist()
    positions = sorted(set(p8 + np.linspace(0, 3886, 26).round().astype(int).tolist()))
    selected, old8 = [costs[i][1] for i in positions], [costs[i][1] for i in p8]
    selection = dict(
        selected_uids=selected,
        positions=positions,
        old8_selected_uids=old8,
        population_sha256=h,
        series_counts=[costs[i][0] for i in positions],
    )
    old_selection = dict(
        ordered_fitting_uids=uids, selected_uids=old8, series_counts=list(counts.items())
    )
    pixels = {u: dict(n=1, img_sha256="1" * 64, meta_sha256="2" * 64) for u in selected}
    old_rows = [dict(uid=u, windows=1, images_sha256="1" * 64, meta_sha256="2" * 64) for u in old8]
    logits = np.tile(np.linspace(-2, 2, 32)[:, None], (1, 12))
    probability = 1 / (1 + np.exp(-logits))
    old_logits = {u: logits[selected.index(u)][None].tolist() for u in old8}
    receipt = dict(
        status="PASS",
        studies=32,
        scored=32,
        fallback={},
        timed_out=False,
        amp="fp32",
        models=["E_prior_s42_fall.pt"],
        spatial_models=1,
        stats={"windows": 32},
    )

    def summarize(a, b):
        return {"mean_gap": float(np.abs(a - b).mean())}

    return [
        selection,
        pixels,
        probability,
        logits,
        old_selection,
        old_rows,
        old_logits,
        uids,
        np.zeros((3887, 2, 12), np.float32),
        receipt,
        summarize,
    ]


def test_complete_recovery_keeps_original_failure_and_missing_resource_evidence(example):
    result = scout.compare(*example)
    assert result["maximum_original8_probability_error"] == 0
    assert result["studies"] == result["windows"] == 32
    assert result["original_terminal_failure_preserved"] is True
    assert result["original_RSS_time_and_total_output_cap_accepted"] is False
    assert result["full_cohort_teacher_generation_or_training_accepted"] is False
    assert result["clinical_accuracy_claim"] is False


@pytest.mark.parametrize("change", ["membership", "inputs", "fallback", "nonfinite", "precision"])
def test_mixed_or_unqualified_outputs_fail(example, change):
    v = copy.deepcopy(example)
    if change == "membership":
        v[0]["selected_uids"] = v[0]["selected_uids"][::-1]
    elif change == "inputs":
        v[1][v[0]["old8_selected_uids"][0]]["img_sha256"] = "3" * 64
    elif change == "fallback":
        v[9]["fallback"] = {"decode": 1}
    elif change == "nonfinite":
        v[3][0, 0] = np.nan
    else:
        i = v[0]["selected_uids"].index(v[0]["old8_selected_uids"][0])
        v[3][i] += 0.01
        v[2][i] = 1 / (1 + np.exp(-v[3][i]))
    with pytest.raises(ValueError):
        scout.compare(*v)
