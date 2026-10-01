"""Verify the best available completed reference for the live native384 run.

No native outcome or strict causal comparison is invented. This reads original
bindings and separately verified fixed-final reference evidence only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
NATIVE = ROOT / "initiative/audits/native384-production-preparation-v1/gpu-initial-progress"
SAMPLING = ROOT / "initiative/audits/sampling-fixed-final-v1"
REFERENCE = SAMPLING / "uniform"
C9 = "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
SAMPLING_SOURCE = "7ac7a9779d3c8bfd756a44eceb6414e9bb763b85"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    native, uniform = read(NATIVE / "run.json"), read(REFERENCE / "run.json")
    n, u = native["binding"], uniform["binding"]
    assert {k for k in n.keys() | u.keys() if n.get(k) != u.get(k)} == {
        "cache_side",
        "config",
        "code_sha256",
    }
    assert {
        k
        for k in n["config"].keys() | u["config"].keys()
        if n["config"].get(k) != u["config"].get(k)
    } == {"side"}
    assert (n["config"]["side"], u["config"]["side"]) == (384, 224)
    assert (n["cache_side"], u["cache_side"]) == (384, 256)
    assert n["config"]["slice_sampling"] == u["config"]["slice_sampling"] == "uniform"
    assert n["config"]["augment"] is u["config"]["augment"] is False
    assert n["config"]["encoder"] == u["config"]["encoder"]
    assert native["training_studies"] == uniform["training_studies"] == 2247
    assert native["outer_studies"] == uniform["outer_studies"] == 1156
    assert native["held_out_confirmation_and_gold_used"] is False
    assert uniform["held_out_confirmation_and_gold_used"] is False
    changed = sorted(k for k in n["code_sha256"] if n["code_sha256"][k] != u["code_sha256"][k])
    assert changed == ["competitive/data.py", "competitive/infer.py", "competitive/train.py"]
    for source, binding in ((C9, n), (SAMPLING_SOURCE, u)):
        for name, expected in binding["code_sha256"].items():
            raw = subprocess.run(
                ["git", "show", f"{source}:rsna/{name}"],
                cwd=ROOT,
                capture_output=True,
                check=True,
            ).stdout
            assert hashlib.sha256(raw).hexdigest() == expected
    diff = subprocess.run(
        ["git", "diff", C9, SAMPLING_SOURCE, "--", *("rsna/" + k for k in changed)],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    diff_path = AUDIT / "original-policy-source.diff"
    if diff_path.exists():
        assert diff_path.read_bytes() == diff
    else:
        diff_path.write_bytes(diff)
    native_inputs = read(NATIVE / "native-input-provenance.json")
    uniform_inputs = read(REFERENCE / "sampling-input-provenance.json")
    assert native_inputs["strict_deterministic_algorithms"] is True
    assert uniform_inputs["strict_determinism"] is True
    assert (
        native_inputs["environment_archive_sha256"]
        == uniform_inputs["environment"]["archive_sha256"]
    )
    assert native_inputs["scientific_source_git"] == C9
    assert uniform_inputs["source_git"] == SAMPLING_SOURCE
    result, comparison = read(REFERENCE / "result.json"), read(SAMPLING / "comparison.json")
    assert result["status"] == uniform["status"] == "completed"
    assert result["checkpoint_selection"] == "fixed_final_epoch_no_outer_selection"
    assert result["optimizer_steps"] == 2810
    assert digest(REFERENCE / "outer_predictions.csv") == result["predictions_sha256"]
    assert result["metrics"]["macro_auc"] == comparison["uniform_macro_auc"]
    snapshot = read(REFERENCE / "epoch-checkpoints/epoch-010.json")
    assert snapshot["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert snapshot["epoch"] == 10 and snapshot["optimizer_steps"] == 2810
    files = [
        NATIVE / "run.json",
        NATIVE / "native-input-provenance.json",
        REFERENCE / "run.json",
        REFERENCE / "result.json",
        REFERENCE / "sampling-input-provenance.json",
        REFERENCE / "outer_predictions.csv",
        REFERENCE / "epoch-checkpoints/epoch-010.json",
        SAMPLING / "comparison.json",
        SAMPLING / "recompute.py",
        diff_path,
    ]
    record = {
        "schema_version": 1,
        "status": "best_available_fixed_final_reference_verified_with_comparison_limitations",
        "native_job": 867367,
        "reference_job": 866719,
        "native_source_git": C9,
        "reference_source_git": SAMPLING_SOURCE,
        "native_model_input_side": 384,
        "reference_model_input_side": 224,
        "native_cache_side": 384,
        "reference_cache_side": 256,
        "same_accepted_release_masks_folds_and_train_outer_rosters": True,
        "same_full_encoder_provenance_and_all_other_training_settings": True,
        "both_actual_launch_provenances_require_strict_determinism": True,
        "same_original_environment_archive": True,
        "reference_fixed_final_internal_macro_auc": comparison["uniform_macro_auc"],
        "native_fixed_final_internal_macro_auc": None,
        "native_score_or_terminal_claimed": False,
        "changed_source_modules": changed,
        "source_diff_review": "Added stratified policy and evaluation remapping; both actual roles use uniform, whose remapping is identity. No changed model/adapter/loss/optimizer/sampler implementation.",
        "direct_native_initial_learned_tensor_equality_receipt_collected": False,
        "strict_causal_resolution_only_trial_verified": False,
        "remaining_limitations": [
            "Separately reconstructed cache; prior reconstructed256 engineering difference was bounded, not proven clinically irrelevant.",
            "Native initial learned-state hashes have not been directly compared with the reference's actual initialization receipt.",
            "One seed/fold and report-derived labels; no gold, confirmation or Kaggle outcome for native384.",
        ],
        "next_result_policy": "Verify original native fixed-final terminal/bindings/checkpoint/prediction bytes; compare all12 targets paired by whole leakage component with this reference, keep undefined draws and limitations. Repeat a promising effect on a separately matched seed/fold before promotion.",
        "protected_confirmation_and_gold_used": False,
        "files": {
            p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in files
        },
    }
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = AUDIT / "reference-acceptance.json"
    if target.exists():
        assert target.read_text() == text
    else:
        target.write_text(text)
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "status",
                    "reference_fixed_final_internal_macro_auc",
                    "native_fixed_final_internal_macro_auc",
                    "strict_causal_resolution_only_trial_verified",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
