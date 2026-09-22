# Frozen competitive development folds

These three experiment folds partition only the accepted `train` split: **3,904 studies in 3,761 conservative leakage components**. The original data release, masks and accepted splits remain unchanged. The 441 weak-validation studies, 58 historically exposed gold studies and four quarantined studies are absent. Their labels do not guide this overlay. Full release integrity was validated through `load_verified_release` before train selection and writing; the resulting overlay was then loaded and replay-validated.

The fixed seed is `20260922`. `fold.csv` is bound by `folds.json`, which records the accepted source manifest hash, release binding, deterministic membership hash, training observed-label/mask fingerprint, algorithm version, and all per-fold target study/component counts. Masked values have no role in membership or metrics. Every train row stays represented, including fully masked rows; training and `observed_train` evaluation may select rows with at least one observed target later.

| Fold | Studies | Components | Fracture-positive components | MCL-positive components | Baker-positive components | Synovitis-negative components |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1,301 | 1,247 | 2 | 2 | 3 | 5 |
| 1 | 1,302 | 1,259 | 1 | 3 | 3 | 5 |
| 2 | 1,301 | 1,255 | 2 | 2 | 3 | 5 |

All twelve targets have at least one observed positive and negative in every fold. This enables AUC calculation; it does not establish precise estimates for rare targets. Component counts are conservative leakage units, not verified patient counts. A component can contain positive, negative and masked rows, so those component counts need not sum to the total.

Membership SHA-256: `8699248177042dc2e95b1e1947e25cbe5bc39e42c43ee4919398afc219665e07`.
CSV SHA-256: `fe590e5072a0d82abeb377598aeef0021bbe77fe4529a19b9826a7d40d322050`.
Release binding SHA-256: `e2ecb283915a5b1810bfc9a2f6e1dda1f1d0d762f91cf2487140af3d05cb6c35`.

## Use

Always call `rsna.release.load_verified_release(studies_path, artifact_root)` on the complete manifest first. Then `rsna.competitive.folds.load_folds(path, manifest, binding)` accepts this directory, `fold.csv`, or `folds.json` and validates exact train membership, group isolation, source binding, class counts, masks and deterministic assignment. Never edit a frozen overlay to accommodate a score. Creating a new overlay requires an explicit new output directory and declared protocol change.

`evaluate_oof(manifest, overlay, predictions, seed=..., membership="observed_train")` accepts concatenated predictions from all three outer folds. Prediction columns are `study_uid`, `fold`, `seed`, then the twelve named target probabilities in canonical target order. Each UID must appear exactly once; the seed must match all rows. `observed_train` requires every train study with any observed target. `all_train` instead requires every accepted train study. Missing folds, diagnostic subsets, duplicates, reserved UIDs, incorrect fold/group metadata, mixed seeds and nonfinite/out-of-range scores are rejected. Use a separate call for each model seed.

The primary target AUC is the positive-negative-pair-weighted mean of within-fold AUCs. The full twelve-target macro exists only when all twelve target aggregates exist. Pooled OOF is explicitly secondary because fold score-scale differences can create misleading cross-fold rankings. No held-out calibration or normalization is performed.

`paired_group_bootstrap` accepts two complete, same-seed prediction tables and resamples leakage components with replacement within folds. The same component multiplicities weight both models; components, folds and seeds are never mixed. It reports percentile intervals for A-minus-B, counts undefined draws rather than replacing them with 0.5, and hashes the shared draws. Intervals describe uncertainty conditional on the fitted models and on defined resamples, not the entire search procedure. Seed variability must be reported separately. `leave_one_positive_component_out` provides a no-refit sensitivity check for fracture, MCL and Baker by default.

## Evidence and limits

The focused synthetic suite has 30 passing cases, including row-order determinism, rare support, leakage and membership tampering, masked-value invariance, a score-scale pooling trap, paired bootstrap and undefined resamples. Ruff and mypy cover the new modules. The real full-release validation and fold construction completed locally. No competitive predictions, gold/confirmation evaluation, model selection, weights or GPU training were performed for this fold/evaluation task.

The stratifier is a deterministic multilabel heuristic with safe coverage repair, not a global optimality proof. It balances positive and negative component presence, then study size, and fails rather than silently claiming complete support when it cannot retain feasible class coverage. Frozen code and test hashes belong to the enclosing Git experiment milestone.

Author/executor: `/root/competition_contract`. CLI: `python -m rsna.competitive.folds --studies initiative/releases/report-supervision-v1/studies.parquet --artifact-root . --output initiative/experiments/competitive-v1/folds --seed 20260922 --folds 3` (already completed; rerunning this writer refuses to replace existing files).


The evaluation CLI requires explicitly named prediction files (no glob/latest-run discovery):

```sh
python -m rsna.competitive.evaluation \
  --studies initiative/releases/report-supervision-v1/studies.parquet \
  --artifact-root . --folds initiative/experiments/competitive-v1/folds \
  --predictions /experiment/fold0/outer_predictions.csv /experiment/fold1/outer_predictions.csv /experiment/fold2/outer_predictions.csv \
  --seed 20260922 --membership observed_train --output /experiment/evaluation-new
```

Substitute actual completed experiment paths. This writes a sorted combined prediction table, primary/secondary metrics, and `evaluation.json` with exact portable input, fold, release and code hashes plus dependency versions. It refuses to overwrite an evaluation and does not select models or checkpoints. Diagnostic subsets fail the complete OOF membership guard. The CLI was exercised only with synthetic predictions for this implementation task.
