# Blended per-target label source (v2 + public severity)

## Status

Artifact built, sha-bound, loader-verified — queued behind the GPU slot like
the other paired arms.

## What this is

A `label_source_v1` overlay assembled cell-by-cell from two bound sources:

- **v2 review states** on 8 targets — ACL, MCL, Medial/Lateral Meniscus,
  PF OA, Synovitis, Contusion, Fracture — mapped with the baseline binary-mask
  semantics: `positive`→1.0, `negative`→0.0, `uncertain`/`not_mentioned`→masked.
- **Public `llm_labels_v4_blend` probabilities** on the 4 severity-qualified
  targets where its recall measured better — Medial OA, Lateral OA, Effusion,
  Baker's — passed through as soft [0,1] supervision, mask 1.

Artifacts (all three outer folds, sha-bound and loader-verified):

- `blended_v2plus_public.fold0.overlay.json` — sha256
  `6a93fe7860ece65cc1beb0d690620198b331bca89039f0be34e01605f49c193d`,
  2,247 fitting studies, 19,186/26,964 supervised cells, 9,405 cells
  differing from the v2 baseline.
- `blended_v2plus_public.fold1.overlay.json` — sha256 `71cc8f91…`,
  2,278 fitting studies.
- `blended_v2plus_public.fold2.overlay.json` — sha256 `64ee2cc5…`,
  2,281 fitting studies.

## Why

`GOLD-ARBITRATION.md` measured each source against the 58 expert labels:
v2 wins on meniscus/synovitis/ligament precision+recall; public wins on
severity-qualified targets (Effusion recall 1.00 vs v2's 0.41, Medial OA 0.93
vs 0.50, Baker's 0.92 vs 0.60) — v2's protocol severity bar is set above
gold's on exactly those four targets. Neither source is uniformly better, so
a per-target assignment is the natural third arm beside the straight
public-vs-v2 contrast.

## IMPORTANT: gold-informed selection caveat

The target assignment was informed by gold-58 agreement measurements. This is
an **upper-bound probe**, not a shippable label policy: its selection used
evaluation-only data, so its outcome cannot be fairly arbitrated on gold or
any gold-derived holdout. The artifact carries `label_source_experiment_only`
status like the other arms. Its honest use is measuring the headroom a
per-target best-source policy could provide — if the arm clears both straight
arms by a margin that survives the disjoint development set, that justifies a
properly adjudicated v2.1 severity re-review (not gold-derived labels).

## Protocol

Identical to `label-source-ablation-v1`: same release, folds, outer fold,
recipe — only the label table changes. Three-way comparison vs the two
straight arms (v2 baseline, public overlay) on the disjoint development set
with paired intervals.

## Prediction (from the arbitration table)

If the semantics hypothesis is right, this arm should beat the public-only
arm on meniscus/synovitis AUCs and match it on effusion/OA — i.e., strictly
dominate where the mechanisms differ. A null result vs the better of the two
straight arms means per-cell label mixing adds noise without signal.
