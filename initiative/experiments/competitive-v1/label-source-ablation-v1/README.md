# Label-source ablation v1 — public LLM labels vs v2 reviewed labels

## Question

Does the v2 human-reviewed label table (4,407 reports, 52,884
evidence-bound decisions) beat the public `llm_labels_v4_blend` table that
trained the copied 0.936/0.942 ensembles — same model, same folds, same
evaluation?

Three independent external sources say label quality, not model capacity, is
the field's ceiling (rank-18 write-up: extractor ceiling ~0.83 per-class
agreement; pilkwang: "the ceiling ... sits in the supervision"; public DINOv2
small-vs-base ablation: capacity doesn't raise the ceiling at 224px). This is
the single most decision-relevant experiment in the program.

## Design — strictly paired

| | control | candidate |
|---|---|---|
| release | `initiative/releases/report-supervision-v1` (unchanged) | same |
| folds | `initiative/experiments/competitive-v1/folds/fold.csv` | same |
| outer fold | fold 0, 1,156 studies — identical evaluation frame | same |
| fitting | 2,247 studies | 2,247 studies (same roster) |
| fitting labels | v2 reviewed values + masks | `llm_labels_v4_blend` values, mask=1 everywhere |
| model/recipe | canonical competitive recipe, unchanged | identical |

The swap is admitted through the existing `fitting_supervision` mechanism with
policy kind `label_source_v1`; the overlay below covers every fitting study so
no cell silently keeps its original supervision. Evaluation masks and the
release receipt are never touched. `enabled: false` runs the control arm
through the same code path (applied_cells=0).

## Artifacts

- `llm_labels_v4_blend.fold0.overlay.json` — bound overlay for fold 0.
  sha256 `2feb33c22e7b7eef46640cbc9b3ebd62f95047a376eda8f053e6f6fb71805dfe`
  (2,247 fitting studies, 26,954/26,964 cells swapped on admission; the 10
  unchanged cells are coincidentally identical values).
- `initiative/public-labels/` — the four verified public label CSVs with
  `provenance.json` (source `Shio-Koube/knee3@373f4e1d`, per-file sha256).
- `rsna/competitive/label_source.py` — overlay admission/loader.
- `scripts/research/build_label_source_overlay.py` — overlay builder; rerun
  with `--fold N` for additional folds or `--source` a different public table.

## Run (on GPU host)

```bash
# candidate arm
python -m rsna.competitive.train ... \
  --fitting-supervision initiative/experiments/competitive-v1/label-source-ablation-v1/llm_labels_v4_blend.fold0.overlay.json \
  # config: fitting_supervision = {"kind": "label_source_v1",
  #   "sha256": "2feb33c22e7b7eef46640cbc9b3ebd62f95047a376eda8f053e6f6fb71805dfe",
  #   "enabled": true}
# control arm: identical run with "enabled": false
```

Decision rule: candidate beats control only if the outer-fold macro-AUC delta
exceeds the paired bootstrap interval that already governs this program —
and any superiority claim needs a public submission as arbiter, per the
dual-track validation doctrine in `docs/research-frontier-20261006.md`.

## Follow-ons queued by the same mechanism

- `train_labels.csv` (same dataset, alternate soft table) — second arm, same
  fold.
- Graded/ordinal v2 targets (uncertain cells at intermediate values instead
  of masked) — tests whether v2's uncertainty grades are exploitable, not
  just safer.
- frankenstein tables need clipping to [0,1] before admission (max 1.2745).
