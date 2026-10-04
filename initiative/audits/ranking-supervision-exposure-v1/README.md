# Ranking needs actual comparison opportunities

Before implementing the next proposed auxiliary ranking objective, replayed the
exact original fitting metadata and sampler at seeds20260922/20261004. This is
a **CPU metadata diagnostic**, not training, clinical truth or an accuracy gain.
No live numerical source or either pending seed2 job was changed. Original
accepted labels/masks/folds, base weights and protected evaluation remain intact.

## Measured constraint

The actual trainer forwards **one study at a time**, then accumulates eight
microbatches per optimizer update. A same-target pairwise loss added inside the
current forward/backward loop would therefore have **zero positive/negative
study pairs for every target**. Gradient accumulation does not automatically
construct comparisons across those separate forward passes.

Even if comparisons were collected over the eight-study effective batch, the
original strict labels supply cross-component positive/negative pairs in only
15/843 updates for MCL(1.78%),5–7/843 for fracture(0.59–0.83%),17–18/843 for
Baker's(2.02–2.14%) and16–22/843 for synovitis(1.90–2.61%). Synovitis is limited
by observed negatives, not positives. Medial meniscus is much denser:820–826/843.
See [all twelve targets](all12-support.md) and `comparison-support-v2.json`.
Current strict BCE still has its ordinary per-study gradients; this constraint
applies specifically to a future study-pair ranking objective.

These are original **control** labels. Effusion has16 observed fitting positives
here; the separately reviewed31-cell candidate is not replayed by this diagnostic.
The calculation counts binary clinical-target polarity opportunities, not yet-
admitted descriptive concepts or severity levels. Sparse binary support does not
prove that an independently reviewed concept/ranking task would have no support.

## Reproducible boundary

`scripts/research/audit_ranking_exposure.py` checks the completed control run,
whole release SHA, canonical full folds, original sampler SHA and both complete
fitting/outer roster hashes. It projects only training metadata/masks for cohort
membership, then reads target values for the2247 fitting UIDs only. Outer target
values, protected confirmation/gold, report text and MRI bytes are not decoded.
Exact sampler keys preserve once-per-epoch real-study exposure and masked tail
padding. Same-leakage-component pairs are excluded from the comparison counts.
No individual case records or report decisions appear in the aggregate output.

The first formatted replay reproduced the original result bytes exactly; see
`replay-acceptance.json`. The enhanced current replay preserves all original
counts and verifies all843 actual GPU update observed-cell counters, including
eight original/resumed overlaps. CPU Torch2.14.0 and original2.14.0+cu130 share
the exact same-version CPU sampler. See `replay-acceptance-v2.json`; old outputs
remain immutable. Sixteen tests check forward/effective-batch distinction,
padding, masking, component exclusion, deterministic replay, fitting-only target
projection, original log overlap/counters and rejection of altered original
source/release/roster/fold bindings.
Software acceptance does not establish that ranking improves MRI classification.

```bash
PYTHONPATH=. python -B scripts/research/audit_ranking_exposure.py \
  --studies initiative/releases/report-supervision-v1/studies.parquet \
  --folds initiative/experiments/competitive-v1/folds/fold.csv \
  --original-run initiative/audits/effusion-supervision-screen-final-v1/control/run.json \
  --output artifacts/ranking-exposure-new/comparison-support.json
```

## Research decision

Reject a naive within-forward ranking add-on and an arbitrary increase in its
weight. The next mechanism first needs independently admitted fitting-only
concepts/partial orders, then a measured comparison schedule with the same MRI
encoding budget for control/candidate. Unknown findings stay unknown; do not
manufacture negatives or duplicate rare positives without an explicit sampling
and weighting hypothesis. Report comparison density per target and gradient
strength before reserving GPU time. Existing strict BCE remains the anchor.

No ranking objective, concept annotation, GPU pilot or accuracy benefit is
implemented/admitted by this finding. The queued31-cell supervision replication
continues independently. The result identifies an execution constraint for the
next idea, not a confirmed explanation of Kaggle scores.
