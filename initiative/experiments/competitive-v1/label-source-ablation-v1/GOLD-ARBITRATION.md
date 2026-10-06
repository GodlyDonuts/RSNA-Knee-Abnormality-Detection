# Gold-58 arbitration: public vs v2 label semantics

Direct measurement: for the 58 expert-labeled studies, compare each label
source's hard call against gold, per target. v2's *stored* labels equal gold
by construction (0/696 differing cells — gold rows keep original values), so
v2 is evaluated on its **`__state`-derived** calls: `positive`→1,
`negative`→0, `uncertain`/`not_mentioned`→abstain. Public is thresholded at
0.5. Gold is evaluation-only — this table informs review priorities, it does
not arbitrate recipe selection (N=58, per-target positives as low as 1–24).

## Results

| Target | gold pos | v2 recall | v2 prec | public recall | public prec | v2 abstained |
|--------|---------:|----------:|--------:|--------------:|------------:|-------------:|
| ACL | 24 | 0.93 | 1.00 | 0.96 | 0.79 | 19 |
| MCL | 9 | 1.00 | 1.00 | 1.00 | 0.56 | 20 |
| Medial Meniscus | 26 | 0.96 | 0.88 | 0.88 | 0.82 | 8 |
| Lateral Meniscus | 23 | 0.89 | 0.89 | 0.78 | 0.82 | 15 |
| Medial OA | 15 | **0.50** | 1.00 | **0.93** | 0.67 | 35 |
| Lateral OA | 11 | **0.67** | 0.67 | **0.82** | 0.53 | 39 |
| PF OA | 21 | 1.00 | 1.00 | 0.81 | 0.68 | 38 |
| Effusion | 35 | **0.41** | 0.88 | **1.00** | 0.69 | 23 |
| Synovitis | 27 | **1.00** | 0.73 | **0.48** | 0.76 | 43 |
| Baker's | 12 | **0.60** | 1.00 | **0.92** | 0.69 | 43 |
| Contusion | 19 | 0.90 | 0.69 | 0.84 | 0.57 | 39 |
| Fracture | 18 | n/a (6 decided) | — | 0.78 | 0.67 | 52 |

On the 322 cells v2 decided: v2 accuracy 0.894, public accuracy (same cells)
0.863. On all 696 cells: public accuracy 0.818.

## What this revises

The tier-0 and OA triage docs framed v2's semantics as systematically better
calibrated to the protocol. **Gold arbitration says the truth is
target-family-split, not uniform:**

- **v2 wins decisively on meniscus + synovitis + ligament precision.**
  Its morphology-inclusive meniscal semantics beat public's tear-only boundary
  on both recall AND precision (0.96/0.89 vs 0.88/0.78 recall). The tier-0
  adjudication stands: public systematically misses degenerative meniscal
  findings that gold counts. v2's positives are the most trustworthy calls in
  either source — precision 0.88–1.00 on decided cells almost everywhere.

- **public wins on severity-qualified targets — the "over-call" was
  correct.** Gold counts *mild* effusion (e.g. "leve derrame articular"),
  mild osteophyte/cartilage findings, and small Baker's cysts as positive.
  v2's protocol severity bar excludes exactly those, so its recall collapses:
  Effusion 0.41, Medial OA 0.50, Baker's 0.60. The 501 `public_over_calls_negative`
  OA cells are therefore probably **mostly genuine positives that v2 wrongly
  calls below-threshold-negative**, not public over-calls — the opposite of
  the triage doc's first-pass framing (kept as evidence but corrected here).
  v2's Effusion severity assumption is the single clearest protocol error vs
  gold.

- **public's weakness is precision, not recall** (0.53–0.82 almost
  everywhere) — it hallucinates positives on cells where gold is negative.
  v2's weakness is abstention + strict severity — 39–75% of gold cells
  abstained on Synovitis/Baker's/Fracture, and its decided negatives
  over-fire on severity-qualified targets.

## Implications

1. **Neither source is uniformly better → per-target source selection is a
   real candidate policy.** If the pattern holds at scale, a blend taking
   v2's calls on meniscus/synovitis/ligament cells and public's calls on
   effusion/OA/Baker's cells would capture the best of each. This is a new,
   cheap experiment arm alongside the straight public-vs-v2 ablation — the
   `label_source_v1` overlay mechanism already supports arbitrary cell-level
   label tables.
2. **v2.1 review should relax the severity bar on Effusion/OA/Baker's**, not
   just review conflicts: gold semantics appear to count mild/trace findings
   the protocol excludes. The below-threshold `negative` state on those three
   targets is the highest-risk systematic error class in the v2 release —
   501 OA cells + the effusion/Baker's equivalents are the re-review pool.
3. **The public-vs-v2 ablation now predicts a per-target split result** —
   public-label fit should win on Effusion/OA/Baker's AUC, v2-label fit on
   meniscus/synovitis. If the fit shows exactly that split, it confirms the
   semantics hypothesis and directly justifies the blended overlay.
4. **Small-N caveat**: per-target positive counts are 1–35; the recall gaps
   on Effusion (0.41 vs 1.0) and Medial OA (0.50 vs 0.93) are far outside
   noise, but per-target precision estimates (e.g. v2 ACL 1.00 on 15 gold
   positives) have wide intervals. Treat directionally, confirm via the
   paired fit.
