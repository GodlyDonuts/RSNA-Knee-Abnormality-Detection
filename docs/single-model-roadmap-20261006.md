# Single-model roadmap — RSNA Knee Abnormality Detection (2026-10-06)

Owner direction: a highly sophisticated **single model first**; ensembles later.
This document synthesizes the frontier survey (`research-frontier-20261006.md`),
the gold-58 label arbitration, and the v2.1 adjudication results into the
ordered experiment sequence most likely to reach a competition-grade solo
model.

## What we now know

1. **Supervision, not capacity, is the binding constraint.** Three external
   sources agree (rank-18 write-up, pilkwang, public backbone ablation). Our
   gold-58 arbitration confirms it internally and makes it precise: the two
   label sources split the 12 targets — v2 wins meniscus/synovitis/ligaments
   (degenerative morphology the public extractor misses), public wins the 4
   severity targets (gold counts mild findings that v2's severity bar
   excludes). The label-source question is not "which source is better" but
   "which severity boundary does the official metric reward" — and that is a
   measurable question, now packaged.
2. **Label precision is tiered.** Gold-58 validation of the v2.1 adjudication:
   positive-call precision 0.46 (pass 1) / 0.75 (pass 2) / 0.08 (pass 3).
   Pass-3 flips are hypothesis-generating, not labels to trust blindly —
   exactly why they are a separate arm rather than folded into one release.
3. **Resolution is near its ceiling at 224 px.** Corpus-wide census: 99.6% of
   series fail the Nyquist bound at 224 px; failure plateaus ~10% at ≥448 px
   (acquisition-limited beyond that). The useful ladder is 336 → 448.
4. **Laterality is unnormalized.** Only 43% of studies carry an L/R tag;
   geometric fallback covers the rest (~94–99% tag agreement). External
   measurement: +3.2pt attention-alignment gain from canonical-side mirroring.
   Our current 50% random flips teach the model to ignore side — the opposite
   of normalization.
5. **Masked cells are supervision, not noise.** Synovitis is 89% masked; the
   masked-cell budget IS that target's training signal. SPML/PU losses
   (entropy-max, consistency EMA) are the principled treatment.
6. **Architecture is already at the externally-validated design.**
   `CompetitiveStudyNet` = per-slice scores → per-target series attention over
   plane slots → per-target output heads. Matches pilkwang's per-diagnosis
   query attention. Max pooling is the control arm. No architecture novelty
   needed before the supervision questions are answered.

## The experiment ladder (ordered)

Each step is a paired, single-variable change on the same backbone/folds.
Steps 1–3 need one GPU slot each; 4–6 are multipliers applied to whatever
wins earlier.

### Step 1 — Label-source ablation (decides the supervision basis)

Seven arms, all packaged with sha-bound run configs (PR #4,
`run-configs/label-source-*`):

| arm | question |
|-----|----------|
| `base-v2-control` | current baseline |
| `label-source-public` | does the public extractor beat v2? |
| `label-source-blended` | gold-informed hybrid (v2 on 8, public on 4) — upper-bound probe |
| `label-source-v21` | do all 7.9k adjudicated flips help or hurt? |
| `label-source-v21pn` | does explicit negation supervision help? |
| `label-source-v21hc` | pass-2-only flips (0.75-precision tier) |
| `graded-u05` | does supervising uncertain cells at 0.5 beat masking? |

Preregistered predictions + decision rules are in `ABLATION-ANALYSIS.md`
(per-target split expected; v21hc ≥ v21; v21pn > v21 on fluid targets;
graded-u05 < v21). The outcome selects the label basis for everything below.

### Step 2 — Canonical-side normalization

`canonical-sides.fold{0,1,2}` configs bound. Deterministic mirror to a common
side; pair against whichever label basis wins step 1. Expected: helps the 5
side-dependent targets (MCL, medial/lateral meniscus, medial/lateral OA).

### Step 3 — Resolution ladder 336 → 448 (one fold)

On the winning configuration only. Stop at the first rung that fails to beat
224 on the arbitration holdout — pixel cost is quadratic and the plateau is
documented (~10% failure ≥448 px, acquisition-limited). Trade slices before
pixels: through-plane pitch (~4 mm) can't be fixed.

### Step 4 — Masked-cell losses

`masked-cell-em005` config bound (entropy-max on masked cells, weight 0.05).
Consistency-EMA implemented as `masked_cell.kind:"ema_consistency"` —
EMA-teacher (α=0.999) MSE on masked-cell logits, ramped 20%, teacher
checkpointed for exact resume; `masked-cell-consistency` config emitted.
Most load-bearing for Synovitis/sparse targets.

### Step 5 — Knee-domain SSL pretrain (one-time, day-scale)

~820k unlabeled knee slices, scanner/protocol matched to test. OAI-DINO and
MRI-CORE both show domain-matched DINO pretraining beats generic supervised
pretraining in low-label regimes. Highest ceiling item on the list; also the
most expensive. Run once when a long GPU window exists.

### Step 6 — Attention-vs-max pooling + local-dependency term

Ablation grid on the pooling level: `pooling-max` config emitted (max is
the control; attention is default). Third arm implemented: `slice_smooth`
(SmMIL-style total-variation penalty on the gated per-slice attention
scores — a tear occupies a contiguous run, not scattered singletons);
`slice-smooth-smmil` config emitted at weight 0.1.

## Decision discipline

- Arbitration: the 1,156-study weak-label holdout + bootstrap intervals
  (per-target split resolution is limited — expect target-level, not
  macro-level, winners).
- Gold-58 never arbitrates anything. SE is wider than every delta we chase.
- `leakage_group` stays load-bearing in splits (template-identical reports).
- No TTA by default (2026 systematic study: consistent degradation on
  medical benchmarks); any TTA is validated per-model first.
- No calibration/threshold work — AUC is rank-invariant.

## What reaching ~0.95 solo requires

Honest framing: the best *solo* public results are ~0.90–0.93 territory; the
0.936/0.942 scores are ~10-member rank-fused ensembles. The gap a single
model must close is covered by supervision (steps 1–2), resolution (step 3),
domain pretraining (step 5), and plane-aware aggregation (step 6) — in that
order of expected size. If the ablation confirms the per-target split, a
single model trained on the blended/v21hc basis at 336–448 px with
canonical sides is the realistic 0.95 candidate. Ensemble fusion on top of
that is the later move, exactly as the owner directed.
