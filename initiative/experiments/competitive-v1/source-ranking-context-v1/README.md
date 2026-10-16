# Study-pair ranking from the strong MRI foundation

Prospective protocol, October16,2026. Full-parent139111324 training and all
existing serving/entry owners remain unchanged. This is a separate one-CPU
experiment reusing the already accepted frozen MRI features, with no new CNN
encoding, source download or GPU allocation.

## Hypothesis and fixed comparison

The current head optimizes anchored probability BCE, while competition ROC AUC
depends on between-study ordering. Test a fixed ranking surrogate against the
already completed uniform context head. Same3,887fitting studies, exact two-view
features/labels/masks/teacher,402,956parameter graph, seed20261016, batch16,
10epochs/2,430updates, LR3e-4/warmup-cosine, AdamW/WD.01/clip1, bounded±.5logit
correction. The underlying licensed0.949MRI parent remains unchanged.

Let B be the old anchored loss:75%detached-parent BCE+25%observed-source BCE,
with missing source influence returned to the teacher. For each target, form
distinct-study pairs in the current minibatch. Pair weight is max(y_i-y_j,0),
and either missing source target excludes the pair. Equal labels impose no
order. Scores are stable log-odds of the **mean of the two view probabilities**,
matching the inference-time order, not the mean of their logits. Pair risk is
the weighted mean of softplus(-(s_i-s_j)), normalized separately per target.

Replace exactly half of the observed-source point BCE with that pair risk,
preserving observation coverage and the full detached-teacher contribution.
Targets without valid pairs keep the exact original loss, including singleton
final batches. The ranking fraction is fixed.5; no strength/epoch sweep.
Both uniform and candidate use the same study permutation; no label-balanced
sampling, modified mask or invented clinical negative is introduced.

The actual candidate must pass fresh full/stop1/resume processes, complete
model/AdamW/LR/cursor/allRNG/history and exact prediction-byte equality on the
original32engineering fitting feature records. A uniform engineering arm must
reproduce the old engine's complete numerical state and predictions. The full
completed original uniform model is then reused; do not repeat its full fit or
publication. Preserve all ten new immutable candidate epochs privately once.

## Interpretation and decision

This external parent was trained on our internal cohort, so neither its fitting
loss nor local weak-label AUC is independent validation. The primary next test
is an exact fixed-final MRI-only Kaggle comparison after actual inference and
runtime acceptance, within fresh live allowance. Do not select an epoch using
Gold/confirmation, fit blend weights on exposed holdouts, infer accuracy from
training loss or submit diagnostics. No competition entry is dispatched by the
training wrapper. A bad result retains the strong parent and negative evidence.

This mechanism can fail: weak source order can be wrong, the small batch may
undersample rare contrasts, macro normalization increases noisy-target influence,
and a±.5correction may be insufficient. It cannot repair missing image evidence.

The conceptual precedents are [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)
and [medical deep AUC optimization](https://arxiv.org/abs/2012.03173). This
implementation is a weighted pairwise logistic contrast; it is **not** the latter
paper's margin min-max algorithm, a replication of its results, or a new theory.

Fresh funding must retain all prior607,095,306,179required node bytes, then add
256MiBcheckpoint/recovery/publication and32MiBcontrols. Zero shared pixel/model
growth or deletion credit. Original feature/label/index byte proofs and license
are reused; patient arrays stay on the authorized owning node. Source state is
not yet actually CPU-admitted, fitted, privately preserved or Kaggle-scored.
