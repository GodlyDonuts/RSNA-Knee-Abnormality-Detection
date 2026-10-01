# Use submissions to learn which DINOv3 change helps

The owner requested experimental submissions to determine whether greater adapter
rank produces better results. `learning-board.json` records the question,
comparison, training population and actual internal/public evidence separately.
It creates no training job, submission or new label release.

At the authenticated October 1 04:01 UTC read, rank64 entry **56740377** still
returned no hidden status, error or score. It was submitted once; never repeat it.
The scored references remain sampled rank32 **0.769 / 56697342** and
all-development rank32 **0.792 / 56711823**. The latter uses 3,403 training
studies instead of 2,247 and cannot isolate an adapter-rank effect.

| Completed experiment | Internal fixed-final AUC | Public Kaggle score |
| --- | ---: | ---: |
| Rank32, 16 slices, fold0 | 0.862094 | 0.769 |
| Rank64, 16 slices, same fold0 | 0.834050 | Awaiting actual outcome |
| Rank32, every eligible slice, same fold0 | 0.860340 | Not submitted |
| Rank32, all-development final fit | No remaining internal holdout | 0.792 |

Internal AUC uses accepted report-derived labels and is not the Kaggle metric
on the hidden competition studies. All twelve target weights remain intact.

## What the rank result currently says

The rank64 internal macro difference is **−0.028044**. MCL has just two positive
components and contributes approximately 92% of the total macro decline. ACL
also declined by **0.058295**, with 85 positive components; the result cannot be
dismissed entirely as the rare MCL target. Ten targets have paired bootstrap
intervals that include zero. The two models also consume different amounts of
initialization randomness, so this is a fixed-recipe comparison rather than a
pure causal estimate of rank.

`paired-component-uncertainty.json` independently replays the exact original
prediction, label, result and run bindings, then makes 4,000 paired bootstrap
draws of the **1,106 leakage components** containing the same 1,156 held-out
studies. Every member of a sampled component retains its labels/masks and its
component multiplicity for both models. No gold or reserved confirmation
targets are used. Weighted/tied AUCs are tested against sklearn.

The macro percentile range on the **2,836 defined draws** is
**[−0.050331, −0.006980]**. The other **1,164 draws** lose at least one target's
positive or negative class and have undefined all-twelve-target macro AUC.
They are explicitly recorded; no target is dropped and no replacement draw is
silently requested. These conditional percentiles describe this particular
weak-label development pool. They do not resolve the two-positive population,
head-initialization uncertainty, different seeds/folds or unseen competition
label semantics. Rank32 remains the measured working reference; do not claim
that higher rank generally hurts, or that training longer must repair rank64.

Replay to a **new** output location:

```bash
PYTHONNOUSERSITE=1 artifacts/qa-env/bin/python -m scripts.rank_learning \
  --output /tmp/rsna-rank-learning-replay.json --draws 4000 --seed 20261001
```

## The next comparisons

The already submitted rank64 E10 answers the existing fixed-recipe capacity
question. Wait for its actual result before another capacity submission.
The next prepared training comparison is original-source **384-input rank32**,
keeping capacity, train/outer membership and slice/series policy fixed. Its full
cache is being rebuilt by the two existing CPU lanes; no native production GPU
job exists yet. Deterministic-runtime and reconstructed-256 rounding differences
are documented numerical caveats. Do not reinterpret its result as a rank gain.

A stronger subsequent rank32/rank64 comparison must verify identical initial
study-head tensor hashes and zero-adapter evaluation logits, isolate adapter RNG
from head/data-order RNG, use paired seeds and the same full accepted input
membership, and keep resolution, sampling, effective batch, optimizer/update
schedule and fixed ten-epoch endpoint matched. The board declares two seed
values; this stricter initialization experiment is **protocol-only**, without
an implementation, staged config, reservation or dispatch. A second development
fold is required before a broad rank claim. Rank128 is a later capacity
hypothesis, not the automatic successor to a larger parameter count.

Keep the existing rsLoRA policy **alpha=8, alpha/sqrt(rank)** across ranks.
Its literal multiplier is 1.4142 at rank32 and 1 at rank64; that rank-dependent
factor is the existing parameterization, not an accidental independent alpha
change. Equalizing the multiplier by changing alpha creates a separate scaling
experiment. Adapter strength or learning-rate sweeps must be recorded separately
instead of attributing their gain to rank alone.

Before each worthwhile submission, record the question, exact reference and
expected decision in the board; require completed immutable/private-verified
weights, exact MRI-only offline inference acceptance and fresh live account
allowance. Use the single Newton coordinator/ledger. Five daily slots are a
ceiling, not evidence to submit five similar models. Public outcomes provide
directional evidence; repeated use makes the public leaderboard a development
set. Do not fit a blend, threshold, target selection or checkpoint choice to it.
Protected confirmation and official gold remain untouched.
