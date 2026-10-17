# Equal-rank teacher transport into one trainable MRI classifier

October17 2026. **Target0.97; verified best0.949.** This is a falsifiable mechanism
and numerical preparation, not a trained model, new best score or promise.
The [actual eight-study screen](../../../audits/s6-fitting8-cross-lineage-20261017-v1/README.md)
shows raw probability shifts8–39points but substantially closer ordering. A raw
confidence-gap gate would confound teacher complementarity with score scales.

## Mechanism

Use the two different MRI teachers' **ordering** and express it on the existing
CoAtNet teacher's scale. On the same complete3887 fitting IDs, for each target:

1. Compute A as mean(sigmoid(original CoAtNet logits)) across its two anatomical
   views; B is original S6 probability. Use exact tied midranks for each.
2. Order studies by the fixed equal sum of those ranks. Assign A's sorted
   probabilities to that consensus order. An exact consensus tie gets the mean
   of its occupied A order statistics; there is no case-ID tiebreak.
3. Apply one scalar shift to both A view logits so their mean probability equals
   that transported value. Preserve their relative view log-odds. An explicit
   counted1e-7 endpoint clamp keeps shifts finite.
4. Use those detached teacher logits in the existing25%observed-source/
   75%teacher anchored loss, training every backbone/head parameter. Preserve
   masks; absent labels never become negative. There is no new pairwise-batch
   optimizer, so the admitted per-study gradient accumulation remains feasible.

This transfers consensus ordering without copying S6's very different raw
probability scale. Without consensus ties it retains A's complete probability
multiset; with ties it preserves the unclamped sum, not the variance. It does
**not** clinically calibrate either teacher or turn ranks into disease truth.
A constant B contributes no ordering and leaves A unchanged. Equal and opposite
rankings cancel into ties rather than inventing confident preferences.

The synthetic target builder is [rank_transport.py.txt](rank_transport.py.txt).
Nineteen actual contracts cover independent hand-ranked examples, reversal/
ties, original control loss/gradient identity, monotonic B recalibration,
simultaneous permutation, decoded two-view targets, view log-odds, finite
extremes and refusals. [Tests](check_rank_transport.py.txt), [actual result](checked.json).
They do not admit the full MRI teachers, GPU trainer or clinical accuracy.

## Matched comparison, frozen before fitting

Control: original A view targets, **bit-identical**. Treatment: the fixed A+B
transport above. Same original0.949 full CoAtNet initialization, identical
fitting roster, masks, source supervision, study order, common IID16 draws/
anatomical views, effective8, full encoder/head, AdamW/LR/precision, recovery
semantics and fixed3epochs/1458updates; full94 two-view inference. Freeze the
exact numerical recipe/target SHA before fitting. No target-specific weights,
Gold tuning or best-epoch selection. The running original sampling experiment
is neither changed nor claimed to be this independently admitted contrast.

The old target bundle is pinned and immutable. A **distinct versioned data
adapter** must explicitly bind the new complete teacher bundle; do not replace
the old `parent_logits` file, weaken its hash checks or reuse its output roots.
Actual complete-state full/stop/resume equality, both-rank/scaler coverage and
learned serving equality are required for the final changed data path. The
numerical helper alone cannot establish identity alignment or licence admission.

## Gates and decision

First finish the original S6 offline/public acceptance and entry, and run the
already prepared fixed equal-rank inference comparison. The separate
[32-study scout](../../../audits/s6-fitting32-scout-preparation-20261017-v1/README.md)
checks original input parity/FP16 error/throughput. Only after that evidence may
one explicitly funded job generate complete fitting-cohort S6 outputs; private
population/finite/hash/fallback checks must all pass before target creation.
Eight/32 cost quantiles cannot stand in for the complete empirical rank cohort.

Then fund and admit the matched full-encoder contrast, keeping existing healthy
training intact. Compare fixed final code entries under actual account/runtime
allowance. Released encoders exposed this cohort: internal weak-label agreement
or head folds cannot establish independent OOF, clinical calibration or private
generalization. If S6 is not complementary, transport may preserve shared errors
or harm the stronger model. Retain that negative and keep the actual better model.
No new ordinal trainer GPU or full3887 teacher acquisition is dispatched yet.

## Research basis

Compressing knowledge from multiple teachers into one model follows the general
[knowledge-distillation principle](https://arxiv.org/abs/1503.02531). Ranking
objectives are established, including
[RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/);
this candidate instead transports ranks into pointwise teacher targets to reuse
our accumulated full-encoder training engine. These sources motivate a mechanism;
they provide no evidence for an MRI gain or0.97. We make no algorithmic-novelty
claim. The value must come from the measured complementary MRI information and
an actual controlled result.


## Later actual data-path engineering

The separate [teacher overlay](data-path/README.md) now passed17actual synthetic
CPU tests with the unchanged full-encoder trainer, both-role complete-state
recovery and exact full94 CSVs. It automatically binds the actual target arrays
to checkpoints while preserving original pixel/label/mask objects and refusing
silent target/role changes. This satisfies a CPU engineering step; no complete
S6 teacher bundle, MRI/GPU recovery, production dispatch or gain is implied.
The fixed inference blend also has an [actual staged sole owner](../../../audits/s6-equalrank-deployment-20261017-v1/README.md).

Input visibility remains a limitation: a student cannot recover absent MRI
evidence from better teacher targets alone. Larger S6 window counts do not
establish that this occurs; measure physical/source-position evidence coverage
if the matched target contrast fails. Do not treat teacher disagreement as a
clinical error or block existing healthy experiments on this open question.
