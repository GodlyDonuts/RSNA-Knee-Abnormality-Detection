# The score gap may be partly a model-selection problem

14 October 2026. Read-only challenge to the compact-challenger recommendation, using already-produced aggregate receipts. No clinical rows, private predictions, MRI, weights, new evaluation, training, submission or repository mutation were used.

## Bottom line

The compact CoAtNet comparison has useful deployment-option value, but the available evidence does not make architecture the leading explanation of the remaining .872-to-.95 gap. The strongest rival is **supervision and selection mismatch**: a model can improve competition ranking without improving the particular report-observed population measured by our weak-development endpoint.

That is a plausible explanation, not an identified cause of the remaining gap. We cannot determine its contribution from aggregate scores. The practical correction is to stop treating weak-development AUC as a guaranteed competition-ranking oracle. This does **not** justify replacing the endpoint, accessing protected cohorts, automatically preferring teacher blending, or launching a leaderboard search.

## 1. A more informative comparison was already paid for

The closest supervision comparison is **.783 → .854**, not .804 → .854 or .854 → .872.

- Both relevant fold-0 models use seed 20260922, 2,247 fitting studies, fixed epoch 10 and 2,810 updates. Both preserve the same accepted release binding
- The .783 submission uses checkpoint `2465648917b5…`; .854 uses `ebeb3c394286…`
- The first is not literally the paired-control checkpoint `85f44b7fb588…`. However, the completed shared-control proof establishes exact equality of all 258 common learned tensors, all 2,810 numerical update records and the entire final 1,156-study prediction CSV. Its additional 131,072 patch-summary parameters are all zero
- Deployment metadata matches exactly on the pinned runtime image, shared wheelhouse, two-device exact-FP32 prefetch engine and its full-scale/memory reference hashes
- Whole source projections, graph serializations and submitted source hashes differ. This is strong same-family supervision evidence, not a newly demonstrated byte-identical end-to-end label-only experiment on every hidden input

The weak-development gain is **+.021113**, while the rounded public-score gain is **+.071**. This supports the claim that the native encoder/input family had substantial unrealized value under its older supervision. It does not establish that today's remaining gap can be removed by another label change, nor that the current .872 model shares the fold-0 model's exact properties.

The full-public label improvement is already implemented. Recommending it again would add no value. Likewise, a teacher trained on the same labels can reproduce their systematic errors; this comparison alone does not give teacher blending priority over compact full fine-tuning.

Sources: [paired control result](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/control-result.json), [public-label result](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/public-result.json), [completed control-equivalence proof](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/shared-control-final-20261008-v1/README.md), [.783 preparation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-best-internal-e10-2465648917b5-v1/prepared.json), [.854 preparation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-publicsource-e10-ebeb3c394286-v1/prepared.json), [.783 score](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-best-internal-e10-2465648917b5-v1/scored-hidden-outcome.json), [.854 score](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-publicsource-e10-ebeb3c394286-v1/actual-scored-outcome.json).

### Cheap falsifier, with its exact limit

The deterministic claim “a lower weak-development score means a worse competition candidate” already fails on the observed rank32/rank64 pair: **.862094 → .834050 internally; .769 → .778 publicly**, both fixed epoch-10, fold-0 models. The matched sampling pair also has materially different gain sizes: **+.000911 internally; +.014 publicly**. These are existing outcomes, so no new fit or submission is needed to check that claim.

This is not proof that rank64 is generally better, that sampling truly helps, or that report selection caused the disagreement. Public feedback is rounded and reused; internal support is sparse. MCL contributes approximately 92% of the internal rank decline. Initialization/scaling differ between ranks. These limitations prevent a causal rank claim but do not restore perfect agreement between the observed endpoints.

[Rank comparison and uncertainty](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-rank-learning-v1/README.md), [matched sampling experiment](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/sampling-fixed-final-v1/README.md), [sampling deployment identity](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/kaggle-sampling-e10-comparison-v1/README.md).

## 2. The observation mask is associated with the source's scores

Weak-development AUC compares positives and negatives among cells with accepted report labels. It does not score every finding on every study. A bootstrap resamples that observed evaluation population; it cannot recover performance on omitted cells or resolve different target meanings.

An additional check can be reconstructed entirely from the existing **2,247-study fitting-population aggregates**. Let `q` be the admitted public source score, `n+` and `n−` be accepted observed label counts, and `mu+` and `mu−` its mean scores on those groups:

`mean(q | observed) = (n+ × mu+ + n− × mu−) / (n+ + n−)`

The source file supplies those counts and means, plus the mean on previously masked cells. Reconstructing the sums gives:

| Target | Accepted observed + / − | Reconstructed observed q sum / count | Reconstructed masked q sum / count | Mean observed / masked |
|---|---:|---:|---:|---:|
| Synovitis | 243 / 10 | 231.165 / 253 | 554.7075 / 1,994 | .913696 / .278188 |
| PF OA | 38 / 804 | 164.385 / 842 | 955.610 / 1,405 | .195232 / .680149 |
| ACL | 171 / 1,682 | 306.345 / 1,853 | 240.745 / 394 | .165324 / .611028 |

For auditability, the source's positive/negative means respectively are Synovitis `.9505555555555556 / .018000000000000006`; PF OA `.988157894736842 / .15775497512437808`; ACL `.9938011695906432 / .08109690844233054`. Masked sums above are reconstructed as masked count multiplied by its supplied mean; tiny final-decimal differences reflect floating-point arithmetic. No row-level recomputation was performed.

**What this establishes:** the accepted fitting mask is associated with the public source's score distribution. In particular, only 253/2,247 = 11.26% of fitting Synovitis cells were originally observed. Treating the observed and omitted strata as interchangeable is unsupported.

**What it does not establish:** these are not clinical prevalence estimates, not new outer-cohort measurements, and not proof of clinical missing-not-at-random. The source has its own construction rules and proxies. A pure prevalence shift does not change AUC. Proving a ranking bias requires more than different score means, and these aggregates cannot distinguish source construction, target semantics, class-conditional selection and ordinary variance. They also cannot identify which current public-score targets account for the remaining gap.

[Original aggregate statistics and scope declaration](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-fitting-statistics-v1/statistics.json), [interpretation and severity limitations](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/public-label-source-fitting-statistics-v1/README.md).

## What remains plausible about .872 versus .95+

- **Training population:** current own all-development fitting uses 3,403 studies; the inspected nartaa artifact uses 4,349 RSNA studies plus OAI. That is a confound, not evidence that adding the excluded studies is permissible or sufficient. The .953/.951 author claims say no external data, so OAI cannot explain every high-scoring single
- **Evaluation:** .872 and the author-reported .95+ figures are competition-score claims; .899784 is a different fitted model on report-derived development labels. Subtracting .899784 from a public score does not measure generalization loss
- **Label source:** current targets are a report-probability blend, not documented image-teacher targets. Several strong authors report image pseudo-supervision, but an inspected report-only claim reaches .945–.947. Therefore image pseudo-labels are neither a proven requirement nor a quantified explanation of the whole gap
- **Input:** both native and nartaa nominally select 96 source images. Allocation, crop/physical scale, span, neighboring channels and aggregation differ. “More slices” is not an established explanation. The failed older-source-slot bundle does not isolate those factors
- **Optimization:** full-trunk adaptation, head capacity and training schedule remain plausible. The cancelled dense comparison is not a negative accuracy result; two rounded .872 checkpoints do not prove convergence
- **Deployment:** the own score is tied to exact completed submissions. The >.95 author claims mostly lack equally bound executable/checkpoint receipts; nartaa .949 is better audited but includes SWA and two views. Packaging feasibility and label accuracy are separate questions

These are evidence categories, not additive fractions of the gap. The same-day [single-model audit](rsna-single-model-evidence-2026-10-14.md) supplies the source-specific claims and qualifications. No defensible aggregate-only test can decisively identify the remaining gap's main cause.

## Updated decision rule

Keep the compact candidate as **one possible direct final model**, not as a presumed reproduction of .95. Full fine-tuning, a full-dimensional bag head and a different optimization schedule make it more than copying a backbone name. Conversely, retaining our current labels and native geometry while omitting competitors' teacher, population and selection choices means the public scores provide only a weak quantitative forecast for this exact recipe.

1. Recover already-paid results and retain the current completed qualified incumbent
2. Do not move teacher blending ahead merely because the older supervision change helped. A ready, admitted teacher can earn priority through its actual evidence and lower remaining delivery cost; an unavailable or same-error teacher does not
3. Before any future fit, specify whether the authorized delivery budget includes one fixed downstream competition comparison. Keep all twelve existing weak-development targets and their uncertainty reporting. Do not change labels, drop difficult targets or open protected cohorts
4. An inconclusive or rare-target-driven weak difference should not automatically be treated as proof that a predeclared candidate is competition-inferior. If a single downstream comparison is separately authorized and already budgeted, that fixed comparison may still be informative. This is a proposed decision-policy clarification, not permission to submit or to ignore an existing preregistered rule
5. Without that downstream route, a weak-only compact result remains exploratory. A clear broad regression or failed deployment gate is a practical stopping reason for this candidate; an ambiguous weak result is not a reason for more architectures, new holdouts or repeated public probing. Preserve the better-qualified incumbent

The main change is confidence in what the experiment can decide, not an extra experiment. Its deployment option can be valuable even though its ability to explain or close the .95 gap is unproven.
