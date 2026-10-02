# Training-only public report supervision diagnostic

This is a measured diagnostic, not label admission, clinical validation, or an accuracy gain. The accepted release, masks, folds and healthy production job868093 are unchanged. Public report verdicts were compared only on the exact2247 eligible fold0 fitting studies; all outer/protected/ineligible UIDs are excluded before public target interpretation. A separate accepted-label count covers the actual3403-study all-development run.

## What the current model actually learns

| Target | Positive | Negative | Masked | Positive annotation additionally masked |
| --- | ---: | ---: | ---: | ---: |
| ACL | 256 | 2561 | 586 | 28 |
| Baker's | 9 | 1132 | 2262 | 79 |
| Contusion | 182 | 1042 | 2179 | 104 |
| Effusion | 25 | 1928 | 1450 | 422 |
| Fracture | 5 | 699 | 2699 | 15 |
| Lateral Meniscus | 446 | 2386 | 571 | 18 |
| Lateral OA | 27 | 1507 | 1869 | 1 |
| MCL | 7 | 2762 | 634 | 8 |
| Medial Meniscus | 1244 | 1662 | 497 | 49 |
| Medial OA | 69 | 1410 | 1924 | 0 |
| PF OA | 57 | 1249 | 2097 | 0 |
| Synovitis | 364 | 15 | 3024 | 11 |

These are study counts, not independent patients or image-confirmed prevalence. MCL has seven observed positives, effusion25, Baker's nine and fracture five, despite3403 eligible studies. Synovitis has only15 observed negatives. The existing extraction is conservative and target-specific; more encoder parameters cannot create missing supervision. This supports a supervision bottleneck hypothesis, not a causal explanation of the Kaggle gap.

## Public source and disagreements

The original [Pilkwang dataset](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels), ID11529209/version1/CC0, was downloaded once. Its CSV and metadata/archive SHA256 are bound in diagnostic.json. The ZIP is bounded to8MiB expanded and vendor Python was only statically inspected, never executed. Source raw verdicts are YES/NO/UNK. Its author explicitly describes scores as rank-usable and uncalibrated. The shipped script imports `to_scores` from an absent `llm_labeler` module: the exact numerical mapping is not independently reproduced. Observed score categories must not be asserted to be official severity classes.

The diagnostic cohort is fixed to2247 studies; diagnostic.json contains all12 cross-tabs, observed score/confidence histograms, original masked states, semantic reasons and the membership digest. Counts below refer to that2247-study fitting cohort only:

- Effusion:739 observed negatives receive public YES with score0.68. Those can be mild findings below the competition's moderate/large threshold; this is not739 proven errors.
- MCL:197 negatives receive YES/0.68; the sole negative receiving YES/0.94 is explicitly a chronic tear on direct report reading, so the source positive does not automatically justify relabeling.
- There are272 original-positive effusion annotations currently masked in the fitting cohort, and280 masked effusion cells carry incomplete_reliability_review. These overlapping counts must not be added. In the actual3403-study run,422 positive effusion annotations are masked.
- Public upper-score YES occurs on71 masked effusions,19 masked MCL targets and64 masked Baker's targets. A public score is a selection signal for diagnostic review, not a confidence or truth guarantee.

## Source ancestry and review

Metadata for [Steven's labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) and [JEV-Knee labels](https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels) was inspected without downloading either archive. Their cards describe source blending/target choices tuned on58 or74 gold-like reports. These are neither independent corroborating labelers nor clean evidence for our controlled protected-cohort experiment. None was admitted. JEV-Knee's text labeling API also does not establish an MRI architectural result from the official Jev system.

A deterministic25-record private queue spans four targets, upper-score masked/negative disagreements and positive/negative agreement controls, at most two distinct leakage components per target/category. Raw reports, public case values and case IDs remain in ignored0600 artifacts; only aggregate evidence goes to this audit. Nine upper-score cases received an actual first full-source diagnostic reading. A separate reviewer completed actual independent rereading of the same full sources, blinded to first judgments, public labels and current annotations. Eight of nine proposed states agree: both effusions are positive, both dimensional Baker cysts and both insufficiently qualified MCL injuries remain uncertain, and the chronic MCL stays negative. One fracture acuity judgment differs and remains unresolved. The other fracture receives positive agreement but its prior state-disagreement hold is not discharged by this diagnostic. All18 verbatim quotations, source hashes and the frozen rubric were verified; see independent-source-review-aggregate.json. No state or mask is promoted by this diagnostic.

The two sampled effusion holds explicitly originate from an unresolved proxy-reliability review, not a different positive label or a demonstrated wrong diagnosis. Any discharge must bind the exact original obligation, source rationale and independent assessment; generic agreement alone does not clear a hold.

## Next falsifiable direction

First resolve a bounded training-only reliability scope against the original reports and exact contrary evidence; preserve original judgments and represent any accepted changes as a separate experimental overlay. Compare fixed-final training with and without that overlay using identical encoder initialization, study roster, compute and frozen outer targets. Reject the hypothesis if a target gain fails paired component uncertainty or does not repeat on a matched seed/fold. No gold/confirmation tuning or changed live numerical recipe is authorized by this diagnostic.

An alternative mechanism is a separate coarse finding-presence auxiliary head: it could learn descriptive evidence without declaring every mild/chronic finding to be the competition target. That mechanism is an unimplemented hypothesis; public rank scores must not become clinical BCE probabilities. Neighbor-slice context and anatomy-aware scale remain separate experiments.

## Reproduce and limits

Run `python -m scripts.research.audit_public_report_labels` with the private version1 archive/metadata, accepted studies.parquet, frozen fold.csv/rubric, `--expected-studies 2247`, a new aggregate output and ignored private-review output. The CLI refuses existing outputs, incomplete/duplicate fitting membership, invalid fitting values and oversized/unsafe archives. Nine regression tests include poisoned excluded rows that must never be interpreted, original component isolation, immutable labels and deterministic component-bounded sampling.

No MRI diagnostic reread, clinical improvement, supervision rewrite, new GPU job or Kaggle mutation occurred. Sampling is deliberately diagnostic and cannot estimate a population error rate. The separately executed all-development helper is preserved in measure-live-supervision.original.py.txt; its replay exactly matches live-training-supervision.json, including scope/source and membership hashes.
