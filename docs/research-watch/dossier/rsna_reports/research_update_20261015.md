# RSNA research update: October 15, 2026

This addendum supersedes older status snapshots where specified. Research remained read-only with respect to the competition repository, model artifacts and clinical data. Synthetic arithmetic is distinguished from measured MRI performance.

## Latest committed outcome: scored epoch 8

Pin: `c195690dc03e131ed210953cf1e22fd631eb364f`, committed October 15 at 12:18 UTC. Exact E8 submission 57221755 completed with public score **.863**, 248,792 output bytes. The committed authenticated API observation at 11:39 UTC and refreshed UI observation at 12:02 UTC agree. This supersedes the running snapshot below. The best recorded E5/E6 score remains .872; E8 is .009 lower. These rounded public scores have no paired uncertainty and do not prove overfitting or private-set behavior.

Keep the prepared E10 fixed-final comparison as a bounded reference. This curve provides no basis for indefinite epoch extension. Compact N3 remains the higher-priority new controlled experiment; adapter averaging remains unmeasured on real checkpoints. No new fit, inference or submission was performed by this research review.

[Scored receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c195690dc03e131ed210953cf1e22fd631eb364f/initiative/audits/kaggle-epoch8-direct-submit-20261015-v1/scored.json)

### Earlier running snapshot, retained as history

Pin: `182d31c6c70e439586a3a539e1461eafffacf3ae`, committed 07:47 UTC. The E8 receipt records a single submission at 07:45:56 UTC, script version 358433328, observed as Notebook Running with no score and no numeric submission ID yet. Best recorded public score remains .872. Canonical ledger reconciliation is pending; do not duplicate the submission. This research review did not perform the submission, and the receipt's description of authorization elsewhere does not expand this review's permissions.

[Pinned submission receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/182d31c6c70e439586a3a539e1461eafffacf3ae/initiative/audits/kaggle-epoch8-direct-submit-20261015-v1/submitted.json)

## Corrected compact initialization contract

The proposed CoAtNet checkpoint expects channel mean/std .5/.5. The earlier proposal borrowed conventional ImageNet normalization from an MRI-trained Raptor route while initializing a different generic ImageNet checkpoint. The revised proposal uses the actual initialization's normalization. This corrects an untested contract assumption, not an observed AUC defect. It does not affect the current DINO model.

Preserve native MRI physical rendering and slice selection. Do not copy the natural-image squash pipeline merely because it accompanies the checkpoint. Before any authorized fit, synthetic black/midgray/white must map to -1/0/1 in each channel, with exactly one normalization and train/deployment parity. Stop if the configuration is unresolved or these checks fail. Passing advances only to the existing feasibility gates, not performance promotion.

[Immutable checkpoint configuration](https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k/commit/ee3938f907f7cbee730c05c5303034513bad6010)

## New independent distillation caution

Deveshu04's October 14 release reports Gold58 label AUC .897 to .922 after self-OOF distillation, but public .931 for distilled targets versus .935 after restoring raw report targets. This is an independent author report of public reversal, not a clean verified causal ablation: scored historical configurations and teacher identities are absent, Gold58 was swept, and its weak validation thresholds the changing target table. Its .940 system is fifteen checkpoints, not a new .95 single model. The final training disables teachers.

This strengthens the reason to freeze evaluation targets and compare any teacher blend with an otherwise matched unblended control. It neither proves distillation generally harmful nor overturns our measured full-soft-label improvement. Do not promote on label agreement alone.

[Pinned release and results](https://github.com/Deveshu04/RSNA-Knee-Abnormality-Detection/blob/782eee8562ff38578c7bf94f621d76832d75828d/README.md#L43-L71)

## Adapter averaging: useful measurement before inference

The new helper's intended common-scale FP32 rank32 algebra is sound. CPU test passes do not establish a successful real merge or score gain. A useful additional diagnostic compares compression error with the averaging change itself, rather than only the large overall adapter norm.

For each projection, let D be the Frobenius distance from the exact E5–10 mean to E10; C is the distance from the compressed mean to the exact mean. Report C/D when D is nonzero, plus the final distance to E10. Before casting, optimal rank32 truncation has C <= D because E10 is itself an admissible rank32 approximation. An excessive postcast violation signals a numerical issue. C/D near one is not an accuracy failure.

Synthetic example: six rank1 updates vvT with v=(1,e), e=[0,.1,-.1,.1,-.1,0], average to diag(1,.00666667). Rank1 compression returns the final parent diag(1,0), retaining 99.99556% energy while removing the entire averaging change. This is a mathematical example, not a measurement on the user's weights.

Also record the singular-value gap at the rank32 cutoff: a tied cutoff makes the best approximation nonunique. Bind parent order and input/output hashes in the eventual wrapper. The primary practical comparator is E10 with its own head versus the complete soup with its averaged head. Exact-rank192 versus compressed-rank32 with the same averaged head is a conditional diagnostic, not an obligatory new performance program. No energy threshold proves accuracy.

[Audited helper](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/5b55daa46148f32ae9199e2ad86bce944c510143/rsna/competitive/adapter_soup.py#L66-L102)

## Training-to-inference review

No confirmed score-limiting defect was found in target order, padding masks, normalization, LoRA scaling, strict checkpoint restoration or the native preprocessing contract. Current all-development run 908454 actually trained in FP16, scale128/chunk8; the broad brief's BF16 description is stale for that run. Deployment deliberately uses FP32. This difference is not evidence of damaged ranking.

The earlier 1,300-study FP32 prefetch benchmark reported all 15,600 probabilities exactly equal to the original FP32 path. That excludes corruption for that tested checkpoint; it does not establish parity for every new weight state or hidden input.

[Actual training binding](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/5b55daa46148f32ae9199e2ad86bce944c510143/initiative/audits/daily-kaggle-comparison-queue-20261014-v1/actual-training-bindings.json)

## Decision

### Coverage and pooling follow-up

The newly inspected notebook's same-checkpoint .8784 full-bag versus .8700 random-bag result changes cross-window Transformer context as well as coverage and probability averaging. It cannot establish that more slices alone helped. For complete five-slot inputs, four draws encode 64 occurrences but only 45.39 distinct centers in expectation; adjacent triplets cover about 68.07 of 72 selected slice positions. These are analytical expectations, not measured lesion coverage.

A future permitted numerical diagnostic could encode one fixed union once, then compare joint pooling with disjoint partition pooling, preserving masks/positions and unique images. For fixed contextual features, joint attention equals the partition pooled features weighted by each partition's softmax mass, not their equal mean. Test that identity before interpreting prediction differences. Averaged probabilities and averaged logits are also different. A Transformer must be rerun per partition to test contextualization; reusing full-context features isolates only the final pool. These diagnostics add head compute and do not authorize extra scored validation variants. The native model already uses its admitted slices and hierarchical pooling; this is not a reason to repeat its failed whole-cache experiments.

[Executed comparison and code](https://github.com/Deveshu04/RSNA-Knee-Abnormality-Detection/blob/782eee8562ff38578c7bf94f621d76832d75828d/notebooks/04-train/rsna-radcs-a.ipynb#L726-L897)

Keep one directly deployable compact whole-recipe challenger first, with corrected initialization preprocessing and the same admitted information boundaries. Keep teacher transfer conditional. Recover already-paid outcomes, preserve uncertainty, and do not confuse numerical tests, revised-label agreement or public author claims with an independent competition result.
