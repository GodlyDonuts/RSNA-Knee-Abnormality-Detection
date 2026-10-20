# Current RSNA Knee research state

Updated **October 20, 2026, 10:06 UTC**. This is the current consolidated snapshot; original receipts and live service state govern actions. Historical snapshots remain in Git history, including commit `36f6d01b`. The owner is AFK and authorizes independent work. First place and 0.97 remain unachieved.

## Results and the current gap

| Model | Actual Kaggle score | Entry |
| --- | --- | --- |
| External R6 ensemble reproduction | **0.958** | 57333150 |
| R6 with our paired member | 0.957 | 57397292 |
| Released S6 reproduction | 0.950 | 57300990 |
| Own whole94 / influence IID16 full-encoder adaptations | 0.949 / 0.949 | 57336190 / 57332456 |
| Own paired / higher-LR adaptations | 0.949 / 0.949 | 57392062 / 57394237 |
| Own uniform / source-label adaptations | 0.947 / 0.947 | 57329796 / 57331631 |
| Cross-lineage supervision, epoch 1 / fixed-final epoch 3 | 0.948 / **0.949** | 57418131 / 57422143 |

No original accuracy improvement over the external reproduction has been demonstrated. The latest cross-lineage result ties our earlier adaptation. Its completed evaluator 21960 must not be restarted. [Original sources and scored receipts](../initiative/research/S6-Kaggle-production-route-20261019-v1/).

## Primary experiment: learn predictive MRI evidence

The owner's [research proposal](../initiative/research/predictive-evidence-proposal-20261020-v1/README.md) changes the learning objective, rather than extending parent imitation. The active implementation uses the inherited MRI **CoAtNet** encoder with roughly 73 million trainable parameters and fresh heads; it is not a DINOv3 run.

- **B:** direct classification plus diagnosis-independent reconstruction, with access to actual observed descriptors.
- **C:** diagnosis-conditioned prediction of genuinely withheld 64-dimensional MRI descriptors, aggregated over complete support priors.
- **C0:** C's identical architecture with density-loss weight zero, to test what the auxiliary predictive objective contributes.

Masking happens on original source pixels before rendering, with alias closure and an eight-pixel halo. Fixed encoding/header scaling avoids target-dependent percentile normalization. Both complementary masked contexts use 24 encoder windows at side 320 while retaining all descriptor regions/support mass; their posterior probabilities are averaged. The encoder becomes fully trainable after explicit head-only warmup. These models use no old teacher probabilities or inherited classifier-logit shortcut. Accepted labels/masks/folds stay unchanged; protected gold/confirmation stay unused.

## What actually ran

| Evidence | Verified result | What it does not establish |
| --- | --- | --- |
| Original raw-input pilot 977456 | 67 DICOMs, 1,072 source regions; intervention changed withheld descriptors without changing masked context bytes | Descriptor sufficiency or clinical accuracy |
| Raw cache 978232 | All 16 engineering studies; 2,391,033,794 source bytes, 1,091,912,852 cache bytes, 62,032 regions; original hashes and inputs retained | Full-cohort completion |
| Exact context cache 978371 | 263,296,123 bytes; 58.611 s raw load/render versus 3.142 s cached load on 16 studies; **18.656× CPU input-load speed** with bitwise inputs/metadata/priors/RNG | End-to-end training speed or accuracy |
| Raw-input B/C GPU 978305 | COMPLETED 0:0 in 31:25; 24 updates per arm on 12 fitting studies, four disjoint engineering predictions; complete recovery and CSV equality | A held-out clinical result or submission candidate |
| Cached-input B/C/C0 GPU 978554 | COMPLETED 0:0 in 3:58; four updates per arm on the same 12+4 engineering cases; complete recovery and CSV equality | Full predictive training or T4 acceptance |

In the cached diagnostic, 24 study exposures took **B 20.834 s, C 20.602 s, C0 20.448 s** in the measured fit loop, approximately **0.85–0.87 s per exposure**, including about 4.9 s of CPU context loading. Peak device allocation was approximately 5.89 GB. The uncached 24-update arms took about 438 s for 96 exposures, but batch size, warmup, and diagnostics changed, so this is not an isolated cache-only speed comparison.

Source `36f6d01b` clamps fractional warmup and batches gradient-diagnostic device-to-host transfers while retaining per-tensor FP64 reductions and ordered summation. Original staged sources remain unchanged. [Cached-engine receipt](../initiative/research/predictive-cached-engine-20261020-v1/actual-accepted.json); [raw B/C receipt](../initiative/research/predictive-raw16-learning-20261020-v1/actual-accepted.json).

**The numerical risk is still real:** cached C/C0 classification loss reached approximately 9.79/9.90, and head-gradient norms reached approximately 4,366/4,407. C0 also spikes, so removing density loss alone does not resolve it. B's four update losses stayed finite and decreased, but that is training behavior, not model quality. Fixed-unit covariance, correlated support sums, global gradient clipping, descriptor sufficiency, and the input-scaling shift remain concerns. The distinct prepared stability test zero-initializes the density means in both B and C/C0 and uses the actual 2,916-update production warmup horizon for 24 fitting-only updates. It preserves exact complete-support likelihoods and all previous originals. It is now running as **978709** on evc27 after actual input admission **978708**, source `39e315c6`; it is not yet an actual pass. Do not select a model from the four engineering cases.

## Live work and ownership

| Work | Latest actual state | Sole continuation |
| --- | --- | --- |
| Full original-source raw/cache build **978318**, evc33 | RUNNING; **197 completed** at 10:01 UTC, including 16 reused caches; target 3,887 fitting studies / 720,851 DICOMs | Local owner **47540**; preserve exact verified prefix and partial buffers |
| Exact full-context build **978414**, evc33 | RUNNING; **181 completed** at 10:01 UTC; consumes verified raw cases and reuses the 16 contexts | Local owner **48108**; no new source download |
| Full teacher-pair fit **975297**, evc48 | RUNNING; all three pointwise epochs complete/private-verified; teacher-ranking epoch 1 complete/private-verified, later ranking work continues | Successor **46628**, remaining ranking publications and final CPU compilation only |
| Source-zero Kaggle epoch 3 **139786921 v1** | RUNNING; epochs 1 and 2 already private-verified; fixed-final target 1,458 updates on all 3,887 fitting studies | Successor **46627**, remaining preservation, qualification, one final entry, and outcome |
| Cached-attention diagnostic **977100**, evc48 | COMPLETED 0:0 in 8:43; actual reference/linear six-update recovery passed; sanitized original receipt collected read-only | Engineering only; no full production dispatch |
| Predictive two-T4 probe **139796378 v1** | Pushed once after CPU 978681 source admission and fresh 7,858.46 free VM seconds; actual result unobserved | Private offline 600-second engineering cap; preserves 3,000 seconds for source-zero serving; no competition entry |
| Full predictive B/C fitting | **Not dispatched** | Complete inputs, numerical decision, funding, and serving admission first |

Pointwise final model `da61c534…` was compiled by CPU 977528 but has no new T4 acceptance or score. Teacher-ranking epoch 1 is six-object private-verified by CPU 978495 at HF `de7c492abad0d614f4edeef135dcc5887af96d77`; do not repeat it. Source-zero epoch 2 is private-verified at HF `ee73db671130b6f850f551844e0a16a9ac80fcec`. [Epoch receipts](../initiative/research/teacher-pair-epoch-preservation-20261020-v1/); [source-zero receipts](../initiative/research/source-zero-full-T4-20261020-v1/); [successor controls](../initiative/research/reconnected-training-successors-20261020-v1/).

## Deadline and next decisions

The authenticated competition deadline is **October 22, 2026, 23:59 UTC**. Preserve the qualified baseline and latest final full-scored launch target **October 22, 12:00 UTC**.

1. Keep original training/publication/evaluation owners moving; collect actual fixed-final outcomes once.
2. Finish early two-T4 predictive inference timing, decode/parity and memory checks while CPU inputs build. A fresh-head runtime probe cannot accept trained-model parity or hidden-scale runtime.
3. Resolve C/C0's observed evidence/gradient instability with a bounded fitting-only mechanism test, then freeze the matched recipe. Do not turn engineering loss into an accuracy claim.
4. Admit complete fitting inputs and funded independent H100 B/C runs in parallel, with C0 as the specified ablation. Use measured throughput and deadline capacity, not the proposal's original unmeasured 12–16-hour estimate.
5. Original raw job 978318 has a 12-hour limit ending around **21:07 UTC**. Extension was denied; the healthy job was not paused or restarted. If it times out, reconcile its completed prefix and at most two private source buffers, then continue only missing work. Preserve the context prefix too. [Original wall-limit receipt](../initiative/research/predictive-full-cache-observation-20261020-v1/README.md).

## Preservation and limits

Latest verified small-source mirror: source `de165575`, CPU 978283, **67 objects / 352,183 bytes**, private HF `b6c14f013ea200fa573bad8ff25e4adb5fe24793`, all independently downloaded/size/SHA verified. [Exact receipt](../initiative/audits/hf-predictive-learning-20261020-v3/actual-private-verified.json). Later full-cache/context/warmup/cached-engine/T4 changes need a distinct mirror; do not claim the older receipt covers them. Earlier mirrors and failures remain in original audits and Git history.

Raw/cache funding is **320 GiB total**, 4 GiB owned transient space, 256 MiB controls/proofs, plus a separate 24 GiB future model/publication reservation. Exact contexts have **96 GiB total** including the paid 263 MB, plus 32 MiB controls. Every dispatch must retain all other liabilities, 20 GiB shared/node reserve and 5,000 shared files; no deletion credit. The latest observed shared quota was 1,026,232,780 / 1,048,576,000 kB and 994,617 / 1,000,000 files; refresh it before action.

[Execution continuation](codex-continuation.md) contains exact paths, owners and account controls. Research findings and tests are not substitutes for measured competition/held-out results.
