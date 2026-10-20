# Current RSNA Knee research state

Updated **October 20, 2026, 10:41 UTC**. This is the current consolidated snapshot; original receipts and live service state govern actions. Historical snapshots remain in Git history, including commit `36f6d01b`. The owner is AFK and authorizes independent work. First place and 0.97 remain unachieved.

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

**The startup correction now has actual evidence.** The original cached C/C0 experiment reached exam losses 9.79/9.90 and head-gradient norms 4,366/4,407. Distinct job **978709** completed 0:0 in **13:48**, source `39e315c6`: zero initial density means plus the actual 2,916-update warmup horizon kept the 24-update maximum exam losses at **B 0.8705, C 0.8219, C0 0.8220** and head norms at **7.62, 29.33, 29.41**. Complete model/AdamW/history/all-RNG and prediction-byte recovery passed for all three arms. [Actual receipt](../initiative/research/predictive-stable-warmup-20261020-v1/actual-accepted.json).

The [numerical decision](../initiative/research/predictive-full-fitting-20261020-v1/engineering-decision.json) admits this matched recipe for full fitting after complete inputs and resources. This is not accuracy: the bounded run reached LR 4.94e-6, below the planned 3e-5 peak. Inspect the first 200 full-cohort updates through peak LR. Correlated evidence, descriptor sufficiency, fixed input scaling and inherited exposure remain scientific risks. No model was selected using the four fitting engineering cases.

## Live work and ownership

| Work | Latest actual state | Sole continuation |
| --- | --- | --- |
| Full original-source raw/cache build **978318**, evc33 | RUNNING; **325 completed** at 10:38:47 UTC, including 16 reused caches; target 3,887 fitting studies / 720,851 DICOMs | Local owner **47540**; preserve exact verified prefix and partial buffers |
| Exact full-context build **978414**, evc33 | RUNNING; **311 completed** at 10:38:47 UTC; consumes verified raw cases and reuses the 16 contexts | Local owner **48108**; no new source download |
| Full teacher-pair fit **975297**, evc48 | RUNNING; all three pointwise epochs complete/private-verified; teacher-ranking epoch 1 complete/private-verified, later ranking work continues | Successor **46628**, remaining ranking publications and final CPU compilation only |
| Source-zero Kaggle epoch 3 **139786921 v1** | **COMPLETE**, all 1,458 updates; epoch 3 six-object private verification **978929**, serving compilation **978933** | Owner **46627** ended before account preparation on an obsolete helper pin; sole successor **57842** owns exact qualification **139803454 v1**, pushed once, status unreported |
| Cached-attention diagnostic **977100**, evc48 | COMPLETED 0:0 in 8:43; actual reference/linear six-update recovery passed; sanitized original receipt collected read-only | Engineering only; no full production dispatch |
| Predictive two-T4 v1 **139796378** / v2 **139802342 v1** | v1 timed out at 608.8 s before receipt; **v2 pushed once; UI Queued at 10:41 UTC**, 1,200-second cap, retained stage progress | Canonical v2 coordinator; no competition entry or predictive runtime result yet |
| Full predictive B/C/C0 fitting | **Not dispatched**; CPU **978976** accepted staged source, parent and identical shared initialization | Separate finite fitting owner being prepared; waits for complete 3,887-study context admission |
| Missing-only raw/context continuation | Local finite owner **57158** started from `5119b631`; **no new cluster allocation at launch** | Waits for exact original terminals; at most one raw and one context successor, with exclusive intents |

Pointwise final model `da61c534…` was compiled by CPU 977528 but has no new T4 acceptance or score. Teacher-ranking epoch 1 is six-object private-verified by CPU 978495 at HF `de7c492abad0d614f4edeef135dcc5887af96d77`; do not repeat it. Source-zero epochs 1–3 are all privately verified; final epoch 3 model `d5f583b8…` is at HF **`070bde15c4711dd7ac1c2ef42aaaa04be240fce6`**, with original full-state/checkpoint/1,458-update verification. Its compiled serving sources have **not** yet passed T4 qualification or received a hidden score. [Epoch receipts](../initiative/research/teacher-pair-epoch-preservation-20261020-v1/); [source-zero receipts](../initiative/research/source-zero-full-T4-20261020-v1/); [successor controls](../initiative/research/reconnected-training-successors-20261020-v1/).

## Deadline and next decisions

The authenticated competition deadline is **October 22, 2026, 23:59 UTC**. Preserve the qualified baseline and latest final full-scored launch target **October 22, 12:00 UTC**.

1. Keep original training/publication/evaluation owners moving; collect actual fixed-final outcomes once.
2. Collect the already pushed T4 v2 result once. Its retained measurements precede runtime classification. Fresh pre-push free quota was **5,376.85 VM seconds** against 1,200 for the probe and a retained 3,000 for source-zero final serving; unfinished training was retired only after actual source-zero completion. Public measurement cannot establish trained-model parity or hidden-scale runtime.
3. Use the accepted zero-mean, production-horizon matched recipe. CPU **978976** staged the exact worker/publisher and confirmed all three arms share initialization SHA `dbf69ff8…`, without fitting or reserving GPUs. Complete cache admission and fresh arm funding still precede full dispatch. Each full run has six epochs, 2,916 updates and seven-study final batches.
4. Admit complete fitting inputs and funded independent H100 B/C runs in parallel, with C0 as the specified ablation. Use measured throughput and deadline capacity, not the proposal's original unmeasured 12–16-hour estimate.
5. Original raw job 978318 has a 12-hour limit ending around **21:07 UTC**. Finite owner **57158** now handles the exact terminal-only continuation: completed original means collect only; actual TIMEOUT and empty queue permits one missing-only raw successor, followed by context only after its original stops. It preserves all prefixes and partial buffers, holds no allocation while waiting, and does not retry uncertain mutations. [Continuation owner and receipt](../initiative/research/predictive-full-cache-continuation-20261020-v1/README.md).

## Preservation and limits

Latest verified small-source mirror: source `96b61aeb`, CPU **978765**, **61 objects / 338,414 bytes**, private HF **`ebf7cef1685792afc7788417ff9ac888240acdef`**, all independently downloaded/size/SHA verified under the canonical lock. [Exact receipt](../initiative/audits/hf-predictive-cached-engine-20261020-v4/actual-private-verified.json). The later stability, T4 v2, full-fitting preparation, cache-owner launch and source-zero final milestone are **not yet mirrored**; their distinct local mirror preparation awaits the next committed snapshot and canonical lock. No original pixels, clinical IDs, weights or restricted external source bodies were mirrored. Earlier verified publications remain intact and must not be repeated.

Raw/cache funding is **320 GiB total**, 4 GiB owned transient space, 256 MiB controls/proofs, plus a separate 24 GiB future model/publication reservation. Exact contexts have **96 GiB total** including the paid 263 MB, plus 32 MiB controls. Every dispatch must retain all other liabilities, 20 GiB shared/node reserve and 5,000 shared files; no deletion credit. The active continuation owner retains **60 GiB +570 MiB other node liabilities**, plus remaining raw/context caps and scratch; its conservative shared future budget is **96 MiB /368 files** before the standing floor. CPU preparation separately binds its actual funding. Refresh actual quota before action; no historical free-space snapshot grants capacity.

[Execution continuation](codex-continuation.md) contains exact paths, owners and account controls. Research findings and tests are not substitutes for measured competition/held-out results.
