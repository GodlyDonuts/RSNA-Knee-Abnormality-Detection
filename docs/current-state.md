# Current RSNA Knee research state

Updated **October 20, 2026, 11:20 UTC**. This is the current consolidated snapshot; original receipts and live service state govern actions. Historical snapshots remain in Git history, including commit `36f6d01b`. The owner is AFK and authorizes independent work. First place and 0.97 remain unachieved.

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
| Full original-source raw/cache build **978318**, evc33 | RUNNING at 11:16 UTC, elapsed 2:08:33; **29,919,537,514 raw bytes** observed at 11:20; that observer did not return an exact raw count | Local owner **47540**; preserve exact verified prefix and partial buffers |
| Exact full-context build **978414**, evc33 | RUNNING at 11:16 UTC, elapsed 1:56:15; **464/3,887 contexts /8,126,255,514 bytes** at 11:20; reuses original inputs | Local owner **48108**; no new source download |
| Full teacher-pair fit **975297**, evc48 | RUNNING at 11:16 UTC, elapsed 8:15:50; all three pointwise epochs complete/private-verified; teacher-ranking epochs 1–2 complete/private-verified, final ranking work continues | Successor **46628**, remaining ranking publications and final CPU compilation only |
| Source-zero Kaggle epoch 3 **139786921 v1** | **COMPLETE**, all 1,458 updates; epoch 3 six-object private verification **978929**, serving compilation **978933** | Sole successor **57842** completed qualification **139803454 v1** and public serving **139808525 v1**, then submitted **57435296 once**; hidden outcome/score pending |
| Cached-attention diagnostic **977100**, evc48 | COMPLETED 0:0 in 8:43; actual reference/linear six-update recovery passed; sanitized original receipt collected read-only | Engineering only; no full production dispatch |
| Predictive two-T4 v1 **139796378** / v2 **139802342 v1** | v1 timed out at 608.8 s before receipt; **v2 COMPLETE /671.224 s**, measurements retained, original 550-s target missed | Read-only observer **59539** finished; no trained-model acceptance or competition entry |
| Full predictive B/C/C0 fitting | **Not dispatched**; CPU **978976** accepted staged source, parent and identical shared initialization | Finite owner **59538** live from `307a28b7`, unchanged science `b361ee40`; waits without GPU allocation for actual complete-input receipts |
| Missing-only raw/context continuation | Local finite owner **57158** started from `5119b631`; **no new cluster allocation at launch** | Waits for exact original terminals; at most one raw and one context successor, with exclusive intents |

Pointwise final model `da61c534…` was compiled by CPU 977528 but has no new T4 acceptance or score. Teacher-ranking epochs 1–2 are six-object private-verified. Latest epoch 2 /972 updates: CPU **979162**, model `515e45b3…`, HF **`70ddf1f62be885ea5bd789e3a8488b79954eb900`**. Do not repeat either publication. Source-zero epochs 1–3 are all privately verified; final epoch 3 model `d5f583b8…` is at HF **`070bde15c4711dd7ac1c2ef42aaaa04be240fce6`**, with original full-state/checkpoint/1,458-update verification. Qualification **139803454 v1** now passed all 32 original MRI cases on both T4s with exact graph/input equality and zero probability gap. Public serving **139808525 v1** completed in 16.370 s and its 3-row/13-column CSV was independently checked in RAM. Code entry **57435296** was accepted once; the hidden outcome and score remain unknown. [Exact qualification, public and entry receipts](../initiative/research/source-zero-final-serving-20261020-v2/). [Epoch receipts](../initiative/research/teacher-pair-epoch-preservation-20261020-v1/); [source-zero receipts](../initiative/research/source-zero-full-T4-20261020-v1/); [successor controls](../initiative/research/reconnected-training-successors-20261020-v1/).

## Deadline and next decisions

The authenticated competition deadline is **October 22, 2026, 23:59 UTC**. Preserve the qualified baseline and latest final full-scored launch target **October 22, 12:00 UTC**.

1. Keep original training/publication/evaluation owners moving; collect actual fixed-final outcomes once.
2. Apply the completed [T4 v2 analysis](../initiative/research/predictive-T4-throughput-20261020-v2/actual-runtime-analysis.json): CPU preparation took 52.114 s for three studies versus 1.66–1.67 s warmed inference per GPU. Large uninstrumented source-discovery/model-load intervals preceded decoding; their exact attribution is not established. Memory checks passed (host 3.869 GiB; per-device peak 1.271 GiB), but the probe used fresh heads and does not qualify trained predictions or hidden-scale speed. Do not repeat it.
3. Use the accepted zero-mean, production-horizon matched recipe. CPU **978976** staged the exact worker/publisher and confirmed all three arms share initialization SHA `dbf69ff8…`, without fitting or reserving GPUs. Finite owner **59538** now waits for actual raw/context completion, then admits inputs once and dispatches B/C/C0 once on ordinary two-H100 evc33, one H100/two CPUs/eight hours per arm. Complete cache admission and fresh node/shared quota checks still precede GPU dispatch. Each run has six epochs, 2,916 updates and seven-study final batches; the owner preserves sealed epochs and records the first 200 updates and fixed-final outcomes. [Actual owner start](../initiative/research/predictive-full-fitting-20261020-v1/actual-owner-started.json).
4. Preserve the sole fitting owner and its journals; do not launch duplicate runs. B/C/C0 share the same admitted initialization and inputs. Physical evc33 capacity permits at most two concurrent arms. The owner ends by October 22, 09:00 UTC without cancelling healthy children; any uncertain mutation requires read-only reconciliation.
5. Original raw job 978318 has a 12-hour limit ending around **21:07 UTC**. Finite owner **57158** now handles the exact terminal-only continuation: completed original means collect only; actual TIMEOUT and empty queue permits one missing-only raw successor, followed by context only after its original stops. It preserves all prefixes and partial buffers, holds no allocation while waiting, and does not retry uncertain mutations. [Continuation owner and receipt](../initiative/research/predictive-full-cache-continuation-20261020-v1/README.md).

The predictive [offline serving path](../initiative/research/predictive-final-serving-20261020-v1/README.md) binds actual final epoch-6 exports, qualifies B/C/C0 sequentially in one two-T4 job, and compiles one exact-roster candidate per accepted arm. Prepared corrections replace global input-tree recursion with bounded mount discovery, reuse bitwise-equal descriptors across complementary masks, and apply a 600-second watchdog only to the exact public roster. Synthetic CPU descriptor preparation was 1.428× faster; this is not a Kaggle speed claim. Prospective [deployment controls](../initiative/research/predictive-final-deployment-20261020-v1/) remain local and require final exports, new explicit storage funding, fresh VM budget and actual trained qualification. **No predictive final weights, trained T4 qualification, Dataset or entry exists.**

The [fixed-final evaluation decision](../initiative/research/predictive-full-fitting-20261020-v1/evaluation-decision.md) predeclares principal C−B and C−C0 contrasts at epoch 6. Exact inherited-parent exposure invalidates independent whole-model claims from the weak holdout; Gold58 also has historical exposure. Preserve protected cohorts. Three public rows and aggregate Kaggle scores provide no paired clinical confidence interval. The baseline remains 0.958; no new clinical or competition gain has been demonstrated.

## Preservation and limits

Latest verified small-source mirror: source **`e6862ab2`**, CPU **979171**, **73 objects /569,591 bytes**, private HF **`71cbced934556179c3ca1ef56968b140d14df208`**, all independently downloaded/size/SHA verified under the canonical lock. [Exact receipt](../initiative/audits/hf-predictive-full-preparation-20261020-v5/actual-private-verified.json). It covers stability, T4 v2 preparation, full-fitting/cache owners, source-zero controls and predictive serving through that exact commit. The later T4 terminal analysis, source-zero qualification/entry, evaluation decision and newest serving/deployment edits are not retroactively covered. The later teacher-ranking epoch-2 checkpoint is separately six-object private-verified at its revision above. No original pixels, clinical IDs, weights or restricted external source bodies entered this small-source mirror. Never repeat v5 or earlier verified prefixes.

Raw/cache funding is **320 GiB total**, 4 GiB owned transient space, 256 MiB controls/proofs, plus a separate 24 GiB future model/publication reservation. Exact contexts have **96 GiB total** including the paid 263 MB, plus 32 MiB controls. Every dispatch must retain all other liabilities, 20 GiB shared/node reserve and 5,000 shared files; no deletion credit. The frozen cache-continuation owner retains its original 60 GiB +570 MiB node and 96 MiB/368-file shared bounds. The newly live fitting owner conservatively retains **60 GiB +574 MiB**, remaining raw/context bounds and the standing floor; its fresh admission/dispatch/publication checks require **120 MiB /368 shared files**, including an additional 24 MiB for three combined GPU stdout/stderr allowances. Paid model bytes are counted once. CPU preparation separately binds its actual earlier funding. Refresh actual quota before action; no historical free-space snapshot grants capacity.

Read-only CPU979373 at11:20UTC observed node free510,672,789,504bytes; shared usage1,026,232,968/1,048,576,000kB and994,621/1,000,000files. This is a dated observation, not new allocation authority. [Actual input/storage receipt](../initiative/research/predictive-full-fitting-20261020-v1/actual-input-progress-1120.json).

[Execution continuation](codex-continuation.md) contains exact paths, owners and account controls. Research findings and tests are not substitutes for measured competition/held-out results.
