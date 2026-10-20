# Current RSNA Knee research state

Updated **October 20, 2026, 12:34 UTC**. This is the current consolidated snapshot; original receipts and live service state govern actions. Historical snapshots remain in Git history, including commit `36f6d01b`. The owner is AFK and authorizes independent work. First place and 0.97 remain unachieved.

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
| Source-zero fixed-final epoch 3 | **0.948** | **57435296** |

No original accuracy improvement over the external reproduction has been demonstrated. Cross-lineage epoch 3 ties our earlier 0.949 adaptation; the newly completed source-zero epoch 3 scored **0.948 without an error**. Both evaluator 21960 and source-zero owner 57842 finished and must not be restarted. [Actual source-zero outcome](../initiative/research/source-zero-final-serving-20261020-v2/actual-final-hidden-outcome.json); [other scored receipts](../initiative/research/S6-Kaggle-production-route-20261019-v1/).

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
| Original raw/cache build **978318**, evc33 | **FAILED 1:0 /2:23:25**; alias-only acquisition metadata lookup failed; **522 verified cases plus one intact orphan** retained | Old original/continuation owners stopped; preserve every prefix and source buffer |
| Original exact-context build **978414**, evc33 | **FAILED 1:0 /2:11:07** following raw producer failure; completed contexts retained | Old owner ended; successor must preserve original renderer provenance |
| Full teacher-pair fit **975297**, evc48 | **COMPLETED 0:0 /09:10:37**; all three pointwise and three ranking epochs privately verified; both final models CPU-compiled | Owner **46628 finished**; no repeat fitting, preservation or compilation |
| Source-zero Kaggle epoch 3 **139786921 v1** | Training/private preservation/qualification/public run complete; entry **57435296 COMPLETE, score 0.948, no error** | Owner **57842 finished**; no repeat phase or entry |
| Cached-attention diagnostic **977100**, evc48 | COMPLETED 0:0 in 8:43; actual reference/linear six-update recovery passed; sanitized original receipt collected read-only | Engineering only; no full production dispatch |
| Predictive two-T4 v1 **139796378** / v2 **139802342 v1** | v1 timed out at 608.8 s before receipt; **v2 COMPLETE /671.224 s**, measurements retained, original 550-s target missed | Read-only observer **59539** finished; no trained-model acceptance or competition entry |
| Full predictive B/C/C0 fitting | **No GPU dispatch**; v2 CPU **979667 COMPLETED 0:0**, same original initialization and mixed renderer provenance accepted | Sole finite owner **68792**, source `44cc559e`, waits for all 3,887 contexts and fresh admission; old owner **59538** remains stopped |
| Missing-only raw/context recovery | V3 raw **979664** and context **979665 RUNNING**; observer **979808** at **12:20 UTC** verified **602 raw/600 contexts of 3,887**, including the retained 522-case prefix | Sole owner **68411**, source `91224c24`; failed originals and v1/v2 attempts remain preserved |

Pointwise final `da61c534…` was compiled by CPU **977528**; its new [deployment controls](../initiative/research/pointwise-final-deployment-20261020-v1/) have privately verified **Dataset12731583v1** and **qualification139818636v1 RUNNING**; no entry yet. Teacher-ranking final epoch 3 /1,458 updates was privately verified by CPU **979757**, model `d478a502e6fd33be2ad9d781acd91d3d2b6b252629262031fba8282a91cf5bca`, HF **`60a64c2be5d59f60e458caca1b6f475f82a95fa7`**, then compiled by CPU **979767**. All six pointwise/ranking epochs are verified; owner **46628 finished**. Neither new final model has an actual Kaggle result. [Final ranking publication](../initiative/research/teacher-pair-epoch-preservation-20261020-v1/actual-teacher_rank-epoch-3.json); [compiler](../initiative/research/teacher-pair-final-serving-20261020-v2/actual-teacher_rank-compile.json). Source-zero final `d5f583b8…` is privately verified at HF `070bde15c4711dd7ac1c2ef42aaaa04be240fce6`; its 32-case/two-T4 qualification passed, public run completed in 16.370 seconds, and entry **57435296 scored 0.948** without error. Owner **57842 finished**; this establishes no new gain. [Exact qualification/public/outcome receipts](../initiative/research/source-zero-final-serving-20261020-v2/).

## Pointwise deployment: actual Kaggle qualification running

At October 20, 12:34 UTC, exact qualification **139818636 v1**, `godlydonuts/rsna-pointwise-e3-qual-c398324df327`, is RUNNING. Private model Dataset **12731583 v1** was created once; all seven payload objects were independently streamed back and size/SHA verified. Its fixed-final pointwise model remains `da61c534…`, original epoch 3/1,458 updates. [Exact pushed receipt](../initiative/research/pointwise-final-deployment-20261020-v1/actual-qualification-pushed.json); [Dataset verification](../initiative/research/pointwise-final-deployment-20261020-v1/actual-dataset-verified.json).

Original owner 71909 stopped after a read-only Dataset-status lookup failed immediately after the successful creation response. Read-only CPU 979849 confirmed the original private version1 was READY; no second upload occurred. Distinct finite continuation **72266**, source `5270ab56`, uses frozen account controls `01cbe205` and the original canonical ledger/claim for the remaining qualification, public execution, one code entry and hidden outcome. Never restart the original owner or recreate its Dataset. The notebook has a 600-second qualification cap; the public exact-roster watchdog is 300 seconds and hidden execution keeps nine hours. Preserve 3,000 free VM seconds for predictive deployment. **No pointwise competition entry or score yet.**

## Deadline and next decisions

The authenticated competition deadline is **October 22, 2026, 23:59 UTC**. Preserve the qualified baseline and latest final full-scored launch target **October 22, 12:00 UTC**.

1. Deploy the completed pointwise final through its prepared private, exact-weight route while input recovery continues. It still requires private Dataset verification, actual both-T4 qualification, public acceptance and a single canonical entry; ranking final is also preserved/compiled. Do not restart completed owner46628 or republish any checkpoint.
2. Apply the completed [T4 v2 analysis](../initiative/research/predictive-T4-throughput-20261020-v2/actual-runtime-analysis.json): CPU preparation took 52.114 s for three studies versus 1.66–1.67 s warmed inference per GPU. Large uninstrumented source-discovery/model-load intervals preceded decoding; their exact attribution is not established. Memory checks passed (host 3.869 GiB; per-device peak 1.271 GiB), but the probe used fresh heads and does not qualify trained predictions or hidden-scale speed. Do not repeat it.
3. Keep the accepted zero-mean, production-horizon matched recipe: six epochs, 2,916 updates, seven-study final batches, and shared initialization SHA `dbf69ff8…`. CPU **979667 COMPLETED 0:0** for the independently reviewed [v2 successor](../initiative/research/predictive-full-fitting-20261020-v2/), source `44cc559e947560a635c34fee7b1f2a3b7be3a241`, reproducing original CPU978976 initialization while binding both renderer lineages. Sole owner **68792** is waiting for all 3,887 contexts, exact terminal proofs and fresh resource admission. No full predictive fit has begun. [Actual CPU preparation](../initiative/research/predictive-full-fitting-20261020-v2/actual-cpu-prepared.json).
4. Preserve all existing journals and avoid duplicate dispatch. Planned B/C/C0 use one H100/two CPUs/eight hours per arm on ordinary two-H100 evc33, with at most two concurrently. Inspect the first 200 updates through peak LR and preserve every sealed epoch. A successor needs its own exact control binding; do not restart stopped owner59538.
5. Let the accepted missing-only recovery continue. CPU **979537** proved the alias correction preserves prior successful tensors/priors; CPU **979619** proved exact original CSV bytes and admitted the 522-case prefix plus the orphan. Distinct v3 owner **68411**, source `91224c242c5758fcb25b71c27d0a0b920458ee6b`, now owns raw **979664** and context **979665**. Both remain RUNNING; observer **979808** at **12:20 UTC** verified **602/3,887 raw cases and 600/3,887 contexts**. Failed attempts **979573/979604** remain preserved; do not retry them. [Actual recovery progress](../initiative/research/parallel-serving-account-20261020-v1/actual-input-progress.json); [original prefix admission](../initiative/research/predictive-alias-failure-20261020-v1/actual-prefix-byte-admission.json).

The predictive [offline serving path](../initiative/research/predictive-final-serving-20261020-v1/README.md) binds actual final epoch-6 exports, qualifies B/C/C0 sequentially in one two-T4 job, and compiles one exact-roster candidate per accepted arm. Prepared corrections replace global input-tree recursion with bounded mount discovery, reuse bitwise-equal descriptors across complementary masks, and apply a 600-second watchdog only to the exact public roster. Synthetic CPU descriptor preparation was 1.428× faster; this is not a Kaggle speed claim. Prospective [deployment controls](../initiative/research/predictive-final-deployment-20261020-v1/) remain local and require final exports, new explicit storage funding, fresh VM budget and actual trained qualification. **No predictive final weights, trained T4 qualification, Dataset or entry exists.**

The [fixed-final evaluation decision](../initiative/research/predictive-full-fitting-20261020-v1/evaluation-decision.md) predeclares principal C−B and C−C0 contrasts at epoch 6. Exact inherited-parent exposure invalidates independent whole-model claims from the weak holdout; Gold58 also has historical exposure. Preserve protected cohorts. Three public rows and aggregate Kaggle scores provide no paired clinical confidence interval. The baseline remains 0.958; no new clinical or competition gain has been demonstrated.

## Preservation and limits

Latest verified small-source mirror: source **`17beffadba6e7574bd6da69dbac1de9992e2bd05`**, CPU **979704**, **68 objects /597,670 bytes**, private HF **`91ed4964850174a5c8f677b2a4f9af6e4e979a57`**, all independently re-downloaded/size/SHA verified under the canonical shared lock. [Exact v7 receipt](../initiative/audits/hf-predictive-alias-recovery-20261020-v7/actual-private-verified.json); [manifest](../initiative/audits/hf-predictive-alias-recovery-20261020-v7/manifest.json). This distinct delta after verified v6 preserves the alias fix and actual regression, failed preintent attempts and exact-byte admission, v3 recovery start, full-fitting v2 CPU preparation/owner controls, corresponding serving contracts, source-zero 0.948 outcome and committed guides. Private C0 handoff preparation is outside its scope. No MRI, reports, clinical IDs, private arrays/predictions, model weights or restricted external source bodies were included. Never repeat v7 or earlier verified prefixes; later edits and v7's own verification receipt are not retroactively included.

Raw/cache funding is **320 GiB total**, 4 GiB owned transient space, 256 MiB controls/proofs, plus existing model/publication obligations. Exact contexts have **96 GiB total**, plus 32 MiB controls. The active v3 recovery retains **62 GiB +594 MiB other node liabilities**, unpaid raw/context portions and the standing **20 GiB** node floor. Shared future bounds are **122 MiB /376 files**, retaining the **20 GiB /5,000-file** floor. The extra 1 MiB funds permanent canonical ledger growth; its eight transient shared files reuse the existing serialized claim scope. Prospective pointwise transport separately reserves 640 MiB on evc48. Existing paid bytes count once; original failures confer no deletion credit. Refresh actual quota immediately before each dispatch; old reservations or snapshots are not new allocation authority.

Authenticated read-only CPU **979733** at **12:07 UTC** observed **4,166.208664 free GPU-VM seconds**, no registered active GPU sessions, and **three submissions used/two available**. Prospective pointwise allocation is **600 seconds qualification +300 seconds exact-public watchdog**, retaining **3,000 seconds predictive reserve**; the hidden session stays nine hours. Shared quota remains **1,026,232,968/1,048,576,000 kB and 994,621/1,000,000 files**. Recheck actual capacity/allowance before action; no paid credits. [Actual account receipt](../initiative/research/parallel-serving-account-20261020-v1/actual-account.json).

[Execution continuation](codex-continuation.md) contains exact paths, owners and account controls. Research findings and tests are not substitutes for measured competition/held-out results.
