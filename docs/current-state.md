# Current RSNA Knee research state

Updated October 20, 2026, 03:06 UTC. Goal active; first place and 0.97 remain unachieved. Original receipts and live service state supersede this snapshot.

## Actual competition results

| Model | Kaggle score | Entry |
| --- | --- | --- |
| External R6 reproduction | **0.958** | 57333150 |
| Released S6 reproduction | 0.950 | 57300990 |
| Own whole94 full encoder | 0.949 | 57336190 |
| Own influence IID16 full encoder | 0.949 | 57332456 |
| Own paired / higher-LR adaptations | 0.949 / 0.949 | 57392062 / 57394237 |
| Own uniform / source-label adaptations | 0.947 / 0.947 | 57329796 / 57331631 |
| R6 with own paired member | 0.957 | 57397292 |

No original improvement over the public reproduction is established. Engineering checks, training loss and more parameters are not accuracy. Public teacher/base exposure is not independent OOF evidence.

## Changed-supervision full-model experiment

All 3,887 fitting studies, all 73,136,096 parameters, effective batch eight, original labels/masks/proposals/LRs, fixed three epochs / 1,458 updates. Cross-lineage equal-rank teacher supervision is the changed variable. Scientific source cce4ae78b63ec80b54b806ea2e40d34c95ae15df.

- E1 kernel 139699006 v1 COMPLETE: 486 updates, 4,016.6867 seconds. Model SHA 4a7a9f9b59493d45b2820a9619f40fe4c60aaa77a5f5fcb07699864cc5f789df, 292,873,139 bytes. CPU974836 privately published all six objects once; independent full download/size/SHA passed at HF3d69bf08b182be2d7a6e1fb0389dd8f76b2370c5. CPU974845 delivered its exact publication pointer. Never repeat.
- E2 kernel **139710942 v1 COMPLETE972updates**. CPU975416 privately preserved all six objects with independent full download/size/SHA verification at **HFd2afe8cedcfef8e92f7e06569571bd210d855030**. Model SHA599a47c9df17291630484d426a256f66a91a1968c1ac394d5efbe8dff02fada1. Exact optimizer/RNG continuation, no new recipe; never repeat publication.
- **E3 kernel139719974 v1 RUNNING**, exact sourcec870ded987b8e006abc36f65ed9bfddd64cf9ceb6ed71148429aa3995e3f2410/runtime/all inputs accepted by CPU975698. Original SaveKernel975430 created only a draft; exact version-zero/source-plus-editor-header/offline/T4x2/zero-reservation proof allowed one distinct first executable commit975697. Its response and remote source/runtime were verified before a local initialization-status KeyError; read-only canonical acceptance then reconciled the existing version and released the old claim. No GPU work duplicated. See actual-epoch3-reconciliation-clarification.json; raw old phase receipts are preserved.
- Sole new E3 preservation/compiler owner **25232**, source20ef4a01, waits on this exact version, then publishes/re-download/hash-verifies once and compiles serving. It launches no new training or competition entry. Earlier owners4836/4837 ended; exact old compiler receipt archived before truthful current-owner pointer handoff.

Original publication974799 failed before model/HF mutation at an unnecessarily high file guard. Read-only reconciliation974818 proved no mutation; the distinct explicit recovery974836 retained the funded metadata/file and byte reserves. No originals deleted. Earlier owners702/1613/2264/2861/4838 are ended; never revive them.

Receipts: `initiative/research/S6-Kaggle-production-route-20261019-v1/`. Old influence control0.949 remains the matched-recipe reference; do not retrain it unchanged. The new E1 score is0.948; no final E3 score exists.

## Kaggle evaluation: first new checkpoint submitted

Corrected qualification **139716471 v1 COMPLETE**: real32MRI acceptance on each T4, identical both-view inputs/learned graph, maximum probability difference0.0 against the training graph, peak2017322496bytes per GPU. Corrected public-example notebook **139717947 v1 COMPLETE** produced a verified3study/13column CSV with no decoding/empty-study failures. This is deployment acceptance, not a hidden score.

**Competition entry57418131 accepted ONCE**, exact notebook `godlydonuts/rsna-s6-target-e1-offline-373b46a543cc` v1. The public example CSV was not uploaded. **Actual hidden score0.948, COMPLETE, no error.** This is below the older own influence0.949 and does not establish a supervision gain. Live evaluation owner21960 obtained this outcome after the E3 claim was reconciled; it now waits for actual E3 private preservation/serving compilation before qualification and the fixed-final comparison. Never repeat entry57418131.

Original unpinned qualification139711261 was stopped once and CANCEL_ACKNOWLEDGED; preserve its source/version and never submit or revive it.

Correct canonical runtime:
`gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`

The API initially reported negative unreserved time because the unusable live session carried a large reservation. Its acknowledged cancellation released the reservation. No inference from that earlier negative balance is needed: every subsequent push rechecks authoritative quota/capacity. No paid credits authorized or used by policy. Account pay-to-scale being enabled is not permission to spend.

Controller `artifacts/S6-twoT4-runtime64-reconciled-20261020-v1/` permits E1/E3 only, at most two code entries, actual qualification/public CSV and fresh allowance, actual E1 hidden outcome before E3. Preserve9h hidden limit and600s public-three-study watchdog. Never revive5590 or old unpinned route; never upload the example CSV.

## Ranking mechanism: actual GPU recovery accepted, full comparison not started

Original974708 on evc45 was PREEMPTED after2:01; it completed the pointwise reference/stop1. Distinct normal continuation **974940 COMPLETED0:0 in4:15**, independently collected by974967. Only four missing processes ran; no reference repeated.

Both pointwise and teacher-ranking arms passed complete model/base/AdamW/LR/cursor/allRNG/scaler/history/snapshot equality and exact16x12 two-view full94 MRI prediction bytes, uninterrupted versus resumed. Four updates per arm, all73M parameters, same initial state and different resulting predictions. This is real numerical acceptance, not clinical gain. Receipt SHA1bc918297634a5bc75e3c789cad02fff278a3a8771f54c4d5b40b9f1ea37cb97; see `initiative/research/teacher-pair-earlier45-20261020-v1/actual-normal-continuation-terminal.json`.

Full12-file/38,873,662,650-byte inputs already copied and independently SHA verified once to45. Cached48 CPU975246 proved the original numerical modules and original pixel-file proofs match45; no bulk copy repeated. CPU975286 admitted all3887 inputs and six representative both-view checks, then CPU975296 submitted **GPU975297 ONCE on evc48**. Actual975734 observation now shows RUNNING with15 finite pointwise updates, loss0.5377578/gradient norm2.58435 and empty stderr. Ordinary2H100 node, oneH100/twoCPU/24GiB/12h, sequential matched pointwise/ranking fresh fits. Scheduler estimate was earlier than45, not guaranteed. Old45 job975129 was cancelled only while PENDING/zero runtime after actual cached48 admission; do not revive it. Read `initiative/research/teacher-pair-cached48-20261020-v1/`.

Both full3epoch/1458update objectives use the same original73M initial state, two-view compute, original labels/masks/teachers/proposals/LRs. The old one-view0.949 is not the causal control. The pointwise arm has started; the teacher-ranking arm follows sequentially. Source7f75fd46 is frozen; the faster reader is not part of this queued payload.

## Measured input bottleneck and execution opportunity

CPU974909 COMPLETED0:0: original renderer built94 windows then selected16. AST-derived selected rendering produced bit-identical tensors on32 actual MRI studies /64 plain-mirrored pairs, preserving duplicate/order behavior. Renderer time8.74364s versus1.24940s, **6.99825x renderer-only speedup**. No model/GPU/optimizer/clinical operations occurred. This is neither end-to-end throughput nor accuracy.

Actual renderer receipt: `initiative/research/selected-window-rendering-20261020-v1/actual-CPU-measured.json`.

**Complete private input-integrity coverage now accepted:** original CPU975028 timed out after a flushed2369study prefix; continuation975384 completed remaining rendering but failed at receipt line162 because `old` was overwritten by source bytes. Distinct finalizer **975656** verified the terminal/source bindings and all populated windows, sealed the retained manifest without any new MRI decode/render calls. Actual coverage3887studies ×2views ×94windows,7774 original full-tensor checks; private manifest23384320bytes, SHA85c2fbb342c4018ba77a5027a22b266b12c4ff59a766fd2af517c7c9ff30763f. No pixel signatures exported. Original failed receipts preserved.

Actual CPU975724 admitted the isolated selected reader against the complete manifest: six full-cohort studies/twelve plain-mirrored pairs,16 selected windows each, exact image/label/mask/teacher equality, unchanged CPU RNG. Original full-case loading5.43978s versus selected1.92892s, **2.8201× including decoding/finite/integrity checks**, alternating measurement order. This is a small actual CPU benchmark, not GPU/end-to-end throughput. The faster reader is **not deployed into the already running comparison**. Next admission should establish exact original labels/teacher/proposals/RNG and selected input bytes, then measure end-to-end loading. Do not alter the queued ranking source in place or rerun the completed integrity pass.

## Preservation and next actions

Distinct sourceecab19d8 milestone privately mirrored at HF83e2203c730b9b3bdd9f21c2aa8ebb013bac1d29:31objects/134789bytes independently downloaded/size/SHA verified via974893. Original11272518 milestone HF75c24179d9865e1e52fc656f6dd47465c916e03d:19objects/71961bytes. Never repeat these or earlier verified mirrors. Distinct source **bafd73ec** and91 selected small source/aggregate objects (259553bytes) are now privately mirrored and independently re-downloaded/size/SHA verified at **HFcd200afd06324acce85fa8c3961c40cbdc24232d**, CPU975743, under the canonical shared lock. This includes E1 actual0.948, E2 preservation, accepted E3 draft reconciliation, cached48 dispatch/progress and full manifest/selected-reader admission; no clinical records, private signatures or weights. Never repeat this mirror.

1. Follow actual E3 version139719974 under owner25232 and preserve the final checkpoint once; no further training push.
2. E1 actual0.948 is negative evidence; finish the predeclared final E3 comparison through existing evaluation owner21960.
3. Follow original975297 finite training, checkpoint and terminal outcomes; never revive cancelled975129.
4. Admit the selected-window reader using the completed private manifest before any explicit pending-only source handoff.

Full QA scope:5061passed/6skipped for the recorded base plus selected-reader bytes; later host-port/prefix9checks and finalizer2checks passed separately. Counts are engineering evidence, not accuracy.

[Continuation](codex-continuation.md) records operational bindings. Earlier snapshots and original receipts remain in Git history.
