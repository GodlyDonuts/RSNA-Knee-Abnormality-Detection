# Corrective data preparation, 2026-09-20

The owner has expanded the mission: after data readiness, research and deliver a
competitive training strategy and prioritized experiment plan. Both phases are
required; see [mission-and-training-plan.md](mission-and-training-plan.md).

## Current local release: September 23

The owner's one-off epoch-1 Kaggle baseline and private preservation of **every**
completed training epoch supersede the earlier monitoring-only cadence. Primary
Slurm job **841107** is the ten-epoch, two-H100/four-training-CPU fold-0
run on evc28. Slurm completed `0:0` after all ten epoch boundaries with unchanged
accepted masks, folds, official DINOv3-L weights and training source Git
`b74060b98f01bff1b924070ccba5ce44dd002f40`. Its immutable epoch-2
snapshot SHA is `f56c0d64519fbf2fde5433ef4ff3dcc59d8f01394d80515ebf5cebf97726fa02`.
CPU-only watcher **841134** shares evc28 and captured completed epoch 3 before
the rolling recovery file advanced; its verified snapshot SHA is
`caaa821440a1e880ba95744223ac693d30a5f8d3ad62a4d0a0f29d46e2b494f4`.
The watcher captured epoch 5 with SHA
`8ea11a5b513e369b1cb7e26861b3135b657b3d4b85c02c57ce9a8fc963ef1dd4`.
It captured odd epoch 7 with SHA
`998530993c80d4cb1c632ba9aa5b4ff175d862590b352c9caea8c30fccb05fd8`.
It captured epoch 9 with checkpoint SHA
`41cdb0f8e618b76bdf40962ba43f7b9751204afbb5be015932a3f9b0fda44707`
and completed `0:0`. Even epochs 4/6/8/10 are trainer-scheduled snapshots.
All ten epochs were separately committed to private `GodlyDonuts/RSNA-Data`
and fully downloaded back for SHA/size verification. The exact per-epoch
revisions and small receipts are in the [preservation audit](../initiative/audits/epoch1-baseline-and-epoch-preservation-v1/);
epoch 10 is pinned to revision `8e41da9c831af6bf777daf48b81109257f7e4b82`.
The completed fixed-final-epoch result reports **0.7783102649** fold-0 macro
AUC across twelve targets on 1,156 accepted-train outer-fold studies, with
2,810 optimizer updates. The final prediction file and result are archived in
the preservation audit and have matching SHA-256/provenance. This is an internal
one-fold report-label metric; the Kaggle score remains pending.

The rolling first-epoch checkpoint had already been replaced when the owner
requested an epoch-1 baseline. Separate two-H100 job **841126** resumed the
verified update-26 source and reached the completed `(epoch=1,next_update=0,
step=281)` checkpoint. Torchrun propagated the trainer's deliberate exit 75 as
wrapper failure `1:0`; this did **not** change primary job 841107. CPU-only
job **841157** verified the saved 1,215,174,821-byte checkpoint SHA
`01d56458c54f4f2811ecfd5b997e99baec765e850345001960b953304ff3d28e`
and all 255 replayed update records against the primary run. CPU job **841160**
then hard-linked immutable epoch-1 weights and wrote a bound one-off sidecar.
The private epoch-1 checkpoint was fully downloaded and verified from HF
revision `94d95aa232f41e2859dd56e5134901efd066a406`; inspect its
per-epoch journal before any attempted repeat. The normal
2/4/6/8/10 Kaggle cadence remains unchanged; the explicit baseline is one
additional submission, so six total entries may span multiple quota days.
Epoch-1 Kaggle competition submission **56482890** was made once, after a fresh
live allowance of five; its public score is pending.
The [epoch-1 Kaggle deployment audit](../initiative/audits/kaggle-epoch1-baseline-v1/)
records a private version-1 Dataset (ID 12145591) whose 46 payload files were
fully downloaded and rehashed. The first notebook slug was absent after an
uncertain push and an HTTP 409 recovery. An exact-source operation under a new
slug was accepted as private offline T4 notebook version 1. It completed its
MRI-only three-study public execution and passed a version-bound receipt before
submission. The canonical single ledger preserves both failed intents, the
migration and the successful code submission. This public example execution
does not establish hidden-test runtime or a leaderboard score. The epoch-2
private Dataset was fully downloaded and rehashed; its distinct version-1
notebook completed a version-bound MRI-only public-example receipt and was
submitted once as Kaggle entry **56483451** after a live allowance of four;
the leaderboard score is pending. Epoch 4's private Dataset was created once, fully
downloaded and rehashed in a second read-only reconciliation after the first
encountered a Kaggle API error; its dedicated private notebook version 1 was
then pushed once. Its MRI-only public-example receipt passed, and the exact
version was submitted once as Kaggle entry **56483949** after a live allowance
of two. Epoch 6's private Dataset passed full byte verification; its
dedicated notebook version 1 completed a valid MRI-only receipt and was
submitted once as Kaggle entry **56483731** after a live allowance of three.
The epoch-1, epoch-2, epoch-4 and epoch-6 scores are pending. Epoch 8's private
Dataset passed full byte verification, but its first notebook push is recorded
as uncertain after a Kaggle server error. An exact search associates an unnamed
private placeholder with the original slug; do not repeat that mutation. A
second notebook identity using the same hard-linked Dataset and source was
pushed once as private offline version 1 and is running its public-example
execution. Epoch 10's private Dataset appeared as version 1,
ID 12146648; all 43 payload files were downloaded and hash-verified. Its first
notebook-push dispatch stopped before `SaveKernel` on the unnamed listing item.
A reviewed exact-search guard with a known-notebook positive control then
proved the epoch-10 slug absent, and one CPU-only recovery pushed the unchanged
private offline epoch-10 notebook as version 1. The ledger records `pushed`;
its Kaggle public-example execution is running. Neither epoch has a competition
submission yet. The epoch-10 checkpoint is also fully verified on private
Hugging Face. See the
[epoch-2 audit](../initiative/audits/kaggle-epoch2-v1/),
[epoch-4 audit](../initiative/audits/kaggle-epoch4-v1/),
[epoch-6 audit](../initiative/audits/kaggle-epoch6-v1/) and
[epoch-8 audit](../initiative/audits/kaggle-epoch8-v1/) and
[epoch-10 audit](../initiative/audits/kaggle-epoch10-v1/).

The accepted Kaggle wheelhouse was staged and rehashed in CPU job **841116**.
An isolated pinned Kaggle management environment was installed in CPU job
841121; its import-only probe encountered the SDK's automatic authentication
on import. Recovery job **841131** completed `0:0` using a nonsecret offline
legacy placeholder solely for the symbol import check, with `pip check` and
exact versions. The separate offline Hugging Face upload client passed CPU
job **841139**. Tokens remain in local ignored `.env` and are streamed through
SSH/stdin only during the bounded network phase. The epoch publisher's durable
intents and receipts are under Newton `hf-epoch-publisher-v1/epochs/`.

The user stopped the subagents after reconnecting. Root alone owns further
remote dispatches, transfers and publication; do not restart agents. Preserve
their completed originals and inspect receipts before taking over operations.
Pilot 839216 failed on evc44 during its early driver check (exit 2, 1:59),
before any optimizer update. Keep evc44 out of subsequent training allocations
until allocated-device health is established. After a fresh resource inventory,
root rerouted the **same pending** successor job 839321 to free evc49. Its strict
driver and full private-header guards passed. Job 839321 completed 0:0 on two
H100s/four CPUs, with 26 real optimizer updates over two diagnostic epochs,
same-world-size interruption/resume, elementwise identical final checkpoint
state and byte-identical outer prediction CSVs. The complete original records
and exact timings are in `initiative/audits/dinov3-h100-pilot-v6/`. Serialized
checkpoint files have different hashes; tensor and metadata content matched.
This is a 100-study diagnostic, with no Kaggle submission or competitive score.
The earlier actual Slurm attempts and preflight failures are preserved in
`initiative/audits/newton-slurm-signal-v1/` and
`initiative/audits/newton-slurm-signal-v2/`. The measured fix at Git
`c58090e7c7a238461a5ac69293b32b64b74c6721` passed CI 35799881539,
was byte-verified in private HF revision
`997a7758e2b529e0888a2097c2c511316236bd54`, and is the verified Newton
checkout HEAD. The checkout path `code-stopfix-1f486765` retained its historical
name when updated; rely on its verified HEAD and file hashes.

The real two-H100 Slurm stop job **840527** received one batch USR1 after its
first optimizer update and exited intentionally with code 75 after three
updates. Its supervisor verified the checkpoint stop and both worker signals.
The checkpoint was hash-verified and preserved by hard link. Separate resume
job **840537** completed 0:0 through 26 updates and 100 outer predictions.
One-CPU comparison job **840557** found the final checkpoint payload elementwise
identical to the uninterrupted pilot after excluding the expected changed
`competitive/train.py` source binding; outer prediction bytes also matched.
This closes the distributed recovery gate, not a Kaggle score. Exact receipts
are in `initiative/audits/newton-recovery-production-v1/`.

One-CPU probe **840559** confirmed evc49's XFS `/tmp` has over 850 GB free for
the full cache extraction. The first production wrapper, job 840560, exited in
nine seconds before training because a header receipt used an obsolete control
path. The corrected, exact-script-hash v5 ten-epoch fold-0 job **840562** was
submitted once and reached 22 epoch-0 optimizer updates on 2,247 eligible
optimization studies and 1,156 outer-fold studies. It then made no progress
for over ten minutes: one rank was blocked in CUDA stream synchronization with
its H100 busy while the other rank waited. A single batch USR1 requested a
checkpoint; both workers were signaled, but the 120-second grace expired
without one. Slurm reported `FAILED 2:0`. No epoch or checkpoint can be
resumed from this attempt. The initial evc49 hardware suspicion was provisional.

The exact same committed source, accepted masks/folds, official DINOv3-L weights
and production configuration were staged in a fresh v6 wrapper, excluding
evc38, evc44, evc49 and evc50. Replacement job **840578** was submitted once
and allocated evc40 with two H100s and four CPUs. Its allocated-device driver
probe passed, but it reproduced the **same update-22 loss and CUDA stall** on
evc40. All 22 non-time optimizer events matched job 840562 exactly. A single
batch USR1 again signaled both workers, but no checkpoint completed before the
120-second grace; job 840578 ended `FAILED 2:0`. This rules out a simple
evc49-only explanation. Do not relaunch the unchanged full-data job. CPU-only
job **840735** extracted all 98 selected cache members with one CPU and no GPU,
then failed its exact image-range assertion. One-CPU follow-up job **840800**
completed and found 20 pixels at 1.000000238 in the rank-1 third microbatch;
the other seven studies passed and no sampled pixels were nonfinite or negative.
This points to FP32 interpolation roundoff against the model's strict `[0,1]`
guard. The loader fix clips that roundoff. Job **840938** then verified all
eight studies on Newton with the committed fix: seven image hashes were
unchanged, the affected one now has maximum one, and Slurm completed `0:0`.
Bounded two-H100 job **840991** crossed update 23 and completed `0:0` at
update 26. Its 1,215,174,757-byte checkpoint SHA-256 is
`b50ae5591b7b5b02f36423676f1f7fcbedacd69c8d848172bee459e4ae8d044a`.
The first one-CPU proof job 841089 failed before proof creation because the
absolute script path omitted the pinned Python package. The narrowly repaired
one-CPU job **841091** completed `0:0`, decoded `(epoch=0,next_update=26,step=26)`
and hard-linked the original checkpoint and records. Two-H100/four-CPU job
**841094** failed at evc27's CUDA visibility preflight before training, leaving
the checkpoint and proof unchanged. The single retry **841107** allocated evc28
with evc27 excluded and uses the same proof. Check its
Slurm state, `run.json`, `invocation.json`, epoch sidecars and logs before
acting; do not submit a second continuation. The [fix audit](../initiative/audits/newton-interpolation-fix-v1/)
holds the immutable small proof and dispatch records. The completed epochs and
private preservation results are recorded at the top of this guide; the one
epoch-1 baseline submission is described there.
The prepared Kaggle notebook now runs exact trained inference in 32-study
chunks, checking each receipt and cleaning each temporary image cache. This
bounds test-cache space without changing the pinned `rsna/**/*.py` trainer
or model. Synthetic three-study checkpoint equivalence and local bundle tests
pass; hidden-test runtime and scores are still unmeasured. Stage the new exact
Kaggle code projection separately from the active b740 training checkout after
its packaging commit is verified. Do not use the older monolithic projection
for a newly scheduled epoch bundle.
The exact packaging source projection is verified at
`/home/sa305415/rsna-knee/kaggle/code-projections/c220cc2-b740-v1/source`;
its receipt and private Hugging Face revision are under
`initiative/audits/kaggle-chunked-inference-v1/`. It binds package Git
`c220cc29d32dcb7c1ad008e86ed0975feb245289` to unchanged training Git
`b74060b98f01bff1b924070ccba5ce44dd002f40`. Keep Kaggle coordination on
the one canonical ledger when an eligible epoch snapshot exists.
Both failure audits are in
`initiative/audits/newton-production-stall-v1/` and
`initiative/audits/newton-repeatable-stall-v1/`. The canonical Kaggle ledger
remains untouched because no eligible snapshot exists.

Newton Lustre file quota had reached 1,016,723 files while a separate
interactive one-GPU Bash job 840217 was active. That job ended; the file count
fell to 981,978, under the 1,000,000 soft limit, without deleting or sparsifying
the older checkout. The verified full-cache archive is unaffected. Recheck
quota before further staging or extraction.

The full-cache byte-resume operation is **complete and verified**. The v3 plan
rechecked every one of the 48,772 source files and the interrupted
75,434,622,976-byte remote prefix. Its `completed.json` agrees exactly with the
independent remote full-file/TAR receipt: 82,834,493,440 bytes, SHA
`6acf433e16c4832503103dd045943bbafa01dfb65fde8f56d93911fe1caac338`,
48,772 members and 82,794,366,615 unpacked bytes. See
`initiative/audits/full-cache-newton-stage-v1/` for the immutable small records.
Do not start another transfer. The archive remains on private Newton storage;
its membership will be rechecked when extracted to node-local scratch. The
unchanged production jobs 840562 and 840578 both stalled at update 22 and are
terminal without a checkpoint. The CPU probes 840735 and 840800 isolated an
FP32 interpolation range overshoot in the exact next batch. The fixed loader
passed CPU job 840938; bounded GPU job 840991 is the current gate before full
training can restart. See the [fix audit](../initiative/audits/newton-interpolation-fix-v1/).

The final `initiative/releases/report-supervision-v1/` export and real runtime
acceptance passed. Start with [data-readiness-report.md](data-readiness-report.md),
[training-strategy.md](training-strategy.md), and the final receipt. The exact
manifest is `2bf49fd2a8ae0b7c5af7e8c105fc3f2337845a92654d136fe67bc41a1e52fae7`;
receipt `6453a8ff31373c40556dc857e15fa0025f122195388a2b46c81ecd9cff0b8833`.
The accepted semantic registry is `d7e513e444b9639b0797f6e9034a075256d9b32d33abed41964eabfe32210f36`.
The 569-correction registry is now `7c5d0e294cd770c2e96c1ecc10a277de293f762ae50f719935dda170f31b11b1`;
only its three QA references changed. All whole correction objects are unchanged.

The three global QA gates are resolved against the actual proposal, distinct
assessment, root decision and full-cell hold ledger. Core acceptance contains
573 passing tests. Final runtime acceptance took 235.7 seconds and five commands:
32 training/32 weak-validation diagnostic studies, two epochs/16 steps, complete
checkpoint equality after resume, zero masked gradients, 58 gold predictions and
three exact-template predictions. The full release is checked before any subset
is selected; this is operability evidence, not a competitive model.

Publication and the expanded modeling plan are complete. Use Git `f41879a3` and
private HF `a177562d5f49458e287752b94f085dcb061f6495` under `releases/report-supervision-v1/`.
All 19 promoted objects passed full immutable byte verification; the separate
complete-tree checkpoint `c0b2c8b041120ae6d27891145f1a19e48a6d498f` has two actual restoration proofs
covering 11,145 files. The publication record is
`initiative/provenance/report-supervision-v1-publication.json`.

The owner has selected DINOv3 and authorized one initial two-H100 Newton node
with four CPUs total, up to a second node for useful independent work, and Kaggle
submissions every two epochs within the account-wide five-per-day limit. Read
[competitive-training.md](competitive-training.md), [newton-launch.md](newton-launch.md)
and [kaggle-submissions.md](kaggle-submissions.md). The new competitive package
implements training-only three-fold development, offline pretrained encoders,
target-specific slice/series attention, exact distributed masked loss, recovery,
and scheduled checkpoint packaging. It preserves the accepted release unchanged.

SSH access to Newton and Stokes works. Use account `skattel`, partition `normal`,
and `gpu:nvidia_h100_pcie:2`; credentials live only in ignored local `.env` files.
DINOv3-L access was approved, and authenticated official config and license bytes
are verified in `configs/competitive/weights/dinov3-large.json`.
The official pretrained DINOv3-L completed the two-H100 100-study pilot
described above. At that earlier checkpoint, no competitive ten-epoch run or
submission had completed; the current status is at the top of this guide.
The official config SHA is
`135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5`.
The 1,212,559,808-byte weights passed full SHA verification on Newton in CPU-only
job 838408. The pilot cache also passed streaming and independent remote archive
verification: 2,248 files / 4,077,105,588 unpacked bytes; archive SHA
`8d50e027f66e91beb6afa2b56877b8da59059f6d43e7c13aa0103059d447d715`.
Use `/home/sa305415/rsna-knee/encoder-stage/run-configs/pilot100-fold0.json`
and the archive under the private `cache/` directory. The full 82.8 GB cache
transfer was interrupted; use the byte-bound continuation described above.
The offline Python 3.12 / Torch 2.14 environment
passed installation, relocation and CPU acceptance in jobs 838473 and 838493.
The original Newton deployment used `06816e6b698901bd5148d47a3cdf73252c5a3a89`;
its source, environment and publication receipts are preserved in
`initiative/audits/dinov3-deployment-v1/`. The corrected deployment is the clean
detached commit `e7b08625854de692b9266ccfc25c90fc22169f58`. CPU job 838857 checked
all 11,538 tracked files against their Git blobs and all 75 explicit source pins.
The earlier inventory check counted 23 generated bytecode files; reconciliation
preserved those files before removing the unchanged generated copies. The exact
source proof is in `initiative/audits/newton-source-e7b-verification-v1/`.
Exact-head CI 35766826230 passed and the private HF overlay is byte-verified at
`6f30d9def9b411c3adf2517ec683a22b2adf1bb4`.

The first two-H100 pilot, job 838502 on evc38, stopped at the GPU preflight after
41 seconds, before training. Slurm allocated two GPUs, the driver inventory
reported one, and Torch could not initialize CUDA. A separate bounded probe,
838579 on evc37, passed tiny tensor allocation on both H100s with the same
environment, in both batch and Slurm-step execution. Both original and PCI bus
device ordering worked without changing the scheduler's visible devices. This
points to an evc38-specific issue; its underlying cause is unproven. Exclude
evc38 from our jobs until verified healthy. The recovery-pilot retry is job
838580 on evc37, with the same code/configuration and a fresh output directory.
That retry passed CUDA/NCCL preflight but stopped before training: the official
checkpoint uses `layer.N.*`, while Transformers 5.6.1 expects `model.layer.N.*`.
The complete strict-load failure is preserved. A narrow mapping for the exact
reviewed source now follows the library's own conversion while keeping strict
loading and original weight hashes. CPU job 838689 verified all 415 official
tensors, a finite 224-pixel forward and bit-exact checkpoint-only restoration.
All 905 local tests, Ruff and mypy passed. See
`initiative/audits/dinov3-official-state-compatibility-v1/`. Deploy this corrected
committed source before the next GPU recovery pilot; the older `06816e6` checkout
does not contain the fix. No epoch time or competitive score has been measured
at this checkpoint. Pilot 838868 was submitted once with the corrected source,
but its initial evc37 request had an estimated wait of several hours. The dispatcher
updated that same pending job to available evc50, retaining the original payload,
output and resources. Root authorizes any available dual-H100 node except evc38,
with the same strict preflight and bounded resources. The evc50 attempt
failed its allocated-GPU preflight after 40 seconds: only one GPU was exposed.
No training update occurred. Exclude evc50 as well until verified healthy, and
inspect actual scheduler/dispatch receipts before assuming a replacement exists.
The next bounded pilot, 838921 on evc37, passed both driver and Torch/NCCL checks
and loaded the official model. Its first backward pass failed while Triton built
a helper: `/usr/include/python3.12/Python.h` is absent. No optimizer update or
epoch completed. A matching private development-header/toolchain supplement and
CPU compilation check are required before another GPU attempt. Preserve the
accepted environment archive; do not install into the system interpreter or
silently change model/backend behavior. The exact GPU resume and separate real
Slurm USR1 forwarding rehearsals remain open.

The matching private header repair subsequently passed CPU job 839074 on evc37
in 21 seconds. Compute nodes evc22/evc37 have Python/libpython package release
`3.12.13-2.el8_10`, unlike the login node's `-3`; the first signature-verified
candidate was correctly withheld by that identity check. Two further preserved
checks exposed a `/home` versus `/lustre` path-alias assertion and a generated
`pyconfig-64.h` dependency supplied by the matching runtime package. The complete
private successor contains 215 byte-bound headers. Manual and actual installed
Triton helper builds/imports plus version/pointer/PyObject ABI checks passed;
the environment archive and numerical backend remain unchanged. See
`initiative/audits/newton-runtime-repair-v1/`,
`initiative/audits/newton-python-headers-accepted-v1/`, and the Newton guide for
exact hashes and operational paths. Inspect the sole dispatcher's current pilot
receipts; no optimizer update or epoch is implied by this CPU milestone.

The exact offline Kaggle wheelhouse also passed an actual T4 probe. Its dedicated
private Dataset and notebook remain at version 1, with all small receipts under
`initiative/audits/dinov3-deployment-v1/kaggle-t4-acceptance/`. Reuse the archived
`shared-wheelhouse.json` for epoch bundles; do not upload the same 3.18 GB wheel
archive for every checkpoint. The probe used no competition data or pretrained
weights, and no new competition submission has been made.

The epoch publication coordinator is now integrated under `scripts/kaggle/`.
Its minimal exact-Git projection preserves the checkpoint's runtime hashes, and
one fixed account ledger governs every publication phase. A real private
three-file Dataset and offline CPU notebook proved ZIP directory preservation
and exact mounted bytes; no competition submission or model inference was part
of that probe. The source review, 85 focused passing tests, original findings and
correction, and prior e7b GitHub/HF receipts are preserved under
`initiative/audits/kaggle-epoch-publication-v1/`. Build its deployment projection
from the committed helper version against the actual training commit; do not
mutate the active clean training checkout. Large epoch payloads stay on Newton. All 941 local tests, Ruff and mypy passed
for this integration; root check logs are preserved in the same audit.
That helper milestone is published on GitHub main as
`018a43f4a8cf4061a2cbae493a19a171d15916e9`, with successful exact-head CI
35770990593 and byte-verified private HF revision
`b3b8e6660b0b495e069218b7f841b5479fbf7951`. Its 45-file minimal projection passed
an actual clean-process bootstrap check and preserves the e7b training runtime.
These publication/projection receipts are in the runtime-repair archive.

Remote Lustre has about 212.66 GiB of byte quota available at discovery but only
45,004 remaining file slots; the 48,772-file cache must not be unpacked there.
Consult the current Newton discovery/staging receipts before allocating a GPU.
Local storage must retain at least 20 GiB free. Training, model access and queued
submissions must be checked against actual receipts rather than inferred from
implemented software. Keep gold and reserved confirmation out of model selection;
public leaderboard feedback is now explicitly part of development.

The actual local one-process and two-process CPU/Gloo checks both passed exact
within-epoch recovery, scheduled snapshot eligibility and public-template inference.
Failed earlier disk/rendezvous attempts are preserved in
`initiative/audits/competitive-runtime-acceptance-v1/`; CUDA/NCCL acceptance is
still separate. The active task heartbeat `rsna-dinov3-training-and-submissions`
checks progress every five minutes. It uses actual job/transfer/submission
receipts and one account ledger; it must never duplicate active work.

The correspondence final check clarifies one historical shorthand: among the
24 unresolved cases, 23 non-gold studies have twelve zero masks, and 032:35 is
an official-gold evaluation-only study. Its official masks remain one and no
report supervision enters training. The 18 registered report exclusions are zero.

## Historical checkpoint context before final acceptance

The following records preserve earlier milestones and their then-open work.
Their pending-gate statements are superseded by the current release above.

The local registry contains **569 active accepted corrections**. Wording004
adds 78 across four cases: FIRST14, SECOND62 and two single-item original-author
context amendments. All 491 earlier whole objects and three retirements remain
unchanged. The integration preserves all 63 first/73 second proposals, 27 initial
postfreeze root reconsiderations, nine second rejections and four first state
rejections. Root's 46 complete sources / 552 initial judgments preceded current
case findings.

The [569 integration proof](../initiative/audits/qualified-negation-focused-004-integration-v1/complete569-integration-validation.json)
and [independent replay](../initiative/audits/qualified004-integration-independent-verification-v1/independent-integration-validation.json)
verify exact additions, preserved objects, three pending semantic gates and zero
conflicts across 54 identical-report groups / 204 reports. Their original 548
matches plus four pending-cell account remains historically correct.

A distinct source assessor rejected the four additional root proposals. Root
then reread those three complete sources and withdrew all four in a
[later explicit resolution](../initiative/audits/qualified004-additional-root-resolution-v1/root-later-resolution.json).
Three normal-appearance statements retain their negatives; one dimension-only
Baker cyst retains uncertainty because no calibrated moderate/large threshold
was supplied. These are four later root reconsiderations, not initial agreement.
The later effective 552 states and quotes all match 569; no extra corrections
were registered. The older wording003 additional proposal for 026:03 Lateral OA
remains a real semantic disagreement and is still unregistered.

The latest verified publication is Git
`a815aae64d812596a00b517c8542346280ee85f4` / private HF
`7d6d4cef88215a032cdaa23de5d189aefc0aa99b`: 569 corrections and 10,566 restorable
Git files. Exact-commit CI 35701433274 passed. Its actual receipt and two complete
reconstruction executions are preserved in
`qualified004-verified569-publication-v1/`. The peer proof and local alias describe
one execution; the distinct root proof is the second. The final two bone batches
and conservative supervision disposition work are newer than that checkpoint.
No corrected training export exists yet.

Both cartilage readers completed 260 priority reports and root adjudications
000–005. Both bone readers completed source and comparison passes through 030:
1,549 reports / 3,098 target judgments each. The complete original final two
batches are preserved with their bound completion manifests in
`bone-final-passes-preservation-v1/`. Bone proposals remain
unintegrated until a supported disposition; no reading count is clinical approval.

Root wording005 has frozen all 48 complete neutral attention sources / 576 target
judgments before current case findings: 462 exact nonmissing quotations and 114
empty missing states. Source freeze:
`b66019833e7d2f9773afa83aafdfb2417f314dc5ced5536712520e99cd9ae881`.
The helper preserves two prefreeze literal-quote validation failures and exact
source corrections; neither changed an initial state. Comparison/adjudication
and wording006 attention remain. Snapshot v18 preserves bounded completed
cutoffs; archive copying does not itself approve clinical annotations.

The finite closure map and registry362 evidence ledger link all 4,407 reports /
52,884 cells to original review evidence and targeted follow-ups. The ledger has
187,755 historical records across 416 stages; these are not independent readings.
Missing session identifiers remain null. Its fixed cutoff includes bone000–019
and root002 source judgments, and excludes bone020/root002 final adjudication.
Use the separately preserved later evidence for those scopes; do not silently
rewrite the historical ledger. Its 37 exact worklists identify evidence reuse and
remaining dispositions, not automatic QA clearance or a mandate for another
blanket two-pass campaign. Large ledger members are losslessly compressed with
their original hashes and decompression checks.

A separate conservative supervision policy is being implemented. It will preserve
clinical states and evidence while excluding unresolved report targets from loss.
Its all-cell ledger must reuse genuine adequate prior readings, preserve dissent,
distinguish quote variants from defects, and quantify class depletion before any
release approval. A blanket preliminary exclusion scenario removed every PF OA
positive; that scenario is not approved. Review-family names cannot substitute for
an assessment of the actual full-source target evidence. No further blanket reread
or forced clinical consensus is required merely to withhold unresolved supervision.

Complete and validate the concrete supervision disposition, account for observed
and masked cells, then resolve the three gates against its exact ledger and the
final review/correction hashes.
Recheck the 18 report-supervision exclusions and 24 unresolved all-target masks,
build final leakage-isolated manifests, and run one final-input real optimizer,
exact CPU resume, 58-study gold inference and three-study submission validation.
Keep gold evaluation-only. Reuse unchanged image/cache checks; do not invent a
full-image clinical reread or a GPU benchmark as a new readiness requirement.
After this release passes, deliver the competitive training strategy required by
[mission-and-training-plan.md](mission-and-training-plan.md).

Storage consolidation preserved every source path and byte using independent APFS
copy-on-write files. Original data, caches and publication archives were untouched.
Keep at least 20 GiB free and keep existing NPZ hardlinks immutable.

The cartilage discovery snapshot covers all 4,407 reports/13,221 OA cells, with
6,034 candidate cells. Priority covers 302 cells in 260 reports; all 260 first-reader
sources and comparisons are now complete. Another 2,407 observed-negative
candidate cells in 1,204 reports remain pending; 3,002 negative cells outside lexical
discovery are not certified by that proxy. Of 833 reports whose candidate cells
are all masked, only 749 have all three OA targets masked. Final export must
validate all required masks. None of these inventories closes global semantic QA.

This archive contains all 89 first-reader batches and all 89 independently reviewed
batches: 4,407 reports and 52,884 target decisions. Report-review coverage is
complete; global semantic QA, correspondence and final release checks remain
unfinished. All 462 tests, Ruff and mypy pass. The required correspondence registry now
enforces 18 whole-study report-supervision exclusions, including 069:10 under v6
and 064:46 under v5. The v4 extension
adds 048:09, 048:38 and 053:47 from previously completed source and image evidence.
Original clinical artifacts, reports, images and official gold are preserved.

The consolidated 28-case correspondence queue has four cases with observed
report targets. Root and a distinct assessor separately read their full sources
and accepted retaining current supervision for 021:44, 042:14 and 042:41, subject
to semantic QA. Case 069:10 has a RIGHT report and five explicitly L-tagged
series; the sixth series is unsided. Both accepted whole-study non-gold report
supervision exclusion. Root replay verified all 24 series, 699 source memberships
and 48 endpoint bindings across these four cases. No new pixel or named-side
diagnosis is claimed. The other 24 cases already have all twelve report targets
masked; their identity questions remain unresolved and their zero masks must be
checked again on the final export. See `correspondence-source-header-v1/` and
`docs/correspondence.md`.

The reviewed sampling table preserves all 24,338 retained series and prefers
series 11/12 over 6/5 in 064:14 and series 201 over the broad series 101 in 084:27.
The latter decision uses the previously bounded visual review, not inspection of
all 200 slices in series 101. The loader retains plane coverage and lower-priority
alternatives when capacity permits. All 4,405 unaffected studies preserve the
previous selected order and RNG state in the recorded comparison; eight actual
cache-loader passes returned finite arrays. See `reviewed-series-preferences-v1/`
and `docs/series-preferences.md` for the immutable policy, complete table, hashes,
rebuild command and validation scope. This is a sampling preference, not removal
of images or a change in clinical labels.

Batch085 completes independent report-review coverage. Root read all 50 original
reports and froze 600 personally reasoned decisions before clinical draft access;
prior technical source exposure to 085:14 is disclosed. All 46 initial state
disagreements were adjudicated, yielding 40 changes from the draft and six
reconsidered preread states. All 45 reader flags were considered. Evidence and
unchanged original readings are preserved in `clinical-review-085-root-v1/`.
Case 085:14 contains two undated RIGHT-compatible reports with conflicting
findings; all targets remain masked pending temporal/source correspondence.
The inserted bilateral separator alone does not establish opposite knees.
Duplicated PCL wording in 085:05 is not silently repaired into an ACL finding,
and the unsided singular meniscus in 085:04 does not identify both menisci.
Morphology, cartilage extent, qualified negatives and fracture timing remain
documented for global QA. Full review coverage is not diagnostic certification
or a finished training release.

Batch084 now has a complete independent root review of all 50 reports and 600
target decisions. The complete source pass was frozen before clinical draft
access, with prior technical source exposure to 084:11,27,39 explicitly disclosed.
All 32 initial state disagreements were adjudicated, yielding 28 changes from
the draft and four reconsidered preread states. One medial-meniscus quotation
was extended to include its explicit anatomical identity without changing the
state. The unchanged preread, original reader artifact and full decision history
are preserved in `clinical-review-084-root-v1/`. Negation scope, current bone
bruise with prior fracture, indirect OA mentions and cartilage severity remain
flagged for global semantic QA. The other agent's partial review is untouched.

Batch083 now has a complete independent root review of all 50 reports and 600
target decisions. The original source pass was frozen before draft access. All
43 initial state disagreements were adjudicated, with 34 changes from the draft
and nine reconsidered preread states. Explicitly broad cartilage extent and a
definite free-edge meniscal cleft are retained as reviewed report-level inferences,
with unresolved severity, negation and fracture-acuity issues still masked. The
complete evidence is in `clinical-review-083-root-v1/`; no source or original
reader artifact was altered, and global semantic QA remains required.

Batch082 now has a complete root source-first review: all 50 reports and 600
target decisions were frozen before draft comparison. All 67 initial state
disagreements were adjudicated, changing 42 draft states and reconsidering 27
preread states. The original preread, reader artifacts, exact quotations and
full decision history are preserved in `clinical-review-082-root-v1/`. Cases
082:22 and 082:38 contain unresolved report aggregation; all extracted target
losses stay masked. Their additional correspondence follow-up remains open,
and this review does not change the existing 16-case runtime registry.

The required report-similarity registry joins 038:44 and 045:15 before transitive
gold-neighbor quarantine. Read `docs/report-similarity.md`. The complete final
label export remains unfinished.

The historical 205-correction private Hugging Face checkpoint is
`a70f15a39afd38a994aedc72af774817bf53647e`, corresponding to Git
`7c1225e61974c374245c0656edeb0c47f54582c6` (CI35666396114 passed).
It reconstructs all7179 tracked Git files exactly from the pinned1792 base and
a202,364,633-byte delta. Root and a distinct verifier separately streamed the
base/delta and checked every result path, mode, size, SHA256 and Git blob identity.
All six published objects and both immutable base objects were verified. Original
source image trees, five official metadata files and the root dataset card remain
unchanged. See `qualified001-verified205-publication-v1/` for the actual receipts,
operation, manifest and both restoration proofs. The base remains HF54949cd5 /
Git1792f01; use the pinned v2 helper and its exact fixture-aware scan policy.

That immutable checkpoint contains205 active corrections,18 correspondence
exclusions and the reviewed sampling preference. At that stage, the 229 active corrections,
three retirements and newer semantic snapshots required a later verified publication
receipt; do not infer synchronization from a Git commit alone. Earlier118 and205
checkpoints and their original evidence remain immutable.
The archive also preserves proof that547 staged DICOM copies now share their
byte-identical archived files' physical storage. All paths and bytes remain
unchanged; never write through those shared inodes. The full cache is untouched.

The full-coverage duplicate consistency inspection is preserved under
`post-review-consistency4407-v1/`. Of54 identical normalized-report groups
covering204 studies, four have differing target states. Three group differences
affect observed supervision; no positive-versus-negative pair was found.
Root and a distinct assessor each read all nine complete sources and froze108
assessments before proposal access, with prior documentation exposure disclosed.
Seven independently accepted corrections are now registered: four Effusion
negative-to-uncertain masks and three already-masked Contusion/Fracture
uncertainty-to-unmentioned clarifications. That milestone brought the total to34;
the later focused wording cases are summarized with the current total above.
All277 protected source/review/rubric/prior-case/QA artifacts remain unchanged.
The bounded adjudication and all27 focused tests passed. See
`duplicate-consistency-adjudication-v1/`. The three global semantic QA gates remain
pending: cartilage, bone terminology and qualified-negation/duplicate consistency.

The subsequent full-coverage mechanical candidate inventory is preserved under
`qualified-negation-all4407-candidates-v1/`: after the 34 corrections, all 54 exact
normalized-report groups have consistent target states. Its focused wording
queue contains 539 target candidates across 331 reports; candidate membership is
not a finding of error. For its first 50-report batch, two distinct agents have
frozen all 600 source judgments and preserved their separate post-comparison
reconsiderations. Their source freezes agree on 584 states and differ on 16.
The later root and additional independent adjudications described above now
register the accepted corrections while preserving these earlier proposal states.
Read the separate `qualified-negation-focused-000-source-review-v1/` and
`qualified-negation-focused-000-independent-review-v1/` scopes before use.

The earlier `qualified-negation-focused-000-root-review-v1/` snapshot contains only
the first ten source reports/120 decisions. It remains byte-identical. The separate
completed root archive now includes all five original source packets, the combined
48-report freeze and the full comparison/adjudication history. Historical source
and aggregate exposure remain disclosed. Later batch001 and cartilage snapshots
are preserved as pending work; none closes a global semantic QA gate.

The tenth through thirteenth metadata groups now have two frozen source readings
and root replay of 5,361 source memberships, 167 series metadata records and
334 endpoint geometries. Root also recomputed physical fields and slice-origin
extents. The remaining twelve targeted cases now have two separately frozen visual passes
on150 representative matrices from50 selected arrays. Root replay checked all1698
selected source memberships and each exact displayed matrix. Whole-study metadata
covers65 series/2127 sources;15 unselected arrays/429 sources remain outside pixel
scope. Eight source correspondence issues remain unresolved, despite compatible
sampled main protocols. Case084:27 includes a peripheral second limb in broad
series101; the later sampling preference above preserves its originals. See
`technical-tier4-last-twelve-targeted-v1/` and `last-twelve-independent-visual-v1/`.
The other twelve metadata cases needed no immediate pixels from their examined
triggers alone. The original agent replay failed
because it read the wrong candidate-queue field; that script and all four failed
outputs are preserved alongside the corrected, successful root replay.

The eighth targeted cache review is now preserved with successful root numeric
replay: nine complete selected arrays, 574 whole-study source memberships and
27 exact displayed samples. For 056:16, both independent visual readings of all
146 slices in five selected axial/coronal stacks support compatible primary
anatomy. Root replay verified every displayed matrix and its source bindings.
Root's earlier sparse opposing-fibula interpretation was wrong and is preserved
with its explicit correction. These bounded checks do not justify a new
exclusion or assign a named side from anatomy alone. Four unselected 058:04
series retain metadata-only scope.

Fresh distinct agents are active again. Their completed source/image assessments
are preserved with actual canonical identities; earlier usage-limited agents are
historical context. The full final training release remains unfinished. GitHub
milestones and private Hugging Face preparation checkpoints are published after
validation; the HF revision above is the last verified published snapshot.

The user authorized direct main integration on 2026-09-20 and requested one monorepo without a PR workflow. The corrective pipeline commit is integrated into main at `423733e`; both prior PRs are integrated. The active worktree uses local branch `codex/data-readiness-audit`, tracking `origin/main`. Push verified milestones with `git push origin HEAD:main`.
The legacy archive remains unchanged. Its rubric accepted low-severity findings that do not satisfy the organizer criteria. Continue the new versioned review run, `codex-20260920-rubric-v2`, under `protocols/report_weak_v2.md`; do not resume the old rubric as a corrected-label workflow.

## Current gates and evidence

The current archived milestone includes all 89 completed v2 review batches covering 4,407 reports and 52,884 target decisions. The compiled index is authoritative for coverage. All preread hashes and exact quotations were validated. The v2 full-coverage gate now passes; global semantic adjudication and release validation remain required.

The complete endpoint survey covers 48,772 headers, all 24,386 series and 4,410 studies without request failures. Nine independently examined plane candidates preserve official routing; six accepted overrides in study045:14 have a separately bound effective series table. Four earlier studies contain two visually distinct knee protocols; their aggregate report supervision remains masked. A fifth,075:14, now has verified official-source and independently frozen visual evidence for two knees, but its short unsided report remains unresolved. These technical image checks are not diagnostic MRI rereads.

The corrected full cache has passed `rsna.audit`: 819,635 slices, 24,386 series, 4,410 studies, zero cache warnings, one cross-study nonconstant pixel edge and no released train/test overlap. Exact-series selection retains all 4,407 training studies and 24,338 of 24,371 series; 33 fully equivalent model slots are excluded. All 15 released test series remain. The exact 077:22 pair shares one slot, while the similar 042:40 pair remains separate. Every input cache file was checked again before export. See `initiative/audits/full-image-recovery-v1/` and `docs/series-equivalence.md`.

The sole cross-study image edge is confirmed in all 288 fresh official Kaggle sources: five matching series and 144 exact native matrix pairs. Root independently rechecked full file hashes, CRCs, UIDs, all matrices, 15 parsed geometry/encoding fields and all ten cache arrays. Both assessors viewed ten bound native mid-slice panels independently before sharing visual observations. Preserve both studies in one final leakage component. See `initiative/audits/cross-study-official-duplicate-v1/`.

The original 48,772-record endpoint survey is valid against the corrected inventory because neither corrected SOP was an endpoint and every selected source binding is unchanged. The derivative proof preserves the historical partial-prefix scope; it is not a fresh header read. See `initiative/audits/endpoint-source-reuse-v1/`. Technical correspondence work remains unfinished. The shoulder-only report at052:20 now has two independent visual reviews of three bound cached knee samples, establishing a body-region mismatch in those series. Root verified source equality, all six series/169source memberships, twelve endpoint geometry bindings and all three NPZ/displayed sample bytes. Preserve its existing fully masked report supervision. The two combined bilateral reports at052:01/17 now have complete bounded source/header/cache/visual review: 245 source memberships, 12 complete volumes and 36 displayed samples. Both favor coherent sampled protocols while aggregate correspondence remains unresolved; their required report-supervision exclusions are now enforced by the v3 policy. The source-side conflict at055:10 has now passed bounded technical checks; its report supervision still requires semantic/correspondence adjudication.

The three latest preserved correspondence audits are `initiative/audits/repeated-protocol-six-v1/`, `correspondence-native-075-014-v1/` and `correspondence-shoulder-report-052-020-v1/`. For075:14 all337fresh official sources and11series have root-replayed native fingerprints, dimensions, IPP/IOP/spacing, source hashes/CRCs/UIDs,15recorded source geometry/encoding fields, and33pixel-exact panels. Opposing fibular morphology has concordant source IOP direction. Preserve the no-upstream-mirroring caveat; do not assign a named side from blanks. Prior dark display artifacts remain preserved with explicit native-window-v2 supersession.

The required `initiative/audits/correspondence-release-registry.json` binds the preserved study-level policies. `rsna.labels` now validates exact policy membership, original source identity and verified proof bindings before exporting. All twelve report loss masks are zeroed for excluded non-gold rows; original annotations, official gold evaluation and duplicate groups remain unchanged. Excluded reports are omitted from report-versus-gold agreement. Native proof identity wrappers make the original source-handoff study binding explicit without rewriting the underlying observations. See `docs/correspondence.md`. Whole-study exclusions now cover 18 cases; the exact registry and docs/correspondence.md list them. This does not certify the remaining correspondence queue or compile a final manifest.

Native checks for048:12 and086:13 now include root replay of all659sources,15recorded geometry/encoding fields and60pixel-exact displayed samples. Both independently frozen visual readings support opposing knee protocols compatible with their integrated bilateral reports. The source origins remain83retained official originals and576corrected pinned HF files. Fresh official whole-byte confirmation is deferred for those576 after Kaggle429responses; no claim that all659were freshly downloaded from Kaggle. All sources, source-origin journals, partial-attempt records and explicit deferred list are preserved in `correspondence-native-048-012-v1/` and `correspondence-native-086-013-v1/`. Root independently verified all940files and166original-source aliases in the two native and first two mixed-tag archives.

Twenty-five mixed-tag candidates now have complete bounded reviews across eight preserved audit groups: 011013014, 016021022, 030036042, 042044045, 048052056, 057059062, 065067071 and 075079081. Root independently verified all 3,783 source memberships, 280 endpoint headers, 140 full cache volumes and 420 exact displayed samples, and personally viewed every sample before reading the other visual assessment. Original source texts and literal tags were checked separately. These samples favor coherent main protocols; preserve tag conflicts without named-side correction or automatic bilateral exclusion. Oblique and broad-coverage series retain their stated limits. A root caption-reading mistake for 045:38 is preserved with a separate correction; no source tag or display was changed. The final mixed-tag group preserves a separate correction to a generic agent display sentence: 075:28 has an oblique supplement and 079:31 has none. Six further correspondence cases now have root-replayed numeric and independent visual evidence: 1,105 source memberships, 31 cache volumes and 93 displayed samples. The combined Greek reports at062:21 now have enforced report-supervision exclusion under v3. The explicit performed/requested-side clarification at063:35 remains distinct from unresolved contradictory report blocks; no blanket disposition was applied. Four repeated/temporal cases now have root source/visual freezes and successful
replay of29 arrays,791 source memberships and87 displayed matrices. Cases001:18
and021:03 show no new gross mismatch from their triggers alone. Native follow-up
for064:14 reproduced12 selected cache slices bit-exact from pinned corrected
source DICOMs. Four multi-window native figures and the independent all71-slice
cache inspection support a preference for11/12 over6/5 in their repeated roles,
without discarding originals or claiming equal coverage. Source064:46 joins an
initial examination and explicit recall; root and the distinct assessor accept
whole-study non-gold report-supervision exclusion pending exact temporal linkage.
The v5 policy validates this exact accepted exclusion against all4407 blinded
reports and preserves gold masks;22 new tests reject altered evidence and actions. The original root visual freeze
accidentally stored a PNG hash in its source-preread field; an explicit bound
correction preserves the original and identifies the correct source freeze. It
does not prove historical access order. See `repeated-temporal-001021064-v1/`,
`quality-native-064014-v1/` and `independent-quality-temporal-v1/`.

Thirteen metadata triage groups now preserve 78 cases, 19,351 source memberships,
564 series metadata records and 1,128 endpoint geometries. Forty-six cases need no
immediate pixels from the examined triggers alone; 32 require bounded follow-up.
Twenty have completed bounded source/visual/numeric checks, including all 14
from the first six groups, four seventh-group cases and two eighth-group cases.
The twelve later targeted cases now have the bounded two-reader visual and numeric checks described above; their open source-eligibility decisions remain explicit.
These counts concern the original technical tail, not every newly flagged report.
Recompute the final crosswalk after all queues. Root dispositions for the latest
24 cases are in `tier4-tenth-thirteenth-metadata-crosscheck-v1/root-run-001/`.

Root's accidental prior exposure to the third source group's observations remains
disclosed, with an additional independent source assessor supplying that pass.
Subsequent groups preserve separate source-first freezes. Source anomalies and
within-report contradictions remain for global semantic QA. Ninth060:25same-geometry
701/1301have disjoint stored native fingerprints and both remain. Original newline
and orientation overstatements are retained beside independently verified corrections.

The preceding 3,757-report checkpoint passed 246 tests, including 46 correspondence regressions, plus Ruff and mypy. The initial independent code review found two gaps in policy membership and source identity binding; both were fixed and independently rechecked. Final code hashes, source-only validation of all 4,407 reports against the four exclusions, review history and image-preservation receipts are in `initiative/audits/correspondence-enforcement-v1/`. Root independently reviewed the recovery and inference changes, ran 26 recovery tests and 35 inference/pipeline tests, and verified every reused metadata hash and revision-only semantic delta. Held-out inference now binds the exact training series table, study manifest and cache source revision; separate test inputs record their own provenance. See `docs/source-recovery.md` and `docs/inference-bindings.md`.

Reports042:14 and042:41 explicitly combine both knees. Independent correspondence adjudication accepted 16 additional eligibility masks; the immutable original extractions and both readings remain preserved. These are report-to-image correspondence masks, not claims that the imaged studies contain both knees. That milestone brought the registry to27 accepted target changes; the latest independent duplicate-consistency adjudication brings it to34. Mandatory global cartilage, bone terminology and qualified-negation/duplicate QA remain pending. No original clinical artifact is overwritten.

GitHub and Hugging Face are private checkpoint destinations. HF source revision `0b5e410288232f570c69ba4f44723356e6ae6ab7` replaces exactly two invalid mirror encodings with verified official Kaggle files; the original pinned `52f995f4568e36bfa5fbbaf69a89af41226b4f5d` remains preserved. A verified private preparation snapshot of GitHub `2359718669fa7dc22d394af7b0682f3e5aaf5026` is pinned at HF `a36928407109e9d23ccb694a57765606b792fbc6`, including3,057reviewed reports, the337official075:14sources and shoulder-report correspondence evidence. Its receipt is preserved under `initiative/audits/hf-preparation-checkpoint-2359718/`; root independently streamed all1,843package members and verified exact hashes/membership after60remote object checks. The subsequent 3,207-report milestone at GitHub `1bdcfcc059c036002e15b1ebe64a81a92392d574` was verified privately at HF `d6557ffc2aadc9be050ab4668a1a24ef9b9d8414`. Its preserved receipt under `initiative/audits/hf-preparation-checkpoint-1bdcfcc/` binds 214 verified remote objects and 2,919 independently rechecked package members. The 3,507-report GitHub milestone `88edbf46772d3eaf94e2ec8693b3840528f301d8` is verified at HF `492b1eb363680fe593087f84e9627b1de4ee8680`, with 365 remote objects and 3,268 independently rechecked package members. Its receipt is preserved under `hf-preparation-checkpoint-88edbf4/`. The 3,607-report milestone at GitHub `6f2d07d06111d6170b22cdac1f3b27b4d41f4d38` passed exact-commit CI run35592976831. Its original Hugging Face upload stalled in the transfer client and was stopped with a preserved receipt. Recovery reuses the exact frozen 490-object operation and 873,998,824-byte package with bounded HTTP transport and fresh private-HEAD reconciliation. The successful recovery is now pinned at HF `76a4c0dab20fd49ffd0590a4344eac5b67d8be1f`; its complete verification receipt is preserved as described above. The current 4,407-report archive is a later GitHub preparation checkpoint. No finished training release has been published; checkpoints remain explicitly incomplete until all gates pass.

- All five fresh authenticated Kaggle metadata downloads match `initiative/source/` byte for byte. `rsna.preflight` independently checked source/inventory relationships and recorded 54 duplicate-report groups covering 204 studies.
- `initiative/reviews-v2/` stores completed drafts/reviews, immutable per-artifact identity/hash sidecars, review attention flags, and the compiled progress index. Original blinded inputs remain in `initiative/reviews/inputs/`.
- Use the index to select unfinished batches; preserve every completed preread and adjudication history. Stage unfinished work outside `initiative/` to keep archived milestones stable.
- Every report requires a reader plus a separate source-reading reviewer. Reports alone create weak labels, with ambiguous findings masked. Gold values must not be inspected to tune extraction. All official evaluation studies remain excluded from training.
- Cache version is `series-u16-v3`; old v2 metadata/hashes must not be relabeled v3. The first 100 series / 3,182 slices passed the new decoder and cache audit. This pilot is not the full audit.
- The larger isolated pilot covers 150 training studies plus all 3 public test examples, 848 series and 28,882 slices. Its independent audit found no cache warnings or cross-study pixel duplicates within that subset. See `initiative/audits/pilot-realdata-v3/`.
- The real-data pipeline diagnostic passed 126 optimizer updates over two complete epochs; continuous and resumed loaded checkpoints match exactly, including model, optimizer and RNG state. All three public test predictions match template order and contain 12 finite probabilities. `initiative/audits/pilot-realdata-smoke-v3/` archives the tiny checkpoints, commands, manifest, scope, predictions and hashes. These are diagnostic weights, not competition training initialization. The completed full pixel graph must feed final transitive leakage isolation; final-release smoke checks remain required.
- The baseline streaming job stopped with two invalid DICOM encodings, one in each of two series. Exact official source restoration and a separate recovery completed all 74 affected slices. Do not restart the old supervisor against the invalid baseline inventory. The effective cache is `work/cache256-v3-official-recovery-v1/cache`, with its exact corrected inventory beside it; the full audit is `work/dicom-audit-v3-official-recovery-v1/`. Reused NPZ hardlinks are immutable: never rewrite either old or new cache files in place. Preserve 20 GiB free disk. Source restoration, per-series lineage, independent visual limits, full audit and training/test selection artifacts are archived under `initiative/audits/source-pixel-representation-v1/` and `initiative/audits/full-image-recovery-v1/`.
- The one-header-per-study audit covered 4,410 released studies and found 4,410 distinct nonempty 12-character PatientIDs, all lacking issuer metadata. No repeated identifier groups were found. This does not establish true patient identity or consistency across every header. `initiative/audits/patient-header-audit/` preserves hashed identifiers and source-byte provenance. The exact run script is archived in `initiative/provenance/patient_header_scan.original.py`; it expects a workspace `work/` containing the repository as `recovered/` and uses environment credentials only.
- Assistant visual checks cover six native/cache slice pairs and 48 spaced cached slices, across all 3 planes. No gross crop/blank/encoding artifact was observed in those samples. An edge slice is very dark but later anatomy is useful, so it is retained. Downsampling softens detail; full-resolution source remains authoritative. See `initiative/audits/visual-*.json`.

## Compile and validate new review outputs

Per-artifact sidecars go to `initiative/reviews-v2/provenance/batch-NNN-{draft|reviewed}.json`. Use the actual agent identity, frozen rubric hash, canonical source-report fingerprint, input-file hash, exact UID list, output hash, study count, stage and notes. Review sidecars also bind the exact parent draft SHA256. Never generate missing annotations programmatically; code may assemble and validate individually reasoned decisions.

```bash
python -m rsna.review_progress --inputs initiative/reviews/inputs --root initiative/reviews-v2 --rubric protocols/report_weak_v2.md
# All 89 batches now have both valid passes; this checks review coverage only:
python -m rsna.review_progress --inputs initiative/reviews/inputs --root initiative/reviews-v2 --rubric protocols/report_weak_v2.md --require-complete
```

The compiler checks evidence, identity, source/rubric/parent hashes and coverage. These checks cannot replace semantic review. It never certifies training readiness. Incomplete in-progress artifact/sidecar pairs are not eligible for release.

After the full image and review gates pass, use the updated `rsna.labels` invocation in README. Then perform real-data optimizer, exact resume, gold-inference and public-example submission checks. Preserve masked unknowns and investigate any class collapse; do not invent positives to balance classes.

## Preserve verified progress

Freeze completed batch outputs before updating their archive snapshot. Run `python -m rsna.archive --refresh` then `python -m rsna.archive`, all tests, Ruff and mypy before pushing. The old archive's `--require-complete` concerns the historical v1 run, so use `rsna.review_progress --require-complete` for the v2 run.

The HF source mirror was freshly verified private on2026-09-21. Preserve restricted access for competition reports, labels and images. Validated preparation checkpoints can be mirrored privately with explicit incomplete status; the pinned original source revision stays unchanged. The private GitHub repository remains the code/review handoff. No public cleaned release is authorized by this workflow.
