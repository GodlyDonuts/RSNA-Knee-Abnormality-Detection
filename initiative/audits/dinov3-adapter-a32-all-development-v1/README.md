# A32 all-development final fitting

September 30, 2026. The owner requested an epoch-10 Kaggle submission and a
separate model trained on all usable development data, with its own later
submission. This preserves the rank-32 recipe so additional training membership
is the intended change. The fixed ten-epoch model starts from the same pinned
official base; it does not give previously trained studies an extra ten epochs.

Training source Git: `8911f90910ab8892b8324f6985c18d3ab146ec0c`.
Clean sparse Newton checkout: `/home/sa305415/rsna-knee/code-adapter-a32-all-8911f90`.
Control: `/home/sa305415/rsna-knee/production-launch-adapter-a32-all-8911f90-v1`.
Production output: `/home/sa305415/rsna-knee/runs/dinov3-l16-adapter-a32-all-8911f90-production-v1`.

## Membership and interpretation

Latest preservation: completed **epoch 6 / 2,556 updates** has all four private
objects fully re-downloaded/size/SHA verified at HF
**ce416ddf72340078acbea63934b2a90375171e10**, checkpoint SHA
`6162d0c3e7164862ddc64dd5a0ccece60944e7b35ca73cee61c073491a7fd6e1`.
Epochs 1–6 are preserved; never repeat their publication. CPU retention **865400**
removed only the redundant epoch-006.pt after exact local hash/private proof,
retaining active recovery, sidecar and private bytes. Its former link count was
one and 1,338,164,780 shared bytes were freed. Fresh projected reserve is
21.008990 GiB with six remaining maximum full inodes and both production
deployment pairs budgeted. Latest real optimizer update 2,630 is in epoch 7;
stderr remains empty. No internal holdout metric exists for this final fit.

`fit_scope=all_development` selects the same accepted training split with at
least one observed target: 3,403 studies. It retains the original target masks,
excludes 501 fully masked training rows, and excludes the reserved 441-study
confirmation, 58-study official gold and four quarantined rows. All accepted
folds remain unchanged and are validated, although there is no outer evaluation
pool for this final fit. Former outer-fold studies now receive training, so no
internal holdout metric or training-set AUC is exported. Fixed-final result
metadata has null metrics/evaluation source, and no outer_predictions.csv.

Effective batch eight means **426 updates per epoch**, **4,260 in ten epochs**.
Every real training study appears once per epoch. The final batch is padded,
with padding masked out of the loss; no study is silently dropped. Warmup is
213 updates then cosine, preserving the five-percent schedule fraction. Same
A32 architecture, 10,223,616 encoder adapters, 203,020 head parameters, official
base weights, BF16, two CPUs, one H100, 16 slices, six series and 224 input side.
Gold and confirmation remain excluded. Rank 64 is a separate matched-fold
comparison, not a simultaneous change to this final-fit candidate.

## Acceptance and dispatch state

Local tests verify exact all-development membership/mask preservation,
reserved/gold/leakage rejection, full all-data train-loop completion without an
evaluation loader, and interrupted/resumed equality of model, optimizer and RNG.
The first bounded allocated pilot **865258 FAILED 2:0** on evc29 in driver
acceptance before any training update or output. A distinct retry **865261**
uses evc31 and a distinct pilot-v2 output. It completed **0:0** with four finite optimizer updates. CPU-only verification
**865262 COMPLETED 0:0**. The verification passed
four real finite optimizer updates, completed all-development diagnostic
metadata, all 120 learned adapters, every original base tensor unchanged, and
strict checkpoint-only reconstruction. Production **865263** was submitted once after fresh acceptance and is **RUNNING
on evc31** with one allocated H100/two CPUs and allocated driver checks passed.
Full-pool learning acceptance verified 48 real finite optimizer updates, exact
3,403 training/zero outer binding, rank-32 adapter count and empty stderr. At that
observation the latest loss was 0.3592521 and gradient norm 1.4037193, after
415.3 seconds of optimizer progress. These are training diagnostics, not accuracy.
Its distinct ten-epoch output has not yet produced a completed epoch at this
observation. Do not duplicate it or restart without terminal/recovery review.

The private per-epoch publisher is `scripts/hf/publish_adapter_a32_all_epoch.py`,
pinned SHA `258e1e9e4766146c947fcca6d4886819284f1da25804bbf48ecc5982d1a762bb`.
The separate single coordinator is `scripts/hf/dispatch_adapter_a32_all_epoch.py`.
It binds this exact 3,403-study, 426-update, zero-outer pool and source revision;
fold-0 receipts cannot be reused. Prefix:
`experiments/dinov3-v1/training-epochs/adapter-a32-all-8911f90-seed20260922`.
Every completed production epoch is to be privately uploaded once and fully
re-downloaded/size/SHA verified; existing intents require read-only reconciliation.
Credentials travel from ignored local .env through stdin only, never saved on Newton.

## Storage

First retention **865238** failed before any deletion because five baseline
snapshots in an old prepared plan were already absent. No retention intent or
receipt was created. The revised exact nine-A32-snapshot plan **865250 COMPLETED
0:0**, fully hashing local objects and checking existing private HF full-download
receipts and fresh private LFS hash metadata before reclaiming 12,043,482,444 bytes.
All ten private epochs and the local final remain preserved. Separate **865254
COMPLETED 0:0** removed only six obsolete verified private Kaggle upload ZIPs,
7,294,089,698 bytes; it freshly checked private Dataset identity and full local
ZIP SHA against prior all-payload download receipts. Dataset bundles, source,
final weights, run records, sidecars, manifests, licenses and ledgers are retained.

After the epoch-10 Dataset bundle and ZIP were staged, quota was approximately
1,008,637,564/1,048,576,000 kB and 991,966/1,000,000 files. This gives about
38.09 GiB soft-byte reserve. Budget eleven full-sized checkpoint inodes
(ten immutable epochs plus one atomic replacement), one 1.25 GiB diagnostic
checkpoint, and another 2.50 GiB final Kaggle bundle/ZIP. The projected reserve
is 20.54 GiB including 100 MiB metadata overhead at actual dispatch: fresh quota acceptance remains
mandatory immediately before production dispatch. GPU node-local environment
and cache extraction do not consume shared Lustre quota.

## Whole-series next experiment

The present loss is applied once to each study's twelve predictions. It does
not declare every slice positive. Learned target-specific slice and series
attention can downweight irrelevant slices. A real remaining limitation is
fixed 16-slice sampling and no explicit adjacent-slice interaction. The next
matched-fold experiment should preserve ordered geometry, encode all valid
slices in bounded chunks, and aggregate neighboring-slice context and evidence
into one study representation. A flat montage is not a substitute for depth or
native detail, and changing deployment sampling only would create a training/
inference mismatch. Architecture/coverage changes require their own held-out
comparison before another all-data final fit.

## Verified external milestone

Source/launch, pilot, storage retention, real optimizer acceptance and the exact
epoch-10 code submission are on GitHub main `fcb3c1d`. The private HF milestone
at revision `f51a2182d4384f4928beeb4edc26676b022f7ca2` fully re-downloaded and
size/SHA verified all 69 objects. It records submission **56697342** without
claiming a hidden score. The new production epochs are separate objects and
require their own immutable publication receipts as they finish.
