# Five matched DINOv3 checkpoints, September 29

The owner authorized five evenly spaced checkpoint submissions, newest first.
Use epochs **10, 8, 6, 4 and 2** from the same global-CLS fold-2 run. These are
the five immutable checkpoints at the accepted two-epoch submission cadence.
This keeps architecture, fold, supervision, preprocessing and inference fixed
while comparing training progress. CLS has the stronger fixed-final internal
three-fold estimate (0.7522657162 versus 0.7360400218 for patch attention);
these are report-derived holdouts, not Kaggle scores. Do not claim improvement
before actual scored results exist.

The [campaign](campaign.json) binds every full checkpoint SHA, optimizer step,
fold and training source. All five share training binding
`9ffc775f8b67f0a7c0332576d0fc704b171ee4de3ea2e1867ed2109f19b869ec`.
Inference uses the tested [input recovery](../kaggle-input-recovery-v1/),
packaging Git `f4f90bcf81c20da4b1e53e5132b270ffabb14fbc`, and all 35 original
training-runtime files from `b74060b98f01bff1b924070ccba5ce44dd002f40`.
Raw FP32 sigmoid probabilities remain unchanged. No labels/reports enter Kaggle
inference and the public three-study CSV is not uploaded as a submission.

## Ledger recovery and actual checkpoint restoration

Newton connectivity returned. The initial scheduler view had no active jobs,
and Lustre usage was 1,027,009,792 / 1,048,576,000 kB and 990,891 / 1,000,000
files. Use CPU-only allocations for staging/publication and watch quota before
additional large writes. Preserve credentials only in ignored local `.env`,
streaming the required token through SSH stdin.

The canonical ledger is still
`/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3`. Failed entry 56550973
was reconciled to `remote_outcome=failed`. Epoch 10 contained an old uncertain
push with no version or submission attempt. Exact authenticated profile listing
and positive-controlled exact-slug search show its notebook absent. The
[guarded recovery](reconcile_ledger.py) preserves the entire old operation,
Dataset and absence proofs before queueing a distinct corrected candidate.
It never repeats the old push. Six meaningful SQLite regressions verify that
submitted, changed or remotely existing/ambiguous attempts cannot be requeued.
The actual one-CPU reconciliation job 863338 completed and its receipts are here.

Preparation job 863333 created and validated the epoch-10 bundle, then failed
before creating epoch 8 because earlier checkpoint files had been reclaimed
from the training directory after verified private backup. CPU diagnostic job
863336 confirmed the missing file boundary; neither failure was a Kaggle run or
submission. Do not repeat the successfully prepared epoch-10 bundle.

One-CPU restore job 863396 restored epochs 8 and 6 by verified hardlinks from
existing private bundle copies. Epochs 2 and 4 were streamed directly to Newton
from their exact pinned private Hugging Face revisions and matched by full size
and SHA-256 before atomic installation. No large artifact was downloaded to this
Mac and no weight was republished. The [restore sources](restore-sources.json),
[script](restore_checkpoints.py) and complete receipts preserve this recovery.
CPU-only continuation 863403 completed `0:0` and validated all four remaining
distinct bundles. Transport job 863414 prepares only those four upload archives.

## Private publication and code submissions

Every checkpoint uses a distinct private Dataset and exact-source private offline
T4 notebook. Fully re-download/hash verify every Dataset payload. Reuse only the
accepted shared cp312/cu130 wheelhouse. Preserve license, weight, source, fold,
environment, notebook source and exact saved version in the canonical ledger.
Old failed/uncertain notebooks are never submitted. Every external mutation has
a durable intent; reconcile uncertain results read-only before any retry.

The epoch-10 transport job 863339 completed `0:0`. Its distinct private Dataset
was created exactly once by CPU-only job 863398. Verification job 863406 completed `0:0`: Dataset version 1, ID **12263387**,
all 43 payload objects fully download/hash verified. Job 863413 pushed the exact
private offline T4 notebook once as version 1. Independent source/metadata checks
passed and its public-example run was `RUNNING` on first inspection. The other
publications continue; per-epoch receipts below are authoritative. No competition
submission has been made for this campaign at this initial milestone. The live
allowance initially returned five slots and zero submissions today. Refresh it
immediately before every exact `CreateCodeSubmission` call; never create a
replacement ledger or exceed the authoritative allowance/five-per-day limit.

All **1,064 tests**, Ruff lint/format and mypy for 35 source files pass. The
previous corrected wrapper also passed its 42 recovery/chunking tests under the
exact historical training-runtime projection. Archive and v2 coverage checks
are refreshed as audit receipts are added.

## First three accepted code submissions and five saved notebooks

Epochs 10, 6 and 8 were submitted once as **56663180**, **56663606** and
**56663742** respectively. Their saved version-1 MRI-only public runs passed
matching checkpoint/manifest/output receipts, finite 13-column predictions,
three inferred studies and zero fallbacks. Epoch 8's healthy public CSV hash
exactly matches its earlier metadata-recovery version: the probability transform
and normal-input predictions are unchanged. Epoch 6 was submitted while epoch 8
was still running; the selected epoch set remains 10/8/6/4/2.

At **05:23 UTC**, all three hidden statuses/errors/scores were null and the live
account allowance was two. This is acceptance of code submissions, not a scored
result or proof that a hidden rerun succeeded. Never repeat these submissions.

The first epoch-4 and epoch-2 SaveKernel calls returned uncertain results. Their
latest readback returns HTTP 500 and authenticated exact searches show private
placeholders. Numeric `versionLabel="1"` also returns 404 for the known good
saved epoch-10 notebook; **that probe cannot prove absence or no saved version**.
The original operations remain uncertain, preserved forever and never retried
or submitted. The [current guard](rebind_unversioned_kernel.py) and its
[epoch-2 binding](rebind_epoch2_kernel.py) instead use a known saved-source hash
positive control, exact owned placeholder searches, unchanged private Dataset
proofs and distinct-target absence. They archive the original uncertainty in the
one canonical ledger before one distinct push, without creating another Dataset.
Both new private offline T4 notebooks were saved once as version 1 and their
exact source/privacy/runtime checks passed. They are running public checks:

- `godlydonuts/rsna-knee-epoch4-f2-inputguard-k2-9f85d010fba4`
- `godlydonuts/rsna-knee-epoch2-f2-inputguard-k2-092c3eed133f`

The old `rebind_unversioned_kernel_v1.py` and network v1/v2 files preserve failed
preflight evidence, not runnable continuation instructions. The first shell
preflight had unused undefined variables; its Slurm job failed before Python or
any remote intent. The next preparation stopped on a proto-JSON empty search
page whose repeated field was omitted; it created the derived bundle but did
not archive the old state or call SaveKernel. Verified read-only recovery and
the corrected explicit phase then finished; all original receipts are preserved.
Use the successful v3 helper and exact per-epoch bindings, never those old phases.
Twenty-two SQLite/remote-protocol regressions reject submitted, changed, already
readable, ambiguous, duplicate and wrong-source operations. The full repository
suite has 1,064 passing tests, with lint, format and 35-source-file typing checks.

The immutable snapshot's historical `submitted=false` field is not the live
submission state. The canonical ledger's `state`, submission ID and exact
identity reconciliation are authoritative. Keep two slots for epochs 4 and 2
only after their actual public completion receipts pass. Preserve the accepted
masks/folds, source/weight/license bindings and raw FP32 sigmoid output.

## Completed five-checkpoint submission campaign

All five requested checkpoints were accepted once. All five private Dataset
payloads passed full download/hash verification; every exact-source private
offline T4 notebook version 1 completed MRI-only public inference. Each output
has the expected 13 columns, three inferred studies and zero fallback studies,
finite probabilities, and matching receipt/CSV/manifest hashes. The public
example CSV is never a competition submission. The actual saved code version
regenerates `submission.csv` with Kaggle's hidden inputs.

| Epoch | Accepted submission | Optimizer steps |
| --- | --- | --- |
| 10 | 56663180 | 2,860 |
| 8 | 56663742 | 2,288 |
| 6 | 56663606 | 1,716 |
| 4 | 56663969 | 1,144 |
| 2 | 56664013 | 572 |

The final submission order was 10/6/8/4/2, determined by public notebook readiness;
the selected checkpoints remain evenly spaced. CPU-only job **863548** used the
pinned exact coordinator to reconcile every accepted identity and actual remote
outcome into the one canonical ledger. At **2026-09-29 05:35 UTC**, all five
hidden statuses, errors and scores were null (`pending_or_unknown`). Initial
zero bytes without a terminal error are not evidence of failure. See
[the final live receipt](final-hidden-outcomes.json) and
[exact identity bindings](final-campaign-bindings.json). Live account limits
confirmed five submissions today and zero slots. Never repeat these submissions.

The [read-only outcome helper](reconcile_five.py) rejects changed canonical
identities before recording remote outcomes; it performs no Kaggle mutation.
The historical immutable snapshot's `submitted=false` field remains untouched;
current ledger state and IDs are authoritative. No root cause or score is
inferred from successful public execution. Accepted data masks/folds, original
training/runtime bytes, checkpoint bytes and raw sigmoid probabilities remain
unchanged. No GPU training allocation was used for this campaign.

The three-submission stage and all five saved notebook bindings were preserved
privately at Hugging Face revision
`2d0d249861480160278164fb3c75264caf8dcccc`; all **163** small objects were fully
re-downloaded and size/SHA verified. Preserve the complete campaign in a
separate bound small milestone, with its own no-retry intent and verification.

The **complete five-submission** milestone is privately preserved at Hugging
Face revision `94f33ae7bf0a37328fd807bc5d9d1adfbb8bff9d`, bound to Git source
`5ef7e92`. All **181 objects** were forced-downloaded and matched by byte size
and full SHA-256. The separate intent/result/verification receipts here bind
that immutable revision; no checkpoint weights were uploaded again. GitHub main
contains the complete campaign and both private milestone receipts.
