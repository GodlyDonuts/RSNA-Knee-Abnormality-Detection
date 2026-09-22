# Scheduled Kaggle code submissions

Submit the completed production checkpoints from epochs **2, 4, 6, 8 and 10**.
These are a fixed monitoring cadence, not checkpoint selection using the public
leaderboard, gold labels, or the reserved 441-study confirmation set. There are
at most five submissions per competition/team/day across all runs. Extra ready
checkpoints wait in the shared queue; they do not each receive five slots.

This is a **code competition**. The three public example studies test execution
only. We submit an exact saved notebook version that produces `submission.csv`;
Kaggle reruns that code with its hidden inputs. Never upload the local three-row
CSV as evidence of useful model performance. Competition constraints are in
[the current contract](competition-contract-20260922.md).

## Verified account access and remaining prerequisites

Read-only authenticated checks on September 22, 2026 confirmed `godlydonuts` has
entered competition `154281`. The new helper rechecked at **17:01:03 UTC**:
zero submissions today, five currently available, two lifetime submissions.
The two existing completed submissions are from September 20 and September 2;
this workflow did not create them or inspect their scores.

`GetSubmissionLimits` does **not** supply an exact reset timestamp. Its current
allowance is authoritative; `reset_at=null` and `reset_verified=false` preserve
that limitation. An overflow entry receives a **00:05 UTC inspection time**, not
a claimed quota reset. A fresh server query must permit the next submission.
Do not substitute a locally calculated midnight allowance for this check.

No competition submission has been created. The initial private environment
probe measured Python 3.12.13, PyTorch 2.10.0+cu128, driver 580.159.04 and two T4s.
A subsequent private Dataset, `godlydonuts/rsna-wheels-cp312-e326c50a4a6a`
(ID 12139697, version 1), now holds the exact wheel archive. Its separate private
offline notebook, `godlydonuts/rsna-wheels-test-e326c50a4a6a` version 1, passed
fresh installation of PyTorch 2.14.0+cu130 and Transformers 5.6.1, CUDA tensor
backward, tiny random DINOv3 forward and decoder checks on a T4. Neither probe
accessed competition data or pretrained weights. The small proofs and reusable
`shared-wheelhouse.json` are archived under
`initiative/audits/dinov3-deployment-v1/kaggle-t4-acceptance/`.
There is no eligible production checkpoint yet. A trained epoch snapshot and
successful complete model inference remain required before a competition entry.

## Local preparation

Run packaging and submission coordination on Newton with its validated Python
3.12 environment. The laptop cannot hold another checkpoint plus wheelhouse
above its 20 GiB reserve. Keep large bundles and the single canonical ledger in
the private remote staging area; mirror only small receipts locally. Run from
the preserved repository root. Set `RSNA_KAGGLE_LEDGER` to the one private remote
`$HOME/rsna-knee/kaggle/account-ledger.sqlite3` file before queue operations.
Use one coordinator process; SQLite rollback journaling is used, not WAL, and
Lustre cross-host locking has not been accepted. All commands
use `.env` internally, require private file permissions, and never print the
token. `--env-file` can select another private credential file. The authenticated
username must match `--account` and the ledger identity.

```sh
python scripts/kaggle/workflow.py account
python scripts/kaggle/workflow.py queue RUN/epoch-checkpoints/epoch-002.json
python scripts/kaggle/workflow.py list
```

The queue checks the sidecar and actual `weights_only` checkpoint, the exact
trained `rsna/**/*.py` hash dictionary, release binding, optimizer steps,
completed-epoch marker and configured two-epoch cadence. Diagnostic, zero-update,
partial, hash-mismatched or unscheduled checkpoints are rejected. Do not edit
runtime code after training and silently package the new version: preserve the
exact trained source tree.

Use **one ledger** for this Kaggle account across every run and fold. The default
must be set explicitly through `RSNA_KAGGLE_LEDGER` or `--ledger`; the helper
refuses queue mutations without it. All orchestrators must use that same file
on one submission coordinator. Never create a fresh per-run ledger to evade its
history. SQLite transactions and durable intent records protect concurrent
local callers. Kaggle's live team allowance also accounts for submissions made
outside this workflow. A concurrent outside action can still exhaust quota
between check and request; the resulting uncertain attempt blocks retries.

Prepare a wheelhouse on compatible Linux, retaining an actual offline-install
acceptance. `wheelhouse.json` requires `platform="linux_x86_64"`, Python major
and minor in `python`, a nonempty `cuda` version, `offline_install_verified=true`,
and `files` mapping each relative filename to its SHA256 and byte count. Include
every wheel and `requirements.lock`; the lock must pin all runtime dependencies
and hashes. Include source-bound `acceptance.json`, `resolved_versions`, and
`pyproject_sha256`; the helper requires actual CPU offline-install acceptance
against the exact current project pins. This is an evidence requirement, not permission to claim an unrun
installation passed. Use the chosen Kaggle Python ABI, compatible CUDA wheels
and DICOM codecs. Record the concrete acceptance in the training environment
handoff. No wheels are downloaded by the preparation command.

```sh
python scripts/kaggle/workflow.py prepare \
  --snapshot RUN/epoch-checkpoints/epoch-002.json \
  --encoder-spec PRIVATE/encoder-spec.json \
  --wheelhouse PRIVATE/linux-wheelhouse \
  --license-file PRIVATE/DINOv3-LICENSE.md \
  --license-source https://github.com/facebookresearch/dinov3/blob/COMMIT40/LICENSE.md \
  --license-sha256 LICENSE_SHA256 \
  --output PRIVATE/kaggle-epoch-002 \
  --dataset-slug godlydonuts/rsna-epoch002-CHECKPOINTSHA12 \
  --kernel-slug godlydonuts/rsna-epoch002-CHECKPOINTSHA12
```

Replace `CHECKPOINTSHA12` with the actual lowercase first 12 checkpoint hash
characters. Each checkpoint has its own Dataset and kernel slug. Preparation
creates a new directory only, checks a 20 GiB free-space floor before copying,
and writes `dataset/`, `kernel/`, and `operation.json`. Supply the full original
DINOv3 license with its SHA and immutable official source commit. Redistribution
of model derivatives must retain that agreement; private publication does not
replace these license terms.
[Official DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). The Dataset contains the
trained checkpoint, its source-bound sidecar, official encoder configuration,
exact trained code, dependency lock and wheels. Original pretrained weights are
already in the trained checkpoint; their original hash stays in the checkpoint
lineage, without another copy or download. No reports, gold labels, `.env`,
training images, optimizer-selection scores or unrelated workspace files are
collected. The checkpoint currently includes optimizer state from the immutable
trainer snapshot; it is not rewritten into a smaller unbound model.

## Private Dataset and notebook execution

First run `python scripts/kaggle/workflow.py validate PRIVATE/kaggle-epoch-002/operation.json`
to reject changed or extraneous payload files. The submission coordinator then publishes the prepared `dataset/` directory
as a **new private Dataset**, retaining its actual Dataset ID/version/upload
receipt. Preserve directory structure and disable automatic CSV conversion.
Use the official SDK's `dataset_create_new(..., public=False,
convert_to_csv=False, dir_mode="zip")`, or equivalent private Dataset UI upload.
Disable SDK mutation retries or use a separate durable publication intent before
that call; the installed SDK otherwise retries Dataset creation. On an uncertain
result inspect the exact slug read-only; do not create repeated versions. This
script deliberately does not invoke Dataset creation. Publication is a distinct
coordinator action and must finish before the guarded kernel push.

The official kernel metadata documents Dataset sources as `owner/slug`, rather
than an immutable numeric version selector. Accordingly, require the dedicated
Dataset to remain **private at version 1**, and embed the exact bundle manifest
hash in the notebook. Every payload file is checked in Kaggle before execution.
Version drift or changed bytes fail closed. A different checkpoint gets a new
Dataset, never an overwrite of an old checkpoint Dataset.
[Official Dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md),
[official notebook metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).

```sh
python scripts/kaggle/workflow.py push PRIVATE/kaggle-epoch-002/operation.json
python scripts/kaggle/workflow.py status FULL_CHECKPOINT_SHA256
```

`push` stores an intent before one `SaveKernel` call, then retains the returned
exact notebook version. The notebook is private, explicitly requests `NvidiaTeslaT4`, and has internet disabled,
with a nine-hour session timeout and only the private bundle and competition
as inputs. It installs from the local hash-locked wheelhouse with `--no-index`;
then invokes `rsna.competitive.infer` with `--encoder-config`, `--scheduled-epoch`,
`--precision fp32`, two CPU threads and zero loader workers. FP32 permits a T4
despite BF16 training on H100; numerical/runtime acceptance on the actual Kaggle
GPU remains required. The selected PyTorch 2.14 Linux wheel uses CUDA 13, so
Pascal/P100 is **not** supported by this package. The measured Kaggle T4/driver
580 passed the exact offline runtime probe. Both H100s on Newton evc37 also
passed tiny CUDA tensor checks with the same environment; the pretrained
DINOv3 recovery pilot remains separate.
[NVIDIA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). It discovers studies from Kaggle's current test
template, decodes the provided test DICOMs, and never reads test reports.

A successful public execution writes `kaggle-inference-receipt.json` binding the
manifest, trained checkpoint and output CSV. Its three-example status is explicit.
`status` requires the dedicated kernel's current version to equal the recorded
version before and after status inspection. Live probing found that the installed
SDK ignores a parsed version, while the typed RPC's `versionLabel` returns404
for both tested selectors. The helper therefore fails on any current-version
drift instead of silently accepting another version. Output verification uses `versionNumber`, bounds receipt
size and request time, and never logs signed download URLs.
[Official notebook operations](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

## Submission and reconciliation

```sh
python scripts/kaggle/workflow.py submit FULL_CHECKPOINT_SHA256
python scripts/kaggle/workflow.py status FULL_CHECKPOINT_SHA256
```

Before submitting, verify the exact private notebook source/settings and completed
version, fetch its bound inference receipt, confirm the Dataset remains private
version 1, and obtain a fresh account quota. Submission calls **CreateCodeSubmission**
with the exact kernel owner, slug, version and output filename. It never calls the
local-file upload submission API. The ledger binds checkpoint SHA, kernel version,
unique submission description and returned submission ID. A repeated `submit`
on a recorded success is read-only/idempotent.
[Official code submission API](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md),
[official CLI implementation](https://github.com/Kaggle/kaggle-cli/blob/main/src/kaggle/api/kaggle_api_extended.py).

Any timeout, transport error, missing reference or process crash leaves an intent
or uncertainty record. An uncertain submission blocks all further submissions
from this ledger until **read-only** reconciliation identifies exactly one remote
submission with its unique description. Missing results are not proof of failure;
never automatically resubmit after an empty list. Unexpected/ambiguous results
need investigation. Reconciliation does not clear uncertainty just because a
new day has begun.

```sh
python scripts/kaggle/workflow.py reconcile-push FULL_CHECKPOINT_SHA256 --version EXACT_VERSION
python scripts/kaggle/workflow.py reconcile-submission FULL_CHECKPOINT_SHA256
```

`reconcile-push` checks the exact prepared source/settings at the explicit version.
`reconcile-submission` records status and any public score without selecting or
changing the training plan. No automatic fallback to a diagnostic model, empty
template, alternate epoch or pretrained-only classifier exists.

## Recurring coordinator procedure

At each bounded check, discover new completed `epoch-checkpoints/epoch-*.json`
sidecars for the configured production runs and pass explicit paths to `queue`.
Use the ledger's existing order, rather than ranking by scores. Reconcile pending
attempts first; prepare/publish each newly valid immutable bundle once; check
running notebooks without pushing duplicates; submit completed versions only
while the freshly queried allowance permits it. After quota exhaustion inspect
again after the recorded UTC boundary, then trust only the new server allowance.
Keep the heartbeat quiet while nothing actionable changes. Root owns activation
and actual submissions; this document does not create a background scheduler.

Focused validation covers synthetic quota exhaustion across runs, UTC staleness,
durable timeout/reopen/reconciliation, idempotency, wrong accounts, notebook
tampering, Dataset privacy/version drift, incomplete runs, output checkpoint
mismatch, payload traversal/tampering and diagnostic/unscheduled snapshots.
Live acceptance covers authenticated account/quota reads, the initial environment
probe, and the exact shared-wheelhouse offline T4 probe described above. The
private version-bound production shared-runtime guard passed against those actual
resources. No trained-model result, full hidden-set runtime, score, ranking or
competition success is claimed.

## Linux dependency preparation

`build_wheelhouse.py` runs only in an allocated zero-GPU Slurm job with at most
four CPU cores. It uses node-local `/tmp`, resolves the exact project and
competitive pins to binary wheels from official PyPI, constructs a complete
hash lock, installs a fresh offline Python3.12 environment and checks imports,
CPU tensor backward, a tiny randomly initialized DINOv3 constructor/forward,
and DICOM decoder-plugin availability. That tiny synthetic forward is a
dependency check, never an eligible training or submission artifact. No
pretrained weights or competition images are used.

It persists one wheel archive, one installed-environment archive and small
proofs on private Lustre, then re-extracts the environment locally and repeats
the checks with its relocated interpreter. Invoke `environment/bin/python`
directly and use `python -m pip`/`python -m pytest`; archived executable shebangs
can retain their old temporary path. The measured shared base Python3.12 path
must exist on the eventual node. Basic H100 CUDA and the offline T4 runtime have
now passed; full pretrained model inference and throughput remain separate gates.
The first CPU-only probe failed before inspection due to a missing module-init
path; the corrected login-shell probe passed. Preserve both records.

The offline notebook creates a fresh node-local Python 3.12 environment, installs the complete hashed wheel lock without network access, and invokes that interpreter directly. This isolates the pinned runtime from Kaggle preinstalled optional packages.

## Observing the epoch trajectory

Queue each eligible completed snapshot as it appears while training continues.
Refresh its completed/pending score with `reconcile-submission`, then use `list`
for the complete ledger JSON. Kaggle scoring may lag the running training job.
For an additional read-only CSV/JSON projection:

```sh
python scripts/kaggle/progress.py --ledger "$RSNA_KAGGLE_LEDGER" \
  --history RUN/invocation.json --output NEW_PROGRESS_DIRECTORY
```

Repeat `--history` for preserved invocation histories across resumed segments.
Each row retains epoch, fold, seed, run binding, checkpoint hash, exact notebook
version and submission ID. Loss is the final training update at that snapshot's
exact optimizer step, not an epoch average. Only histories with the same binding
are joined; missing histories and pending/unknown scores stay null (blank in CSV).
Active stdout alone is not treated as bound history. No outer validation, gold,
confirmation labels, checkpoint ranking or training changes are involved. The
projection opens the canonical ledger read-only and preserves a logical snapshot
hash; use a new output directory for each observation.

## Reusing the verified wheelhouse

After the dedicated private T4 wheelhouse probe passes, its
`shared-wheelhouse.json` records the private Dataset ID/version 1, full archive
SHA/size, original wheel-manifest SHA, project SHA and exact private GPU-probe
source/version/result. `gpu_receipt_sha256` hashes canonical JSON, not the
formatted receipt file. Add `--shared-wheelhouse PRIVATE/shared-wheelhouse.json`
to `prepare` to reuse it. Supply the same verified local `--wheelhouse` directory
for preparation checks; its large wheel files are then omitted from the new
checkpoint Dataset. No optimizer or checkpoint bytes are rewritten.

The notebook attaches exactly two private Datasets: the new checkpoint bundle
and the shared immutable wheelhouse. Before a push or submission, the coordinator
rechecks both private version-1 identities, the exact completed GPU probe source,
and its archive-bound acceptance. The notebook independently verifies the full
archive bytes, extracts only confined ordinary files/directories, verifies every
wheel and lock byte, then installs offline in its fresh environment. Version,
source, receipt, manifest, payload or additional-file drift blocks execution.
Each checkpoint's notebook slug must be absent before the one SaveKernel call;
an existing notebook is never intentionally overwritten. The preflight absence
check is not a server-side compare-and-swap; use the single account coordinator.
The shared Dataset avoids repeated multi-gigabyte uploads; the checkpoint bundle
still contains its measured immutable snapshot, exact code/config and license.
