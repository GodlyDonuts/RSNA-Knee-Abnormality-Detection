# Staged remote epoch publication coordinator

Authored by `/root/competition_contract`. This is an ignored preparation artifact,
not an integrated production helper. No production snapshot has been published,
queued or submitted. The separately authorized synthetic structure probe contains
no checkpoint or medical data and uses an offline CPU notebook only.

## Exact code and state

Use `projection-e7b0862/projection.tar` and its bound `projection.json`, generated
from Git `e7b08625854de692b9266ccfc25c90fc22169f58`. The 42 selected files contain
528,632 source bytes; the tar is 563,200 bytes. Its SHA-256 is
`6c40684334eb047dc12a659b9514e99d868bfddcefadf5a8376a3eaa6330018a` and manifest SHA is
`ad3bbf17ed40c274a8ce9c000dc03cae92c165eb23c829a65a120e18ff355188`.
The projection contains all `rsna/**/*.py`, Kaggle helper Python/shell files and
`pyproject.toml`; no Git objects, reports, labels, cache arrays, weights or secrets.
`project_code.py` reads exact Git objects, rejects links/LFS placeholders and
requires training/runtime source dictionaries to match. The older 51df024
projection remains a historical planning artifact and must not be deployed for
checkpoints made with the corrected runtime.

Keep the original training checkout clean. Extract the small projection into a
separate private directory. Verify the tar hash before extraction, all member
paths/modes against the bound manifest, and the manifest hash before use.
`coordinator.py` rechecks all source file hashes and exact Python membership on
every call. All unlisted files (including bytecode/native modules) and links are
rejected before imports; bytecode writing is disabled in that fresh process. An actual snapshot must match that exact `rsna` dictionary, the
accepted release binding and its internally completed epoch. There is no label
selection or access to gold/reserved confirmation in this helper.

One canonical queue is `$HOME/rsna-knee/kaggle/account-ledger.sqlite3`, fixed in
the helper. This must be the only account-wide queue across runs. Keep rollback
journaling, synchronous FULL, mode0600, and one coordinator at a time. An atomic
shared-directory claim prevents concurrent coordinators; a killed process leaves
its claim. Never automatically steal it: first inspect the Slurm job and all
mutation intents. SQLite multi-host locking on Lustre has not been accepted.

## CPU allocation and runtime

Do not allocate a waiting job before an eligible snapshot exists. Once one of
epochs 2, 4, 6, 8 or 10 is complete, dispatch one finite zero-GPU job, initially
2 CPUs, 16 GiB RAM, and a measured bounded wall time (up to60 minutes for a real
checkpoint upload/verification). GPU memory is unnecessary. Run on an allocated
compute node, not the login node; `coordinator.py` requires a Slurm job and no
GPU allocation. Root's sole dispatcher owns actual job submission.

Use the already accepted Python3.12 environment archive from
`$HOME/rsna-knee/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1`:

- `environment-cp312-cu130.tar`: SHA
  `89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338`,
  6,401,699,840 bytes.
- `wheelhouse-cp312-cu130.tar`: SHA
  `e326c50a4a6ac3c2a69423d9b97405952f815e8f55ea7867cf92e3185539416f`,
  3,178,014,720 bytes.

Use the separately hash-pinned `scripts/newton/runtime_environment.py` and
`runtime_archive.py` from the clean corrected training checkout to extract into
node-local scratch. They are deliberately not silently included in the 42-file
projection. Preserve their executed hashes with the job. Verify `/usr/bin/python3.12`
on the node and use the unpacked interpreter directly, not archived entry-point
shebangs. Unset PYTHONPATH; set PYTHONNOUSERSITE=1. Keep the shared environment
archived to avoid inode pressure. The full wheelhouse must also be extracted in
scratch for the existing prepare_bundle validation; model payloads reuse its
already published immutable private Dataset, so those wheels are not uploaded
again at each epoch.

The accepted runtime does not include Kaggle SDK transport dependencies.
Create a separate small **node-local management venv** with Python3.12 and pin
`kaggle==2.2.3`, `kagglesdk==0.1.30`; preserve a dependency receipt for the actual
installation. Do not modify the accepted training environment. The coordinator
runs under the accepted runtime and invokes the management interpreter as an
isolated subprocess only for upload/create. This installation has not been run
as part of this preparation task; it was used for the previous wheelhouse upload.

Measure free space before extraction and each packaging phase. Keep at least
20 GiB free. Account for environment, wheels, the actual immutable checkpoint,
prepared copy, and explicit ZIP simultaneously. A complete training checkpoint
may contain optimizer tensors: do not budget only the pretrained model's1.2 GiB.
Large artifacts never transit the laptop. Prefer node-local prepared copies/ZIP;
retain a few immutable archives and small receipts in the private shared root.
No automatic deletion of original snapshots is authorized by this helper.

## Prepared operation and actual shared runtime

First use the committed `workflow.prepare_bundle(...)` API or its existing
`prepare` command on a real immutable epoch sidecar. Supply the actual encoder
provenance/config, full original DINOv3 license plus immutable source/hash, accepted
wheelhouse directory, new output directory, dedicated model Dataset and kernel
slugs containing the checkpoint SHA prefix, and `--shared-wheelhouse` pointing to
the preserved `shared-wheelhouse.json`. It calls strict snapshot eligibility;
there is no placeholder/diagnostic override.

Stage the small proof file from the archived deployment evidence, preserving SHA
`7ba1b76afd9da427d22a9a0e0b089b9647111d95057e8650ad1887f8ba1c0ee5`.
It pins private Dataset `godlydonuts/rsna-wheels-cp312-e326c50a4a6a`, ID12139697,
version1, archive bytes/hash above, and the actual completed T4 probe. The current
notebook validates archive hash, complete internal manifest and accepted
installed versions before inference. The shared Dataset and probe source/settings
are rechecked online before model Dataset creation and notebook launch.

The transport uses a single explicit `bundle.zip` with full internal paths:
`bundle-manifest.json`, `code/rsna/...`, `encoder/config.json`, `checkpoint.pt`,
`snapshot.json`, `encoder.json`, `wheelhouse.json`, and `DINOv3-LICENSE.md`.
It does not use per-directory implicit ZIPs. The SDK's convert_to_csv argument is
ignored in2.2.3; we do not claim it disables server conversion. Only expected
non-tabular payload types are allowed and every resulting remote byte is checked.
Dataset metadata is sent via the typed create request and is not an extra member.

## Finite phases and credentials

The reviewed helper has five finite phases. All require an exact projection,
manifest, manifest SHA, actual epoch sidecar and prepared operation. They never
launch Slurm jobs or wait for future epochs. For each phase, use the same common
arguments shown here with actual previously verified private paths:

```sh
"$ACCEPTED_PYTHON" coordinator.py prepare-transport \
  --projection "$PROJECTION" --projection-manifest "$PROJECTION_MANIFEST" \
  --projection-sha256 ad3bbf17ed40c274a8ce9c000dc03cae92c165eb23c829a65a120e18ff355188 \
  --operation "$OPERATION" --snapshot "$ACTUAL_EPOCH_SIDECAR"
```

The remaining phases are `create-dataset --management-python "$MANAGEMENT_PYTHON"`,
`reconcile-dataset`, `push-kernel`, and `submit`, with the same common arguments.
The first creates/validates the explicit ZIP locally and records its SHA. Parent
reviews the exact code/operation/transport receipt before first real deployment.
The network phases read a JSON object containing only the Kaggle token from stdin.
Stream it from the existing local0600 secret through the private SSH ControlMaster
and the finite allocated process. Never place tokens in arguments, batch scripts,
Slurm environment exports, journals, response dumps, remote files, or shell logs.
No raw exception/HTTP bodies or signed URLs enter persistent outputs.

`create-dataset` exhausts authenticated owned-Dataset pagination, rejects an
existing or ambiguous target, verifies the exact ZIP, and fsyncs a unique intent
before upload. The final `CreateDataset` is one direct typed SDK call, bypassing
high-level retries. A timeout or error creates an uncertain journal and permanently
blocks another create attempt for that checkpoint. A response alone is insufficient.

`reconcile-dataset` is read-only remotely: require private ID/version1 and READY,
exhaust versioned file membership, accept at most one common archive wrapper,
stream every immutable file and check full SHA/size, then recheck Dataset identity.
Unknown/403/missing/pending results do not permit retries. ZIP download wrappers
are not guessed; unexpected bytes block. Successful verification is frozen and
bound to the exact intent/operation. Repeating reconciliation only accepts an
identical record.

`push-kernel` requires that complete-byte proof plus fresh Dataset identity and
uses the existing durable one-shot notebook intent. The notebook is private,
offline, T4, nine-hour cap, with only the competition and the two exact private
Datasets. Its loader rechecks mounted payload bytes before inference. Scheduled
epoch inference explicitly uses FP32 and strict checkpoint/config reconstruction.
`submit` requires the exact completed notebook version and its bound valid
inference receipt, rechecks live account allowance, and makes one code-competition
submission. There is no local three-row sample submission path.

Use existing `workflow` read-only status/reconciliation and progress projection
between phases. A heartbeat discovers the next completed primary-run epoch while
training continues, never waits until epoch10 to upload everything. Checkpoints
2/4/6/8/10 are the planned trajectory. Five is an account-wide daily maximum,
not a promise of five remaining slots. Query live allowance before each attempt;
unknown reset time/uncertain mutation blocks. Public scores may lag; preserve
null for pending scores and join only matching checkpoint/run/epoch training loss.
Public trajectory feedback is observational; no gold/confirmation selection or
accepted-label edits occur.

## Acceptance and limits

The source projection is actually built from fixed Git. Pure guard tests cover
layout, source/ZIP tamper, unsupported conversion-prone files, disk floor,
pagination, wrong private version, exact remote membership, streamed tamper,
unsafe redirects and durable uncertain-create replay refusal. The actual tiny private Dataset/CPU notebook proof passed: private Dataset
`godlydonuts/rsna-zip-structure-a3ef941019b6`, ID12140092/version1, all3 remote
files fully streamed and then identically verified at the notebook mount. It is
preserved under `artifacts/kaggle-dataset-structure-probe-v1`. The actual notebook
is `godlydonuts/private-zip-structure-preservation-probe`, version1, private,
offline and CPU-only. SaveKernel derived that slug from its title rather than the
requested synthetic slug; the returned identity and exact source were reconciled
read-only, with no second creation. Production preparation already derives title
from the dedicated slug, and this coordinator now explicitly requires that match.
The mounted receipt SHA is
`159862669ace10b6280286989a24bdd306cfa9b183c41d9d0acb42df52c25c27`.
The remote byte receipt SHA is
`bbe196293b02fa4d9d20732e56083713748d28d58fc14693775be9597d57f925`.

This helper is not yet integrated, and no eligible production snapshot exists.
Remaining actual gates are real completed epoch eligibility, one allocated CPU
management environment, actual snapshot-sized resource timing, the exact model
notebook's full pretrained inference and nine-hour hidden-set runtime. Tiny
structure and previous tiny T4 probes do not establish those outcomes or a score.
