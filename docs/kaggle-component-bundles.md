# Checkpoint-only Kaggle payloads over an immutable common bundle (prototype)

Status: code-only prototype from issue #10. No Dataset, notebook, submission,
GPU job, weight download or account mutation has been made with it. CPU tests
are not Kaggle runtime or accuracy evidence. The active public-source notebook
`godlydonuts/rsna-publicsource-offline-e10-ebeb3c394286` v1, Dataset 12548500 v1,
checkpoint `ebeb3c394286`, its finite finish coordinator and Newton 899766 are
untouched. Their pinned projections do not load this code.

## Formats

Standalone `bundle-manifest.json` (schema 1, no `common_bundle` key) is unchanged
and still loads, packages, publishes and runs on the old path.

Optional component format:

- Common Dataset (`scripts/kaggle/components.py`): root `common-manifest.json`
  (`kind=rsna_common_encoder_code_bundle`) plus only `encoder.json`,
  `encoder/*`, `code/*`, `DINOv3-LICENSE.md` and `wheelhouse.json`. It binds
  the competition, checkpoint storage format (delta base present exactly when
  `frozen_base_delta`), licence source/hash, encoder spec hash, code-tree hash,
  shared wheelhouse proof and every member's SHA-256/size. It is non-diagnostic,
  requires private storage and is **not** competition-submission eligible
  (`competition_submission_eligible: false`, issue #13): it is an asset, never a
  checkpoint entry. `extract_common` derives it from an existing, fully verified
  standalone payload tree without acquiring weights (see bootstrap below).
- Checkpoint-only Dataset: `checkpoint.pt`, `snapshot.json`, optional
  `fitting-admission.json`/`baseline-proof.json` and `bundle-manifest.json`
  with `common_bundle`: exact common slug, Dataset id, version 1, manifest
  SHA/bytes, member-table hash/count/total bytes, the full-download
  verification receipt hash, storage format, licence, encoder, code and wheel
  manifest hashes. There is no latest-version lookup and no partial fallback.

`prepare_bundle(..., common_bundle=dir)` takes a directory with
`common-manifest.json` and its `common-verification.json` (the
`publication.reconcile(..., manifest_name="common-manifest.json")` receipt:
private, version 1, `downloaded_all_bytes`, exact files). It re-derives every
common member from current sources and refuses any byte difference (changed
code, licence, base or encoder spec), then copies only checkpoint members. It
requires a shared wheelhouse and a notebook slug/title distinct from the
checkpoint Dataset, common Dataset and wheel Dataset from the start. Two
observed same-identity SaveKernel 409 attempts succeeded only with a distinct
notebook identity; that is operational evidence, not a proven server root
cause. Never retry an ambiguous original.

Kernel attachments are `[checkpoint, common, wheels]`; `check_kernel` keeps the
exact sorted multiset comparison, so order is not identity while missing,
extra or duplicated attachments still fail. Private/offline/T4/source/runtime
checks are unchanged. `verify_remote_dataset` additionally checks the common
Dataset is owned by the account, private, `currentVersionNumber == 1` and has
the bound id. The notebook finds exactly one attached `common-manifest.json`
with the bound bytes, rejects missing/extra/linked/escaping members and
case-insensitive path collisions, then streams every common and checkpoint
member into a fresh `/tmp/rsna-composed-bundle` while hashing. The composed
tree is byte-identical to the standalone tree, so source/model loading,
`encoder_arguments` and sigmoid math run unchanged.

## Node-local publication journals (new operations only)

`coordinator.py --node-local-journal-root /tmp/rsna-kaggle-journal-<ckpt12>-<manifest12>`
is opt-in and accepted only for new component operations with no existing
shared journal (`dataset-publications/<sha>` or its revision). Existing
journals are never moved. The one canonical `account-ledger.sqlite3` and the
shared `coordinator-claim` stay shared. The transport, upload ZIP, intent,
response/uncertain records stay private (0700) under the bound node directory
(host, uid, device, inode). Shared storage receives two small files per
operation instead of a directory plus four or more journals:

- `<sha>-<manifest12>.node-intent.json`, written exclusively before the one
  create RPC (files, transport, binding, `retry_prohibited`). It is never
  rewritten; a second create attempt fails on it.
- `<sha>-<manifest12>.node-verified.json`, the full-byte reconciliation receipt
  later phases require.

An unverified intent marker blocks every other Dataset create, as unverified
shared journals already do. If the owning node directory is lost, creation is
refused and read-only reconciliation proceeds from the shared intent marker.
Any later run without the flag for a marked operation fails instead of falling
back to shared storage. Original receipts should still be preserved into Git
audits.

## Byte/cost comparison (existing receipts only)

From `initiative/audits/kaggle-publicsource-e10-ebeb3c394286-v1/Dataset-byte-verification.json`
(`python scripts/kaggle/components.py <receipt> --candidates N`):

| | bytes |
|---|---:|
| Standalone Dataset 12548500 v1 | 1,338,824,684 |
| Common members (encoder 1,212,560,553 + code 670,187 + licence/spec/wheel manifest) | 1,213,254,920 |
| Checkpoint-only payload (checkpoint 125,556,365 + sidecar/admission + 12,132 B old manifest) | 125,569,764 |
| Upload saved per additional candidate | 90.6% |
| 5 candidates: standalone vs common once + 5 checkpoint-only | 6,694,123,420 vs 1,841,103,740 |

The real component manifest has fewer member rows than the 12,132-byte
standalone manifest, so this slightly overstates the checkpoint-only size.
Across the 14 delta-format standalone Datasets already verified in
`initiative/audits/` (ids 12287425 to 12548500), the frozen encoder was uploaded
14 times (16.98 GB). One common bundle would have saved about 15.76 GB of
upload and Kaggle Dataset storage. Weights still mount on Kaggle at inference time;
the notebook adds one local copy (~1.2 GB) before inference. Its runtime cost has
not been measured.

## Common Dataset coordinator (issue #13, code-only)

Opt-in phases, separate from every checkpoint phase:

```
coordinator.py prepare-common-transport|create-common-dataset \
  --projection P --projection-manifest P.json --projection-sha256 SHA \
  --common-operation DIR/common-operation.json \
  --node-local-journal-root /tmp/rsna-kaggle-common-journal-<common-manifest12> \
  [--management-python PINNED]   # create only; token on stdin as before
coordinator.py reconcile-common-dataset \
  --projection P --projection-manifest P.json --projection-sha256 SHA \
  --common-operation ANY/common-operation.json   # marker-only; no journal root
```

The CLI rejects `--operation/--snapshot/--publication-revision/...` for these
phases and `--common-operation` for checkpoint phases. Original standalone and
checkpoint phases are unchanged.

### Operation format `common-operation.json`

`kind=rsna_common_dataset_operation`, written once (`x` mode) by
`components.prepare_common_operation` next to the extracted `common/` tree. It
binds:

- frozen common bytes: common manifest SHA/bytes, Dataset metadata SHA, member
  table hash/count/total bytes, checkpoint format, licence source/SHA, encoder
  spec SHA, code-tree SHA, wheel manifest SHA and the shared wheelhouse proof;
- `account` = slug owner; the coordinator requires it to equal the
  authenticated account and the existing canonical `account-ledger.sqlite3`
  identity (the ledger must already exist; it is only read/verified, no entry
  is written);
- `source_receipt`: original standalone Dataset slug/id/version 1, SHA of its
  full-download private receipt, its bundle-manifest SHA and checkpoint SHA;
- `regenerated_common_sha256` (must equal the member-table hash);
- `node`: host/uid/device/inode of the common tree. A moved/copied tree or a
  different node fails `load_common_operation`;
- `checkpoint_entry: null`, `competition_submission_eligible: false`,
  `retry_prohibited: true`.

### Phases

All run under the shared `coordinator-claim`, in a CPU-only Slurm allocation,
from one projection (below).

1. `prepare-common-transport`: refuses if either shared marker exists; creates
   the exact 0700 node-local journal, rehashes every tree byte and writes one
   stored ZIP plus `transport.json` (archive SHA/bytes, operation/projection
   SHA, node binding, explicit footprint, `deletion_credit: false`). No network.
2. `create-common-dataset`: refuses if any intent/verified marker exists, if
   the journal is not the preparing node's, or while any other Dataset intent
   (shared journal or node marker) is unverified; read-only shared wheelhouse
   check; then exclusively writes the shared
   `dataset-publications/common-<manifestSHA>.node-intent.json` and calls the
   existing `publication.create_once` (complete owned-listing absence proof,
   node-local `intent.json`, one create, `response.json` or `uncertain.json`).
3. `reconcile-common-dataset` (`coordinator.reconcile_common`): strictly
   read-only and marker-based. It never reads the common tree, the node-local
   journal or the host identity and never calls `load_common_operation`; the
   CLI rejects a journal root for it. It reads the reviewed operation file
   (any path), whose exact bytes must hash to the intent's
   `common_operation_sha256` and whose content must equal the intent's
   embedded `operation`; the intent's account/slug/version 1/private/
   `manifest_name` and member table must equal that operation (member-table
   hash, count, total bytes, wheel manifest, manifest SHA/bytes). Then the
   shared ledger identity is checked and
   `publication.reconcile(..., manifest_name="common-manifest.json")`
   downloads and hashes every byte of the private v1 Dataset against the
   intent table, and `common-<manifestSHA>.node-verified.json` is written
   exclusively (or must equal it). It therefore works after complete loss of
   the creating node (journal, tree and operation directory) and on any host.
   Recovery of a lost operation file: write
   `components.common_operation_text(intent["operation"])`; it must reproduce
   the original hash. Altered operation bytes/fields or altered marker tables
   are refused before any remote call.

Any intent marker makes creation permanently read-only: an unknown or failed
result, a crash between marker and RPC, or an absent Dataset after the marker
all require reconciliation/manual review, never re-create. The unverified
`common-*.node-intent.json` also blocks every checkpoint Dataset create through
the existing `pending_dataset_mutation`.

The verified marker is the `common-verification.json` for
`prepare_bundle(common_bundle=dir)` (copy it with the tree's
`common-manifest.json`). A component checkpoint operation now additionally
requires that marker to exist with exactly its bound `verification_sha256`, and
all component phases (common and checkpoint) require the running coordinator
and `publication.py` to be byte-identical to the projection's copies.

### Footprint to fund (no deletion credit)

Shared storage, per common Dataset: exactly two new JSON files, the intent and
verified markers, each carrying the full member table; the intent also embeds
the full original operation (including the shared wheelhouse proof). For the
63 common rows of the `ebeb3c394286` receipt they are
roughly 12-20 KB and 14 KB (estimates, unmeasured on the actual tree;
synthetic fixtures are asserted below 64 KiB total). Plus the existing ledger
open/identity check; no directory, upload or entry.

Node-local (`/tmp` of the creating node): the extracted common tree
(1,213,254,920 B of members plus manifest/metadata), the stored ZIP (same
bytes plus ZIP headers), and small journal JSON. About 2.43 GB plus the
existing 20 GiB `build_zip` floor. Nothing is credited for deleting the
original standalone payload, Dataset 12548500 or any other tree; they stay.

The notebook's `/tmp/rsna-composed-bundle` copy (about 1.2 GB plus
checkpoint) and its runtime remain **unmeasured until an actual Kaggle run**.

### Bootstrap review

`extract_common` now requires the original standalone payload's full-download
private v1 receipt (`components.standalone_source`) and the candidate's
regenerated common table (`workflow.common_table`, the same source assembly
`prepare_bundle` uses, no copying). Old bytes are reused only if every common
member is byte-identical; differing names are reported and nothing is
written. No older payload is assumed to match. Concretely, the `ebeb3c394286`
receipt's 58 code members already differ from this branch in
`code/rsna/competitive/{data,infer,train}.py`, and this branch has one more
`rsna/*.py` file. Common code is per-candidate code: only candidates whose
training `rsna/`, inference wrappers (`chunked_infer.py`, resident/reviewed
adapters), encoder, licence and wheel manifest regenerate identically can share
one common Dataset. `prepare_bundle` checks this again before copying.

### Proposed single immutable projection

One `project_code.py` projection of the reviewed merge commit (with
`--preserve-training-code` and the candidate's exact training commit) contains
`components.py`, `coordinator.py`, `publication.py`, `workflow.py`,
`notebook_entry.py`, `chunked_infer.py`, `project_code.py`, every other
`scripts/kaggle` file, `rsna/*.py` and `pyproject.toml`
(`coordinator.COMPONENT_PROJECTION` lists the required ones). Run the
coordinator from that projection. Its `projection_sha256` must be
root-reviewed before use; it is not fixed here because it depends on the merge
commit.

### Local CPU demonstration

`tests/test_kaggle_common_coordinator.py` builds a real Git commit of this
checkout's `rsna/`, `scripts/kaggle/` and `pyproject.toml`, projects it with
`project_code.build`, and runs the projection's own coordinator/workflow. With
the real `validate_snapshot` (only the root release validator and delta model
reconstruction are faked, both tested elsewhere) it runs: an older standalone
payload and its receipt through a fake remote, regeneration, operation,
prepare-common-transport, a create whose response is lost after upload,
refused re-create, full-byte reconcile, checkpoint-only prepare, checkpoint
transport/create/reconcile, one push and a refused second push, then notebook
composition from the fake remote's bytes. Other cases cover a second node,
a lost journal, loss of both journal and common tree, complete node loss with
an actual hostname change (`os.uname` differs) and a recovered operation,
host change blocking prepare/create, altered recovered operations and altered
marker tables, absent result after an intent, moved trees, changed members,
account/ledger/projection guards and the regenerated-code bootstrap refusal.
This is engineering evidence only, not runtime or accuracy.

## Remaining actual Kaggle acceptance requirements

1. Code review of this path and a root-reviewed projection. Fresh storage/file-
   quota acceptance funding the footprint above. Then, on one node, a
   `workflow.common_table` regeneration for the first candidate and a single
   `prepare_common_operation` from the verified standalone payload (expected to
   be refused unless its code matches), followed by the three common phases.
2. A first checkpoint-only Dataset, then an exact private offline 2xT4 public
   MRI-only notebook run with a distinct identity. Its full public acceptance
   (receipt/metrics/CSV hash, 13 columns, no fallback, memory) is required
   before any submission through the single coordinator and fresh allowance.
3. Measure the notebook's added composition time and `/tmp` usage on Kaggle.
   Confirm Kaggle's mount layout for three attached Datasets.
