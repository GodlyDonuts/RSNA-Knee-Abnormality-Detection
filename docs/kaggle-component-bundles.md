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
  shared wheelhouse proof and every member's SHA-256/size. It is non-diagnostic
  and requires private storage. `extract_common` derives it from an existing,
  fully verified standalone payload tree without acquiring weights.
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

## Remaining actual Kaggle acceptance requirements

1. Fresh storage/file-quota admission. Then a single `extract_common` of the
   fully verified `ebeb3c394286` payload on the owning node, followed by one
   private common Dataset create through the canonical coordinator. A
   `create-common-dataset` coordinator phase still needs to be added and
   reviewed (publication now supports `manifest_name="common-manifest.json"`).
   Then a full-download reconciliation receipt.
2. A first checkpoint-only Dataset, then an exact private offline 2xT4 public
   MRI-only notebook run with a distinct identity. Its full public acceptance
   (receipt/metrics/CSV hash, 13 columns, no fallback, memory) is required
   before any submission through the single coordinator and fresh allowance.
3. Measure the notebook's added composition time and `/tmp` usage on Kaggle.
   Confirm Kaggle's mount layout for three attached Datasets.
4. A projection/review of the new `components.py` for future coordinator runs.
   Already-running pinned projections are unaffected.
