# Frozen 3,607-report checkpoint recovery preparation

This package prepares recovery of the existing 490-object operation. It does not
publish anything on import, and the preparation did not run its `--execute`
mode. The original tar, operation, manifest, dataset card and original receipt
location remain unchanged. Root separately stopped the original stalled
publisher; its stop record is required and hash-bound before execution.

The original publisher stalled inside `hf_xet`. The installed
`huggingface_hub==1.18.0` client reads `HF_HUB_DISABLE_XET` at import, and
`_commit_api._upload_files` offers only `basic` and `multipart` transfers when it
is set. The recovery sets that flag **before any Hub import**, checks the reviewed
client source hashes, and tests the actual negotiation path with a fake batch
response. The default client has `timeout=None`; recovery replaces it with
finite timeouts and clamps explicit `None` overrides and redirect requests too.

## Exact inputs and scope

- GitHub commit: `6f2d07d06111d6170b22cdac1f3b27b4d41f4d38`.
- Required completed/successful GitHub CI run: `35592976831`, for that exact commit.
- Private dataset: `GodlyDonuts/RSNA-Data`.
- Original expected parent: `492b1eb363680fe593087f84e9627b1de4ee8680`.
- Corrected source revision: `0b5e410288232f570c69ba4f44723356e6ae6ab7`.
- Original operation SHA-256: `8fc97e4f6997f3c702bca1e9bdacc024eb7d1e3671ed9b74c916eb3b2ea50d97`.
- Original tar SHA-256: `515f871a2da9b00572f3fda5f39a302b354b8db5775d5a014f8b19fb960993a6`;
  873,998,824 bytes, 3,610 exact files.
- All 490 uploaded objects together: 1,014,509,104 bytes.

The mapping reconstructs the original publisher's `files_to_upload`: three
files from the existing checkpoint directory, 486 committed audit objects, and
the already prepared `dataset-card.md` mapped to remote `README.md`.
`preparation-local-preflight.json` records full local SHA/size checks, all 486
audit-object comparisons with the pinned Git blobs, and streaming verification
of every tar member against the original manifest, without extraction or
clinical/gold inspection.

Three mutable root policy files are copied from the exact pinned Git blobs in
`frozen-mutable/` (8,410 bytes total). `mutable-policy-snapshot.json` binds the
original local locations, frozen copies, operation SHA and original Git commit.
Recovery uses those copies. The other 483 audit objects are immutable preserved
case evidence and must stay byte-exact. Current review index/progress, archive
manifest and docs are **not** direct upload dependencies: their old versions are
already inside the unchanged tar. Current aggregation may proceed without
altering the frozen package or those 483 historical audit objects.

## Execution after root review

From the workspace root, choose an unused output directory. These commands are
documented for the coordinating agent; preparation did not execute recovery.

```bash
.venv/bin/python work/annotation-staging/hf-review3607-recovery-v1/recover_checkpoint.py \
  --execute \
  --output work/annotation-staging/hf-review3607-recovery-v1/runs/attempt-001 \
  --max-runtime-seconds 1800
```

`--check-local` instead of `--execute` performs an offline-only check and writes
a new preflight record. Every run requires a new output directory. A lock
prevents two recovery runs from executing concurrently. Execution refuses while
PID 51153 exists or another original publisher command is running; it never
stops processes itself. The recorded root stop file must also match its frozen
hash. The token is loaded only from the existing `.env`, never printed or
written to results.

The recovery checks private HEAD, unchanged source tree/CSV identities and exact
CI. If HEAD is still the original parent, it revalidates all local payloads and
all `CommitOperationAdd.upload_info` digests before one parent-guarded commit.
It repeats the PID and HEAD guards immediately before that call. No package
rebuild, broad folder upload, delete operation or new dataset is supported.

If HEAD differs, the recovery performs **no commit**. It requires every one of
the 490 expected objects and the two source trees/five official CSVs to match at
that HEAD. A mismatch blocks recovery. A complete match produces an idempotent
verification receipt for existing publication content, without inferring which
process authored that commit or its actual Git parent. In that branch,
`hf_parent_revision` is null and the original intended parent is preserved as
`expected_hf_parent_revision`.

Every final object is verified at the immutable publication revision: LFS
objects by Hub SHA-256 OID and byte size, ordinary Git files by streaming full
HTTP bytes through SHA-256 and a strict byte count. This preserves the original
publisher's verification scope; it is not a fresh download of every LFS payload.
Both image directory tree IDs and all five official CSV blob IDs must match the
corrected source revision and original parent. Final private HEAD must still
equal the verified revision. Original local objects are rechecked once more.

## Time bounds, failure and outputs

Connect/pool/read/write timeouts are 15/30/120/120 seconds, with two upload
workers. The installed HTTP retry behavior remains in place. A separate
1,800-second default watchdog bounds the whole remote stage, including retry
waits, and terminates this recovery process if needed. Timeout does not establish
whether a commit landed: the next attempt must reconcile fresh HEAD again.

Each new run preserves `local-preflight.json` and append-only `events.jsonl`.
After a returned commit, the journal immediately preserves its revision as
verification-pending. Only complete verification creates
`publication-recovery-receipt.json`. The original
`hf-preparation-checkpoints/.../publication-receipt.json` is never changed.
An error or timeout never claims success or training readiness. Error output
contains fixed codes and exception type, not exception text, HTTP URLs,
authorization headers, subprocess stderr or token values.

The 38 offline regressions cover path/membership/payload corruption, unchanged
tar preservation, Xet-free negotiation, finite `None` timeouts, live/restarted
publisher refusal, all source identity fields, LFS and ordinary-file failures,
private/CI/HEAD/upload-info gates, exact-parent success, idempotent success,
verification-pending failure, immutable receipts and sensitive-error suppression.
No live HTTP request or remote commit was made by these tests.
