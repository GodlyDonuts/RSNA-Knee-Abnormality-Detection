# Epoch 2: SaveKernel returned no identity (draft-aware reconciliation proposal)

Code and tests only. Nothing was run against the cloud while writing this: no RPC, SaveKernel, UI Save Version, Dataset, push or submission. The canonical epoch 2 entry stays `push_uncertain` with its original intent, `kernel_absence_proof` and `kernel_create_response` (all `null`). Root's deferral of epoch 2 and its single finite controller for epochs 3–5 are unchanged.

## Observed facts (root, 2026-10-14)

- Dataset `godlydonuts/rsna-alldev-e002-9422444c2144` was created once and all 68 objects were verified.
- One `SaveKernel` call for `godlydonuts/rsna-alldev-offline-e002-9422444c2144` returned no `kernelId`, `ref` or `versionNumber`. The ledger moved to `push_uncertain`.
- `GetKernel` on the exact slug returns HTTP 403. The profile search returns only the `[Private Notebook]` placeholder.
- In the owner's authenticated UI, the exact dedicated editor opens with the correct inputs and code. Version History says *Starting Fresh, no saved versions*.

## What changed

1. **Capturing the response** (`scripts/kaggle/workflow.py`). `save_kernel_response` replaces the old 3-field capture. It keeps the legacy `kernel_id`, `ref` and `version` keys. It adds bounded and sanitized `error` and `url`, the `invalid*Sources` and `invalidTags` lists (count plus the first 8), and the response key names (up to 32). Token-like (`KGAT_…`, `Bearer …`) and signed-URL substrings are redacted. Text over 240 characters keeps a prefix, its length and a hash of the sanitized text. `push` now also records `kernel_create_exception` (type plus bounded message). Existing ledger records are never rewritten. This only helps future calls; the epoch 2 response was already lost.
2. **Classification** (`coordinator.observe_uncertain_push`, read-only):
   - `committed_version`: `GetKernel` is accessible and has an integer `currentVersionNumber ≥ 1`. That version is immutable, so use the existing `workflow.reconcile_push(version)`, which checks exact source, metadata and sources. Never commit again.
   - `private_draft_owner_attested`: *all* of the following hold:
     - `GetKernel` returns exactly HTTP 403.
     - The exact-slug search returns exactly the placeholder.
     - There is a fresh owner UI attestation (≤ 900 s old, exact key set) that names this account and slug, with `version_history == "starting_fresh_no_saved_versions"`, inputs and code matching, and no UI Save Version or SaveKernel retry performed.
   - `unresolved`: anything else. In particular, **the placeholder alone proves neither absence nor a draft.**
3. **First-commit continuation** (`coordinator.commit_private_draft_once`, coordinator action `commit-private-draft`).
   - **Admission conditions.** All are checked under the account-wide `BEGIN IMMEDIATE` lock:
     - The entry is exactly `push_uncertain`.
     - `kernel_version`, `submission_id` and `description` are absent.
     - There is no prior `draft_commit`.
     - The ledger `operation_sha256` equals both the file digest and `--uncertain-operation-sha256`, and the stored operation equals the reloaded one.
     - The original `kernel_absence_proof` and `dataset` are present, and the original response has `null` id, ref and version. A returned *different* ref is the title-slug case and stays refused.
     - The operation was never superseded, and the owner is the authenticated account.
     - The rebuilt request has the exact slug, and its source hash equals `kernel_source_sha256`.
     - The classification is `private_draft_owner_attested`.
     - `verify_remote_dataset` equals the stored dataset identity.
   - **Commit sequence.**
     1. The function writes `draft_commit` (attestation, observation, request hash, hash of the original response, `expected_version: 1`, `retry_prohibited`) and `state=draft_commit_intent` durably.
     2. It then makes exactly one `SaveKernel` with the **same slug and request**. It never uses a new slug, so it cannot create a second notebook.
   - **Result handling.**
     - Only `versionNumber == 1` (int), no `error`, and `ref == "/code/<exact slug>"` move the entry to `pushed`/`kernel_version=1`.
     - Anything else, including an exception, becomes `draft_commit_uncertain`, with the sanitized response and exception stored.
     - A version of 2 or more means a hidden committed version already existed. That entry stays uncertain and is never submitted.
   - **After the call.** `draft_commit_intent` and `draft_commit_uncertain` can never run the commit again. `reconcile_push` (read-only exact version 1) accepts them, and `daily_queue.plan` routes them to read-only `reconcile-push`. The existing distinct-slug supersession (`push-after-uncertain-placeholder`) still requires `push_uncertain`. It is therefore unavailable after a draft commit, and it should **not** be used for epoch 2, because a draft already occupies the exact slug.
   - **No duplicate run.** A draft has no run. Version 1 is the first and only run. Submission still requires the existing exact public acceptance of version 1 and fresh allowance.

## Residual risk (stated, not hidden)

Kaggle's API hides this slug behind 403, so the only evidence that no committed version exists is the owner's UI Version History. If that observation were wrong, the single commit would return version ≥ 2. The guard then refuses to promote or submit it, but cannot undo the extra version. The 15-minute freshness window and the requirement that nothing changed in between keep this risk small. A UI *Save Version* by the owner is an alternative with the same evidence base. Afterwards, root would reconcile read-only with `reconcile_push(…, 1)`, whose exact-source, metadata and T4 checks still apply. Do not do both.

## Root execution handoff (only when root decides to resume epoch 2)

1. Do not run this while another controller could mutate. Root's current single controller for epochs 3–5 must be terminal or idle between mutations, with no `push_intent`, `submit_intent` or `submit_uncertain` anywhere in the canonical ledger.
2. Have the owner re-open the exact editor `godlydonuts/rsna-alldev-offline-e002-9422444c2144`. Then write an ignored 0600 JSON file with exactly the keys in `DRAFT_ATTESTATION_KEYS` and `observed_at` set to now in UTC ISO format. Commit a copy to this directory afterwards.
3. In one CPU-only Slurm step on evc40, with the exact epoch 2 projection, operation and snapshot, run:
   `coordinator.py commit-private-draft --operation <epoch2 operation.json> --snapshot <sidecar> --projection … --projection-manifest … --projection-sha256 … --uncertain-operation-sha256 <ledger operation_sha256> --draft-attestation <file> --node-local-upload-root /tmp/rsna-kaggle-upload-9422444c2144 --node-local-journal-root <same as before>` (token over stdin, as for the other phases).
4. On `pushed`, continue through the existing phases unchanged: `observe` → exact `accept-public` → `submit` with fresh allowance. On `draft_commit_uncertain` or an error, reconcile read-only only (`reconcile-push` version 1). Never re-run the command or delete an intent.
5. Before step 3, `observe_uncertain_push` can be run read-only to print the classification. It writes nothing.

Tests: `tests/test_kaggle_private_draft_commit.py` covers:
- sanitization and bounds;
- null-identity push capture;
- classification, including placeholder-only → `unresolved`;
- exactly one SaveKernel and preservation of the original record;
- terminal uncertain responses (empty, error, version 2, title-derived ref, transport);
- a crash after the intent leaves no retry;
- refusal of a changed or known identity, operation, source or dataset, and of stale or missing evidence;
- read-only reconciliation of the draft states and daily-queue routing.
