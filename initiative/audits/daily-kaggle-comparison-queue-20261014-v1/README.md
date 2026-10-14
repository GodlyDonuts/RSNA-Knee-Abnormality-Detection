# Daily Kaggle comparison queue: all-development epochs 1–5

Five immutable RSNA-only public-label native384 rank32 all-development epochs of
original job 908454 (evc40, source `1bcd6c0e`, 3,403 fitting / 0 outer studies,
`label_source_final_v1`, official DINOv3-L base). All five are already privately
published and fully download/hash verified. Original training stays unchanged.
Nothing here is a cloud action: no RPC, job, transfer, upload or submission was
run while preparing this code.

- `queue.json`: the exact owner queue (checkpoint SHA, bytes, steps, binding,
  fixed slugs). Mutable labels such as "latest" are not admitted.
- `root-sidecar-bindings.json`: root's sanitized evc40 sidecar read
  ([issue 64 comment](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/issues/64#issuecomment-6164379085)).

## Admission

`scripts/kaggle/reviewed_infer.py` role `all_development_public_source_epoch`
accepts only `reviewed_infer.queue_admission(epoch)`, compared key-for-key. It
checks that the checkpoint SHA matches its epoch, the full binding/code hashes
and source Git, epoch/step/next_update, scope 3403/0, the exact supervision
receipt (overlay, release, folds, rosters, 40,816 cells, Shio-Koube/knee3
revision and CSV hash), the official base, rank32/native384, and that no
external, private, diagnostic or public-input keys are present. The original
`infer.validate_inference_checkpoint` then runs on a metadata-only view.
Learned weights and numerical sources are never rewritten. Default, effusion
and public-final10 admissions are unchanged and cannot authorize these
checkpoints.

## Driver

`scripts/kaggle/daily_queue.py` handles one candidate at a time, in epoch order:
`prepare → prepare-transport → create-dataset → reconcile-dataset → push-kernel
→ public-acceptance → submit`. It uses the original `workflow.prepare_bundle`,
`coordinator.run`, the accepted FP32 prefetch prefix and the canonical ledger.

- Before each phase that is local or mutates the cloud, the driver writes an
  exclusive intent file, and it writes an exclusive result after. An intent
  with no result stops the queue for read-only reconciliation. `reconcile`
  closes it only from authoritative evidence: Dataset byte verification, ledger
  `pushed` or `submitted`, or a workflow quota deferral (whose intent is
  archived, not deleted).
- A ledger `push_*`/`submit_*` intent or uncertain state, or any ledger entry
  this queue did not create, stops the queue. `submitted`/scored entries are
  skipped and never resubmitted.
- Before `create-dataset`, `push-kernel` and `submit`, the driver reads a fresh
  account and calls the original `workflow.allowance` (live quota ≤120 s old,
  local UTC-day count, account-wide uncertain refusal).
- `submit` requires exact public acceptance first: 3 studies, 13 columns, the
  expected UIDs, values that are finite and in [0,1], zero fallback, RSS ≤29 GiB,
  and matching checkpoint, manifest and CSV hashes. Only the original
  CreateCodeSubmission is used. The public CSV is never uploaded.
- A run stops after one submission or at any non-run decision.

## Root execution handoff

On the owning node's bounded CPU allocation:

1. Build the exact projection: `scripts/kaggle/project_code.py --package-commit
   <merged main> --training-commit 1bcd6c0e2e2a47d6894d5c545d6950e0c6f0d5b9
   --preserve-training-code`. Record the projection SHA.
2. Write an ignored root config (0600) with: `projection`,
   `projection_manifest`, `projection_sha256`, `ledger`
   (`/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3`), `snapshots`
   (`{"1": .../epoch-001.json, …}` original sidecars), `encoder_spec`,
   `wheelhouse`, `shared_wheelhouse`, `license_file`/`license_source`/
   `license_sha256`, `runtime_image`, `management_python`,
   `reference_fullscale_receipt_sha256` `057b05b2…`,
   `reference_memory_receipt_sha256` `d11d6133…`, `expected_public_uids`, and
   `accepted_prefetch_helper_hashes` copied from the accepted ebeb3c394286
   `INFERENCE_PREFETCH_HASHES`. If the helpers differ, preparation refuses.
3. Fund node/shared storage freshly as in the reference preparation (one
   candidate's bundle at a time).
4. `python -B scripts/kaggle/daily_queue.py --queue <this queue.json>
   --journal-root <node journal> --config <config> --plan-only`, with
   `{"token": …}` on stdin, then without `--plan-only`. Repeat after the public
   notebook completes. Any `read_only_reconciliation` → inspect, run the
   workflow `reconcile-push`/`reconcile-submission` CLI or `daily_queue.reconcile`
   as appropriate. Never delete journals or rerun a phase.
