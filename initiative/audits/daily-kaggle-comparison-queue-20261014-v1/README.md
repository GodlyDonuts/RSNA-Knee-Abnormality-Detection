# Exact daily Kaggle queue for all-development epochs (issue 64)

Code only. No cloud RPC, job, transfer, upload or submission was run while preparing it. Original job 908454, its checkpoints and the numerical sources are unchanged.

## What changed

- `scripts/kaggle/reviewed_infer.py` has a new role, `public_source_all_development_epoch`. It admits one completed scheduled epoch of 908454 and nothing else. It pins source `1bcd6c0e`, run binding `5c8e1473…` (shared by every epoch), counters `epoch / step = 426·epoch / next_update = 0`, 3403 fitting / 0 outer (empty roster), the `label_source_final_v1` overlay `54b5a652`, the Shio-Koube/knee3 CSV and revision, release, folds, official DINOv3-L base, native384, rank 32 and frozen-base delta. It also checks the exact checkpoint SHA, the training-code hash, the private HF prefix and MRI-only inference. The admission must contain exactly the listed keys. Diagnostic, private-research, external-initialization, public-input and unknown-source weights are still refused. The original `infer.validate_inference_checkpoint` then checks a metadata-only view. Weights are never rewritten. The existing effusion and `public_source_final10` paths are unchanged.
- `scripts/kaggle/daily_queue.py` contains:
  - `load_queue` / `admission_for`. Admissions come only from an exact entry of `queue.json`. Each entry must match a committed private full-download `verified.json`: same SHA, byte count, revision, prefix and all 5 members. The receipt itself is pinned by its file SHA.
  - `accept_public`: the mandatory exact public acceptance. It requires the exact checkpoint, manifest and admission; the `two_device_exact_FP32_prefetch_v1` engine; zero fallback; total RSS ≤ 29 GiB; CSV bytes matching both receipts; 13 columns; the expected UID order; and probabilities that are finite and in [0, 1].
  - `plan` / `Driver`: a finite queue that handles one candidate at a time. The phases are `prepare → create-dataset → reconcile-dataset → push-kernel → observe → accept-public → submit → reconcile-outcome`. Before every cloud mutation the driver reads a fresh allowance and the canonical ledger twice; it refuses if they changed and stops when the allowance is 0. Each mutation has an exclusive fsynced intent. On restart, an intent with no result, or any `push_*`, `submit_*`, `pushed` or `submitted` state, leads only to read-only reconciliation or a halt. Nothing is retried or resubmitted. A `failed` outcome halts before the next candidate. A `pending_or_unknown` or scored outcome lets the next candidate proceed. The public three-row CSV is never uploaded.
  - `node_phase`: the owning-node side of the phases. It reuses the original `coordinator.run` (`create-dataset`, `reconcile-dataset`, `push-kernel`, `submit`) and `workflow.status` / `reconcile_push` (version 1) / `reconcile_submission`, plus the reference `prepare_remote` steps: `workflow.py prepare --inference-engine resident_two_gpu_fp32 --fitting-admission`, the FP32 prefetch prefix, `validate` and `prepare-transport`. Before `coordinator.run submit`, `submit` collects the output again and must reproduce the driver's acceptance exactly.
- `tests/test_daily_kaggle_queue.py`: tests for exact admissions; rejection of a changed queue, checkpoint, binding, code, supervision, base, source, counters, private, external, diagnostic, public-input, outer or extra-key; exact public acceptance; one run of each phase; the account limit and per-run cap; crash and restart at prepare, create, push and submit; and existing-state reconciliation.

## Root execution handoff

1. **Commit `queue.json`** in this directory. Epochs 1–3 already have committed receipts. Epochs 4 and 5 need their own `verified.json` receipts committed first, from the original private publication; do not publish again. Format:
   ```json
   {"schema_version":1,"queue_id":"daily-kaggle-comparison-queue-20261014-v1",
    "status":"owner_authorized_exact_daily_kaggle_queue","original_job":908454,
    "scientific_source_git":"1bcd6c0e2e2a47d6894d5c545d6950e0c6f0d5b9",
    "training_binding_sha256":"5c8e14732143a479073b43a24ff7851775a1dd550dc8deb0a9291b8f8091597e",
    "account":"godlydonuts","public_csv_upload":false,"automatic_mutation_retry":false,
    "entries":[{"epoch":1,"checkpoint_sha256":"74a24c64…","checkpoint_bytes":125556557,
      "optimizer_steps":426,"training_code_sha256":"<object_sha(binding.code_sha256)>",
      "private_publication":{"repo_id":"GodlyDonuts/RSNA-Data","revision":"424827b0…","prefix":"…/epoch-001","file_count":5},
      "verified_receipt":"initiative/audits/final-public-source-epoch1-20261013-v1/verified.json",
      "verified_receipt_sha256":"<sha256 of that file>",
      "dataset_slug":"godlydonuts/rsna-alldev-e001-74a24c6468a3",
      "kernel_slug":"godlydonuts/rsna-alldev-offline-e001-74a24c6468a3"}, …]}
   ```
   Take `training_code_sha256` from the actual bound checkpoint: compute `reviewed_infer.object_sha(checkpoint["binding"]["code_sha256"])` read-only on evc40, or copy it from the original run binding. Never guess it. Then run `python scripts/kaggle/daily_queue.py validate-queue`, which must report epochs `[1..5]`.
2. **Projection.** Build it like the reference: `rsna/` must be byte-exact at `1bcd6c0e`, because `validate_snapshot` compares `binding.code_sha256`. `scripts/kaggle/` comes from this PR's merge commit. Stage the reviewed `prepare_envelope` JSON (`tar`, `tar_sha`, `manifest`, `projection_sha`, `encoder`, `helpers`, `fullscale_sha` `057b05b2…`, `memory_sha` `d11d6133…`) on the owning node.
3. **Node config.** Write it as an ignored 0600 file on evc40, where the 908454 checkpoints live. Fields: `node`; `work_root` (e.g. `/tmp/rsna-kaggle-daily-queue-20261014-v1`); `kaggle_base` (`/home/sa305415/rsna-knee/kaggle`, which holds the canonical ledger); `python`; `management_python`; `snapshot_dir` (`/tmp/rsna-final-public-source-fit-evc40-20261013-v1/production/epoch-checkpoints`); `encoder_spec`; `wheelhouse`; `shared_wheelhouse`; `license_file`; `license_source`; `license_sha256`; `runtime_image`; `min_free_bytes`; `prepare_envelope`; `expected_uids` (the three public test UIDs in order). Fund node storage before you prepare.
4. **Transport.** The `--phase-command` is a root-owned argv. It reads the request JSON from stdin, adds `"token"` from the ignored `.env` in RAM, and forwards it over the stable SSH master into one genuine one-CPU, zero-GPU Slurm step on evc40 running `daily_queue.py node-phase --config <file>`. `coordinator.run` refuses GPU allocations.
5. **Run.** `daily_queue.py plan --journal artifacts/daily-kaggle-queue-20261014-v1/journal --phase-command '[…]'` is read-only. Then run `drive` with the same arguments plus `--expected-uids <json>` and optionally `--max-submissions N`. If anything halts or crashes, inspect the journal, ledger and remote state read-only. Never delete an intent to force a repeat.

There is no checkpoint selection from public scores; each epoch is a separate measurement. Public scores are recorded as operational results only.

Root review additions: the actual five entries and all five private verification receipts are now present. Standalone queue Dataset transport uses the existing node-local journal protocol, narrowed to exact daily-queue slugs with explicit fitting-admission hashes; unrelated standalone or revision operations remain rejected. Only two flat durable markers per Dataset reach shared storage. Root tests require an uncertain create to reconcile without a second create and refuse fallback to shared journals. The five-candidate funding decision preserves the20GiB/5000-file floor, retires only the completed twoT4-create future metadata obligation (already-paid markers remain physical) and retains all other obligations. Fresh quota is still required before mutation.

Actual root validation: exact five-entry queue validates;111focused tests and176broader queue/component/old-admission/submission tests pass. The root node-journal regression checks uncertain mutation preservation, one-create-only reconciliation and rejection of unsupported standalone scope. Publication to main waits for the required full checks; no competition submission is claimed.
