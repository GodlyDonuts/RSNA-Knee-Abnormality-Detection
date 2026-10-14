# Root handoff: exact all-development epoch queue (issue #64)

Code only. This branch performed no Kaggle/HF/Slurm RPC, transfer, upload,
job or submission. Original job `908454`, its checkpoints and every earlier
Dataset/kernel/ledger version are unchanged.

## What is admitted

`scripts/kaggle/reviewed_infer.py` role `public_source_all_development_epoch`
admits exactly the five epochs in `ALL_DEVELOPMENT_EPOCHS` (checkpoint SHA +
private HF revision) and nothing else. The admission JSON must equal
`all_development_admission(epoch)` byte-for-byte in content; extra or edited
keys are rejected. The checkpoint itself must carry:

- run binding hash `5c8e1473…597e` (production-admission
  `initial_run_binding_sha256`) and rsna module map hash `86db55a2…2b7e`
  (recomputed from the committed `actual-CPU-accepted.json` `code/rsna/*`
  modules; the real checkpoint is the final arbiter, so a mismatch fails closed
  at `validate`)
- source Git `1bcd6c0e…d5b9`, seed 20260922, `epoch == N`, `step == 426·N`,
  `scheduled_epoch`, optimizer boundary, `next_update == 0`
- `fit_scope=all_development`, 3403 fitting / empty outer roster, release,
  folds, source CSV, `label_source_final_v1` overlay `54b5a652…`
- official DINOv3-L base (source id/revision/weights/config SHA), rank 32,
  native 384, frozen-base delta format
- no diagnostic limit, external initialization, private-research flag, public
  input cache or external weight publication

The original `infer.validate_inference_checkpoint` then runs unchanged on a
metadata-only copy with only the fitting-supervision fields removed. Learned
weights and numerical sources are never rewritten.

## Driver

```
python scripts/kaggle/exact_queue.py plan --queue initiative/audits/daily-kaggle-comparison-queue-20261014-v1/queue.json \
    --journal artifacts/daily-kaggle-queue-20261014-v1 --executor <root executor argv>
python scripts/kaggle/exact_queue.py step  …same args…
```

`step` performs at most one phase of the earliest unfinished epoch and exits.
`--advance-after outcome` (default) starts epoch N+1 only after epoch N's
reconciled outcome is `scored`/`failed`/`complete_unscored`;
`--advance-after submitted` starts it once the submission reference is
reconciled. The queue file must list exactly the five owner entries (mutable
`deployment_state` labels are ignored; state comes only from the journal and
canonical ledger).

Phases per epoch, in order: `validate`, `queue`, `prepare`,
`prepare-transport`, `create-dataset`*, `reconcile-dataset`, `push-kernel`*,
`public-acceptance`, `submit`*, `reconcile-submission`. Before each `*`
mutation the driver reads `account` (fresh, ≤120 s) and `ledger` again and
applies the original `workflow.allowance`; any `push_*`/`submit_intent`/
`submit_uncertain` entry anywhere blocks every mutation. `submit` also needs
the driver's own exact public-acceptance record whose manifest equals the
ledger operation manifest.

Restart rules: an interrupted `create-dataset` goes to read-only
`reconcile-dataset`; interrupted `push-kernel` (ledger `push_intent`/
`push_uncertain`) blocks for root `workflow.py reconcile-push`; any
`submit_intent`/`submit_uncertain`/`submitted` ledger entry only ever runs
`reconcile-submission`; interrupted local once-phases (`queue`, `prepare`,
`prepare-transport`) block for root review. Nothing is retried.

## Executor contract (root-owned, ignored, Newton CPU-only)

Reads one JSON request on stdin, prints one JSON object, exits non-zero on any
error. Every result echoes `phase` and `checkpoint_sha256`. Requests carry
`candidate` and the durable `fitting_admission` path written by the driver.

| phase | existing implementation (same as source-e10 `phase_remote.py`) |
| --- | --- |
| `ledger` | read canonical `account-ledger.sqlite3`; return `{"entries": [...]}` |
| `account` | `remote.account()` (GetSubmissionLimits) |
| `validate` | `workflow.py validate --fitting-admission` on the verified HF prefix/revision snapshot |
| `queue` | `workflow.py queue --fitting-admission` |
| `prepare` | `workflow.py prepare --fitting-admission`; return `manifest_sha256` |
| `prepare-transport`, `create-dataset`, `reconcile-dataset`, `push-kernel` | `coordinator.py` phases, management interpreter, single canonical lock, resident `two_device_exact_FP32_prefetch_v1` |
| `public-acceptance` | `workflow.status`; return `kernel_status`, `receipt`, `metrics`, `submission_csv` text and the 3 `expected_public_uids` |
| `submit` | `coordinator.py submit` (one `CreateCodeSubmission` of the notebook) |
| `reconcile-submission` | `workflow.py reconcile-submission` (read-only) |

Kernel identities: one fresh title-derived slug per epoch (e.g.
`rsna-alldev-offline-e01-<sha12>`); never reuse/overwrite an earlier slug.
The public three-row CSV is only read for acceptance and is never uploaded.
