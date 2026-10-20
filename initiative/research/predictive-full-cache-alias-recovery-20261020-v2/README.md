# Alias recovery after the purged-job queue query

Prepared locally; **not dispatched**. This distinct successor preserves the
failed v1 attempt and its journal.

V1 raw job **979573** failed before creating its output directory because
`squeue -j 978318` returned an error after the terminal job was purged from
Slurm's live controller. CPU **979578** independently verified both v1 output
roots absent. No source/cache work was performed by that attempt.

All queue reads now use `squeue -u sa305415` and filter exact requested job IDs.
Accounting still must contain each exact terminal job; an unrelated running
job cannot invalidate a completed job, and a queued target still prohibits
recovery. No missing accounting result is interpreted as success.

The launcher adds required `--prior-attempt-receipt` and
`--prior-attempt-sha256` arguments to the v1 regression-bound launch contract.
It requires actual 979573 FAILED 1:0 and the byte-bound absence receipt.
Both workers independently recheck the prior terminal and absent v1 roots
before creating their distinct v2 roots:

- `/tmp/rsna-predictive-full-raw-alias-recovery-20261020-v2`
- `/tmp/rsna-predictive-full-context-alias-recovery-20261020-v2`

The source correction, retained-prefix/orphan/missing-only behavior, mixed
original/corrected context provenance, two raw lanes/four CPUs and one context
CPU, 20-hour limits and 30-hour finite owner are unchanged from v1.
The existing 121 MiB/376-file shared and 62 GiB + 594 MiB other-node reservations
are reused; v1 created no node output allocation. Raw/context/scratch bounds
and free reserves still apply.

Only this new directory receives future `actual-owner-raw_collect.json` and
`actual-owner-context_collect.json` receipts. V1 controls and failed receipts
remain immutable. Nothing here allocates a GPU or changes model numerics.
