# Prepared resilient predictive fitting orchestration

This is an executable **local preparation**, not evidence of a launch. Root owns
the exclusive idle-owner handoff, frozen launcher and any actual start.
`fitting.run(...)` replaces only orchestration. Scientific source `44cc559e`,
the original HTTP program and payload attribution `2b834507`, initialized
tensors, worker, publisher, recipe, original journal and output paths remain
unchanged. Existing deployment and read-only observer consumers retain their
original receipt interfaces.

The entry is **prephase-only**: it requires the original owner intent and refuses
existing phase intents, results, transports, prior deadline decisions or an
ended owner. It cannot resume a partly executed controller.

After B/C submission, the main thread durably records their exact job map before
starting one isolated C0 placement thread. Each remote call gets its own deep
payload copy, including actual B/C dispatches for paid-log accounting. C0 uses
the original transfer/dispatch interfaces once. Only the main thread merges a
verified C0 dispatch and publishes epochs, so publication remains serial.

- C0 placement exceptions are unresolved **operations**, never invented failed
  GPU jobs. B/C monitoring and publication continue.
- Per-node progress exceptions retain one unresolved-observation receipt per
  incident. Other nodes continue; only read-only observation is retried.
  Recovered observations are recorded separately. Stale observations cannot
  establish that every known job has finished.
- Any publication exception retains its original intent and stops further
  publication mutations. Observation of submitted jobs continues; no upload is
  retried. Final return exceptions are likewise preserved without repetition.
- Six original verified C0 epoch proofs precede its one-time final return.
  Training, transfer and submission retries remain disabled.

New-stage cutoff calculations use the prepared pure policy from `15863fa1`:
three hours deployment, eight hours per parallel training job, two hours
scheduler allowance, one hour final publication/return, 90 minutes initial CPU
controls, 150 minutes C0 transfer and 45 minutes C0 dispatch control. The
preservation loop remains finite until **October 22 noon UTC**, separately from
the earlier stop-new-stage decisions.

These cutoffs govern **invoking** a new stage. The unchanged remote program may
wait for a CPU allocation before submitting a GPU job. The two-hour scheduler
allowance covers combined CPU-control and GPU scheduling as a planning estimate;
it is not an absolute bound or a guarantee of actual `sbatch`/GPU start time.
No healthy submitted job is cancelled when a launch cutoff passes. At the
preservation expiry, exact originals and unresolved operation/thread state are
retained for read-only reconciliation.

Focused tests run the actual orchestration against fake phase services. They
cover C0 blocking while B/C publishes, transfer/dispatch/return uncertainty,
per-node parser failure and recovery, serial publication, payload isolation,
deadline crossings, no invented jobs, and refusal of pre-existing phases.
They are operational tests, not training or accuracy evidence.
