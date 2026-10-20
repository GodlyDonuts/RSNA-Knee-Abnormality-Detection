# Prepared deadline admission and preservation separation

**Local preparation only.** No owner is launched or modified, and this code cannot
dispatch, cancel, transfer or publish. Existing frozen science `44cc559e` and live
owners remain unchanged. Parent must integrate and freeze any operational
successor after reconciling the original idle owner.

The current portable owner accepts cache completion until October 22 09:00 UTC
(`predictive-c0-portable-placement-20261020-v1/owner.py:284–300`), despite
eight-hour training jobs and a preceding C0 transfer. That timestamp is a
preservation-owner expiry, not a safe latest training start. Current cache
progress does not establish that this late-arrival failure will occur.

`policy.py` computes separate stage cutoffs from the fixed **October 22 noon UTC
latest entry target**. Callers supply explicit wall-clock allowances for
qualification/public execution, scheduler waiting, context transfer, admission,
and final preservation/return. B/C and C0 training run concurrently; the policy
uses the longer path, not three sequential eight-hour fits. It also rejects a
new launch when the proposed preservation owner expires before final readiness.
Its result is prospective and never grants mutation authority.

An illustrative planning budget in the tests uses:

- Three hours for deployment, retaining final readiness by October 22 09:00 UTC.
- Eight hours per training job, two hours scheduler allowance and one hour for
  final publication/return.
- Ninety minutes initial B/C CPU admission/dispatch; 150 minutes C0 transfer and
  45 minutes C0 dispatch control.

Those values yield cache-admission cutoff **October 21 17:15 UTC**, transfer
cutoff **October 21 18:45 UTC**, and already-prepared B/C or C0 dispatch cutoff
**October 21 22:00 UTC**. These are proposed allowances, not an accepted
operational decision or guarantees about queue time. Root may explicitly select
different justified allowances before integration. The 3,360-second free
GPU-VM requirement remains a separate account check.

After the cutoff, preserve and observe every already-submitted original job.
Unknown state requires read-only reconciliation. A timeout preserves completed
epochs and records the failure; it does not authorize an automatic restart.
The training core supports exact resume, but the current production worker
does not wire a resume checkpoint into its `fit()` call.

## Minimal operational successor boundaries

1. Keep the existing exact-once `invoke` and transfer intent/result interfaces.
   Invoke the pure stage assessment before *new* admission/dispatch/transfer
   intents. Keep mutation failure handling separate from read-only polls.
2. After the original B/C submission result is durable, create a persistent
   dispatched-job map immediately. Enter its observation/publication loop even
   if C0 has not been submitted.
3. Run the one-time C0 transfer/admission/dispatch path as a distinct placement
   task owned by the same finite coordinator. Only that task may create C0
   intents; the main loop alone publishes epochs. Record a typed placement
   failure/uncertain outcome and stop C0 mutations without terminating B/C
   observation/publication. Never retry a transfer with an existing intent.
   The transport/CPU controls already accept immutable payloads; use separate
   payload copies rather than sharing mutable `all_dispatched_jobs` between
   tasks. Do not run two node/account publications simultaneously.
4. Observe only actual durable dispatches. A missing or failed C0 placement is
   a separate outcome, not a fictitious failed GPU job. Keep the read-only
   first-200 observer on those same exact dispatch receipts.
5. Once C0 completes and its original six publication proofs are verified,
   return its final objects independently of another arm still running.
   Record per-arm final/failed state while preserving fixed-final comparisons.
6. Use one cutoff for new launch stages and a later finite preservation expiry.
   If a healthy job remains active at preservation expiry, retain its exact
   journal and emit a continuation-needed receipt; never cancel it or erase its
   outputs. A root-owned successor must reconcile before taking ownership.

The current synchronous `self.transfer()` at portable owner line 299 precedes
the publication loop. A transfer exception therefore stops B/C preservation
despite valid running B/C jobs. The independent observer still reads them but
cannot publish; the proposed boundary removes this coupling without changing
the model, optimizer, input bytes, initialized tensors or scientific recipe.

`placement_failure.py::continuation_after_exception` is the executable narrow
transition for those boundaries. It takes only already-verified durable B/C
(and, if known, C0) dispatch receipts. It records the C0 **operation as
unresolved**, never invents a failed C0 model or Slurm job, and keeps the exact
known jobs in the observation/publication set. All automatic C0 transfer,
dispatch, return and training retries remain false.

Future integration anchor: frozen `926b6577` portable `Owner.run()` immediately
after line 298 establishes the B/C job map; lines 299–302 perform C0 placement.
An operational successor can catch only that placement task's exception,
durably record this transition once, and continue the existing publication loop
over known job maps. It must not let this handler swallow a B/C publication
failure, make up an uncertain C0 job ID, or restart the failed operation.
Successful jobs still need their actual final proofs. No live code currently
uses this prepared transition.
