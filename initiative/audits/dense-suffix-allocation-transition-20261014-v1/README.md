# Dense-suffix explicit allocation transition v1 (PREPARED, not staged/funded/dispatched)

Issue #73. Original full dense-first production runs inside Slurm 899766 on evc31
(one H100/two CPUs, 13500-second bounded slice, SIGUSR1 -> checkpoint/exit 75). The
parent ends October 14 18:01:47 UTC; extension was denied. `evc31` is MAINT/RESERVED
for cudaCheck. This record prepares an explicit, separately versioned, checkpoint-
continuous transition of the SAME run. It funds no GPU, copies nothing and claims no
accuracy. Root alone stages, funds, transfers, dispatches and publishes, once.

Coordinator: `scripts/newton/transition_dense_suffix_allocation.py`, schema
`dense-suffix-allocation-transition-v1`, staged by root as
`ROOT/allocation-transitions-v1/coordinator.py`. Tests:
`tests/test_dense_suffix_transition.py`.

## Unchanged original proof

`ROOT/launch.py` (5ff1ed77…), `ROOT/publish.py` (5a5878e0…), production funding
(0f5a2d17…), `production-admission.json`, CPU receipt 158c242f…, recovery receipt
cb860282…, source 0ecece12, configs, cache, overlay, folds, official base, runtime and
every original journal stay byte-identical. The original launcher still refuses any
placement other than evc31/899766; the transition never edits or rebinds it. The
canonical trainer `rsna.competitive.train` is unchanged and resumes through the
launcher's own executor with `--resume ROOT/production/dense/checkpoint.pt`.

## Phases (each fail-closed, each journal exclusive and never retried)

1. `record` — zero-GPU step inside original 899766 on evc31, after the original
   launcher has written an ACTUAL `returncode 75, resumable true` terminal whose intent
   and terminal carry job 899766. Refuses an absent, non-contiguous, uncertain (intent
   without terminal), failed, completed or non-resumable terminal, a later arm already
   started, or an existing record. Re-admits through the pinned original launcher
   (`admitted`, `static_admission`, `verify_resumable`, `validate_binding`), then reads
   the complete state: frozen-base-delta schema 2, exact binding SHA, optimizer
   boundary, AdamW state, finite LR, AMP scaler, one RNG stream and
   `step == epoch*281 + next_update` with `0 < step < 2810` (step-0/initial state is
   refused). Writes `001/record.json` once: original funding/admission/CPU receipt/
   launcher/publisher/source/config hashes, terminal/run/invocation/checkpoint SHAs,
   and an exact path/bytes/SHA/mode manifest of the original root, admission tree,
   public cache, overlay, runtime and shared `/home` inputs (artifact studies/folds/
   series and official base) at their ORIGINAL paths. Hashing ~34 GB of node-local
   members must finish before the parent ends.
2. Root separately funds `001/replacement-funding.json`: binds the record, coordinator,
   original funding/admission/CPU receipt and predecessor 899766; a new parent job,
   node, assigned GPU, CPUs, runtime python under the original runtime path, slice and
   invocation bounds, destination envelope, existing liabilities, 20 GiB reserve,
   free-file reserve and zero shared growth. `relocation` must equal `node != evc31`.
3. Relocation only: root transports every node-local member to the IDENTICAL original
   path (no alias rebinding) and writes `001/transport-receipt.json` binding record,
   manifest, source/destination node, member count/bytes, `original_state_deleted
   false`, `automatic_retry false`. Shared `/home` inputs are verified, not copied.
   Owning-node replacement reuses the private inputs in place and must have no receipt.
4. `admit` — zero-GPU step inside the funded replacement parent: exact funding/receipt,
   every member identical with no extra member, byte/file reserve; writes
   `001/destination-admission.json` once.
5. `resume` — GPU step in the replacement parent. Before any intent it rechecks the
   original journal/controls/epoch snapshots, scheduler (job RUNNING on the funded node
   with at least one hour left), node state (MAINT/MAINTENANCE/RESERVED/DRAIN/DOWN never
   bypassed), at most two concurrent GPU nodes, exactly one assigned H100 via
   `SLURM_STEP_GPUS`/`CUDA_VISIBLE_DEVICES`, runtime python, and byte/file reserve. The
   first replacement invocation must see the recorded terminal/run/invocation/checkpoint
   bytes and hard-links the exit-75 checkpoint to `001/exit75-checkpoint-01.pt` before
   the trainer atomically replaces `checkpoint.pt`. Intents/terminals continue the
   invocation numbering under `001/journal/<arm>/`; the original journal is never
   written. Dense then matched suffix-R32 fixed10; completion writes `001/terminal.json`
   with `accuracy_claimed false`.
6. `verify` (read-only) and `publish` — every epoch snapshot is bound to its run binding
   (dense bound to the recorded original binding) with `optimizer_steps == 281*epoch`
   and exact SHA; lineage contiguous. `publish` calls the pinned original publisher
   unchanged (same journal/prefix/admission/funding checks) and only binds its
   one-CPU zero-GPU placement to the admitted replacement node.

## Root's measured feasibility (not funding)

Comment 6169070398: source cache 27,714,909,689 bytes, runtime 6,329,569,496 bytes,
admission tree 1,019,235 bytes; evc40 `/tmp` free 669,135,454,208 bytes; the cache
path does not exist on evc40. Original 908454 runs on evc40; with 899766 on evc31 that
is already two GPU nodes, so a relocation GPU step on evc40 adds no third node.
