# Dense-suffix checkpoint-continuous allocation transition (PREPARED, not staged/dispatched)

Issue 73. The full dense-first production (source 0ecece12, launcher 5ff1ed77…, publisher
5a5878e0…, funding 0f5a2d17…) runs inside original Slurm 899766 on evc31, which ends
October 14 18:01:47 UTC; extension was denied. `scripts/newton/dense_suffix_allocation_transition.py`
(`dense_suffix_allocation_transition_v1`) is a separate coordinator that lets a separately
funded replacement parent continue the exact checkpoint. It dispatches nothing, funds no GPU
and claims no accuracy. The original launcher, its funding/static admission, journals, configs,
checkpoints, source bytes and private epoch publication lineage are re-admitted unchanged and
never rewritten, deleted or rebound; the original launcher keeps refusing the new placement.

Root-only order (none performed here): stage the coordinator as
`ROOT/allocation-transitions/coordinator.py`; after the actual original exit-75 terminal and
after 899766 is no longer live, write `transition-01/replacement-funding.json` for one funded
replacement parent; run `coordinator.py admit` (no CUDA) then bounded `coordinator.py fit
--slice-seconds S` steps inside that parent.

`admit` writes one exclusive `transition.json` only for the latest actual exit-75 terminal with
returncode 75, `resumable: true`, launcher-canonical journal bytes, finite contiguous non-zero
optimizer history, exact checkpoint SHA/bytes, run/invocation SHA and binding SHA, plus original
run/source/config/roster/base/runtime/static-admission/funding hashes and every predecessor
journal byte. Refused: absent, failed, killed, uncertain or step-0 terminals; a still-live
predecessor parent; changed predecessor bytes; stale/changed funding or placement; a duplicate
transition for the same terminal (unless the funding explicitly supersedes one unused
transition); more than 4 transitions.

`fit` re-admits all of that, then before every resumed/fresh GPU intent rechecks the genuine
scheduler record (replacement RUNNING on its node, one GPU, no reservation, predecessor not
live), node state (MAINTENANCE/RESERVED/DRAIN/DOWN never bypassed), at most two GPU nodes
running (evc40 plus the replacement), the assigned single H100, `SLURM_JOB_ID`/CPUs, and byte
plus inode reserve. It resumes only the admitted checkpoint through the unchanged trainer's
`--resume` (AdamW/LR/counters/RNG restored by the trainer); initial-state restart of an admitted
arm is refused. New invocations are journaled only under `transition-kk/journal/`; the matched
suffix-R32 arm then starts fresh in sequence. A later exit 75 may be chained by transition k+1.

Owning-node placement reuses private inputs/runtime in place. Relocation requires root's exact
`transport.json` (every input/runtime/checkpoint file at the same absolute path, size and SHA)
and a matching `destination-admission.json` with `path_rebinding: false`; differing or extra
bytes are refused before the first resumed intent.

Tests: `tests/test_dense_suffix_allocation_transition.py` (77) plus the unchanged
`tests/test_dense_suffix_production.py` (77).
