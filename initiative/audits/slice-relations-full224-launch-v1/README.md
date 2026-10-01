# Full224 relation diagnostic launch preparation

The pinned wrapper `scripts/newton/launch_relation_gradient_probe.py` has two
distinct exclusive phases. First a one-CPU/zero-GPU allocation materializes
the exact scientific source and admitted runtime on node-local disk, copies
all original cached series for the first sixteen actual training studies, and
verifies the full224-pixel tensors, geometry, protocol flags, accepted labels
and masks. It creates `cpu-input-verified.json` without constructing an encoder,
taking an optimizer update or evaluating outer studies.

Only a later one-H100/two-CPU measurement phase can use that exact completed
CPU receipt. Its SHA must be supplied by the reviewed GPU launch. That phase
reconstructs the same node-local inputs independently, binds source-manifest
bytes/configs/runtime and rejects any different tensors, label masks or source.
It then runs the three matched gradient controls described in the earlier
slice-relations-gradient-preparation-v1 audit. This is not actual two-epoch
trainer recovery or a competitive accuracy result.

Both phases use a Git source TAR with an exact regular-file/hash membership
map, without Git alternates or package downloads. The accepted cache and
runtime archives remain on authorized private storage. All extracted pixels,
source and the40403-file runtime live on node-local disk and are cleaned after
the phase. The cache copy has a2GiB bound and must preserve20GiB node-local
reserve; the GPU's observations remain224 pixels/six series/sixteen slices.

The live sampling source7ac7a97 is unchanged. No third GPU may be allocated
while both sampling roles run. Require fresh CPU prerequisite/source/byte/file
funding before staging. A stage, dispatch or mode runtime marker is single-use;
reconcile actual jobs and receipts before any further mutation. Use a frozen
copy of the accepted original training-run binding so its later terminal
metrics cannot invalidate the CPU→GPU input record.

Thirty-four local tests cover actual derivative normalization/control replay,
strict source byte admission and the new bootstrap/CPU→GPU binding. These are
synthetic software tests. At this source-preparation milestone no actual CPU
phase or H100 phase has yet been dispatched; subsequent original dispatch and
terminal receipts, rather than this statement, establish later state.

## Actual CPU admission and dependent GPU dispatch, October 1

CPU job866843 completed0:0 in10:11 on evc21. The exact source is
4acd6fa57a193cf8a4b5bb813866d98d358e1e5a (48 regular source files).
It materialized317,211,763 bytes of original cache on node-local disk and decoded
16 real accepted training studies at224 pixels. The original runtime, complete
training roster, cache envelopes, classification labels/masks, geometry and all
sixteen tensor hashes passed. Exact CPU receipt SHA is
a6e6c02db933388b9c65242f31785f0cf60aa41f7b828933c0f81993748db1d5.
See terminal/ and the unchanged original stage/dispatch journals. No official
encoder was constructed in CPU input mode, and no optimizer update occurred.

One H100/two-CPU,32GiB,one-hour diagnostic job866846 was submitted ONCE. It is
PENDING for afterok:866716 on evc32, so it acquires no third GPU and leaves both
sampling roles intact. Exact source/control and CPU receipt are bound by the
original gpu/ journals/coordinator; fresh funding leaves20.7199GiB/5742 files
after all current obligations. Never repeat this dispatch or overwrite its
control. GPU acceptance, actual gradient measurements, trainer recovery and
clinical accuracy remain outstanding. A failed parent will leave the dependency
unsatisfied; reconcile actual state before changing anything.

The later GPU run must reproduce CPU tensor/source/config hashes before loading
the encoder. It compares zero/physical/shuffled gradients at a common initial
state on sixteen training studies, with zero optimizer updates and no checkpoint.
This is a loss-scale diagnostic, not production throughput or held-out accuracy.
